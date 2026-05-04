from __future__ import annotations

"""Visualize pathology report hierarchy graph JSON files.

This replaces the older single-purpose visualization scripts with one entry
point that can render either one graph or a two-dataset comparison figure.
"""

import argparse
import json
from pathlib import Path
from textwrap import shorten

from pathology_report_extraction.common.pdf_utils import ensure_dir
from pathology_report_extraction.common.pipeline_defaults import DEFAULT_HIERARCHY_GRAPH_ROOT, DEFAULT_OUTPUT_ROOT

DEFAULT_GRAPH_ROOT = DEFAULT_HIERARCHY_GRAPH_ROOT
DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "graph_visualizations"


def _load_plot_deps():
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle, FancyArrowPatch, Patch, Rectangle
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Visualization requires matplotlib and networkx. Install them before rendering hierarchy graph figures."
        ) from exc

    return plt, nx, Line2D, Circle, FancyArrowPatch, Patch, Rectangle

NODE_STYLE = {
    "document": {"color": "#4C78A8", "size": 3000, "shape": "s"},
    "section": {"color": "#72B7B2", "size": 2200, "shape": "o"},
    "sentence": {"color": "#F2CF5B", "size": 520, "shape": "o"},
}

COLORS = {
    "document": "#4C78A8",
    "section": "#72B7B2",
    "sentence": "#F2CF5B",
    "parent": "#2F2F2F",
    "next": "#9A9A9A",
    "panel_bg": "#FBFBFB",
    "panel_edge": "#D7D7D7",
    "text_dark": "#1F1F1F",
    "muted": "#555555",
}


def load_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def find_first_graph(dataset: str | None = None) -> Path:
    search_root = DEFAULT_GRAPH_ROOT / dataset if dataset else DEFAULT_GRAPH_ROOT
    candidates = sorted(search_root.rglob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"No graph JSON files found under {search_root}")
    return candidates[0]


def get_section_nodes(payload: dict) -> list[dict]:
    return [node for node in payload["nodes"] if node["node_type"] == "section"]


def truncate_text(text: str, width: int = 52) -> str:
    return shorten(" ".join((text or "").split()), width=width, placeholder="...")


def build_graph(payload: dict) -> tuple[nx.DiGraph, dict[int, tuple[float, float]], dict[int, str]]:
    _, nx, _, _, _, _, _ = _load_plot_deps()
    graph = nx.DiGraph()
    nodes = payload["nodes"]
    edges = payload["edges"]

    document_node = next(node for node in nodes if node["node_type"] == "document")
    section_nodes = get_section_nodes(payload)
    sentence_nodes = [node for node in nodes if node["node_type"] == "sentence"]

    sentences_by_section: dict[int, list[dict]] = {}
    for sentence_node in sentence_nodes:
        section_index = sentence_node["section_index"]
        sentences_by_section.setdefault(section_index, []).append(sentence_node)

    positions: dict[int, tuple[float, float]] = {}
    labels: dict[int, str] = {}

    graph.add_node(document_node["node_index"], **document_node)
    positions[document_node["node_index"]] = (0.0, 2.2)
    labels[document_node["node_index"]] = f'Document\n{payload["document_id"]}'

    section_count = max(len(section_nodes), 1)
    section_gap = 2.8
    x_start = -((section_count - 1) * section_gap) / 2.0

    for idx, section_node in enumerate(section_nodes):
        node_index = section_node["node_index"]
        section_x = x_start + idx * section_gap
        positions[node_index] = (section_x, 1.0)
        labels[node_index] = (
            f'{section_node["section_title"]}\n'
            f'({section_node["sentence_count"]} sentences)'
        )
        graph.add_node(node_index, **section_node)

        section_sentences = sentences_by_section.get(section_node["section_index"], [])
        sentence_count = max(len(section_sentences), 1)
        sentence_gap = 0.52 if sentence_count > 1 else 0.0
        sentence_x_start = section_x - ((sentence_count - 1) * sentence_gap) / 2.0

        for sent_offset, sentence_node in enumerate(section_sentences):
            sent_idx = sentence_node["node_index"]
            sent_x = sentence_x_start + sent_offset * sentence_gap
            positions[sent_idx] = (sent_x, -0.18)
            labels[sent_idx] = ""
            graph.add_node(sent_idx, **sentence_node)

    for edge in edges:
        graph.add_edge(
            edge["source_index"],
            edge["target_index"],
            edge_type=edge["edge_type"],
            source_type=edge["source_type"],
            target_type=edge["target_type"],
        )

    return graph, positions, labels


def draw_single_graph(payload: dict, output_png: Path) -> None:
    plt, nx, Line2D, _, _, Patch, _ = _load_plot_deps()
    graph, positions, labels = build_graph(payload)
    ensure_dir(output_png.parent)

    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=180)
    ax.set_facecolor("#FBFBFB")

    parent_edges = [(u, v) for u, v, d in graph.edges(data=True) if d["edge_type"] == "parent"]
    next_edges = [(u, v) for u, v, d in graph.edges(data=True) if d["edge_type"] == "next"]

    nx.draw_networkx_edges(graph, positions, edgelist=parent_edges, edge_color="#2F2F2F", width=1.8, arrows=False, ax=ax)
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=next_edges,
        edge_color="#A0A0A0",
        width=1.2,
        style="dashed",
        alpha=0.9,
        arrows=False,
        ax=ax,
    )

    for node_type, style in NODE_STYLE.items():
        node_ids = [n for n, attrs in graph.nodes(data=True) if attrs["node_type"] == node_type]
        nx.draw_networkx_nodes(
            graph,
            positions,
            nodelist=node_ids,
            node_color=style["color"],
            node_size=style["size"],
            node_shape=style["shape"],
            linewidths=1.2,
            edgecolors="#333333",
            ax=ax,
        )

    nx.draw_networkx_labels(
        graph,
        positions,
        labels=labels,
        font_size=9,
        font_weight="bold",
        font_family="DejaVu Sans",
        ax=ax,
    )

    sentence_preview = []
    for node in payload["nodes"]:
        if node["node_type"] == "sentence" and len(sentence_preview) < 3:
            sentence_preview.append(f'S{node["sentence_index"] + 1}: {truncate_text(node.get("text", ""), 70)}')

    title = (
        "Representative Pathology Report Hierarchy Graph\n"
        f'{payload["dataset"]} | {payload["document_id"]} | '
        f'{payload["node_counts"]["section"]} sections | {payload["node_counts"]["sentence"]} sentences'
    )
    ax.set_title(title, fontsize=15, fontweight="bold", pad=18)

    legend_items = [
        Patch(facecolor=NODE_STYLE["document"]["color"], edgecolor="#333333", label="Document node"),
        Patch(facecolor=NODE_STYLE["section"]["color"], edgecolor="#333333", label="Section node"),
        Patch(facecolor=NODE_STYLE["sentence"]["color"], edgecolor="#333333", label="Sentence node"),
        Line2D([0], [0], color="#2F2F2F", lw=1.8, label="Parent edge"),
        Line2D([0], [0], color="#A0A0A0", lw=1.2, linestyle="--", label="Next edge"),
    ]
    ax.legend(handles=legend_items, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=10)

    preview_text = "Sentence examples:\n" + "\n".join(sentence_preview)
    ax.text(
        1.01,
        0.42,
        preview_text,
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        ha="left",
        linespacing=1.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#FFFFFF", "edgecolor": "#DDDDDD"},
    )

    ax.text(
        0.0,
        -0.12,
        "Layout uses the real graph structure from the generated hierarchy graph JSON.\n"
        "Document at top, sections in middle, sentences at bottom.",
        transform=ax.transAxes,
        fontsize=9,
        color="#444444",
        ha="left",
        va="top",
    )

    ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def draw_arrow(ax, start: tuple[float, float], end: tuple[float, float], *, color: str, dashed: bool = False) -> None:
    _, _, _, _, FancyArrowPatch, _, _ = _load_plot_deps()
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-",
        mutation_scale=1.0,
        linewidth=1.6 if not dashed else 1.2,
        linestyle="--" if dashed else "-",
        color=color,
        alpha=0.95 if not dashed else 0.9,
    )
    ax.add_patch(patch)


def draw_dataset_panel(ax, payload: dict, *, x_center: float, panel_width: float, panel_title: str) -> None:
    _, _, _, Circle, _, _, Rectangle = _load_plot_deps()
    section_nodes = get_section_nodes(payload)
    document_id = payload["document_id"]
    section_count = payload["node_counts"]["section"]
    sentence_count = payload["node_counts"]["sentence"]

    panel_left = x_center - panel_width / 2
    panel_bottom = 0.06
    panel_height = 0.84

    panel = Rectangle(
        (panel_left, panel_bottom),
        panel_width,
        panel_height,
        facecolor=COLORS["panel_bg"],
        edgecolor=COLORS["panel_edge"],
        linewidth=1.0,
    )
    ax.add_patch(panel)

    ax.text(
        x_center,
        0.88,
        f"{panel_title}\n{document_id}",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color=COLORS["text_dark"],
    )
    ax.text(
        x_center,
        0.82,
        f"{section_count} sections | {sentence_count} sentences",
        ha="center",
        va="center",
        fontsize=10.5,
        color=COLORS["muted"],
    )

    doc_w, doc_h = 0.09, 0.085
    doc_x, doc_y = x_center - doc_w / 2, 0.68
    doc = Rectangle(
        (doc_x, doc_y),
        doc_w,
        doc_h,
        facecolor=COLORS["document"],
        edgecolor=COLORS["text_dark"],
        linewidth=1.2,
    )
    ax.add_patch(doc)
    ax.text(x_center, doc_y + doc_h / 2, "Document", ha="center", va="center", fontsize=10, fontweight="bold")

    section_y = 0.49
    sec_gap = panel_width / max(len(section_nodes), 1)
    sec_xs = [panel_left + sec_gap * (i + 0.5) for i in range(len(section_nodes))]

    for sec_x, section in zip(sec_xs, section_nodes):
        circle = Circle((sec_x, section_y), 0.035, facecolor=COLORS["section"], edgecolor=COLORS["text_dark"], linewidth=1.1)
        ax.add_patch(circle)
        label = f'{section["section_title"]}\n({section["sentence_count"]})'
        ax.text(sec_x, section_y, label, ha="center", va="center", fontsize=8.6, fontweight="bold")
        draw_arrow(ax, (x_center, doc_y), (sec_x, section_y + 0.035), color=COLORS["parent"])

    for left_x, right_x in zip(sec_xs[:-1], sec_xs[1:]):
        draw_arrow(ax, (left_x + 0.04, section_y), (right_x - 0.04, section_y), color=COLORS["next"], dashed=True)

    sentence_y = 0.19
    for sec_x, section in zip(sec_xs, section_nodes):
        count = section["sentence_count"]
        visible = min(count, 8)
        gap = 0.035
        start_x = sec_x - ((visible - 1) * gap) / 2
        sentence_positions = []
        for idx in range(visible):
            sx = start_x + idx * gap
            sentence_positions.append((sx, sentence_y))
            circ = Circle((sx, sentence_y), 0.012, facecolor=COLORS["sentence"], edgecolor=COLORS["text_dark"], linewidth=0.9)
            ax.add_patch(circ)

        if sentence_positions:
            for pos in sentence_positions:
                draw_arrow(ax, (sec_x, section_y - 0.037), (pos[0], pos[1] + 0.013), color=COLORS["parent"])
            for left_pos, right_pos in zip(sentence_positions[:-1], sentence_positions[1:]):
                draw_arrow(ax, (left_pos[0] + 0.014, left_pos[1]), (right_pos[0] - 0.014, right_pos[1]), color=COLORS["next"], dashed=True)

        if count > visible:
            ax.text(sec_x, 0.13, f"... {count - visible} more", ha="center", va="center", fontsize=8.5, color=COLORS["muted"])


def build_mermaid(left_payload: dict, right_payload: dict, left_label: str, right_label: str) -> str:
    left_sections = [node["section_title"] for node in get_section_nodes(left_payload)]
    right_sections = [node["section_title"] for node in get_section_nodes(right_payload)]
    lines = [
        "```mermaid",
        "flowchart LR",
        '    A["Shared Structure"] --> B["Document"]',
        '    B --> C["Section"]',
        '    C --> D["Sentence"]',
        f'    E["{left_label} Representative"] --> F["' + "<br/>".join(left_sections) + '"]',
        f'    G["{right_label} Representative"] --> H["' + "<br/>".join(right_sections) + '"]',
        "```",
    ]
    return "\n".join(lines)


def write_comparison_md(left_payload: dict, right_payload: dict, output_md: Path, output_png: Path, left_label: str, right_label: str) -> None:
    left_sections = [node["section_title"] for node in get_section_nodes(left_payload)]
    right_sections = [node["section_title"] for node in get_section_nodes(right_payload)]

    md = f"""# {left_label} vs {right_label} Hierarchy Overview

- {left_label} representative: `{left_payload["document_id"]}`
- {right_label} representative: `{right_payload["document_id"]}`
- Shared node levels: `Document -> Section -> Sentence`
- Shared edge types: `parent`, `next`
- {left_label} sections in this representative sample: `{", ".join(left_sections)}`
- {right_label} sections in this representative sample: `{", ".join(right_sections)}`
- Figure: `{output_png.name}`

{build_mermaid(left_payload, right_payload, left_label, right_label)}
"""
    ensure_dir(output_md.parent)
    output_md.write_text(md, encoding="utf-8")


def draw_comparison(left_payload: dict, right_payload: dict, output_png: Path, left_label: str, right_label: str) -> None:
    plt, _, Line2D, _, _, Patch, _ = _load_plot_deps()
    ensure_dir(output_png.parent)
    fig, ax = plt.subplots(figsize=(16, 9), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.set_facecolor("white")

    ax.text(
        0.5,
        0.965,
        f"{left_label} vs {right_label} Pathology Report Hierarchy Overview",
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
        color=COLORS["text_dark"],
    )
    ax.text(
        0.5,
        0.93,
        "Real representative samples from generated text hierarchy graph outputs",
        ha="center",
        va="top",
        fontsize=11,
        color=COLORS["muted"],
    )

    draw_dataset_panel(ax, left_payload, x_center=0.26, panel_width=0.43, panel_title=left_label)
    draw_dataset_panel(ax, right_payload, x_center=0.74, panel_width=0.43, panel_title=right_label)

    legend_handles = [
        Patch(facecolor=COLORS["document"], edgecolor=COLORS["text_dark"], label="Document node"),
        Patch(facecolor=COLORS["section"], edgecolor=COLORS["text_dark"], label="Section node"),
        Patch(facecolor=COLORS["sentence"], edgecolor=COLORS["text_dark"], label="Sentence node"),
        Line2D([0], [0], color=COLORS["parent"], lw=1.6, label="Parent edge"),
        Line2D([0], [0], color=COLORS["next"], lw=1.2, linestyle="--", label="Next edge"),
    ]
    ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=5, frameon=False, fontsize=10)

    plt.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize text hierarchy graph JSON outputs.")
    parser.add_argument("mode", choices=("single", "compare"), nargs="?", default="single")
    parser.add_argument("--graph_json", type=Path, default=None, help="Single graph JSON to render.")
    parser.add_argument("--brca_json", type=Path, default=None, help="BRCA graph JSON for comparison mode.")
    parser.add_argument("--kirc_json", type=Path, default=None, help="KIRC graph JSON for comparison mode.")
    parser.add_argument("--left_label", default="BRCA", help="Left panel label in comparison mode.")
    parser.add_argument("--right_label", default="KIRC", help="Right panel label in comparison mode.")
    parser.add_argument("--output_png", type=Path, default=None, help="Output PNG path.")
    parser.add_argument("--output_md", type=Path, default=None, help="Output Markdown path for comparison mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "single":
        graph_json = args.graph_json or find_first_graph("KIRC")
        payload = load_payload(graph_json)
        output_png = args.output_png or DEFAULT_OUTPUT_DIR / f"{graph_json.stem}_hierarchy_graph.png"
        draw_single_graph(payload, output_png)
        print(f"Graph PNG written to: {output_png}")
        return

    left_json = args.brca_json or find_first_graph(args.left_label)
    right_json = args.kirc_json or find_first_graph(args.right_label)
    left_payload = load_payload(left_json)
    right_payload = load_payload(right_json)
    output_png = args.output_png or DEFAULT_OUTPUT_DIR / f"{args.left_label}_vs_{args.right_label}_hierarchy_overview.png"
    output_md = args.output_md or DEFAULT_OUTPUT_DIR / f"{args.left_label}_vs_{args.right_label}_hierarchy_overview.md"

    draw_comparison(left_payload, right_payload, output_png, args.left_label, args.right_label)
    write_comparison_md(left_payload, right_payload, output_md, output_png, args.left_label, args.right_label)
    print(f"Comparison PNG written to: {output_png}")
    print(f"Comparison Markdown written to: {output_md}")


if __name__ == "__main__":
    main()
