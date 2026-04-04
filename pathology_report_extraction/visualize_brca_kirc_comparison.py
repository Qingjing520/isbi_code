from __future__ import annotations

"""Render a BRCA vs KIRC hierarchy overview figure from real graph JSON files."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Patch, Rectangle

from pdf_utils import ensure_dir


DEFAULT_BRCA_JSON = Path(
    r"D:\Tasks\isbi_code\pathology_report_extraction\Output\text_hierarchy_graphs_masked\BRCA\TCGA-AQ-A04L.40B58A7A-2A05-4AAF-B1D4-B418E4A56115.json"
)
DEFAULT_KIRC_JSON = Path(
    r"D:\Tasks\isbi_code\pathology_report_extraction\Output\text_hierarchy_graphs_masked\KIRC\TCGA-6D-AA2E.2F9C29AE-B1C6-4A2B-9CC8-E5113138F864.json"
)
DEFAULT_OUTPUT_PNG = Path(
    r"D:\Tasks\isbi_code\pathology_report_extraction\Output\graph_visualizations\BRCA_vs_KIRC_hierarchy_overview.png"
)
DEFAULT_OUTPUT_MD = Path(
    r"D:\Tasks\isbi_code\pathology_report_extraction\Output\graph_visualizations\BRCA_vs_KIRC_hierarchy_overview.md"
)


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


def get_section_nodes(payload: dict) -> list[dict]:
    return [node for node in payload["nodes"] if node["node_type"] == "section"]


def draw_arrow(ax, start: tuple[float, float], end: tuple[float, float], *, color: str, dashed: bool = False) -> None:
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

    # Document node
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

    # Section nodes
    section_y = 0.49
    sec_gap = panel_width / max(len(section_nodes), 1)
    sec_xs = [panel_left + sec_gap * (i + 0.5) for i in range(len(section_nodes))]

    for sec_x, section in zip(sec_xs, section_nodes):
        circle = Circle((sec_x, section_y), 0.035, facecolor=COLORS["section"], edgecolor=COLORS["text_dark"], linewidth=1.1)
        ax.add_patch(circle)
        label = f'{section["section_title"]}\n({section["sentence_count"]})'
        ax.text(sec_x, section_y, label, ha="center", va="center", fontsize=8.6, fontweight="bold")
        draw_arrow(ax, (x_center, doc_y), (sec_x, section_y + 0.035), color=COLORS["parent"])

    # Next edges between sections
    for left_x, right_x in zip(sec_xs[:-1], sec_xs[1:]):
        draw_arrow(ax, (left_x + 0.04, section_y), (right_x - 0.04, section_y), color=COLORS["next"], dashed=True)

    # Sentence nodes per section
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


def build_mermaid(brca_payload: dict, kirc_payload: dict) -> str:
    brca_sections = [node["section_title"] for node in get_section_nodes(brca_payload)]
    kirc_sections = [node["section_title"] for node in get_section_nodes(kirc_payload)]
    lines = [
        "```mermaid",
        "flowchart LR",
        '    A["Shared Structure"] --> B["Document"]',
        '    B --> C["Section"]',
        '    C --> D["Sentence"]',
        '    E["BRCA Representative"] --> F["' + "<br/>".join(brca_sections) + '"]',
        '    G["KIRC Representative"] --> H["' + "<br/>".join(kirc_sections) + '"]',
        "```",
    ]
    return "\n".join(lines)


def write_summary_md(brca_payload: dict, kirc_payload: dict, output_md: Path, output_png: Path) -> None:
    brca_sections = [node["section_title"] for node in get_section_nodes(brca_payload)]
    kirc_sections = [node["section_title"] for node in get_section_nodes(kirc_payload)]

    md = f"""# BRCA vs KIRC Hierarchy Overview

- BRCA representative: `{brca_payload["document_id"]}`
- KIRC representative: `{kirc_payload["document_id"]}`
- Shared node levels: `Document -> Section -> Sentence`
- Shared edge types: `parent`, `next`
- BRCA sections in this representative sample: `{", ".join(brca_sections)}`
- KIRC sections in this representative sample: `{", ".join(kirc_sections)}`
- Figure: `{output_png.name}`

{build_mermaid(brca_payload, kirc_payload)}
"""
    ensure_dir(output_md.parent)
    output_md.write_text(md, encoding="utf-8")


def draw_comparison(brca_payload: dict, kirc_payload: dict, output_png: Path) -> None:
    ensure_dir(output_png.parent)
    fig, ax = plt.subplots(figsize=(16, 9), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.set_facecolor("white")

    ax.text(
        0.5,
        0.965,
        "BRCA vs KIRC Pathology Report Hierarchy Overview",
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
        color=COLORS["text_dark"],
    )
    ax.text(
        0.5,
        0.93,
        "Real representative samples from the generated text hierarchy graph outputs",
        ha="center",
        va="top",
        fontsize=11,
        color=COLORS["muted"],
    )

    draw_dataset_panel(ax, brca_payload, x_center=0.26, panel_width=0.43, panel_title="BRCA")
    draw_dataset_panel(ax, kirc_payload, x_center=0.74, panel_width=0.43, panel_title="KIRC")

    legend_handles = [
        Patch(facecolor=COLORS["document"], edgecolor=COLORS["text_dark"], label="Document node"),
        Patch(facecolor=COLORS["section"], edgecolor=COLORS["text_dark"], label="Section node"),
        Patch(facecolor=COLORS["sentence"], edgecolor=COLORS["text_dark"], label="Sentence node"),
        Line2D([0], [0], color=COLORS["parent"], lw=1.6, label="Parent edge"),
        Line2D([0], [0], color=COLORS["next"], lw=1.2, linestyle="--", label="Next edge"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=5,
        frameon=False,
        fontsize=10,
    )

    plt.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a BRCA vs KIRC hierarchy comparison figure.")
    parser.add_argument("--brca_json", type=Path, default=DEFAULT_BRCA_JSON, help="Representative BRCA graph JSON.")
    parser.add_argument("--kirc_json", type=Path, default=DEFAULT_KIRC_JSON, help="Representative KIRC graph JSON.")
    parser.add_argument("--output_png", type=Path, default=DEFAULT_OUTPUT_PNG, help="Output PNG path.")
    parser.add_argument("--output_md", type=Path, default=DEFAULT_OUTPUT_MD, help="Output Markdown path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    brca_payload = load_payload(args.brca_json)
    kirc_payload = load_payload(args.kirc_json)
    draw_comparison(brca_payload, kirc_payload, args.output_png)
    write_summary_md(brca_payload, kirc_payload, args.output_md, args.output_png)
    print(f"Comparison PNG written to: {args.output_png}")
    print(f"Comparison Markdown written to: {args.output_md}")


if __name__ == "__main__":
    main()
