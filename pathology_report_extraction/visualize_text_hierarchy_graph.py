from __future__ import annotations

"""将文本层次图 JSON 渲染成直观的层次结构图片。

默认读取一份代表性样本，输出 PNG 到 Output/graph_visualizations。
"""

import argparse
import json
from pathlib import Path
from textwrap import shorten

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pdf_utils import ensure_dir


DEFAULT_GRAPH_JSON = Path(
    r"D:\Tasks\isbi_code\pathology_report_extraction\Output\text_hierarchy_graphs_masked\KIRC\TCGA-6D-AA2E.2F9C29AE-B1C6-4A2B-9CC8-E5113138F864.json"
)
DEFAULT_OUTPUT_DIR = Path(
    r"D:\Tasks\isbi_code\pathology_report_extraction\Output\graph_visualizations"
)

NODE_STYLE = {
    "document": {"color": "#4C78A8", "size": 3000, "shape": "s"},
    "section": {"color": "#72B7B2", "size": 2200, "shape": "o"},
    "sentence": {"color": "#F2CF5B", "size": 520, "shape": "o"},
}


def load_graph_payload(graph_json: Path) -> dict:
    return json.loads(graph_json.read_text(encoding="utf-8"))


def truncate_text(text: str, width: int = 52) -> str:
    return shorten(" ".join((text or "").split()), width=width, placeholder="...")


def build_graph(payload: dict) -> tuple[nx.DiGraph, dict[int, tuple[float, float]], dict[int, str]]:
    graph = nx.DiGraph()
    nodes = payload["nodes"]
    edges = payload["edges"]

    document_node = next(node for node in nodes if node["node_type"] == "document")
    section_nodes = [node for node in nodes if node["node_type"] == "section"]
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


def draw_graph(payload: dict, output_png: Path) -> None:
    graph, positions, labels = build_graph(payload)
    ensure_dir(output_png.parent)

    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=180)
    ax.set_facecolor("#FBFBFB")

    parent_edges = [(u, v) for u, v, d in graph.edges(data=True) if d["edge_type"] == "parent"]
    next_edges = [(u, v) for u, v, d in graph.edges(data=True) if d["edge_type"] == "next"]

    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=parent_edges,
        edge_color="#2F2F2F",
        width=1.8,
        arrows=False,
        ax=ax,
    )
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
            sentence_preview.append(
                f'S{node["sentence_index"] + 1}: {truncate_text(node.get("text", ""), 70)}'
            )

    title = (
        f'Representative Pathology Report Hierarchy Graph\n'
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
    ax.legend(
        handles=legend_items,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=10,
    )

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize one text hierarchy graph as a PNG image.")
    parser.add_argument(
        "--graph_json",
        type=Path,
        default=DEFAULT_GRAPH_JSON,
        help="Path to one graph JSON file under Output/text_hierarchy_graphs_masked.",
    )
    parser.add_argument(
        "--output_png",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to Output/graph_visualizations/<stem>_graph.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_json = Path(args.graph_json)
    payload = load_graph_payload(graph_json)

    if args.output_png is None:
        output_png = DEFAULT_OUTPUT_DIR / f"{graph_json.stem}_hierarchy_graph.png"
    else:
        output_png = Path(args.output_png)

    draw_graph(payload, output_png)
    print(f"Graph PNG written to: {output_png}")


if __name__ == "__main__":
    main()
