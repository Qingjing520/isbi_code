from __future__ import annotations

"""Build sentence + ontology graphs from concept-enhanced text graphs.

This script derives a lighter auxiliary graph branch for the
"sentence + ontology" setting:

Document -> Sentence -> Concept

It intentionally removes section nodes and section-only co-occurrence edges so
the graph branch focuses on ontology relations while the original sentence
branch keeps raw text semantics.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch

from pathology_report_extraction.common.pdf_utils import ensure_dir, write_json
from pathology_report_extraction.common.pipeline_defaults import (
    CONCEPT_GRAPH_OUTPUT_SUBDIRS,
    DEFAULT_OUTPUT_ROOT,
    SENTENCE_ONTOLOGY_GRAPH_OUTPUT_SUBDIRS,
)


LOGGER = logging.getLogger("build_sentence_ontology_graphs")

DEFAULT_INPUT_DIR = DEFAULT_OUTPUT_ROOT / CONCEPT_GRAPH_OUTPUT_SUBDIRS["masked"]
DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / SENTENCE_ONTOLOGY_GRAPH_OUTPUT_SUBDIRS["masked"]
FEATURE_DIM = 512

NODE_TYPE_TO_ID = {"document": 0, "sentence": 1, "concept": 2}
EDGE_TYPE_TO_ID = {
    "parent": 0,
    "next": 1,
    "mention": 2,
    "ontology": 3,
    "same_sentence": 4,
}
DEFAULT_EDGE_TYPE_WEIGHTS = {
    "parent": 1.0,
    "next": 0.25,
    "mention": 1.0,
    "ontology": 0.7,
    "same_sentence": 0.25,
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_graph_jsons(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*.json")
        if path.is_file()
        and path.name not in {"run_summary.json"}
        and not path.name.startswith("pipeline_run_summary_")
    )


def _empty_feature_matrix(dim: int = FEATURE_DIM) -> torch.Tensor:
    return torch.empty((0, dim), dtype=torch.float32)


def _empty_long_matrix() -> torch.Tensor:
    return torch.empty((0, 2), dtype=torch.long)


def _source_node_lists(source_json: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    nodes = list(source_json.get("nodes", []) or [])
    document_node = next((node for node in nodes if node.get("node_type") == "document"), {})
    sentence_nodes = [node for node in nodes if node.get("node_type") == "sentence"]
    concept_nodes = [node for node in nodes if node.get("node_type") == "concept"]
    return document_node, sentence_nodes, concept_nodes


def _remap_concept_to_concept_edges(
    source_tensor: dict[str, Any],
    source_edge_type_mapping: dict[str, int],
    relation_name: str,
    sentence_count: int,
    section_count: int,
    concept_count: int,
) -> set[tuple[int, int, str]]:
    edge_type_id = source_edge_type_mapping.get(relation_name)
    if edge_type_id is None:
        return set()

    edge_index = source_tensor.get("edge_index")
    edge_type = source_tensor.get("edge_type")
    node_type = source_tensor.get("node_type")
    if not isinstance(edge_index, torch.Tensor) or not isinstance(edge_type, torch.Tensor) or not isinstance(node_type, torch.Tensor):
        return set()

    concept_node_type_id = 3
    concept_start_old = 1 + section_count + sentence_count
    concept_start_new = 1 + sentence_count
    edges: set[tuple[int, int, str]] = set()

    src_all = edge_index[0].tolist()
    dst_all = edge_index[1].tolist()
    type_all = edge_type.tolist()
    node_type_all = node_type.tolist()

    for src, dst, rel_id in zip(src_all, dst_all, type_all):
        if int(rel_id) != int(edge_type_id):
            continue
        if not (0 <= int(src) < len(node_type_all) and 0 <= int(dst) < len(node_type_all)):
            continue
        if int(node_type_all[int(src)]) != concept_node_type_id or int(node_type_all[int(dst)]) != concept_node_type_id:
            continue

        src_concept_index = int(src) - concept_start_old
        dst_concept_index = int(dst) - concept_start_old
        if not (0 <= src_concept_index < concept_count and 0 <= dst_concept_index < concept_count):
            continue

        edges.add(
            (
                concept_start_new + src_concept_index,
                concept_start_new + dst_concept_index,
                relation_name,
            )
        )

    return edges


def build_sentence_ontology_graph(
    source_json: dict[str, Any],
    source_tensor: dict[str, Any],
    source_json_path: Path,
    source_tensor_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    document_feature = torch.as_tensor(source_tensor.get("document_feature", torch.empty((FEATURE_DIM,), dtype=torch.float32))).float()
    if document_feature.dim() == 2 and document_feature.shape[0] == 1:
        document_feature = document_feature.squeeze(0)
    if document_feature.dim() != 1:
        raise ValueError(f"document_feature must be 1-D in {source_tensor_path}")

    sentence_features = torch.as_tensor(source_tensor.get("sentence_features", _empty_feature_matrix()), dtype=torch.float32)
    concept_features = torch.as_tensor(source_tensor.get("concept_features", _empty_feature_matrix()), dtype=torch.float32)
    concept_ids = list(source_tensor.get("concept_ids", []) or [])
    concept_ic = torch.as_tensor(source_tensor.get("concept_ic", torch.empty((0,), dtype=torch.float32))).float()
    concept_depth = torch.as_tensor(source_tensor.get("concept_depth", torch.empty((0,), dtype=torch.float32))).float()
    concept_direct_mentions = torch.as_tensor(
        source_tensor.get("concept_direct_mentions", torch.empty((0,), dtype=torch.float32))
    ).float()
    concept_sentence_links = [list(map(int, links or [])) for links in list(source_tensor.get("concept_sentence_links", []) or [])]

    sentence_count = int(sentence_features.shape[0])
    concept_count = int(concept_features.shape[0])
    if concept_count != len(concept_sentence_links):
        concept_sentence_links = concept_sentence_links[:concept_count]
        if len(concept_sentence_links) < concept_count:
            concept_sentence_links.extend([[] for _ in range(concept_count - len(concept_sentence_links))])

    section_count = int(source_json.get("node_counts", {}).get("section", 0))
    source_edge_type_mapping = {
        str(name): int(value)
        for name, value in dict(source_json.get("edge_type_mapping", {}) or {}).items()
        if value is not None
    }

    nodes: list[dict[str, Any]] = []
    document_node, source_sentence_nodes, source_concept_nodes = _source_node_lists(source_json)
    nodes.append(
        {
            "node_index": 0,
            "node_id": "doc_0",
            "node_type": "document",
            "level": 0,
            "title": document_node.get("title") or source_json.get("document_id") or source_json.get("file_name") or "document",
        }
    )

    for sentence_index in range(sentence_count):
        source_meta = source_sentence_nodes[sentence_index] if sentence_index < len(source_sentence_nodes) else {}
        nodes.append(
            {
                "node_index": len(nodes),
                "node_id": f"sent_{sentence_index}",
                "node_type": "sentence",
                "level": 1,
                "sentence_index": sentence_index,
                "section_index": source_meta.get("section_index"),
                "section_title": source_meta.get("section_title"),
                "section_sentence_index": source_meta.get("section_sentence_index"),
                "text": source_meta.get("text"),
            }
        )

    for concept_index in range(concept_count):
        source_meta = source_concept_nodes[concept_index] if concept_index < len(source_concept_nodes) else {}
        concept_id = str(concept_ids[concept_index]) if concept_index < len(concept_ids) else str(source_meta.get("concept_id", f"concept_{concept_index}"))
        nodes.append(
            {
                "node_index": len(nodes),
                "node_id": f"concept_{concept_id}",
                "node_type": "concept",
                "level": 2,
                "concept_id": concept_id,
                "concept_name": source_meta.get("concept_name", concept_id),
                "depth": int(float(concept_depth[concept_index].item())) if concept_index < len(concept_depth) else int(source_meta.get("depth", 0) or 0),
                "ic": float(concept_ic[concept_index].item()) if concept_index < len(concept_ic) else float(source_meta.get("ic", 0.0) or 0.0),
                "direct_mention_count": int(float(concept_direct_mentions[concept_index].item())) if concept_index < len(concept_direct_mentions) else int(source_meta.get("direct_mention_count", 0) or 0),
                "is_direct": bool(source_meta.get("is_direct", False)),
                "is_ancestor_only": bool(source_meta.get("is_ancestor_only", False)),
                "sentence_indices": concept_sentence_links[concept_index] if concept_index < len(concept_sentence_links) else [],
            }
        )

    edge_rows: list[dict[str, Any]] = []
    edge_tuples: set[tuple[int, int, str]] = set()

    for sentence_index in range(sentence_count):
        edge_tuples.add((0, 1 + sentence_index, "parent"))
    for sentence_index in range(sentence_count - 1):
        edge_tuples.add((1 + sentence_index, 1 + sentence_index + 1, "next"))
    for concept_index, sentence_indices in enumerate(concept_sentence_links):
        concept_node_index = 1 + sentence_count + concept_index
        for sentence_index in sentence_indices:
            if 0 <= int(sentence_index) < sentence_count:
                edge_tuples.add((1 + int(sentence_index), concept_node_index, "mention"))

    edge_tuples.update(
        _remap_concept_to_concept_edges(
            source_tensor=source_tensor,
            source_edge_type_mapping=source_edge_type_mapping,
            relation_name="ontology",
            sentence_count=sentence_count,
            section_count=section_count,
            concept_count=concept_count,
        )
    )
    edge_tuples.update(
        _remap_concept_to_concept_edges(
            source_tensor=source_tensor,
            source_edge_type_mapping=source_edge_type_mapping,
            relation_name="same_sentence",
            sentence_count=sentence_count,
            section_count=section_count,
            concept_count=concept_count,
        )
    )

    sorted_edges = sorted(edge_tuples, key=lambda item: (EDGE_TYPE_TO_ID[item[2]], item[0], item[1]))
    for src, dst, edge_name in sorted_edges:
        edge_rows.append(
            {
                "source_index": int(src),
                "target_index": int(dst),
                "edge_type": edge_name,
                "edge_weight": DEFAULT_EDGE_TYPE_WEIGHTS.get(edge_name, 1.0),
                "source_type": nodes[int(src)]["node_type"],
                "target_type": nodes[int(dst)]["node_type"],
            }
        )

    if sorted_edges:
        edge_index = torch.tensor([[src, dst] for src, dst, _ in sorted_edges], dtype=torch.long).t().contiguous()
        edge_type = torch.tensor([EDGE_TYPE_TO_ID[name] for _src, _dst, name in sorted_edges], dtype=torch.long)
        edge_weight = torch.tensor([DEFAULT_EDGE_TYPE_WEIGHTS.get(name, 1.0) for _src, _dst, name in sorted_edges], dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)
        edge_weight = torch.empty((0,), dtype=torch.float32)

    node_features = torch.cat(
        [
            document_feature.unsqueeze(0),
            sentence_features.float(),
            concept_features.float(),
        ],
        dim=0,
    )
    node_type = torch.tensor(
        [NODE_TYPE_TO_ID["document"]]
        + [NODE_TYPE_TO_ID["sentence"]] * sentence_count
        + [NODE_TYPE_TO_ID["concept"]] * concept_count,
        dtype=torch.long,
    )

    section_features = _empty_feature_matrix()
    graph_tensor_payload = {
        "document_feature": document_feature.float(),
        "section_features": section_features,
        "sentence_features": sentence_features.float(),
        "concept_features": concept_features.float(),
        "concept_ids": concept_ids,
        "concept_ic": concept_ic,
        "concept_depth": concept_depth,
        "concept_direct_mentions": concept_direct_mentions,
        "concept_sentence_links": concept_sentence_links,
        "node_features": node_features.float(),
        "node_type": node_type,
        "edge_index": edge_index,
        "edge_type": edge_type,
        "edge_weight": edge_weight,
        "sentence_to_section": torch.as_tensor(source_tensor.get("sentence_to_section", torch.empty((0,), dtype=torch.long)), dtype=torch.long),
        "section_spans": _empty_long_matrix(),
        "node_type_mapping": NODE_TYPE_TO_ID,
        "edge_type_mapping": EDGE_TYPE_TO_ID,
    }

    graph_json_payload = {
        "document_id": source_json.get("document_id"),
        "file_name": source_json.get("file_name"),
        "dataset": source_json.get("dataset"),
        "filter_mode": source_json.get("filter_mode"),
        "source_pdf": source_json.get("source_pdf"),
        "source_sentence_view_json": source_json.get("source_sentence_view_json"),
        "source_preprocessed_json": source_json.get("source_preprocessed_json"),
        "source_concept_json": source_json.get("source_concept_json"),
        "source_concept_graph_json": str(source_json_path),
        "source_concept_graph_pt": str(source_tensor_path),
        "embedding_path": source_json.get("embedding_path"),
        "pooling": "mean",
        "feature_dim": FEATURE_DIM,
        "graph_variant": "sentence_ontology_graph",
        "graph_cleanup": source_json.get("graph_cleanup", {}),
        "node_type_mapping": NODE_TYPE_TO_ID,
        "edge_type_mapping": EDGE_TYPE_TO_ID,
        "node_count": len(nodes),
        "edge_count": len(edge_rows),
        "node_counts": {
            "document": 1,
            "sentence": sentence_count,
            "concept": concept_count,
        },
        "concept_count": concept_count,
        "concept_annotation_summary": dict(source_json.get("concept_annotation_summary", {}) or {}),
        "nodes": nodes,
        "edges": edge_rows,
    }
    return graph_tensor_payload, graph_json_payload


def process_all_documents(input_dir: Path, output_dir: Path, limit: int | None = None) -> dict[str, Any]:
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    ensure_dir(output_root)

    log_path = output_root / "graph.log"
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    json_paths = iter_graph_jsons(input_root)
    if limit is not None:
        json_paths = json_paths[:limit]

    LOGGER.info("Discovered %s concept graph JSON files under %s", len(json_paths), input_root)
    summaries: list[dict[str, Any]] = []

    for index, json_path in enumerate(json_paths, start=1):
        tensor_path = json_path.with_suffix(".pt")
        LOGGER.info("[%s/%s] Building sentence-ontology graph for %s", index, len(json_paths), json_path)
        try:
            if not tensor_path.exists():
                raise FileNotFoundError(f"Missing source tensor file: {tensor_path}")

            source_json = load_json(json_path)
            source_tensor = torch.load(tensor_path, map_location="cpu")
            if not isinstance(source_tensor, dict):
                raise TypeError(f"Expected dict payload in {tensor_path}")

            graph_tensor_payload, graph_json_payload = build_sentence_ontology_graph(
                source_json=source_json,
                source_tensor=source_tensor,
                source_json_path=json_path,
                source_tensor_path=tensor_path,
            )

            relative_path = json_path.relative_to(input_root)
            output_json_path = output_root / relative_path
            output_pt_path = output_root / relative_path.with_suffix(".pt")
            ensure_dir(output_json_path.parent)
            graph_json_payload["graph_tensor_path"] = str(output_pt_path)
            torch.save(graph_tensor_payload, output_pt_path)
            write_json(output_json_path, graph_json_payload)

            summaries.append(
                {
                    "document_id": graph_json_payload.get("document_id"),
                    "file_name": graph_json_payload.get("file_name"),
                    "dataset": graph_json_payload.get("dataset"),
                    "source_graph_json": str(json_path),
                    "source_graph_pt": str(tensor_path),
                    "output_json": str(output_json_path),
                    "output_pt": str(output_pt_path),
                    "sentence_count": int(graph_json_payload["node_counts"]["sentence"]),
                    "concept_count": int(graph_json_payload["node_counts"]["concept"]),
                    "status": "success",
                }
            )
        except Exception as exc:
            LOGGER.exception("Failed to build sentence-ontology graph for %s", json_path)
            summaries.append(
                {
                    "document_id": json_path.stem,
                    "file_name": json_path.name,
                    "dataset": None,
                    "source_graph_json": str(json_path),
                    "source_graph_pt": str(tensor_path),
                    "output_json": None,
                    "output_pt": None,
                    "sentence_count": 0,
                    "concept_count": 0,
                    "status": "failed",
                    "error": str(exc),
                }
            )

    successes = [item for item in summaries if item["status"] == "success"]
    failures = [item for item in summaries if item["status"] == "failed"]
    dataset_counts: dict[str, int] = {}
    for item in successes:
        dataset = item.get("dataset") or "UNKNOWN"
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1

    run_summary = {
        "input_dir": str(input_root),
        "output_dir": str(output_root),
        "graph_variant": "sentence_ontology_graph",
        "node_type_mapping": NODE_TYPE_TO_ID,
        "edge_type_mapping": EDGE_TYPE_TO_ID,
        "edge_type_weights": DEFAULT_EDGE_TYPE_WEIGHTS,
        "total_graph_files": len(json_paths),
        "success_count": len(successes),
        "failure_count": len(failures),
        "datasets": dataset_counts,
        "files": summaries,
    }
    write_json(output_root / "run_summary.json", run_summary)
    LOGGER.info(
        "Completed sentence-ontology graph build | total=%s | success=%s | failed=%s",
        len(json_paths),
        len(successes),
        len(failures),
    )
    return run_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build sentence + ontology auxiliary graphs from concept-enhanced text graphs."
    )
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = process_all_documents(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        limit=args.limit,
    )
    print(
        "Sentence-ontology graph build finished | "
        f"total={summary['total_graph_files']} | "
        f"success={summary['success_count']} | "
        f"failed={summary['failure_count']}"
    )


if __name__ == "__main__":
    main()
