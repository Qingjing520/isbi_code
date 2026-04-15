from __future__ import annotations

"""Audit ontology concept annotations and optional concept-graph outputs.

This script summarizes one or more concept-annotation directories, optionally
cross-checks aligned concept-graph JSON files, and writes a cohort-level JSON
report that is useful for:

- checking zero-concept / zero-sentence documents
- inspecting the most frequent direct and true-path-expanded concepts
- comparing BRCA vs KIRC concept prevalence
- validating that concept graph counts still match the upstream annotations
"""

import argparse
import json
import math
import statistics
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_JSON = Path(
    r"F:\Tasks\isbi_code\pathology_report_extraction\Output\ontology_audits\ontology_concept_audit_summary.json"
)


def ensure_dir(path: Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_annotation_jsons(annotation_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in Path(annotation_dir).rglob("*.json")
        if path.is_file()
        and path.name not in {"run_summary.json"}
        and not path.name.startswith("pipeline_run_summary_")
    )


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _safe_median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def _safe_p95(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(v) for v in values)
    idx = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
    return float(ordered[idx])


def _top_counter_items(
    counter: Counter[str],
    metadata: dict[str, dict[str, Any]] | None = None,
    total_docs: int | None = None,
    top_k: int = 25,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for concept_id, count in counter.most_common(top_k):
        item: dict[str, Any] = {
            "concept_id": concept_id,
            "count": int(count),
        }
        if total_docs:
            item["doc_frequency_ratio"] = float(count / max(total_docs, 1))
        if metadata and concept_id in metadata:
            item.update(metadata[concept_id])
        items.append(item)
    return items


def _label_from_dir(path: Path) -> str:
    parts = [part for part in path.parts if part]
    if parts:
        return parts[-1]
    return path.name or str(path)


def audit_annotation_dir(
    annotation_dir: Path,
    graph_dir: Path | None = None,
    top_k: int = 25,
) -> dict[str, Any]:
    annotation_root = Path(annotation_dir)
    graph_root = Path(graph_dir) if graph_dir is not None else None
    files = iter_annotation_jsons(annotation_root)

    sentence_counts: list[int] = []
    section_counts: list[int] = []
    mention_counts: list[int] = []
    direct_concept_counts: list[int] = []
    concept_counts: list[int] = []
    graph_node_counts: list[int] = []
    graph_edge_counts: list[int] = []
    graph_concept_counts: list[int] = []

    zero_sentence_documents: list[str] = []
    zero_concept_documents: list[str] = []
    graph_mismatch_documents: list[dict[str, Any]] = []
    missing_graph_documents: list[str] = []

    dataset_counts: Counter[str] = Counter()
    section_title_counter: Counter[str] = Counter()
    matched_term_counter: Counter[str] = Counter()
    concept_doc_freq: Counter[str] = Counter()
    direct_concept_doc_freq: Counter[str] = Counter()
    concept_mention_freq: Counter[str] = Counter()
    concept_metadata: dict[str, dict[str, Any]] = {}

    documents_with_ancestor_expansion = 0

    for path in files:
        payload = load_json(path)
        document_id = str(payload.get("document_id") or path.stem)
        dataset = str(payload.get("dataset") or "")
        dataset_counts[dataset or "unknown"] += 1

        sentence_count = int(payload.get("sentence_count", 0))
        section_count = int(payload.get("section_count", 0))
        mention_count = int(payload.get("mention_count", 0))
        direct_concept_count = int(payload.get("direct_concept_count", 0))
        concept_count = int(payload.get("concept_count", 0))

        sentence_counts.append(sentence_count)
        section_counts.append(section_count)
        mention_counts.append(mention_count)
        direct_concept_counts.append(direct_concept_count)
        concept_counts.append(concept_count)

        if sentence_count == 0:
            zero_sentence_documents.append(document_id)
        if concept_count == 0:
            zero_concept_documents.append(document_id)
        if concept_count > direct_concept_count:
            documents_with_ancestor_expansion += 1

        concepts = payload.get("concepts", []) or []
        mentions = payload.get("mentions", []) or []

        direct_in_doc: set[str] = set()
        all_in_doc: set[str] = set()

        for concept in concepts:
            concept_id = str(concept.get("concept_id", "")).strip()
            if not concept_id:
                continue
            all_in_doc.add(concept_id)
            concept_metadata.setdefault(
                concept_id,
                {
                    "concept_name": str(concept.get("concept_name", "")).strip(),
                    "depth": int(concept.get("depth", 0)),
                    "ic": float(concept.get("ic", 0.0)),
                },
            )
            if bool(concept.get("is_direct", False)):
                direct_in_doc.add(concept_id)

        for concept_id in all_in_doc:
            concept_doc_freq[concept_id] += 1
        for concept_id in direct_in_doc:
            direct_concept_doc_freq[concept_id] += 1

        for mention in mentions:
            concept_id = str(mention.get("concept_id", "")).strip()
            if concept_id:
                concept_mention_freq[concept_id] += 1
            matched_term = str(mention.get("matched_term", "")).strip()
            if matched_term:
                matched_term_counter[matched_term] += 1
            section_title = str(mention.get("section_title", "")).strip()
            if section_title:
                section_title_counter[section_title] += 1

        if graph_root is not None:
            graph_path = graph_root / path.relative_to(annotation_root)
            if not graph_path.exists():
                missing_graph_documents.append(document_id)
                continue

            graph_payload = load_json(graph_path)
            graph_concept_count = int(graph_payload.get("concept_count", 0))
            graph_node_count = int(graph_payload.get("node_count", 0))
            graph_edge_count = int(graph_payload.get("edge_count", 0))
            graph_concept_counts.append(graph_concept_count)
            graph_node_counts.append(graph_node_count)
            graph_edge_counts.append(graph_edge_count)

            if graph_concept_count != concept_count:
                graph_mismatch_documents.append(
                    {
                        "document_id": document_id,
                        "annotation_concept_count": concept_count,
                        "graph_concept_count": graph_concept_count,
                        "annotation_json": str(path),
                        "graph_json": str(graph_path),
                    }
                )

    total_docs = len(files)
    unique_concepts = set(concept_doc_freq)
    unique_direct_concepts = set(direct_concept_doc_freq)

    summary: dict[str, Any] = {
        "annotation_dir": str(annotation_root),
        "graph_dir": str(graph_root) if graph_root is not None else None,
        "total_documents": total_docs,
        "datasets": dict(dataset_counts),
        "zero_sentence_document_count": len(zero_sentence_documents),
        "zero_sentence_documents": zero_sentence_documents[: min(25, len(zero_sentence_documents))],
        "zero_concept_document_count": len(zero_concept_documents),
        "zero_concept_documents": zero_concept_documents[: min(25, len(zero_concept_documents))],
        "documents_with_ancestor_expansion": documents_with_ancestor_expansion,
        "unique_concepts": len(unique_concepts),
        "unique_direct_concepts": len(unique_direct_concepts),
        "sentence_count_stats": {
            "mean": _safe_mean(sentence_counts),
            "median": _safe_median(sentence_counts),
            "p95": _safe_p95(sentence_counts),
            "max": max(sentence_counts) if sentence_counts else 0,
        },
        "section_count_stats": {
            "mean": _safe_mean(section_counts),
            "median": _safe_median(section_counts),
            "p95": _safe_p95(section_counts),
            "max": max(section_counts) if section_counts else 0,
        },
        "mention_count_stats": {
            "mean": _safe_mean(mention_counts),
            "median": _safe_median(mention_counts),
            "p95": _safe_p95(mention_counts),
            "max": max(mention_counts) if mention_counts else 0,
        },
        "direct_concept_count_stats": {
            "mean": _safe_mean(direct_concept_counts),
            "median": _safe_median(direct_concept_counts),
            "p95": _safe_p95(direct_concept_counts),
            "max": max(direct_concept_counts) if direct_concept_counts else 0,
        },
        "concept_count_stats": {
            "mean": _safe_mean(concept_counts),
            "median": _safe_median(concept_counts),
            "p95": _safe_p95(concept_counts),
            "max": max(concept_counts) if concept_counts else 0,
        },
        "top_concepts_by_document_frequency": _top_counter_items(
            concept_doc_freq,
            metadata=concept_metadata,
            total_docs=total_docs,
            top_k=top_k,
        ),
        "top_direct_concepts_by_document_frequency": _top_counter_items(
            direct_concept_doc_freq,
            metadata=concept_metadata,
            total_docs=total_docs,
            top_k=top_k,
        ),
        "top_concepts_by_mentions": _top_counter_items(
            concept_mention_freq,
            metadata=concept_metadata,
            total_docs=None,
            top_k=top_k,
        ),
        "top_matched_terms": [
            {"matched_term": term, "count": int(count)}
            for term, count in matched_term_counter.most_common(top_k)
        ],
        "top_sections_with_mentions": [
            {"section_title": section_title, "count": int(count)}
            for section_title, count in section_title_counter.most_common(top_k)
        ],
    }

    if graph_root is not None:
        summary["graph_crosscheck"] = {
            "missing_graph_document_count": len(missing_graph_documents),
            "missing_graph_documents": missing_graph_documents[: min(25, len(missing_graph_documents))],
            "graph_concept_mismatch_count": len(graph_mismatch_documents),
            "graph_concept_mismatches": graph_mismatch_documents[: min(25, len(graph_mismatch_documents))],
            "graph_node_count_stats": {
                "mean": _safe_mean(graph_node_counts),
                "median": _safe_median(graph_node_counts),
                "p95": _safe_p95(graph_node_counts),
                "max": max(graph_node_counts) if graph_node_counts else 0,
            },
            "graph_edge_count_stats": {
                "mean": _safe_mean(graph_edge_counts),
                "median": _safe_median(graph_edge_counts),
                "p95": _safe_p95(graph_edge_counts),
                "max": max(graph_edge_counts) if graph_edge_counts else 0,
            },
            "graph_concept_count_stats": {
                "mean": _safe_mean(graph_concept_counts),
                "median": _safe_median(graph_concept_counts),
                "p95": _safe_p95(graph_concept_counts),
                "max": max(graph_concept_counts) if graph_concept_counts else 0,
            },
        }

    summary["_concept_doc_freq"] = dict(concept_doc_freq)
    summary["_direct_concept_doc_freq"] = dict(direct_concept_doc_freq)
    summary["_concept_metadata"] = concept_metadata
    return summary


def build_pairwise_comparisons(
    cohort_summaries: dict[str, dict[str, Any]],
    top_k: int = 25,
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []

    for left_name, right_name in combinations(sorted(cohort_summaries), 2):
        left = cohort_summaries[left_name]
        right = cohort_summaries[right_name]

        left_total = max(int(left.get("total_documents", 0)), 1)
        right_total = max(int(right.get("total_documents", 0)), 1)
        left_doc_freq = {str(k): int(v) for k, v in left.get("_concept_doc_freq", {}).items()}
        right_doc_freq = {str(k): int(v) for k, v in right.get("_concept_doc_freq", {}).items()}
        left_direct_doc_freq = {str(k): int(v) for k, v in left.get("_direct_concept_doc_freq", {}).items()}
        right_direct_doc_freq = {str(k): int(v) for k, v in right.get("_direct_concept_doc_freq", {}).items()}
        metadata = {
            **{str(k): v for k, v in right.get("_concept_metadata", {}).items()},
            **{str(k): v for k, v in left.get("_concept_metadata", {}).items()},
        }

        left_unique = sorted(set(left_doc_freq) - set(right_doc_freq))
        right_unique = sorted(set(right_doc_freq) - set(left_doc_freq))
        shared = sorted(set(left_doc_freq) & set(right_doc_freq))

        left_distinctive: list[dict[str, Any]] = []
        right_distinctive: list[dict[str, Any]] = []
        for concept_id in set(left_doc_freq) | set(right_doc_freq):
            left_ratio = float(left_doc_freq.get(concept_id, 0) / left_total)
            right_ratio = float(right_doc_freq.get(concept_id, 0) / right_total)
            item = {
                "concept_id": concept_id,
                "left_doc_frequency_ratio": left_ratio,
                "right_doc_frequency_ratio": right_ratio,
                "ratio_diff": left_ratio - right_ratio,
            }
            if concept_id in metadata:
                item.update(metadata[concept_id])
            left_distinctive.append(item)
            right_distinctive.append(item)

        left_distinctive.sort(key=lambda item: item["ratio_diff"], reverse=True)
        right_distinctive.sort(key=lambda item: item["ratio_diff"])

        left_direct_unique = sorted(set(left_direct_doc_freq) - set(right_direct_doc_freq))
        right_direct_unique = sorted(set(right_direct_doc_freq) - set(left_direct_doc_freq))
        direct_shared = sorted(set(left_direct_doc_freq) & set(right_direct_doc_freq))

        left_direct_distinctive: list[dict[str, Any]] = []
        right_direct_distinctive: list[dict[str, Any]] = []
        for concept_id in set(left_direct_doc_freq) | set(right_direct_doc_freq):
            left_ratio = float(left_direct_doc_freq.get(concept_id, 0) / left_total)
            right_ratio = float(right_direct_doc_freq.get(concept_id, 0) / right_total)
            item = {
                "concept_id": concept_id,
                "left_doc_frequency_ratio": left_ratio,
                "right_doc_frequency_ratio": right_ratio,
                "ratio_diff": left_ratio - right_ratio,
            }
            if concept_id in metadata:
                item.update(metadata[concept_id])
            left_direct_distinctive.append(item)
            right_direct_distinctive.append(item)

        left_direct_distinctive.sort(key=lambda item: item["ratio_diff"], reverse=True)
        right_direct_distinctive.sort(key=lambda item: item["ratio_diff"])

        comparisons.append(
            {
                "left_cohort": left_name,
                "right_cohort": right_name,
                "left_total_documents": left_total,
                "right_total_documents": right_total,
                "shared_concept_count": len(shared),
                "left_unique_concept_count": len(left_unique),
                "right_unique_concept_count": len(right_unique),
                "shared_direct_concept_count": len(direct_shared),
                "left_unique_direct_concept_count": len(left_direct_unique),
                "right_unique_direct_concept_count": len(right_direct_unique),
                "left_unique_concept_examples": [
                    {"concept_id": concept_id, **metadata.get(concept_id, {})}
                    for concept_id in left_unique[:top_k]
                ],
                "right_unique_concept_examples": [
                    {"concept_id": concept_id, **metadata.get(concept_id, {})}
                    for concept_id in right_unique[:top_k]
                ],
                "left_unique_direct_concept_examples": [
                    {"concept_id": concept_id, **metadata.get(concept_id, {})}
                    for concept_id in left_direct_unique[:top_k]
                ],
                "right_unique_direct_concept_examples": [
                    {"concept_id": concept_id, **metadata.get(concept_id, {})}
                    for concept_id in right_direct_unique[:top_k]
                ],
                "most_left_distinctive_all_concepts": left_distinctive[:top_k],
                "most_right_distinctive_all_concepts": right_distinctive[:top_k],
                "most_left_distinctive_direct_concepts": left_direct_distinctive[:top_k],
                "most_right_distinctive_direct_concepts": right_direct_distinctive[:top_k],
            }
        )

    return comparisons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit ontology concept annotations and optional concept graph outputs."
    )
    parser.add_argument(
        "--annotation_dir",
        type=Path,
        action="append",
        required=True,
        help="Concept annotation directory. Pass multiple times to compare cohorts.",
    )
    parser.add_argument(
        "--graph_dir",
        type=Path,
        action="append",
        default=None,
        help="Optional aligned concept-graph directory. If provided, pass the same number of times as --annotation_dir.",
    )
    parser.add_argument(
        "--cohort_name",
        action="append",
        default=None,
        help="Optional cohort label for the matching --annotation_dir. Defaults to the directory name.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Where to write the audit summary JSON.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=25,
        help="How many top concepts / terms to keep in the summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    annotation_dirs = list(args.annotation_dir)
    graph_dirs = list(args.graph_dir or [])
    cohort_names = list(args.cohort_name or [])

    if graph_dirs and len(graph_dirs) != len(annotation_dirs):
        raise ValueError("--graph_dir must be omitted or provided the same number of times as --annotation_dir.")
    if cohort_names and len(cohort_names) != len(annotation_dirs):
        raise ValueError("--cohort_name must be omitted or provided the same number of times as --annotation_dir.")

    while len(graph_dirs) < len(annotation_dirs):
        graph_dirs.append(None)  # type: ignore[arg-type]

    if not cohort_names:
        cohort_names = [_label_from_dir(path) for path in annotation_dirs]

    cohort_summaries: dict[str, dict[str, Any]] = {}
    for cohort_name, annotation_dir, graph_dir in zip(cohort_names, annotation_dirs, graph_dirs):
        summary = audit_annotation_dir(
            annotation_dir=annotation_dir,
            graph_dir=graph_dir,
            top_k=int(args.top_k),
        )
        cohort_summaries[str(cohort_name)] = summary

    comparisons = build_pairwise_comparisons(cohort_summaries, top_k=int(args.top_k))

    output_payload: dict[str, Any] = {
        "cohorts": cohort_summaries,
        "comparisons": comparisons,
    }

    for summary in output_payload["cohorts"].values():
        summary.pop("_concept_doc_freq", None)
        summary.pop("_direct_concept_doc_freq", None)
        summary.pop("_concept_metadata", None)

    write_json(Path(args.output_json), output_payload)
    print(
        f"Ontology audit finished | cohorts={len(cohort_summaries)} | "
        f"comparisons={len(comparisons)} | output={Path(args.output_json)}"
    )


if __name__ == "__main__":
    main()
