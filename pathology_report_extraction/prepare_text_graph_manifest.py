from __future__ import annotations

"""Prepare a training manifest from text graph outputs.

This script matches case-level text graphs to slide-level labels and writes
one row per (slide_id, text_graph) sample. It is intended as the bridge
between the preprocessing pipeline and later model training.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from config.config import get_path, get_stage_config, get_value, load_yaml_config


DEFAULT_OUTPUT_ROOT = Path(r"D:\Tasks\isbi_code\pathology_report_extraction\Output")
DEFAULT_GRAPH_DIR = DEFAULT_OUTPUT_ROOT / "text_hierarchy_graphs_masked"
DEFAULT_LABEL_CSV = Path(r"D:\Tasks\pathologic_stage.csv")
DEFAULT_SPLIT_CSV: Path | None = None
DEFAULT_MANIFEST_DIR = DEFAULT_OUTPUT_ROOT / "manifests"
DEFAULT_MANIFEST_OUTPUT_SUBDIR = "manifests"


def ensure_dir(path: Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_graph_jsons(graph_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in graph_dir.rglob("*.json")
        if path.is_file()
        and path.name not in {"run_summary.json"}
        and not path.name.startswith("pipeline_run_summary_")
    )


def load_split_map(split_csv: Path | None) -> tuple[dict[str, str], dict[str, str]]:
    if split_csv is None or not split_csv.exists():
        return {}, {}

    with split_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    slide_to_split: dict[str, str] = {}
    case_to_split: dict[str, str] = {}

    if {"slide_id", "split"}.issubset(set(fieldnames)):
        for row in rows:
            slide_id = (row.get("slide_id") or "").strip()
            split = (row.get("split") or "").strip()
            if slide_id and split:
                slide_to_split[slide_id] = split
        return slide_to_split, case_to_split

    if {"case_id", "split"}.issubset(set(fieldnames)):
        for row in rows:
            case_id = (row.get("case_id") or "").strip()
            split = (row.get("split") or "").strip()
            if case_id and split:
                case_to_split[case_id] = split
        return slide_to_split, case_to_split

    candidate_split_cols = [name for name in fieldnames if name in {"train", "val", "test"}]
    for split_name in candidate_split_cols:
        for row in rows:
            value = (row.get(split_name) or "").strip()
            if value:
                slide_to_split[value] = split_name

    return slide_to_split, case_to_split


def load_graph_records(graph_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    graph_json_paths = iter_graph_jsons(graph_dir)
    records: list[dict[str, Any]] = []
    duplicate_case_ids: list[str] = []

    seen_case_counts: Counter[str] = Counter()

    for graph_json_path in graph_json_paths:
        payload = load_json(graph_json_path)
        case_id = str(payload.get("document_id") or "").strip()
        if not case_id:
            continue

        seen_case_counts[case_id] += 1
        if seen_case_counts[case_id] == 2:
            duplicate_case_ids.append(case_id)

        graph_pt_path = graph_json_path.with_suffix(".pt")
        node_counts = payload.get("node_counts", {})

        records.append(
            {
                "report_id": graph_json_path.stem,
                "case_id": case_id,
                "document_id": case_id,
                "dataset": payload.get("dataset"),
                "file_name": payload.get("file_name"),
                "filter_mode": payload.get("filter_mode"),
                "source_pdf": payload.get("source_pdf"),
                "source_preprocessed_json": payload.get("source_preprocessed_json"),
                "source_sentence_view_json": payload.get("source_sentence_view_json"),
                "source_embedding_pt": payload.get("embedding_path"),
                "source_concept_json": payload.get("source_concept_json") or "",
                "graph_json": str(graph_json_path),
                "graph_pt": str(graph_pt_path),
                "node_count": int(payload.get("node_count", 0)),
                "edge_count": int(payload.get("edge_count", 0)),
                "section_count": int(node_counts.get("section", 0)),
                "sentence_count": int(node_counts.get("sentence", 0)),
                "concept_count": int(node_counts.get("concept", 0)),
                "has_concept_level": bool(isinstance(payload.get("node_type_mapping"), dict) and "concept" in payload.get("node_type_mapping", {})),
            }
        )

    return records, sorted(duplicate_case_ids)


def load_label_rows(label_csv: Path) -> list[dict[str, Any]]:
    with label_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = set(reader.fieldnames or [])

    required = {"case_id", "slide_id", "label"}
    if not required.issubset(fieldnames):
        raise ValueError(f"Label CSV must contain columns {required}. Got: {sorted(fieldnames)}")

    normalized: list[dict[str, Any]] = []
    for row in rows:
        case_id = (row.get("case_id") or "").strip()
        slide_id = (row.get("slide_id") or "").strip()
        label_raw = row.get("label")
        if not case_id or not slide_id or label_raw in (None, ""):
            continue
        normalized.append(
            {
                "case_id": case_id,
                "slide_id": slide_id,
                "label": int(label_raw),
            }
        )
    return normalized


def derive_default_output_csv(graph_dir: Path, output_dir: Path) -> Path:
    stem = graph_dir.name
    if stem.startswith("text_hierarchy_graphs_"):
        suffix = stem.replace("text_hierarchy_graphs_", "", 1)
        return output_dir / f"text_graph_manifest_{suffix}.csv"
    if stem.startswith("text_concept_graphs_"):
        suffix = stem.replace("text_concept_graphs_", "", 1)
        return output_dir / f"text_graph_manifest_{suffix}.csv"
    return output_dir / f"{stem}_manifest.csv"


def write_manifest_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    ensure_dir(output_csv.parent)
    fieldnames = [
        "case_id",
        "slide_id",
        "label",
        "split",
        "dataset",
        "report_id",
        "file_name",
        "filter_mode",
        "graph_pt",
        "graph_json",
        "source_embedding_pt",
        "source_concept_json",
        "source_sentence_view_json",
        "source_preprocessed_json",
        "source_pdf",
        "node_count",
        "edge_count",
        "section_count",
        "sentence_count",
        "concept_count",
        "has_concept_level",
        "image_pt",
        "image_exists",
    ]

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def prepare_manifest(
    graph_dir: Path,
    label_csv: Path,
    output_csv: Path,
    split_csv: Path | None = None,
    image_dir: Path | None = None,
) -> dict[str, Any]:
    graph_root = Path(graph_dir)
    label_csv = Path(label_csv)
    output_csv = Path(output_csv)
    image_dir = Path(image_dir) if image_dir is not None else None

    if not graph_root.exists():
        raise FileNotFoundError(f"Graph directory not found: {graph_root}")
    if not label_csv.exists():
        raise FileNotFoundError(f"Label CSV not found: {label_csv}")

    graph_records, duplicate_case_ids = load_graph_records(graph_root)
    label_rows = load_label_rows(label_csv)
    slide_to_split, case_to_split = load_split_map(split_csv)

    graphs_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in graph_records:
        graphs_by_case[record["case_id"]].append(record)

    manifest_rows: list[dict[str, Any]] = []
    unmatched_label_rows = 0
    matched_case_ids: set[str] = set()

    for label_row in label_rows:
        case_id = label_row["case_id"]
        slide_id = label_row["slide_id"]
        label = label_row["label"]

        matched_graphs = graphs_by_case.get(case_id, [])
        if not matched_graphs:
            unmatched_label_rows += 1
            continue

        split = slide_to_split.get(slide_id) or case_to_split.get(case_id) or ""

        for graph_record in matched_graphs:
            row = {
                "case_id": case_id,
                "slide_id": slide_id,
                "label": label,
                "split": split,
                "dataset": graph_record.get("dataset") or "",
                "report_id": graph_record["report_id"],
                "file_name": graph_record.get("file_name") or "",
                "filter_mode": graph_record.get("filter_mode") or "",
                "graph_pt": graph_record["graph_pt"],
                "graph_json": graph_record["graph_json"],
                "source_embedding_pt": graph_record.get("source_embedding_pt") or "",
                "source_concept_json": graph_record.get("source_concept_json") or "",
                "source_sentence_view_json": graph_record.get("source_sentence_view_json") or "",
                "source_preprocessed_json": graph_record.get("source_preprocessed_json") or "",
                "source_pdf": graph_record.get("source_pdf") or "",
                "node_count": graph_record["node_count"],
                "edge_count": graph_record["edge_count"],
                "section_count": graph_record["section_count"],
                "sentence_count": graph_record["sentence_count"],
                "concept_count": graph_record["concept_count"],
                "has_concept_level": graph_record["has_concept_level"],
                "image_pt": "",
                "image_exists": "",
            }
            if image_dir is not None:
                image_pt = image_dir / f"{slide_id}.pt"
                row["image_pt"] = str(image_pt)
                row["image_exists"] = image_pt.exists()
            manifest_rows.append(row)

        matched_case_ids.add(case_id)

    unmatched_graph_case_ids = sorted(set(graphs_by_case) - matched_case_ids)

    write_manifest_csv(manifest_rows, output_csv)

    dataset_counts = Counter(row["dataset"] for row in manifest_rows)
    split_counts = Counter(row["split"] or "unspecified" for row in manifest_rows)
    label_counts = Counter(str(row["label"]) for row in manifest_rows)
    image_exists_count = sum(1 for row in manifest_rows if row.get("image_exists") is True)
    rows_with_concepts = sum(1 for row in manifest_rows if int(row.get("concept_count", 0)) > 0)
    rows_with_concept_level = sum(1 for row in manifest_rows if str(row.get("has_concept_level", "")).lower() in {"true", "1"})
    total_concept_nodes = sum(int(row.get("concept_count", 0)) for row in manifest_rows)

    summary = {
        "graph_dir": str(graph_root),
        "label_csv": str(label_csv),
        "split_csv": str(split_csv) if split_csv is not None else None,
        "image_dir": str(image_dir) if image_dir is not None else None,
        "output_csv": str(output_csv),
        "output_summary_json": str(output_csv.with_name(output_csv.stem + "_summary.json")),
        "total_graph_files": len(graph_records),
        "unique_graph_cases": len(graphs_by_case),
        "duplicate_graph_case_ids": duplicate_case_ids,
        "total_label_rows": len(label_rows),
        "matched_rows": len(manifest_rows),
        "matched_cases": len(matched_case_ids),
        "unmatched_label_rows": unmatched_label_rows,
        "unmatched_graph_case_count": len(unmatched_graph_case_ids),
        "unmatched_graph_case_ids": unmatched_graph_case_ids,
        "datasets": dict(dataset_counts),
        "splits": dict(split_counts),
        "labels": dict(label_counts),
        "image_exists_count": image_exists_count,
        "rows_with_concept_level": rows_with_concept_level,
        "rows_with_concepts": rows_with_concepts,
        "total_concept_nodes": total_concept_nodes,
    }
    if summary["matched_rows"] == 0:
        summary["warning"] = (
            "No graph cases matched the provided label CSV. "
            "Check whether the label file belongs to the same cancer cohorts "
            "and whether case_id naming matches the graph document_id values."
        )
    write_json(Path(summary["output_summary_json"]), summary)
    return summary


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=None, help="Optional YAML config preset.")
    pre_args, _ = pre_parser.parse_known_args()
    raw_config, config_path = load_yaml_config(pre_args.config)
    stage_block = raw_config.get("prepare_text_graph_manifest") if isinstance(raw_config.get("prepare_text_graph_manifest"), dict) else {}
    config = get_stage_config(raw_config, "prepare_text_graph_manifest")
    defaults_block = raw_config.get("defaults") if isinstance(raw_config.get("defaults"), dict) else {}
    preprocess_config = get_stage_config(raw_config, "preprocess")
    filter_mode = str(get_value(preprocess_config, "filter_mode", "masked"))
    output_root = get_path(defaults_block, "output_root", DEFAULT_OUTPUT_ROOT, config_path)

    graph_input_default = get_path(stage_block, "graph_dir", DEFAULT_GRAPH_DIR, config_path)
    if stage_block and stage_block.get("graph_dir") in (None, ""):
        graph_output_subdirs = {
            "masked": "text_hierarchy_graphs_masked",
            "no_diagnosis": "text_hierarchy_graphs_no_diagnosis",
            "no_diagnosis_masked": "text_hierarchy_graphs_no_diagnosis_masked",
            "full": "text_hierarchy_graphs_full",
        }
        graph_output_subdirs.update(get_value(get_stage_config(raw_config, "build_text_hierarchy_graphs"), "output_subdirs", {}) or {})
        graph_input_default = output_root / graph_output_subdirs.get(filter_mode, graph_output_subdirs["masked"])

    manifest_output_dir_default = get_path(stage_block, "output_dir", DEFAULT_MANIFEST_DIR, config_path)
    if stage_block and stage_block.get("output_dir") in (None, ""):
        manifest_output_subdir = str(get_value(config, "output_subdir", DEFAULT_MANIFEST_OUTPUT_SUBDIR))
        manifest_output_dir_default = output_root / manifest_output_subdir

    parser = argparse.ArgumentParser(description="Prepare a slide-level training manifest from text hierarchy graph outputs.")
    parser.add_argument("--config", type=Path, default=pre_args.config, help="Optional YAML config preset.")
    parser.add_argument("--graph_dir", type=Path, default=graph_input_default, help="Directory containing text graph .pt/.json files.")
    parser.add_argument("--label_csv", type=Path, default=get_path(stage_block, "label_csv", DEFAULT_LABEL_CSV, config_path), help="CSV containing case_id, slide_id, label.")
    parser.add_argument("--split_csv", type=Path, default=None if get_value(stage_block, "split_csv", None) is None else get_path(stage_block, "split_csv", DEFAULT_LABEL_CSV, config_path), help="Optional split CSV.")
    parser.add_argument("--image_dir", type=Path, default=None if get_value(stage_block, "image_dir", None) is None else get_path(stage_block, "image_dir", DEFAULT_OUTPUT_ROOT, config_path), help="Optional image feature directory for recording image .pt paths.")
    parser.add_argument("--output_dir", type=Path, default=manifest_output_dir_default, help="Directory to save manifest CSV and summary JSON.")
    parser.add_argument("--output_csv", type=Path, default=None, help="Optional explicit output CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_csv = args.output_csv or derive_default_output_csv(args.graph_dir, args.output_dir)
    summary = prepare_manifest(
        graph_dir=args.graph_dir,
        label_csv=args.label_csv,
        output_csv=output_csv,
        split_csv=args.split_csv,
        image_dir=args.image_dir,
    )
    print(
        f"Prepared manifest | rows={summary['matched_rows']} | "
        f"matched_cases={summary['matched_cases']} | "
        f"output={summary['output_csv']}"
    )
    if summary.get("warning"):
        print(summary["warning"])


if __name__ == "__main__":
    main()
