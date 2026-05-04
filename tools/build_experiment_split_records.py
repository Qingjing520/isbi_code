"""Build split-level result indexes for the organized experiment tree.

The experiment tree keeps full training artifacts under ``runs/``.  The sibling
``records/`` directory is intentionally narrower: it should contain only stable
split-level indexes that are easy to compare across methods.

By default this script performs a dry run.  Pass ``--apply`` to write
``split_results.csv/json/md`` and archive old record folders into
``runs/_legacy_records``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DATASETS = ("BRCA", "KIRC", "LUSC")
METHODS = (
    "sentence-only",
    "sentence-ontology",
    "sentence-hierarchical-graph",
    "sentence-hierarchical-graph-ontology",
)

MODE_TO_METHOD = {
    "sentence_pt": "sentence-only",
    "sentence": "sentence-only",
    "hierarchy_graph": "sentence-hierarchical-graph",
    "dual_text": "sentence-hierarchical-graph",
    "dual_text_concept_graph": "sentence-hierarchical-graph-ontology",
}

RESULT_FIELDS = [
    "dataset",
    "method",
    "run_name",
    "split_idx",
    "status",
    "acc",
    "auc",
    "source_type",
    "source_path",
    "text_mode",
    "ontology_variant",
    "text_dir",
    "text_graph_feature",
    "text_use_graph_structure",
    "split_file",
    "graph_manifest_csv",
    "exp_dir",
    "best_path",
    "train_size",
    "test_size",
    "fusion_gate_mean",
    "doc_attention_entropy_mean",
    "doc_attention_max_mean",
]


@dataclass(frozen=True)
class MethodDir:
    dataset: str
    method: str
    path: Path

    @property
    def runs_dir(self) -> Path:
        return self.path / "runs"

    @property
    def records_dir(self) -> Path:
        return self.path / "records"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments-root",
        default="experiments",
        type=Path,
        help="Organized experiments root containing dataset directories.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write split result records and archive old records content.",
    )
    return parser.parse_args()


def resolve_inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    root_resolved = root.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise RuntimeError(f"Refusing to operate outside {root_resolved}: {resolved}") from exc
    return resolved


def to_repo_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def split_idx_from_value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    match = re.search(r"(\d+)", text)
    return match.group(1) if match else text


def split_idx_from_path(path: Path) -> str:
    match = re.search(r"split[_-](\d+)", str(path), flags=re.IGNORECASE)
    return match.group(1) if match else ""


def infer_ontology_variant(run_name: str, graph_manifest_csv: str = "") -> str:
    text = f"{run_name} {graph_manifest_csv}".lower()
    variants = (
        "full_multi_ontology",
        "ncit_snomed_mapped",
        "ncit_only",
        "ncit_do",
    )
    for variant in variants:
        if variant in text:
            return variant
    return ""


def normalize_mode(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def method_for_row(default_method: str, row: dict[str, Any]) -> str:
    mode = normalize_mode(str(row.get("mode") or row.get("text_mode") or ""))
    return MODE_TO_METHOD.get(mode, default_method)


def method_dirs(experiments_root: Path) -> list[MethodDir]:
    dirs: list[MethodDir] = []
    for dataset in DATASETS:
        for method in METHODS:
            path = experiments_root / dataset / method
            if path.exists():
                dirs.append(MethodDir(dataset=dataset, method=method, path=path))
    return dirs


def config_for_split(run_dir: Path, split_idx: str, exp_dir: str = "") -> dict[str, Any]:
    candidates: list[Path] = []
    if split_idx:
        candidates.append(run_dir / f"split_{split_idx}" / "config.json")
    if exp_dir:
        exp_path = Path(exp_dir)
        if not exp_path.is_absolute():
            exp_path = run_dir / exp_path
        candidates.append(exp_path / "config.json")
    for candidate in candidates:
        if candidate.exists():
            return read_json(candidate)
    return {}


def config_fields(config: dict[str, Any]) -> dict[str, str]:
    data = config.get("data") if isinstance(config.get("data"), dict) else {}
    output = config.get("output") if isinstance(config.get("output"), dict) else {}
    return {
        "text_mode": str(data.get("text_mode", "")),
        "text_dir": str(data.get("text_dir", "")),
        "text_graph_feature": str(data.get("text_graph_feature", "")),
        "text_use_graph_structure": str(data.get("text_use_graph_structure", "")),
        "split_file": str(data.get("split_file", "")),
        "graph_manifest_csv": str(data.get("graph_manifest_csv", "")),
        "exp_dir": str(output.get("exp_dir", "")),
    }


def row_from_csv(
    *,
    dataset: str,
    method: str,
    run_name: str,
    source_path: Path,
    source_type: str,
    row: dict[str, str],
    repo_root: Path,
    run_dir: Path | None = None,
) -> dict[str, str]:
    split_idx = split_idx_from_value(row.get("split_idx") or row.get("split"))
    exp_dir = row.get("exp_dir", "")
    config = config_for_split(run_dir, split_idx, exp_dir) if run_dir else {}
    config_values = config_fields(config)
    text_mode = row.get("mode") or row.get("text_mode") or config_values["text_mode"]
    graph_manifest_csv = row.get("graph_manifest_csv") or config_values["graph_manifest_csv"]

    record = {
        "dataset": dataset,
        "method": method,
        "run_name": run_name,
        "split_idx": split_idx,
        "status": row.get("status", "recorded"),
        "acc": row.get("acc", ""),
        "auc": row.get("auc", ""),
        "source_type": source_type,
        "source_path": to_repo_path(source_path, repo_root),
        "text_mode": text_mode,
        "ontology_variant": infer_ontology_variant(run_name, graph_manifest_csv),
        "text_dir": row.get("text_dir") or config_values["text_dir"],
        "text_graph_feature": row.get("text_graph_feature") or config_values["text_graph_feature"],
        "text_use_graph_structure": row.get("text_use_graph_structure")
        or config_values["text_use_graph_structure"],
        "split_file": row.get("split_file") or config_values["split_file"],
        "graph_manifest_csv": graph_manifest_csv,
        "exp_dir": exp_dir or config_values["exp_dir"],
        "best_path": row.get("best_path", ""),
        "train_size": row.get("train_size", ""),
        "test_size": row.get("test_size", ""),
        "fusion_gate_mean": row.get("fusion_gate_mean", ""),
        "doc_attention_entropy_mean": row.get("doc_attention_entropy_mean", ""),
        "doc_attention_max_mean": row.get("doc_attention_max_mean", ""),
    }
    return {field: str(record.get(field, "")) for field in RESULT_FIELDS}


def best_log_record(log_path: Path) -> dict[str, Any]:
    best: dict[str, Any] = {}
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return best

    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        target = event.get("target")
        if not isinstance(target, dict) or "auc" not in target:
            continue
        try:
            auc = float(target["auc"])
        except (TypeError, ValueError):
            continue
        if not best or auc > float(best["auc"]):
            best = {
                "epoch": str(event.get("epoch", "")),
                "acc": str(target.get("acc", "")),
                "auc": str(target.get("auc", "")),
                "analysis": target.get("analysis") if isinstance(target.get("analysis"), dict) else {},
            }
    return best


def row_from_log(
    *,
    dataset: str,
    method: str,
    run_name: str,
    split_dir: Path,
    repo_root: Path,
) -> dict[str, str] | None:
    log_path = split_dir / "log.jsonl"
    if not log_path.exists():
        return None
    best = best_log_record(log_path)
    if not best:
        return None

    config_values = config_fields(read_json(split_dir / "config.json"))
    analysis = best.get("analysis") if isinstance(best.get("analysis"), dict) else {}
    split_idx = split_idx_from_path(split_dir)
    graph_manifest_csv = config_values["graph_manifest_csv"]
    record = {
        "dataset": dataset,
        "method": method,
        "run_name": run_name,
        "split_idx": split_idx,
        "status": f"from_log_best_epoch_{best.get('epoch', '')}",
        "acc": str(best.get("acc", "")),
        "auc": str(best.get("auc", "")),
        "source_type": "log.jsonl",
        "source_path": to_repo_path(log_path, repo_root),
        "text_mode": config_values["text_mode"],
        "ontology_variant": infer_ontology_variant(run_name, graph_manifest_csv),
        "text_dir": config_values["text_dir"],
        "text_graph_feature": config_values["text_graph_feature"],
        "text_use_graph_structure": config_values["text_use_graph_structure"],
        "split_file": config_values["split_file"],
        "graph_manifest_csv": graph_manifest_csv,
        "exp_dir": config_values["exp_dir"],
        "best_path": "",
        "train_size": "",
        "test_size": "",
        "fusion_gate_mean": str(analysis.get("fusion_gate_mean", "")),
        "doc_attention_entropy_mean": str(analysis.get("doc_attention_entropy_mean", "")),
        "doc_attention_max_mean": str(analysis.get("doc_attention_max_mean", "")),
    }
    return {field: str(record.get(field, "")) for field in RESULT_FIELDS}


def run_name_for_legacy(record_dir: Path, row: dict[str, str]) -> str:
    mode = normalize_mode(row.get("mode", ""))
    if mode and not record_dir.name.lower().endswith(f"__{mode}"):
        return f"{record_dir.name}__{mode}"
    return record_dir.name


def collect_split_records(experiments_root: Path, repo_root: Path) -> dict[tuple[str, str], list[dict[str, str]]]:
    collected: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)

    for method_dir in method_dirs(experiments_root):
        if method_dir.runs_dir.exists():
            for run_dir in sorted(child for child in method_dir.runs_dir.iterdir() if child.is_dir()):
                if run_dir.name == "_legacy_records":
                    continue
                for csv_name, source_type in (("summary.csv", "summary.csv"), ("comparison_results.csv", "comparison_results.csv")):
                    csv_path = run_dir / csv_name
                    if not csv_path.exists():
                        continue
                    for row in read_csv_rows(csv_path):
                        target_method = method_for_row(method_dir.method, row)
                        collected[(method_dir.dataset, target_method)].append(
                            row_from_csv(
                                dataset=method_dir.dataset,
                                method=target_method,
                                run_name=run_dir.name,
                                source_path=csv_path,
                                source_type=source_type,
                                row=row,
                                repo_root=repo_root,
                                run_dir=run_dir,
                            )
                        )

                for split_dir in sorted(run_dir.glob("split_*")):
                    if not split_dir.is_dir():
                        continue
                    record = row_from_log(
                        dataset=method_dir.dataset,
                        method=method_dir.method,
                        run_name=run_dir.name,
                        split_dir=split_dir,
                        repo_root=repo_root,
                    )
                    if record:
                        collected[(method_dir.dataset, method_dir.method)].append(record)

        for legacy_csv, source_type in legacy_comparison_files(method_dir):
            record_dir = legacy_csv.parent
            for row in read_csv_rows(legacy_csv):
                target_method = method_for_row(method_dir.method, row)
                run_name = run_name_for_legacy(record_dir, row)
                collected[(method_dir.dataset, target_method)].append(
                    row_from_csv(
                        dataset=method_dir.dataset,
                        method=target_method,
                        run_name=run_name,
                        source_path=legacy_csv,
                        source_type=source_type,
                        row=row,
                        repo_root=repo_root,
                        run_dir=None,
                    )
                )

    return {key: dedupe_records(rows) for key, rows in collected.items()}


def legacy_comparison_files(method_dir: MethodDir) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    if method_dir.records_dir.exists():
        files.extend(
            (path, "legacy_record_comparison_results.csv")
            for path in sorted(method_dir.records_dir.glob("**/comparison_results.csv"))
        )
    legacy_records_dir = method_dir.runs_dir / "_legacy_records"
    if legacy_records_dir.exists():
        files.extend(
            (path, "archived_legacy_record_comparison_results.csv")
            for path in sorted(legacy_records_dir.glob("**/comparison_results.csv"))
        )
    return files


def dedupe_records(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    priority = {
        "summary.csv": 40,
        "comparison_results.csv": 30,
        "legacy_record_comparison_results.csv": 25,
        "archived_legacy_record_comparison_results.csv": 25,
        "log.jsonl": 10,
    }
    best_by_key: dict[tuple[str, str, str, str], dict[str, str]] = {}

    for row in rows:
        key = (row["dataset"], row["method"], row["run_name"], row["split_idx"])
        current = best_by_key.get(key)
        if current is None:
            best_by_key[key] = row
            continue
        if priority.get(row["source_type"], 0) > priority.get(current["source_type"], 0):
            best_by_key[key] = row

    return sorted(
        best_by_key.values(),
        key=lambda row: (
            row["dataset"],
            row["method"],
            row["run_name"],
            int(row["split_idx"]) if row["split_idx"].isdigit() else 10**9,
            row["split_idx"],
        ),
    )


def write_records(records_dir: Path, rows: list[dict[str, str]]) -> None:
    records_dir.mkdir(parents=True, exist_ok=True)
    csv_path = records_dir / "split_results.csv"
    json_path = records_dir / "split_results.json"
    md_path = records_dir / "split_results.md"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(rows), encoding="utf-8")


def render_markdown(rows: list[dict[str, str]]) -> str:
    lines = [
        "# Split Results",
        "",
        "This file is generated by `tools/build_experiment_split_records.py`.",
        "",
        "| run_name | split_idx | acc | auc | status | source_type | ontology_variant |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    if not rows:
        lines.append("| _no split results found_ |  |  |  |  |  |  |")
    for row in rows:
        lines.append(
            "| {run_name} | {split_idx} | {acc} | {auc} | {status} | {source_type} | {ontology_variant} |".format(
                **{field: row.get(field, "") for field in RESULT_FIELDS}
            )
        )
    lines.append("")
    return "\n".join(lines)


def archive_legacy_records(method_dir: MethodDir, experiments_root: Path, dry_run: bool) -> list[str]:
    actions: list[str] = []
    records_dir = method_dir.records_dir
    if not records_dir.exists():
        return actions

    keep_names = {"split_results.csv", "split_results.json", "split_results.md"}
    legacy_items = [item for item in records_dir.iterdir() if item.name not in keep_names]
    if not legacy_items:
        return actions

    legacy_root = method_dir.runs_dir / "_legacy_records"
    resolve_inside(legacy_root, experiments_root)

    for item in legacy_items:
        resolve_inside(item, experiments_root)
        target = legacy_root / item.name
        suffix = 1
        while target.exists():
            target = legacy_root / f"{item.stem}_{suffix}{item.suffix}" if item.is_file() else legacy_root / f"{item.name}_{suffix}"
            suffix += 1
        actions.append(f"archive {item} -> {target}")
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(item), str(target))
    return actions


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd()
    experiments_root = args.experiments_root
    resolve_inside(experiments_root, repo_root)

    all_actions: list[str] = []

    if args.apply:
        for method_dir in method_dirs(experiments_root):
            all_actions.extend(archive_legacy_records(method_dir, experiments_root, dry_run=False))

    collected = collect_split_records(experiments_root, repo_root)

    for method_dir in method_dirs(experiments_root):
        rows = collected.get((method_dir.dataset, method_dir.method), [])
        all_actions.append(
            f"write {len(rows):4d} rows -> {method_dir.records_dir / 'split_results.csv'}"
        )
        if args.apply:
            write_records(method_dir.records_dir, rows)
        else:
            all_actions.extend(archive_legacy_records(method_dir, experiments_root, dry_run=True))

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"{mode}: experiment split records")
    for action in all_actions:
        print(f"- {action}")
    if not args.apply:
        print("Pass --apply to write files and archive legacy records.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
