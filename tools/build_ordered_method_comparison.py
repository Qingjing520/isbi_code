from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
OUTPUT_DIR = REPO_ROOT / "experiment_records" / "comparisons" / "ordered_splits"

DATASETS = ("BRCA", "KIRC", "LUSC")
METHODS = (
    "sentence-only",
    "sentence-ontology",
    "sentence-hierarchical-graph",
    "sentence-hierarchical-graph-ontology",
)

METHOD_LABELS = {
    "sentence-only": "sentence-only",
    "sentence-ontology": "sentence + ontology",
    "sentence-hierarchical-graph": "sentence + hierarchical graph",
    "sentence-hierarchical-graph-ontology": "sentence + hierarchical graph + ontology",
}

LUSC_BASELINE_RUN = (
    EXPERIMENTS_ROOT
    / "LUSC"
    / "sentence-only"
    / "runs"
    / "LUSC_150splits_sentence_only"
)

RUNNING_RUNS = {
    ("LUSC", "sentence-ontology"): (
        EXPERIMENTS_ROOT
        / "LUSC"
        / "sentence-ontology"
        / "runs"
        / "lusc_sentence_ontology_0splits"
    ),
    ("LUSC", "sentence-hierarchical-graph"): (
        EXPERIMENTS_ROOT
        / "LUSC"
        / "sentence-hierarchical-graph"
        / "runs"
        / "lusc_sentence_hierarchical_graph_0splits"
    ),
    ("LUSC", "sentence-hierarchical-graph-ontology"): (
        EXPERIMENTS_ROOT
        / "LUSC"
        / "sentence-hierarchical-graph-ontology"
        / "runs"
        / "lusc_sentence_hierarchical_graph_ontology_0splits"
    ),
}

COMPARISON_FIELDS = [
    "dataset",
    "method",
    "method_label",
    "status",
    "official_splits",
    "started_splits",
    "acc",
    "auc",
    "delta_auc_vs_sentence_only",
    "source",
    "notes",
]


@dataclass(frozen=True)
class SplitMetric:
    split_idx: int
    acc: float
    auc: float
    is_final: bool
    source: Path


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def float_or_none(value: Any) -> float | None:
    try:
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def split_idx_from_name(path: Path) -> int | None:
    try:
        return int(path.name.split("_")[-1])
    except ValueError:
        return None


def metric_from_log(split_dir: Path) -> SplitMetric | None:
    split_idx = split_idx_from_name(split_dir)
    log_path = split_dir / "log.jsonl"
    if split_idx is None or not log_path.exists():
        return None

    best_acc: float | None = None
    best_auc: float | None = None
    final_acc: float | None = None
    final_auc: float | None = None

    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None

    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        target = event.get("target")
        if not isinstance(target, dict):
            continue

        acc = float_or_none(target.get("acc"))
        auc = float_or_none(target.get("auc"))
        if acc is None or auc is None:
            continue

        if event.get("type") == "final_evaluation":
            final_acc = acc
            final_auc = auc
            continue

        if best_auc is None or auc > best_auc:
            best_acc = acc
            best_auc = auc

    if final_acc is not None and final_auc is not None:
        return SplitMetric(split_idx=split_idx, acc=final_acc, auc=final_auc, is_final=True, source=log_path)
    if best_acc is not None and best_auc is not None:
        return SplitMetric(split_idx=split_idx, acc=best_acc, auc=best_auc, is_final=False, source=log_path)
    return None


def metrics_from_run(run_dir: Path) -> list[SplitMetric]:
    if not run_dir.exists():
        return []
    metrics = []
    for split_dir in sorted(run_dir.glob("split_*"), key=lambda path: split_idx_from_name(path) or 10**9):
        if not split_dir.is_dir():
            continue
        metric = metric_from_log(split_dir)
        if metric is not None:
            metrics.append(metric)
    return metrics


def metrics_from_records(dataset: str, method: str) -> list[SplitMetric]:
    records_path = EXPERIMENTS_ROOT / dataset / method / "records" / "split_results.csv"
    metrics: list[SplitMetric] = []
    for row in read_csv_rows(records_path):
        if row.get("status") == "skipped":
            continue
        acc = float_or_none(row.get("acc"))
        auc = float_or_none(row.get("auc"))
        split_idx = float_or_none(row.get("split_idx"))
        if acc is None or auc is None or split_idx is None:
            continue
        source = Path(row.get("source_path") or records_path)
        if not source.is_absolute():
            source = REPO_ROOT / source
        metrics.append(
            SplitMetric(
                split_idx=int(split_idx),
                acc=acc,
                auc=auc,
                is_final=True,
                source=source,
            )
        )
    return sorted(metrics, key=lambda item: item.split_idx)


def mean_std(values: list[float]) -> str:
    if not values:
        return ""
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.4f} +/- {std:.4f}"


def mean_value(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def source_for(metrics: list[SplitMetric], fallback: Path | None = None) -> str:
    if metrics:
        parent = metrics[0].source.parent
        if parent.name.startswith("split_"):
            return str(parent.parent)
        return str(parent)
    return str(fallback) if fallback is not None else ""


def official_metrics(dataset: str, method: str) -> tuple[list[SplitMetric], str, str]:
    if (dataset, method) == ("LUSC", "sentence-only"):
        metrics = metrics_from_run(LUSC_BASELINE_RUN)
        final_count = sum(1 for item in metrics if item.is_final)
        note = (
            "historical 150-split baseline; features correspond to "
            "F:\\Tasks\\Text_Sentence_extract_features\\LUSC_text after file migration; "
            "149 splits have final_evaluation, 1 uses best logged epoch"
        )
        status = "complete" if len(metrics) == 150 else "partial"
        if final_count != len(metrics):
            status = "complete-with-note"
        return metrics, status, note

    metrics = metrics_from_records(dataset, method)
    if metrics:
        return metrics, "complete", ""

    return [], "not-run", ""


def running_metrics(dataset: str, method: str) -> list[SplitMetric]:
    run_dir = RUNNING_RUNS.get((dataset, method))
    if run_dir is None:
        return []
    return metrics_from_run(run_dir)


def format_delta(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:+.4f}"


def build_rows() -> list[dict[str, str]]:
    baselines: dict[str, float | None] = {}
    official_cache: dict[tuple[str, str], tuple[list[SplitMetric], str, str]] = {}

    for dataset in DATASETS:
        metrics, status, note = official_metrics(dataset, "sentence-only")
        official_cache[(dataset, "sentence-only")] = (metrics, status, note)
        baselines[dataset] = mean_value([item.auc for item in metrics]) if metrics else None

    rows: list[dict[str, str]] = []
    for dataset in DATASETS:
        for method in METHODS:
            metrics, status, note = official_cache.get((dataset, method), official_metrics(dataset, method))
            if not metrics and (dataset, method) in RUNNING_RUNS:
                partial = running_metrics(dataset, method)
                started = len(partial)
                final_count = sum(1 for item in partial if item.is_final)
                status = "running" if started else "queued"
                note = f"official comparison pending; {started} splits started, {final_count} finalized"
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "status": status,
                        "official_splits": "0",
                        "started_splits": str(started),
                        "acc": "",
                        "auc": "",
                        "delta_auc_vs_sentence_only": "",
                        "source": source_for(partial, RUNNING_RUNS[(dataset, method)]),
                        "notes": note,
                    }
                )
                continue

            auc_mean = mean_value([item.auc for item in metrics])
            baseline = baselines.get(dataset)
            delta = None
            if auc_mean is not None and baseline is not None and method != "sentence-only":
                delta = auc_mean - baseline

            if not metrics and method == "sentence-ontology" and dataset in {"BRCA", "KIRC"}:
                note = "not run yet with the new sentence-ontology graph product"

            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "status": status,
                    "official_splits": str(len(metrics)),
                    "started_splits": str(len(metrics)),
                    "acc": mean_std([item.acc for item in metrics]),
                    "auc": mean_std([item.auc for item in metrics]),
                    "delta_auc_vs_sentence_only": format_delta(delta),
                    "source": source_for(metrics),
                    "notes": note,
                }
            )
    return rows


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(rows: list[dict[str, str]]) -> str:
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Ordered Method Comparison",
        "",
        f"Generated: {generated}",
        "",
        "This table aligns the four text-side experiment classes. Rows marked `running`, `queued`, or `not-run` are excluded from official comparison metrics until their splits finish.",
        "",
        "| Dataset | Method | Status | Official splits | ACC | AUC | Delta AUC vs sentence-only |",
        "|---|---|---|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            "| {dataset} | {method_label} | {status} | {official_splits} | {acc} | {auc} | {delta_auc_vs_sentence_only} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Sources And Notes",
            "",
            "| Dataset | Method | Source | Notes |",
            "|---|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| {dataset} | {method_label} | `{source}` | {notes} |".format(**row)
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    rows = build_rows()
    write_csv(rows, OUTPUT_DIR / "ordered_split_method_comparison.csv")
    (OUTPUT_DIR / "ordered_split_method_comparison.md").write_text(
        render_markdown(rows),
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} rows to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
