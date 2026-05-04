from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def _safe_mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _safe_std(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return float(math.sqrt(var))


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_details(experiment_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    root_analysis = experiment_dir / "analysis.json"
    if root_analysis.exists():
        with root_analysis.open("r", encoding="utf-8") as f:
            analysis = json.load(f)
        for item in analysis.get("sample_details", []):
            row = dict(item)
            row["split_idx"] = 0
            row["confidence"] = abs(float(row.get("prob_late", 0.5)) - 0.5)
            rows.append(row)

    for split_dir in sorted(experiment_dir.glob("split_*")):
        if not split_dir.is_dir():
            continue
        try:
            split_idx = int(split_dir.name.split("_")[-1])
        except ValueError:
            continue

        analysis_path = split_dir / "analysis.json"
        if not analysis_path.exists():
            continue

        with analysis_path.open("r", encoding="utf-8") as f:
            analysis = json.load(f)

        for item in analysis.get("sample_details", []):
            row = dict(item)
            row["split_idx"] = split_idx
            row["confidence"] = abs(float(row.get("prob_late", 0.5)) - 0.5)
            rows.append(row)
    return rows


def _section_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        title = str(row.get("top_section_title") or "UNKNOWN")
        counter[title] += 1
    return dict(counter.most_common())


def _metric_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gates = [x for row in rows if (x := _as_float(row.get("fusion_gate"))) is not None]
    graph_weights = [x for row in rows if (x := _as_float(row.get("graph_branch_weight"))) is not None]
    entropies = [x for row in rows if (x := _as_float(row.get("doc_attention_entropy"))) is not None]
    top_weights = [x for row in rows if (x := _as_float(row.get("top_section_attention"))) is not None]
    return {
        "n": len(rows),
        "fusion_gate_mean": _safe_mean(gates),
        "fusion_gate_std": _safe_std(gates),
        "fusion_gate_is_sentence_branch_weight": True,
        "graph_branch_weight_mean": _safe_mean(graph_weights),
        "graph_branch_weight_std": _safe_std(graph_weights),
        "doc_attention_entropy_mean": _safe_mean(entropies),
        "doc_attention_entropy_std": _safe_std(entropies),
        "top_section_attention_mean": _safe_mean(top_weights),
        "top_section_attention_std": _safe_std(top_weights),
        "top_section_counts": _section_counts(rows),
    }


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "split_idx",
        "slide_id",
        "label",
        "pred",
        "prob_late",
        "correct",
        "confidence",
        "fusion_gate",
        "sentence_branch_weight",
        "graph_branch_weight",
        "top_section_title",
        "top_section_index",
        "top_section_attention",
        "doc_attention_entropy",
        "section_attention_mix",
        "document_attention_mix",
        "graph_json_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect per-sample dual_text gate/attention diagnostics from split analysis.json files."
    )
    parser.add_argument("--experiment_dir", required=True, help="Directory containing split_*/analysis.json files.")
    parser.add_argument(
        "--output_dir",
        default="",
        help="Directory to write summary files. Defaults to the experiment directory.",
    )
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    output_dir = Path(args.output_dir) if args.output_dir else experiment_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_details(experiment_dir)
    correct_rows = [row for row in rows if bool(row.get("correct"))]
    wrong_rows = [row for row in rows if not bool(row.get("correct"))]

    _write_rows_csv(output_dir / "dual_text_sample_details.csv", rows)

    wrong_by_confidence = sorted(wrong_rows, key=lambda row: float(row.get("confidence", 0.0)), reverse=True)
    summary = {
        "experiment_dir": str(experiment_dir),
        "sample_count": len(rows),
        "correct_count": len(correct_rows),
        "wrong_count": len(wrong_rows),
        "overall": _metric_block(rows),
        "correct": _metric_block(correct_rows),
        "wrong": _metric_block(wrong_rows),
        "most_confident_wrong": wrong_by_confidence[:20],
    }
    with (output_dir / "dual_text_sample_analysis_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
