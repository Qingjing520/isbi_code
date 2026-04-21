from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from configs.config import get_config
from train import load_and_eval


DATASET_DEFAULTS = {
    "BRCA": {
        "split_dir": r"D:\Tasks\Split_Table\BRCA",
        "label_file": r"D:\Tasks\Pathologic_Stage_Label\BRCA_pathologic_stage.csv",
        "image_dir": r"D:\Tasks\WSI_extract_features\BRCA_WSI_extract_features",
        "sentence_text_dir": r"D:\Tasks\Text_Sentence_extract_features\BRCA_text",
        "graph_text_dir": r"D:\Tasks\isbi_code\pathology_report_extraction\Output\text_hierarchy_graphs_masked\BRCA",
        "sentence_exp_root": r"D:\Tasks\isbi_code\experiments\brca_text_mode_comparison_5splits_node_features\sentence_pt",
        "dual_exp_root": r"D:\Tasks\isbi_code\experiments\brca_dual_text_readout_v2_gate001_10splits",
    },
    "KIRC": {
        "split_dir": r"D:\Tasks\Split_Table\KIRC",
        "label_file": r"D:\Tasks\Pathologic_Stage_Label\KIRC_pathologic_stage.csv",
        "image_dir": r"D:\Tasks\WSI_extract_features\KIRC_WSI_extract_features",
        "sentence_text_dir": r"D:\Tasks\Text_Sentence_extract_features\KIRC_text",
        "graph_text_dir": r"D:\Tasks\isbi_code\pathology_report_extraction\Output\text_hierarchy_graphs_masked\KIRC",
        "sentence_exp_root": r"D:\Tasks\isbi_code\experiments\kirc_text_mode_comparison_20splits_node_features\sentence_pt",
        "dual_exp_root": r"D:\Tasks\isbi_code\experiments\kirc_dual_text_readout_v2_gate001_10splits",
    },
}


def _mean(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def _std(values: list[float]) -> float | None:
    return float(statistics.stdev(values)) if len(values) > 1 else (0.0 if values else None)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "split_idx",
        "slide_id",
        "label",
        "sentence_pred",
        "dual_pred",
        "sentence_prob_late",
        "dual_prob_late",
        "sentence_correct",
        "dual_correct",
        "benefit_type",
        "graph_branch_weight",
        "fusion_gate",
        "top_section_title",
        "top_section_attention",
        "doc_attention_entropy",
        "graph_json_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_dual_details(split_dir: Path) -> dict[str, dict[str, Any]]:
    analysis_path = split_dir / "analysis.json"
    if not analysis_path.exists():
        raise FileNotFoundError(f"Missing dual_text analysis file: {analysis_path}")
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    details = analysis.get("sample_details", [])
    return {str(row["slide_id"]): row for row in details}


def _evaluate_dual_details(cfg_path: Path, defaults: dict[str, str], split_idx: int, ckpt_path: Path) -> dict[str, dict[str, Any]]:
    cfg = get_config(str(cfg_path))
    cfg.data.split_file = str(Path(defaults["split_dir"]) / f"split_{split_idx}.csv")
    cfg.data.label_file = defaults["label_file"]
    cfg.data.image_dir = defaults["image_dir"]
    cfg.data.text_mode = "dual_text"
    cfg.data.text_dir = defaults["sentence_text_dir"]
    cfg.data.sentence_text_dir = defaults["sentence_text_dir"]
    cfg.data.graph_text_dir = defaults["graph_text_dir"]
    cfg.data.graph_manifest_csv = ""
    cfg.train.num_workers = 0
    cfg.output.exp_dir = str(ckpt_path.parent)
    metrics = load_and_eval(cfg, str(ckpt_path))
    analysis = metrics.get("analysis", {})
    details = analysis.get("sample_details", [])
    if not details:
        raise RuntimeError(f"No dual_text sample details returned for split {split_idx}: {ckpt_path}")
    with (ckpt_path.parent / "analysis.json").open("w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)
    return {str(row["slide_id"]): row for row in details}


def _evaluate_sentence_details(cfg_path: Path, defaults: dict[str, str], split_idx: int, ckpt_path: Path) -> dict[str, dict[str, Any]]:
    cfg = get_config(str(cfg_path))
    cfg.data.split_file = str(Path(defaults["split_dir"]) / f"split_{split_idx}.csv")
    cfg.data.label_file = defaults["label_file"]
    cfg.data.image_dir = defaults["image_dir"]
    cfg.data.text_mode = "sentence_pt"
    cfg.data.text_dir = defaults["sentence_text_dir"]
    cfg.data.sentence_text_dir = defaults["sentence_text_dir"]
    cfg.data.graph_text_dir = defaults["graph_text_dir"]
    cfg.data.graph_manifest_csv = ""
    cfg.train.num_workers = 0
    cfg.output.exp_dir = str(ckpt_path.parent)
    metrics = load_and_eval(cfg, str(ckpt_path))
    details = metrics.get("analysis", {}).get("sample_details", [])
    if not details:
        raise RuntimeError(f"No sentence_pt sample details returned for split {split_idx}: {ckpt_path}")
    return {str(row["slide_id"]): row for row in details}


def _block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    graph_weights = [v for row in rows if (v := _as_float(row.get("graph_branch_weight"))) is not None]
    top_weights = [v for row in rows if (v := _as_float(row.get("top_section_attention"))) is not None]
    entropies = [v for row in rows if (v := _as_float(row.get("doc_attention_entropy"))) is not None]
    return {
        "n": len(rows),
        "graph_branch_weight_mean": _mean(graph_weights),
        "graph_branch_weight_std": _std(graph_weights),
        "top_section_attention_mean": _mean(top_weights),
        "top_section_attention_std": _std(top_weights),
        "doc_attention_entropy_mean": _mean(entropies),
        "doc_attention_entropy_std": _std(entropies),
        "top_section_counts": dict(Counter(str(row.get("top_section_title") or "UNKNOWN") for row in rows).most_common(20)),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {
        "benefited": [r for r in rows if r["benefit_type"] == "dual_correct_sentence_wrong"],
        "harmed": [r for r in rows if r["benefit_type"] == "sentence_correct_dual_wrong"],
        "both_correct": [r for r in rows if r["benefit_type"] == "both_correct"],
        "both_wrong": [r for r in rows if r["benefit_type"] == "both_wrong"],
    }
    return {
        "total_samples": len(rows),
        "group_counts": {k: len(v) for k, v in groups.items()},
        "groups": {k: _block(v) for k, v in groups.items()},
        "benefit_minus_harm": len(groups["benefited"]) - len(groups["harmed"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare sentence_pt and dual_text sample-level predictions to find hierarchy-benefited reports."
    )
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_DEFAULTS))
    parser.add_argument("--config", default=r"D:\Tasks\isbi_code\configs\config.yaml")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--num_splits", type=int, default=10)
    parser.add_argument("--split_offset", type=int, default=0)
    parser.add_argument("--sentence_exp_root", default="")
    parser.add_argument("--dual_exp_root", default="")
    args = parser.parse_args()

    dataset = args.dataset.upper()
    defaults = dict(DATASET_DEFAULTS[dataset])
    if args.sentence_exp_root:
        defaults["sentence_exp_root"] = args.sentence_exp_root
    if args.dual_exp_root:
        defaults["dual_exp_root"] = args.dual_exp_root

    output_dir = Path(args.output_dir) if args.output_dir else Path(defaults["dual_exp_root"]) / "benefit_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for split_idx in range(args.split_offset, args.split_offset + args.num_splits):
        sentence_ckpt = Path(defaults["sentence_exp_root"]) / f"split_{split_idx}" / "best_model.pt"
        dual_split_dir = Path(defaults["dual_exp_root"]) / f"split_{split_idx}"
        if not sentence_ckpt.exists():
            raise FileNotFoundError(f"Missing sentence_pt checkpoint: {sentence_ckpt}")
        if not dual_split_dir.exists():
            raise FileNotFoundError(f"Missing dual_text split directory: {dual_split_dir}")

        sentence_details = _evaluate_sentence_details(Path(args.config), defaults, split_idx, sentence_ckpt)
        try:
            dual_details = _load_dual_details(dual_split_dir)
        except FileNotFoundError:
            dual_ckpt = dual_split_dir / "best_model.pt"
            if not dual_ckpt.exists():
                raise
            dual_details = _evaluate_dual_details(Path(args.config), defaults, split_idx, dual_ckpt)
        common_ids = sorted(set(sentence_details) & set(dual_details))
        if not common_ids:
            raise RuntimeError(f"No overlapping slide IDs for split {split_idx}.")

        for slide_id in common_ids:
            sentence = sentence_details[slide_id]
            dual = dual_details[slide_id]
            sentence_correct = _as_bool(sentence.get("correct"))
            dual_correct = _as_bool(dual.get("correct"))
            if dual_correct and not sentence_correct:
                benefit_type = "dual_correct_sentence_wrong"
            elif sentence_correct and not dual_correct:
                benefit_type = "sentence_correct_dual_wrong"
            elif dual_correct and sentence_correct:
                benefit_type = "both_correct"
            else:
                benefit_type = "both_wrong"

            rows.append(
                {
                    "split_idx": split_idx,
                    "slide_id": slide_id,
                    "label": int(dual.get("label", sentence.get("label", -1))),
                    "sentence_pred": int(sentence.get("pred", -1)),
                    "dual_pred": int(dual.get("pred", -1)),
                    "sentence_prob_late": _as_float(sentence.get("prob_late")),
                    "dual_prob_late": _as_float(dual.get("prob_late")),
                    "sentence_correct": sentence_correct,
                    "dual_correct": dual_correct,
                    "benefit_type": benefit_type,
                    "fusion_gate": _as_float(dual.get("fusion_gate")),
                    "graph_branch_weight": _as_float(dual.get("graph_branch_weight")),
                    "top_section_title": dual.get("top_section_title"),
                    "top_section_attention": _as_float(dual.get("top_section_attention")),
                    "doc_attention_entropy": _as_float(dual.get("doc_attention_entropy")),
                    "graph_json_path": dual.get("graph_json_path"),
                }
            )
        print(f"[{dataset}] split={split_idx:03d} compared {len(common_ids)} samples", flush=True)

    _write_csv(output_dir / "dual_text_benefit_details.csv", rows)
    _write_csv(
        output_dir / "benefited_samples.csv",
        [r for r in rows if r["benefit_type"] == "dual_correct_sentence_wrong"],
    )
    _write_csv(
        output_dir / "harmed_samples.csv",
        [r for r in rows if r["benefit_type"] == "sentence_correct_dual_wrong"],
    )
    summary = {
        "dataset": dataset,
        "splits": list(range(args.split_offset, args.split_offset + args.num_splits)),
        "sentence_exp_root": defaults["sentence_exp_root"],
        "dual_exp_root": defaults["dual_exp_root"],
        "summary": _summarize(rows),
    }
    (output_dir / "dual_text_benefit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
