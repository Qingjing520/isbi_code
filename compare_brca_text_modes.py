from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

from configs.config import get_config
from datasets.feature_dataset import MultiModalFeatureDataset
from train import load_and_eval, train_one_split


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"
DEFAULT_SPLIT_DIR = Path(r"D:\Tasks\Split_Table\BRCA")
DEFAULT_LABEL_FILE = Path(r"D:\Tasks\Pathologic_Stage_Label\BRCA_pathologic_stage.csv")
DEFAULT_IMAGE_DIR = Path(r"D:\Tasks\WSI_extract_features\BRCA_WSI_extract_features")
DEFAULT_SENTENCE_TEXT_DIR = Path(r"D:\Tasks\Text_Sentence_extract_features\BRCA_text")
DEFAULT_HIERARCHY_TEXT_DIR = (
    PROJECT_ROOT / "pathology_report_extraction" / "Output" / "text_hierarchy_graphs_masked" / "BRCA"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "brca_text_mode_comparison_5splits"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compare BRCA sentence_pt vs hierarchy_graph text modes across multiple splits."
    )
    ap.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    ap.add_argument("--split_dir", type=str, default=str(DEFAULT_SPLIT_DIR))
    ap.add_argument("--label_file", type=str, default=str(DEFAULT_LABEL_FILE))
    ap.add_argument("--image_dir", type=str, default=str(DEFAULT_IMAGE_DIR))
    ap.add_argument("--sentence_text_dir", type=str, default=str(DEFAULT_SENTENCE_TEXT_DIR))
    ap.add_argument("--hierarchy_text_dir", type=str, default=str(DEFAULT_HIERARCHY_TEXT_DIR))
    ap.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--num_splits", type=int, default=5)
    ap.add_argument("--split_offset", type=int, default=0)
    ap.add_argument("--hierarchy_feature", type=str, default="section_features")
    ap.add_argument(
        "--train_num_workers",
        type=int,
        default=0,
        help="Force dataloader num_workers for this comparison. Default 0 for Windows stability.",
    )
    ap.add_argument("--skip_existing", action="store_true")
    return ap.parse_args()


def _dataset_size(
    split_csv: str,
    label_csv: str,
    image_dir: str,
    text_dir: str,
    text_mode: str,
    text_graph_feature: str,
    mode: str,
) -> int:
    ds = MultiModalFeatureDataset(
        split_csv=split_csv,
        label_csv=label_csv,
        image_dir=image_dir,
        text_dir=text_dir,
        text_mode=text_mode,
        text_graph_feature=text_graph_feature,
        mode=mode,
    )
    return len(ds)


def _safe_mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return None
    return float(statistics.mean(vals))


def _safe_std(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if len(vals) < 2:
        return 0.0 if vals else None
    return float(statistics.stdev(vals))


def _mode_text_dir(args: argparse.Namespace, mode_name: str) -> str:
    if mode_name == "sentence_pt":
        return os.path.abspath(args.sentence_text_dir)
    return os.path.abspath(args.hierarchy_text_dir)


def _mode_cfg(base_cfg: Any, args: argparse.Namespace, split_idx: int, mode_name: str) -> Any:
    cfg = copy.deepcopy(base_cfg)
    cfg.data.split_file = os.path.abspath(os.path.join(args.split_dir, f"split_{split_idx}.csv"))
    cfg.data.label_file = os.path.abspath(args.label_file)
    cfg.data.image_dir = os.path.abspath(args.image_dir)
    cfg.data.text_dir = _mode_text_dir(args, mode_name)
    cfg.data.text_mode = mode_name
    cfg.data.text_graph_feature = (
        args.hierarchy_feature if mode_name == "hierarchy_graph" else "node_features"
    )
    cfg.train.num_workers = int(args.train_num_workers)
    cfg.output.exp_dir = os.path.abspath(
        os.path.join(args.output_dir, mode_name, f"split_{split_idx}")
    )
    return cfg


def _load_final_metrics(exp_dir: str) -> dict[str, float] | None:
    log_path = os.path.join(exp_dir, "log.jsonl")
    if not os.path.exists(log_path):
        return None

    final_record = None
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("type") == "final_evaluation":
                final_record = record

    if not final_record:
        return None

    return {
        "acc": float(final_record["target"]["acc"]),
        "auc": float(final_record["target"]["auc"]),
    }


def _train_or_resume(cfg: Any, skip_existing: bool) -> dict[str, float]:
    os.makedirs(cfg.output.exp_dir, exist_ok=True)
    best_ckpt = os.path.join(cfg.output.exp_dir, "best_model.pt")

    if skip_existing and os.path.exists(best_ckpt):
        metrics = _load_final_metrics(cfg.output.exp_dir)
        if metrics is None:
            metrics = load_and_eval(cfg, best_ckpt)["target"]
        return {"acc": float(metrics["acc"]), "auc": float(metrics["auc"])}

    best_path = train_one_split(cfg)
    metrics = load_and_eval(cfg, best_path)["target"]
    return {"acc": float(metrics["acc"]), "auc": float(metrics["auc"])}


def _write_aggregate(output_dir: str, split_count: int, hierarchy_feature: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        grouped.setdefault(row["mode"], []).append(row)

    aggregate = {
        "dataset": "BRCA",
        "num_splits": split_count,
        "hierarchy_feature": hierarchy_feature,
        "modes": {},
    }
    for mode_name, rows in grouped.items():
        accs = [float(r["acc"]) for r in rows]
        aucs = [float(r["auc"]) for r in rows]
        aggregate["modes"][mode_name] = {
            "runs": len(rows),
            "acc_mean": _safe_mean(accs),
            "acc_std": _safe_std(accs),
            "auc_mean": _safe_mean(aucs),
            "auc_std": _safe_std(aucs),
        }

    aggregate_path = os.path.join(output_dir, "comparison_summary.json")
    with open(aggregate_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)
    return aggregate


def _write_results_csv(output_dir: str, records: list[dict[str, Any]]) -> None:
    csv_path = os.path.join(output_dir, "comparison_results.csv")
    fieldnames = [
        "split_idx",
        "mode",
        "text_dir",
        "text_graph_feature",
        "train_size",
        "test_size",
        "acc",
        "auc",
        "exp_dir",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    base_cfg = get_config(args.config)

    split_indices = list(range(args.split_offset, args.split_offset + args.num_splits))
    modes = ["sentence_pt", "hierarchy_graph"]
    records: list[dict[str, Any]] = []

    plan = {
        "dataset": "BRCA",
        "split_indices": split_indices,
        "sentence_text_dir": os.path.abspath(args.sentence_text_dir),
        "hierarchy_text_dir": os.path.abspath(args.hierarchy_text_dir),
        "hierarchy_feature": args.hierarchy_feature,
        "label_file": os.path.abspath(args.label_file),
        "image_dir": os.path.abspath(args.image_dir),
        "output_dir": os.path.abspath(args.output_dir),
    }
    with open(os.path.join(args.output_dir, "comparison_plan.json"), "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)

    for split_idx in split_indices:
        for mode_name in modes:
            cfg = _mode_cfg(base_cfg, args, split_idx, mode_name)
            train_size = _dataset_size(
                cfg.data.split_file,
                cfg.data.label_file,
                cfg.data.image_dir,
                cfg.data.text_dir,
                cfg.data.text_mode,
                cfg.data.text_graph_feature,
                mode="train",
            )
            test_size = _dataset_size(
                cfg.data.split_file,
                cfg.data.label_file,
                cfg.data.image_dir,
                cfg.data.text_dir,
                cfg.data.text_mode,
                cfg.data.text_graph_feature,
                mode="test",
            )

            print(
                f"[BRCA compare] split={split_idx:03d} mode={mode_name} "
                f"train={train_size} test={test_size}",
                flush=True,
            )
            metrics = _train_or_resume(cfg, skip_existing=args.skip_existing)
            record = {
                "split_idx": split_idx,
                "mode": mode_name,
                "text_dir": cfg.data.text_dir,
                "text_graph_feature": cfg.data.text_graph_feature,
                "train_size": train_size,
                "test_size": test_size,
                "acc": float(metrics["acc"]),
                "auc": float(metrics["auc"]),
                "exp_dir": cfg.output.exp_dir,
            }
            records.append(record)

            summary_line = (
                f"split={split_idx:03d}, mode={mode_name}, acc={record['acc']:.4f}, "
                f"auc={record['auc']:.4f}"
            )
            print(f"[BRCA compare] {summary_line}", flush=True)

            with open(
                os.path.join(args.output_dir, "comparison_results.jsonl"),
                "a",
                encoding="utf-8",
            ) as f:
                f.write(json.dumps(record) + "\n")

            _write_results_csv(args.output_dir, records)
            _write_aggregate(args.output_dir, len(split_indices), args.hierarchy_feature, records)
    _write_results_csv(args.output_dir, records)

    aggregate = _write_aggregate(args.output_dir, len(split_indices), args.hierarchy_feature, records)

    print("\n[BRCA compare] Done.", flush=True)
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == "__main__":
    main()
