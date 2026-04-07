from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from configs.config import get_config
from train import load_and_eval, train_one_split


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"
DEFAULT_SPLIT_DIR = Path(r"D:\Tasks\Split_Table\KIRC")
DEFAULT_LABEL_FILE = Path(r"D:\Tasks\Pathologic_Stage_Label\KIRC_pathologic_stage.csv")
DEFAULT_IMAGE_DIR = Path(r"D:\Tasks\WSI_extract_features\KIRC_WSI_extract_features")
DEFAULT_SENTENCE_TEXT_DIR = Path(r"D:\Tasks\Text_Sentence_extract_features\KIRC_text")
DEFAULT_GRAPH_TEXT_DIR = (
    PROJECT_ROOT / "pathology_report_extraction" / "Output" / "text_hierarchy_graphs_masked" / "KIRC"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "kirc_dual_text_3splits"


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _format_seconds(seconds: float) -> str:
    total = int(round(float(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _infer_dataset_name(args: argparse.Namespace) -> str:
    if str(args.dataset_name).strip():
        return str(args.dataset_name).strip().upper()

    candidates = [
        os.path.basename(os.path.normpath(args.split_dir)),
        os.path.basename(os.path.normpath(args.sentence_text_dir)),
        os.path.basename(os.path.normpath(args.graph_text_dir)),
        os.path.basename(args.label_file),
    ]
    for item in candidates:
        upper = str(item).upper()
        for key in ("BRCA", "KIRC", "LUSC"):
            if key in upper:
                return key
    return "UNKNOWN"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run dual_text experiments across multiple splits and skip finished splits by default."
    )
    ap.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    ap.add_argument("--split_dir", type=str, default=str(DEFAULT_SPLIT_DIR))
    ap.add_argument("--label_file", type=str, default=str(DEFAULT_LABEL_FILE))
    ap.add_argument("--image_dir", type=str, default=str(DEFAULT_IMAGE_DIR))
    ap.add_argument("--sentence_text_dir", type=str, default=str(DEFAULT_SENTENCE_TEXT_DIR))
    ap.add_argument("--graph_text_dir", type=str, default=str(DEFAULT_GRAPH_TEXT_DIR))
    ap.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--num_splits", type=int, default=3)
    ap.add_argument("--split_offset", type=int, default=0)
    ap.add_argument("--dataset_name", type=str, default="")
    ap.add_argument("--train_num_workers", type=int, default=0)
    ap.add_argument(
        "--gate_reg_weight",
        type=float,
        default=None,
        help="Optional dual_text gate regularization weight. If set, encourages graph branch participation.",
    )
    ap.add_argument(
        "--graph_weight_target",
        type=float,
        default=None,
        help="Optional target average graph branch weight for dual_text gate regularization.",
    )
    ap.add_argument(
        "--force_rerun",
        action="store_true",
        help="Ignore existing best_model.pt files and rerun those splits from scratch.",
    )
    return ap.parse_args()


def _safe_mean(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def _safe_std(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


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


def _build_cfg(base_cfg: Any, args: argparse.Namespace, split_idx: int, output_dir: str) -> Any:
    cfg = copy.deepcopy(base_cfg)
    cfg.data.split_file = os.path.abspath(os.path.join(args.split_dir, f"split_{split_idx}.csv"))
    cfg.data.label_file = os.path.abspath(args.label_file)
    cfg.data.image_dir = os.path.abspath(args.image_dir)
    cfg.data.text_mode = "dual_text"
    cfg.data.text_dir = os.path.abspath(args.sentence_text_dir)
    cfg.data.sentence_text_dir = os.path.abspath(args.sentence_text_dir)
    cfg.data.graph_text_dir = os.path.abspath(args.graph_text_dir)
    cfg.data.text_use_graph_structure = False
    if args.gate_reg_weight is not None:
        cfg.loss.dual_text_gate_reg_weight = float(args.gate_reg_weight)
    if args.graph_weight_target is not None:
        cfg.loss.dual_text_graph_weight_target = float(args.graph_weight_target)
    cfg.train.num_workers = int(args.train_num_workers)
    cfg.output.exp_dir = os.path.abspath(output_dir)
    return cfg


def _write_results_csv(output_dir: str, rows: list[dict[str, Any]]) -> None:
    csv_path = os.path.join(output_dir, "comparison_results.csv")
    fieldnames = [
        "split_idx",
        "acc",
        "auc",
        "best_path",
        "exp_dir",
        "status",
        "fusion_gate_mean",
        "graph_branch_weight_mean",
        "doc_attention_entropy_mean",
        "doc_attention_max_mean",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(output_dir: str, dataset_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    accs = [float(row["acc"]) for row in rows]
    aucs = [float(row["auc"]) for row in rows]
    gate_means = [float(row["fusion_gate_mean"]) for row in rows if row.get("fusion_gate_mean") is not None]
    graph_branch_weight_means = [
        float(row["graph_branch_weight_mean"]) for row in rows if row.get("graph_branch_weight_mean") is not None
    ]
    doc_attn_entropy_means = [
        float(row["doc_attention_entropy_mean"])
        for row in rows
        if row.get("doc_attention_entropy_mean") is not None
    ]
    doc_attn_max_means = [
        float(row["doc_attention_max_mean"])
        for row in rows
        if row.get("doc_attention_max_mean") is not None
    ]
    summary = {
        "dataset": dataset_name,
        "num_splits": len(rows),
        "mode": "dual_text",
        "acc_mean": _safe_mean(accs),
        "acc_std": _safe_std(accs),
        "auc_mean": _safe_mean(aucs),
        "auc_std": _safe_std(aucs),
        "analysis": {
            "fusion_gate_mean": _safe_mean(gate_means),
            "fusion_gate_std": _safe_std(gate_means),
            "fusion_gate_is_sentence_branch_weight": True,
            "graph_branch_weight_mean": _safe_mean(graph_branch_weight_means),
            "graph_branch_weight_std": _safe_std(graph_branch_weight_means),
            "doc_attention_entropy_mean": _safe_mean(doc_attn_entropy_means),
            "doc_attention_max_mean": _safe_mean(doc_attn_max_means),
        },
    }
    summary_path = os.path.join(output_dir, "comparison_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def _train_or_resume(cfg: Any, force_rerun: bool) -> tuple[dict[str, float], str]:
    os.makedirs(cfg.output.exp_dir, exist_ok=True)
    best_ckpt = os.path.join(cfg.output.exp_dir, "best_model.pt")

    if not force_rerun and os.path.exists(best_ckpt):
        metrics = _load_final_metrics(cfg.output.exp_dir)
        if metrics is None:
            payload = load_and_eval(cfg, best_ckpt)
            metrics = payload["target"]
            analysis = payload.get("analysis", {})
        else:
            analysis_path = os.path.join(cfg.output.exp_dir, "analysis.json")
            analysis = {}
            if os.path.exists(analysis_path):
                with open(analysis_path, "r", encoding="utf-8") as f:
                    analysis = json.load(f)
        return {
            "acc": float(metrics["acc"]),
            "auc": float(metrics["auc"]),
            "analysis": analysis,
        }, "reused"

    best_path = train_one_split(cfg)
    payload = load_and_eval(cfg, best_path)
    metrics = payload["target"]
    analysis = payload.get("analysis", {})
    with open(os.path.join(cfg.output.exp_dir, "analysis.json"), "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)
    return {
        "acc": float(metrics["acc"]),
        "auc": float(metrics["auc"]),
        "analysis": analysis,
    }, "trained"


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    base_cfg = get_config(args.config)
    dataset_name = _infer_dataset_name(args)
    split_indices = list(range(args.split_offset, args.split_offset + args.num_splits))
    all_start = time.perf_counter()
    rows: list[dict[str, Any]] = []

    plan = {
        "dataset": dataset_name,
        "split_indices": split_indices,
        "split_dir": os.path.abspath(args.split_dir),
        "label_file": os.path.abspath(args.label_file),
        "image_dir": os.path.abspath(args.image_dir),
        "sentence_text_dir": os.path.abspath(args.sentence_text_dir),
        "graph_text_dir": os.path.abspath(args.graph_text_dir),
        "output_dir": os.path.abspath(args.output_dir),
        "gate_reg_weight": args.gate_reg_weight,
        "graph_weight_target": args.graph_weight_target,
        "force_rerun": bool(args.force_rerun),
    }
    with open(os.path.join(args.output_dir, "comparison_plan.json"), "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)

    for split_idx in split_indices:
        split_start = time.perf_counter()
        exp_dir = os.path.join(args.output_dir, f"split_{split_idx}")
        cfg = _build_cfg(base_cfg, args, split_idx, exp_dir)

        print("")
        print("=" * 100)
        print(f"[{_ts()}] [{dataset_name} dual_text] START split={split_idx:03d}")
        print("=" * 100)
        print("")

        metrics, status = _train_or_resume(cfg, force_rerun=bool(args.force_rerun))
        row = {
            "split_idx": split_idx,
            "acc": float(metrics["acc"]),
            "auc": float(metrics["auc"]),
            "best_path": os.path.join(exp_dir, "best_model.pt"),
            "exp_dir": exp_dir,
            "status": status,
            "fusion_gate_mean": metrics.get("analysis", {}).get("fusion_gate_mean"),
            "graph_branch_weight_mean": metrics.get("analysis", {}).get("graph_branch_weight_mean"),
            "doc_attention_entropy_mean": metrics.get("analysis", {}).get("doc_attention_entropy_mean"),
            "doc_attention_max_mean": metrics.get("analysis", {}).get("doc_attention_max_mean"),
        }
        rows.append(row)

        with open(os.path.join(args.output_dir, "comparison_results.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        _write_results_csv(args.output_dir, rows)
        _write_summary(args.output_dir, dataset_name, rows)

        print(
            f"[{_ts()}] [{dataset_name} dual_text] split={split_idx:03d} | "
            f"status={status} | acc={row['acc']:.4f} | auc={row['auc']:.4f}",
            flush=True,
        )
        print(
            f"[{_ts()}] [{dataset_name} dual_text] END split={split_idx:03d} | "
            f"duration={_format_seconds(time.perf_counter() - split_start)}",
            flush=True,
        )
        if split_idx != split_indices[-1]:
            print("")
            print("")
            print("")

    summary = _write_summary(args.output_dir, dataset_name, rows)
    _write_results_csv(args.output_dir, rows)
    print("")
    print(
        f"[{_ts()}] [{dataset_name} dual_text] Done. total_duration="
        f"{_format_seconds(time.perf_counter() - all_start)}",
        flush=True,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
