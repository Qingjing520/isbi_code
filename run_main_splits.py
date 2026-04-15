from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import statistics
from pathlib import Path
from typing import Any

from configs.config import get_config
from train import load_and_eval, train_one_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the main training entry across multiple split CSVs using one base config."
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--split_dir", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--split_offset", type=int, default=0)
    parser.add_argument(
        "--graph_manifest_template",
        type=str,
        default="",
        help=(
            "Optional format string for split-specific graph manifests, "
            "for example ...\\kirc_concept_graph_manifest_split{split_idx}.csv"
        ),
    )
    parser.add_argument(
        "--train_num_workers",
        type=int,
        default=-1,
        help="Override dataloader num_workers. Use -1 to keep the config value.",
    )
    parser.add_argument(
        "--force_rerun",
        action="store_true",
        help="Ignore existing checkpoints and rerun split training from scratch.",
    )
    return parser.parse_args()


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
    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
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


def _build_cfg(base_cfg: Any, args: argparse.Namespace, split_idx: int) -> Any:
    cfg = copy.deepcopy(base_cfg)
    cfg.data.split_file = os.path.abspath(os.path.join(args.split_dir, f"split_{split_idx}.csv"))
    if str(args.graph_manifest_template).strip():
        cfg.data.graph_manifest_csv = os.path.abspath(
            str(args.graph_manifest_template).format(split_idx=split_idx)
        )
    cfg.output.exp_dir = os.path.abspath(os.path.join(args.output_root, f"split_{split_idx}"))
    if args.train_num_workers >= 0:
        cfg.train.num_workers = int(args.train_num_workers)
    return cfg


def _train_or_resume(cfg: Any, force_rerun: bool) -> tuple[dict[str, float], str]:
    os.makedirs(cfg.output.exp_dir, exist_ok=True)
    best_ckpt = os.path.join(cfg.output.exp_dir, "best_model.pt")

    if not force_rerun and os.path.exists(best_ckpt):
        metrics = _load_final_metrics(cfg.output.exp_dir)
        if metrics is None:
            payload = load_and_eval(cfg, best_ckpt)
            metrics = payload["target"]
        return {
            "acc": float(metrics["acc"]),
            "auc": float(metrics["auc"]),
        }, "reused"

    best_path = train_one_split(cfg)
    payload = load_and_eval(cfg, best_path)
    metrics = payload["target"]
    return {
        "acc": float(metrics["acc"]),
        "auc": float(metrics["auc"]),
    }, "trained"


def _write_results(output_root: str, rows: list[dict[str, Any]]) -> None:
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split_idx", "status", "acc", "auc", "exp_dir", "split_file", "graph_manifest_csv"],
        )
        writer.writeheader()
        writer.writerows(rows)

    accs = [float(row["acc"]) for row in rows]
    aucs = [float(row["auc"]) for row in rows]
    summary = {
        "num_splits": len(rows),
        "acc_mean": _safe_mean(accs),
        "acc_std": _safe_std(accs),
        "auc_mean": _safe_mean(aucs),
        "auc_std": _safe_std(aucs),
        "rows": rows,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main() -> None:
    args = parse_args()
    base_cfg = get_config(args.config)
    rows: list[dict[str, Any]] = []

    for split_idx in range(args.split_offset, args.split_offset + args.num_splits):
        cfg = _build_cfg(base_cfg, args, split_idx)
        print(f"[split {split_idx}] start | split_file={cfg.data.split_file}")
        if getattr(cfg.data, "graph_manifest_csv", ""):
            print(f"[split {split_idx}] manifest={cfg.data.graph_manifest_csv}")
        metrics, status = _train_or_resume(cfg, force_rerun=args.force_rerun)
        row = {
            "split_idx": split_idx,
            "status": status,
            "acc": float(metrics["acc"]),
            "auc": float(metrics["auc"]),
            "exp_dir": cfg.output.exp_dir,
            "split_file": cfg.data.split_file,
            "graph_manifest_csv": getattr(cfg.data, "graph_manifest_csv", ""),
        }
        rows.append(row)
        print(
            f"[split {split_idx}] done | status={status} | "
            f"acc={row['acc']:.4f} | auc={row['auc']:.4f}"
        )

    _write_results(args.output_root, rows)
    print(f"[summary] output_root={os.path.abspath(args.output_root)}")


if __name__ == "__main__":
    main()
