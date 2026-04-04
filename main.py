from __future__ import annotations
import argparse
import os
import json

from configs.config import get_config
from train import train_one_split, load_and_eval

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    ap.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        default="train",
        help="train: train then test best checkpoint; test: only test an existing checkpoint",
    )
    ap.add_argument(
        "--ckpt",
        type=str,
        default="experiments/split0/best_model.pt",
        help="checkpoint path for --mode test. If empty, use <exp_dir>/best_model.pt",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = get_config(args.config)
    os.makedirs(cfg.output.exp_dir, exist_ok=True)

    if args.mode == "train":
        best_path = train_one_split(cfg)
        metrics = load_and_eval(cfg, best_path)

        epoch = metrics.get("epoch", None)
        if epoch is not None and epoch >= 0:
            print(f'\nFinal evaluation (best checkpoint = "epoch": {epoch}):')
        else:
            print("\nFinal evaluation (best checkpoint):")
        print(f"  Target ACC={metrics['target']['acc']:.4f}, AUC={metrics['target']['auc']:.4f}")

        # 追加一条最终评估记录到 log.jsonl 中，便于后续统一查看
        log_path = os.path.join(cfg.output.exp_dir, "log.jsonl")
        final_record = {
            "type": "final_evaluation",
            "best_epoch": int(epoch) if epoch is not None else None,
            "target": {
                "acc": float(metrics["target"]["acc"]),
                "auc": float(metrics["target"]["auc"]),
            },
        }
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(final_record) + "\n")
        except Exception as e:
            print(f"[warn] Failed to append final evaluation to log.jsonl: {e}")

    else:  # test only
        ckpt_path = args.ckpt.strip()
        if ckpt_path == "":
            ckpt_path = os.path.join(cfg.output.exp_dir, "best_model.pt")

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        metrics = load_and_eval(cfg, ckpt_path)

        epoch = metrics.get("epoch", None)
        if epoch is not None and epoch >= 0:
            print(f'Final evaluation (checkpoint = "epoch": {epoch}):')
        else:
            print("Final evaluation (checkpoint):")
        print(f"  Target ACC={metrics['target']['acc']:.4f}, AUC={metrics['target']['auc']:.4f}")


if __name__ == "__main__":
    main()