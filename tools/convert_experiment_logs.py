from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENTS_ROOT = REPO_ROOT / "experiments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert old verbose split log.jsonl files into concise train.log files "
            "matching the current CMD training output style."
        )
    )
    parser.add_argument("--experiments_root", type=Path, default=DEFAULT_EXPERIMENTS_ROOT)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write converted train.log files. Without this flag, only report what would change.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing train.log files. By default existing train.log files are skipped.",
    )
    parser.add_argument(
        "--compact-jsonl",
        action="store_true",
        help=(
            "Also rewrite log.jsonl without bulky sample_details. The original is backed up "
            "as log.full.jsonl.bak unless --no-backup is passed."
        ),
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create log.full.jsonl.bak when --compact-jsonl is used.",
    )
    parser.add_argument(
        "--timestamp-mode",
        choices=["none", "mtime"],
        default="none",
        help=(
            "Old log.jsonl does not contain per-epoch timestamps. Use 'none' for honest "
            "reconstructed lines, or 'mtime' to prefix all lines with the log file mtime."
        ),
    )
    parser.add_argument(
        "--skip-recent-minutes",
        type=float,
        default=5.0,
        help="Skip log.jsonl files modified within this many minutes to avoid active runs. Use 0 to disable.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap for testing the converter on the first N split logs.",
    )
    return parser.parse_args()


def split_idx_from_dir(split_dir: Path) -> int | None:
    match = re.match(r"^split_(\d+)$", split_dir.name)
    return int(match.group(1)) if match else None


def timestamp_from_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def format_prefix(timestamp: str | None) -> str:
    return f"[{timestamp}] " if timestamp else ""


def float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def compact_target(target: Any) -> dict[str, Any]:
    if not isinstance(target, dict):
        return {"acc": 0.0, "auc": 0.0}
    compact: dict[str, Any] = {
        "acc": float_value(target.get("acc")),
        "auc": float_value(target.get("auc")),
    }
    analysis = target.get("analysis")
    if isinstance(analysis, dict):
        compact_analysis = {key: value for key, value in analysis.items() if key != "sample_details"}
        if compact_analysis:
            compact["analysis"] = compact_analysis
    return compact


def compact_event(event: dict[str, Any]) -> dict[str, Any]:
    compact = dict(event)
    if "target" in compact:
        compact["target"] = compact_target(compact["target"])
    return compact


def read_events(log_path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
    return events


def split_file_from_config(split_dir: Path, split_idx: int | None) -> str:
    cfg = load_json(split_dir / "config.json")
    data = cfg.get("data") if isinstance(cfg, dict) else None
    split_file = data.get("split_file") if isinstance(data, dict) else ""
    if split_file:
        return str(split_file)

    dataset = ""
    parts = split_dir.parts
    if "experiments" in parts:
        exp_index = parts.index("experiments")
        if exp_index + 1 < len(parts):
            dataset = parts[exp_index + 1]
    if dataset and split_idx is not None:
        return str(Path(r"F:\Tasks\Split_Table") / dataset / f"split_{split_idx}.csv")
    return ""


def mode_string(event: dict[str, Any]) -> str:
    epoch = int(event.get("epoch", 0))
    warmup_epochs = int(event.get("warmup_epochs", 3))
    if epoch < warmup_epochs:
        return "warmup(cls only)"
    decay = float_value(event.get("align_decay_factor"), default=1.0)
    return f"align(decay={decay:.3f})"


def epoch_line(
    event: dict[str, Any],
    timestamp: str | None,
    best_epoch: int | None = None,
    best_acc: float | None = None,
) -> str:
    epoch = int(event.get("epoch", 0))
    train = event.get("train") if isinstance(event.get("train"), dict) else {}
    target = compact_target(event.get("target"))
    logged_best_epoch = event.get("best_epoch_so_far")
    if logged_best_epoch is not None:
        try:
            best_epoch = int(logged_best_epoch)
        except (TypeError, ValueError):
            pass
    logged_best_acc = event.get("best_acc_so_far")
    if logged_best_acc is not None:
        best_acc = float_value(logged_best_acc, default=float("nan"))
    best_epoch_value = int(best_epoch) if best_epoch is not None and best_epoch >= 0 else -1
    best_acc_value = float_value(best_acc, default=float("nan"))
    return (
        f"{format_prefix(timestamp)}[Epoch {epoch:03d}] {mode_string(event)} "
        f"train(loss={float_value(train.get('loss_total')):.4f}, "
        f"cls={float_value(train.get('loss_cls')):.4f}, "
        f"txt={float_value(train.get('loss_txt')):.4f}, "
        f"concept={float_value(train.get('loss_concept')):.4f}, "
        f"node={float_value(train.get('loss_node')):.4f}, "
        f"topo={float_value(train.get('loss_topo')):.4f}, "
        f"text_topo={float_value(train.get('loss_text_topology')):.4f}, "
        f"gate={float_value(train.get('loss_gate')):.4f}, "
        f"graph_w={float_value(train.get('graph_weight')):.3f}) | "
        f"tgt(acc={float_value(target.get('acc')):.4f}, "
        f"auc={float_value(target.get('auc')):.4f}) | "
        f"best(auc={float_value(event.get('best_metric_so_far')):.4f}, "
        f"acc={best_acc_value:.4f}, epoch={best_epoch_value:03d})"
    )


def reconstructed_train_log(split_dir: Path, events: list[dict[str, Any]], timestamp: str | None) -> list[str]:
    split_idx = split_idx_from_dir(split_dir)
    split_file = split_file_from_config(split_dir, split_idx)
    lines: list[str] = []
    if split_idx is not None and split_file:
        lines.append(f"[split {split_idx}] start | split_file={split_file}")
    elif split_idx is not None:
        lines.append(f"[split {split_idx}] start")
    else:
        lines.append("[split] start")

    best_seen: float | None = None
    best_target: dict[str, Any] | None = None
    best_epoch: int | None = None
    best_acc: float | None = None
    final_target: dict[str, Any] | None = None

    for event in events:
        if event.get("type") == "final_evaluation":
            final_target = compact_target(event.get("target"))
            continue
        if "epoch" not in event:
            continue

        epoch = int(event.get("epoch", 0))
        target = compact_target(event.get("target"))
        best_metric = float_value(event.get("best_metric_so_far"))
        if best_seen is None or best_metric > best_seen + 1e-12:
            best_seen = best_metric
            best_target = target
            best_epoch = epoch
            best_acc = float_value(target.get("acc"))
            lines.append(
                f"{format_prefix(timestamp)}[best] save best @ epoch={epoch:03d}, "
                f"auc={best_metric:.4f}, acc={best_acc:.4f}"
            )

        lines.append(epoch_line(event, timestamp, best_epoch, best_acc))
        lines.append("")

        if bool(event.get("early_stop")):
            lines.append(
                f"{format_prefix(timestamp)}[EarlyStop] stop at epoch={epoch:03d} "
                f"(best auc={float_value(event.get('best_metric_so_far')):.4f}, "
                f"acc={float_value(best_acc):.4f}, epoch={int(best_epoch or 0):03d})"
            )

    done_target = final_target or best_target
    status = "trained" if (split_dir / "best_model.pt").exists() else "from_log"
    if split_idx is not None and done_target is not None:
        lines.append(
            f"[split {split_idx}] done | status={status} | "
            f"acc={float_value(done_target.get('acc')):.4f} | "
            f"auc={float_value(done_target.get('auc')):.4f}"
        )
    elif split_idx is not None:
        lines.append(f"[split {split_idx}] done | status={status}")
    if best_epoch is not None and final_target is None:
        # Keep this out of the log body; downstream comparison tools already know
        # how to use best logged epochs when final_evaluation is absent.
        _ = best_epoch
    lines.extend(["", ""])
    return lines


def is_recent(path: Path, minutes: float) -> bool:
    if minutes <= 0:
        return False
    age_seconds = datetime.now().timestamp() - path.stat().st_mtime
    return age_seconds < minutes * 60.0


def convert_one(log_path: Path, args: argparse.Namespace) -> str:
    if is_recent(log_path, args.skip_recent_minutes):
        return "skipped_recent"

    split_dir = log_path.parent
    train_log_path = split_dir / "train.log"
    if train_log_path.exists() and not args.overwrite:
        return "skipped_existing_train_log"

    events = read_events(log_path)
    if not events:
        return "skipped_no_events"

    timestamp = timestamp_from_mtime(log_path) if args.timestamp_mode == "mtime" else None
    train_lines = reconstructed_train_log(split_dir, events, timestamp)

    if args.apply:
        train_log_path.write_text("\n".join(train_lines).rstrip() + "\n", encoding="utf-8")

        if args.compact_jsonl:
            backup_path = split_dir / "log.full.jsonl.bak"
            if not args.no_backup and not backup_path.exists():
                shutil.copy2(log_path, backup_path)
            compact_lines = [json.dumps(compact_event(event), ensure_ascii=False) for event in events]
            log_path.write_text("\n".join(compact_lines) + "\n", encoding="utf-8")

    return "converted"


def iter_log_paths(experiments_root: Path) -> list[Path]:
    return sorted(experiments_root.rglob("split_*/log.jsonl"))


def main() -> int:
    args = parse_args()
    experiments_root = args.experiments_root.resolve()
    log_paths = iter_log_paths(experiments_root)
    if args.max_files > 0:
        log_paths = log_paths[: args.max_files]

    counts: dict[str, int] = {}
    for log_path in log_paths:
        status = convert_one(log_path, args)
        counts[status] = counts.get(status, 0) + 1

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"{mode} convert experiment logs")
    print(f"experiments_root={experiments_root}")
    print(f"log_files_seen={len(log_paths)}")
    for key in sorted(counts):
        print(f"{key}={counts[key]}")
    if not args.apply:
        print("No files were written. Re-run with --apply to create train.log files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
