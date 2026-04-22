from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = REPO_ROOT / "experiment_records" / "reports" / "ordered_5split" / "plan.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch ordered 5-split experiment progress.")
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_summary(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def latest_split_epoch(run_dir: Path, split_idx: int) -> str:
    log_path = run_dir / f"split_{split_idx}" / "log.jsonl"
    if not log_path.exists():
        return "not started"
    latest = ""
    try:
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") == "final_evaluation":
                target = record.get("target", {})
                return "final ACC={:.4f} AUC={:.4f}".format(
                    float(target.get("acc", 0.0)),
                    float(target.get("auc", 0.0)),
                )
            if "epoch" in record and "target" in record:
                target = record.get("target", {})
                latest = "epoch={} ACC={:.4f} AUC={:.4f} best={:.4f}".format(
                    record.get("epoch"),
                    float(target.get("acc", 0.0)),
                    float(target.get("auc", 0.0)),
                    float(record.get("best_metric_so_far", target.get("auc", 0.0))),
                )
    except Exception as exc:
        return f"log unreadable: {exc}"
    return latest or "log exists"


def main() -> int:
    args = parse_args()
    plan = load_json(args.plan)
    if not plan:
        print(f"No plan found: {args.plan}")
        return 1

    print(f"Ordered 5-Split Watch | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Plan: {args.plan}")
    print(f"New splits per task: {plan.get('new_splits_per_task', plan.get('num_splits', ''))}")
    print(f"Split offset: {plan.get('split_offset', '')}")
    print("")

    for task in plan.get("tasks", []):
        dataset = task.get("dataset", "")
        method = task.get("method", "")
        run_dir = Path(task.get("run_dir", ""))
        requested = task.get("requested_split_indices") or plan.get("split_indices", [])
        completed_before = task.get("completed_before", [])
        summary_rows = read_summary(run_dir / "summary.csv")
        print(f"{dataset} / {method}")
        if completed_before:
            print(f"  completed before this run: {completed_before}")
        if requested:
            print(f"  requested this run: {requested}")
        if summary_rows:
            for row in summary_rows:
                print(
                    "  split{split_idx}: {status} ACC={acc} AUC={auc}".format(
                        split_idx=row.get("split_idx", ""),
                        status=row.get("status", ""),
                        acc=row.get("acc", ""),
                        auc=row.get("auc", ""),
                    )
                )
        else:
            for split_idx in requested:
                print(f"  split{split_idx}: {latest_split_epoch(run_dir, int(split_idx))}")
        print(f"  run: {run_dir}")
        print("")

    skipped = plan.get("skipped", [])
    if skipped:
        print("Skipped")
        for item in skipped:
            print(f"  {item.get('dataset')} / {item.get('method')}: {item.get('reason')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
