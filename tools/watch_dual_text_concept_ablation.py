from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / "experiments" / "dual_text_concept_graph_ablation_logs"
SUMMARY_MD = REPO_ROOT / "experiment_records" / "dual_text_concept_graph_ablation" / "summary.md"


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def load_jsonl_last(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    last = None
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
    return last


def parse_current_run() -> dict[str, str]:
    payload: dict[str, str] = {}
    for line in read_text(RUN_DIR / "current_run.txt").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        payload[key.strip().lstrip("\ufeff")] = value.strip()
    return payload


def process_alive(pid: str) -> bool | None:
    if not pid:
        return None

    command = (
        f"Get-Process -Id {int(pid)} -ErrorAction SilentlyContinue | "
        "Select-Object -ExpandProperty Id"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    return bool(result.stdout.strip())


def latest_started_task() -> str:
    log = read_text(RUN_DIR / "full_ablation.out.log")
    matches = re.findall(r"\[(\d\d:\d\d:\d\d)\] start ([^\r\n]+)", log)
    if not matches:
        return "unknown"
    timestamp, task = matches[-1]
    return f"{timestamp} {task}"


def format_float(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return "NA"


def print_summary(exp: Path) -> None:
    summary_path = exp / "summary.json"
    if not summary_path.exists():
        return

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        print("  summary: unreadable")
        return

    print(
        "  summary: "
        f"splits={summary.get('num_splits')} "
        f"ACC={format_float(summary.get('acc_mean'))}+/-{format_float(summary.get('acc_std'))} "
        f"AUC={format_float(summary.get('auc_mean'))}+/-{format_float(summary.get('auc_std'))}"
    )


def print_split_progress(exp: Path, split_idx: int) -> None:
    last = load_jsonl_last(exp / f"split_{split_idx}" / "log.jsonl")
    if last is None:
        print(f"  split{split_idx}: not started")
        return

    target = last.get("target", {})
    if last.get("type") == "final_evaluation":
        print(
            f"  split{split_idx}: final "
            f"ACC={format_float(target.get('acc'))} "
            f"AUC={format_float(target.get('auc'))}"
        )
        return

    print(
        f"  split{split_idx}: epoch={last.get('epoch')} "
        f"ACC={format_float(target.get('acc'))} "
        f"AUC={format_float(target.get('auc'))} "
        f"best={format_float(last.get('best_metric_so_far'))}"
    )


def print_experiment_progress() -> None:
    experiments = sorted(REPO_ROOT.glob("experiments/*dual_text_concept_graph_*_3splits_nw0"))
    if not experiments:
        print("No dual_text concept-graph experiments found yet.")
        return

    for exp in experiments:
        print(f"\n{exp.name}")
        print_summary(exp)
        for split_idx in range(3):
            print_split_progress(exp, split_idx)


def print_completed_rows() -> None:
    lines = read_text(SUMMARY_MD).splitlines()
    table_lines = [
        line
        for line in lines
        if line.startswith("| ") and "Dataset" not in line and "---" not in line
    ]
    if not table_lines:
        return

    print("\nCompleted summary rows:")
    for line in table_lines[-12:]:
        print(line)


def render_once() -> None:
    current_run = parse_current_run()
    pid = current_run.get("PID", "")

    print(f"Dual Text + Concept Graph Ablation Watch | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Main PID: {pid or 'unknown'} | alive: {process_alive(pid)}")
    print(f"Latest task: {latest_started_task()}")
    if current_run.get("STDOUT"):
        print(f"stdout: {current_run['STDOUT']}")
    if current_run.get("STDERR"):
        print(f"stderr: {current_run['STDERR']}")

    print_experiment_progress()
    print_completed_rows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch dual_text + concept graph ablation progress.")
    parser.add_argument("--interval", type=int, default=30, help="Refresh interval in seconds.")
    parser.add_argument("--once", action="store_true", help="Print once and exit.")
    args = parser.parse_args()

    while True:
        clear_screen()
        render_once()
        if args.once:
            return
        time.sleep(max(5, int(args.interval)))


if __name__ == "__main__":
    main()
