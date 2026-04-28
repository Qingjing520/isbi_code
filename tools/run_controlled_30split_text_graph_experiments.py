from __future__ import annotations

"""Run the requested controlled 30-split text-graph experiments sequentially.

This wrapper targets a total number of completed splits per dataset/method.
It continues from the next split after the currently completed split count and
keeps training output visible in the calling CMD window.
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import run_ordered_split_experiments as ordered


PYTHON = Path(r"F:\Anaconda\envs\pytorch\python.exe")
REPORT_DIR = REPO_ROOT / "experiment_records" / "reports" / "controlled_30split"


@dataclass(frozen=True)
class RequestedTask:
    dataset: str
    method: str
    experiment_tag: str = ""


REQUESTED_TASKS = [
    RequestedTask("BRCA", "sentence-only"),
    RequestedTask("BRCA", "sentence-ontology", "compact_auxgw20_residual"),
    RequestedTask("BRCA", "sentence-hierarchical-graph", "compact_auxgw20_residual"),
    RequestedTask("BRCA", "sentence-hierarchical-graph-ontology", "compact_auxgw20_residual"),
    RequestedTask("KIRC", "sentence-ontology", "compact_auxgw20_residual"),
    RequestedTask("KIRC", "sentence-hierarchical-graph", "compact_auxgw20_residual"),
    RequestedTask("KIRC", "sentence-hierarchical-graph-ontology", "compact_auxgw20_residual"),
    RequestedTask("LUSC", "sentence-ontology", "compact_auxgw20_residual"),
    RequestedTask("LUSC", "sentence-hierarchical-graph", "compact_auxgw20_residual"),
    RequestedTask("LUSC", "sentence-hierarchical-graph-ontology", "compact_auxgw20_residual"),
]


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequentially run BRCA/KIRC/LUSC controlled text-graph experiments up to 30 splits."
    )
    parser.add_argument("--target_splits", type=int, default=30)
    parser.add_argument("--ontology_variant", type=str, default="ncit_do")
    parser.add_argument("--train_num_workers", type=int, default=0)
    parser.add_argument("--hierarchy_graph_weight_max", type=float, default=0.2)
    parser.add_argument("--ontology_graph_weight_max", type=float, default=0.2)
    parser.add_argument("--graph_weight_target", type=float, default=0.1)
    parser.add_argument("--fusion_mode", type=str, default="residual", choices=["convex", "residual"])
    parser.add_argument("--no_section_title_embedding", action="store_true")
    parser.add_argument("--force_train", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def completed_split_indices_strict(run_dir: Path) -> list[int]:
    """Count split folders with a checkpoint, not merely a partial log file."""
    indices: list[int] = []
    if not run_dir.exists():
        return indices
    for split_dir in run_dir.glob("split_*"):
        if not split_dir.is_dir():
            continue
        suffix = split_dir.name.rsplit("_", 1)[-1]
        if not suffix.isdigit():
            continue
        if (split_dir / "best_model.pt").exists():
            indices.append(int(suffix))
    return sorted(set(indices))


def build_manifest_template(dataset: str, method: str, variant: str, split_indices: list[int], dry_run: bool) -> str:
    if method == "sentence-ontology":
        graph_dir = ordered.sentence_ontology_graph_dir(dataset, variant)
        if not dry_run:
            ordered.ensure_graph_manifests(
                dataset=dataset,
                indices=split_indices,
                graph_dir=graph_dir,
                manifest_path_fn=lambda dataset_name, split_idx, v=variant: ordered.sentence_ontology_manifest_path(
                    dataset_name,
                    v,
                    split_idx,
                ),
            )
        return str(
            ordered.OUTPUT_ROOT
            / "manifests"
            / "ablation"
            / variant
            / f"{dataset.lower()}_sentence_ontology_graph_manifest_split{{split_idx}}.csv"
        )

    if method == "sentence-hierarchical-graph-ontology":
        graph_dir = ordered.concept_graph_dir(dataset, variant)
        if not dry_run:
            ordered.ensure_graph_manifests(
                dataset=dataset,
                indices=split_indices,
                graph_dir=graph_dir,
                manifest_path_fn=lambda dataset_name, split_idx, v=variant: ordered.concept_manifest_path(
                    dataset_name,
                    v,
                    split_idx,
                ),
            )
        return str(
            ordered.OUTPUT_ROOT
            / "manifests"
            / "ablation"
            / variant
            / f"{dataset.lower()}_concept_graph_manifest_split{{split_idx}}.csv"
        )

    return ""


def make_config(task: RequestedTask, run_dir: Path, manifest_template: str, args: argparse.Namespace) -> Path:
    if task.method == "sentence-only":
        tag = ""
        use_section_title = False
        fusion_mode = "convex"
    else:
        tag = task.experiment_tag
        use_section_title = not bool(args.no_section_title_embedding)
        fusion_mode = str(args.fusion_mode)

    return ordered.write_config(
        dataset=task.dataset,
        method=task.method,
        variant=args.ontology_variant,
        exp_dir=run_dir,
        graph_manifest_template=manifest_template,
        hierarchy_graph_weight_max=args.hierarchy_graph_weight_max,
        ontology_graph_weight_max=args.ontology_graph_weight_max,
        graph_weight_target=args.graph_weight_target,
        use_section_title_embedding=use_section_title,
        fusion_mode=fusion_mode,
        experiment_tag=tag,
    )


def run_task(
    task: RequestedTask,
    run_dir: Path,
    config_path: Path,
    split_indices: list[int],
    manifest_template: str,
    args: argparse.Namespace,
) -> None:
    cmd = [
        str(PYTHON),
        "-u",
        str(REPO_ROOT / "run_main_splits.py"),
        "--config",
        str(config_path),
        "--split_dir",
        str(ordered.DATASETS[task.dataset]["split_dir"]),
        "--output_root",
        str(run_dir),
        "--num_splits",
        str(len(split_indices)),
        "--split_offset",
        str(split_indices[0]),
        "--train_num_workers",
        str(args.train_num_workers),
    ]
    if manifest_template:
        cmd.extend(["--graph_manifest_template", manifest_template])
    if args.force_train:
        cmd.append("--force_rerun")

    ordered.ensure_dir(run_dir)
    command_text = " ".join(cmd)
    orchestration_log = run_dir / "controlled_30split_runner.log"
    with orchestration_log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{now()}] START {task.dataset} {task.method}\n")
        handle.write(command_text + "\n")

    print(f"\n[{now()}] START {task.dataset} / {task.method}", flush=True)
    print(command_text, flush=True)
    completed = subprocess.run(cmd, cwd=REPO_ROOT, text=True)
    with orchestration_log.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now()}] END exit={completed.returncode}\n")
    if completed.returncode != 0:
        raise RuntimeError(f"Training failed: {task.dataset} / {task.method}. See {orchestration_log}")
    print(f"[{now()}] DONE  {task.dataset} / {task.method}", flush=True)


def task_plan_row(task: RequestedTask, args: argparse.Namespace) -> dict[str, Any]:
    run_dir = ordered.planned_run_dir(task.dataset, task.method, experiment_tag=task.experiment_tag)
    completed = completed_split_indices_strict(run_dir)
    remaining = max(0, int(args.target_splits) - len(completed))
    start = (max(completed) + 1) if completed else 0
    split_indices = list(range(start, start + remaining))
    return {
        "dataset": task.dataset,
        "method": task.method,
        "experiment_tag": task.experiment_tag,
        "run_dir": str(run_dir),
        "completed_count": len(completed),
        "completed_indices": completed,
        "remaining_to_target": remaining,
        "requested_split_indices": split_indices,
    }


def write_plan(path: Path, payload: dict[str, Any]) -> None:
    ordered.ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    plan_rows = [task_plan_row(task, args) for task in REQUESTED_TASKS]
    plan = {
        "created_at": now(),
        "target_splits": int(args.target_splits),
        "ontology_variant": args.ontology_variant,
        "fusion_mode": args.fusion_mode,
        "use_section_title_embedding": not bool(args.no_section_title_embedding),
        "hierarchy_graph_weight_max": float(args.hierarchy_graph_weight_max),
        "ontology_graph_weight_max": float(args.ontology_graph_weight_max),
        "graph_weight_target": float(args.graph_weight_target),
        "train_num_workers": int(args.train_num_workers),
        "tasks": plan_rows,
    }
    write_plan(REPORT_DIR / "plan.json", plan)

    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        print(f"[{now()}] dry run complete; no training launched.", flush=True)
        return 0

    for task, row in zip(REQUESTED_TASKS, plan_rows):
        split_indices = [int(x) for x in row["requested_split_indices"]]
        if not split_indices:
            print(f"[{now()}] SKIP {task.dataset} / {task.method}: already has {args.target_splits} splits.", flush=True)
            continue

        reason = ordered.unavailable_reason(task.dataset, task.method, args.ontology_variant, split_indices)
        if reason:
            raise RuntimeError(f"Cannot run {task.dataset} / {task.method}: {reason}")

        run_dir = Path(row["run_dir"])
        manifest_template = build_manifest_template(
            dataset=task.dataset,
            method=task.method,
            variant=args.ontology_variant,
            split_indices=split_indices,
            dry_run=False,
        )
        config_path = make_config(task, run_dir, manifest_template, args)
        run_task(task, run_dir, config_path, split_indices, manifest_template, args)

        ordered_task = ordered.Task(
            dataset=task.dataset,
            method=task.method,
            run_name=run_dir.name,
            run_dir=run_dir,
            config_path=config_path,
            split_indices=split_indices,
            completed_before=[int(x) for x in row["completed_indices"]],
            experiment_tag=task.experiment_tag,
            graph_manifest_template=manifest_template,
        )
        ordered_task = ordered.rename_run_dir_to_completed_count(ordered_task)
        ordered.write_records_from_summary(ordered_task, args.ontology_variant)

        plan_rows = [task_plan_row(item, args) for item in REQUESTED_TASKS]
        plan["tasks"] = plan_rows
        plan["updated_at"] = now()
        write_plan(REPORT_DIR / "plan.json", plan)

    print(f"[{now()}] all requested controlled 30-split tasks are complete or up to target.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
