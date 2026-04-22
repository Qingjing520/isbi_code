from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required to generate training configs.") from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(r"F:\Anaconda\envs\pytorch\python.exe")
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
OUTPUT_ROOT = REPO_ROOT / "pathology_report_extraction" / "Output"
CONFIG_ROOT = REPO_ROOT / "configs" / "generated" / "ordered_5split"
DEFAULT_ONTOLOGY_VARIANT = "ncit_snomed_mapped"

DATASET_ORDER = ("BRCA", "KIRC", "LUSC")
METHOD_ORDER = (
    "sentence-only",
    "sentence-ontology",
    "sentence-hierarchical-graph",
    "sentence-hierarchical-graph-ontology",
)

METHOD_LABELS = {
    "sentence-only": "sentence-only",
    "sentence-ontology": "sentence + ontology",
    "sentence-hierarchical-graph": "sentence + hierarchical graph",
    "sentence-hierarchical-graph-ontology": "sentence + hierarchical graph + ontology",
}

DATASETS = {
    "BRCA": {
        "split_dir": Path(r"F:\Tasks\Split_Table\BRCA"),
        "label_file": Path(r"F:\Tasks\Pathologic_Stage_Label\BRCA_pathologic_stage.csv"),
        "image_dir": Path(r"F:\Tasks\WSI_extract_features\BRCA_WSI_extract_features"),
        "sentence_text_dir": Path(r"F:\Tasks\Text_Sentence_extract_features\BRCA_text"),
    },
    "KIRC": {
        "split_dir": Path(r"F:\Tasks\Split_Table\KIRC"),
        "label_file": Path(r"F:\Tasks\Pathologic_Stage_Label\KIRC_pathologic_stage.csv"),
        "image_dir": Path(r"F:\Tasks\WSI_extract_features\KIRC_WSI_extract_features"),
        "sentence_text_dir": Path(r"F:\Tasks\Text_Sentence_extract_features\KIRC_text"),
    },
    "LUSC": {
        "split_dir": Path(r"F:\Tasks\Split_Table\LUSC"),
        "label_file": Path(r"F:\Tasks\Pathologic_Stage_Label\LUSC_pathologic_stage.csv"),
        "image_dir": Path(r"F:\Tasks\WSI_extract_features\LUSC_WSI_extract_features"),
        "sentence_text_dir": Path(r"F:\Tasks\Text_Sentence_extract_features\LUSC_text"),
    },
}


@dataclass(frozen=True)
class Task:
    dataset: str
    method: str
    run_name: str
    run_dir: Path
    config_path: Path
    graph_manifest_template: str = ""

    @property
    def records_dir(self) -> Path:
        return EXPERIMENTS_ROOT / self.dataset / self.method / "records"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run clean, ordered 5-split experiments in the organized experiments tree."
    )
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_ORDER), choices=DATASET_ORDER)
    parser.add_argument("--methods", nargs="+", default=list(METHOD_ORDER), choices=METHOD_ORDER)
    parser.add_argument("--num_splits", type=int, default=5)
    parser.add_argument("--split_offset", type=int, default=0)
    parser.add_argument("--ontology_variant", type=str, default=DEFAULT_ONTOLOGY_VARIANT)
    parser.add_argument("--train_num_workers", type=int, default=0)
    parser.add_argument("--force_train", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def method_root(dataset: str, method: str) -> Path:
    return EXPERIMENTS_ROOT / dataset / method


def run_dir(dataset: str, method: str) -> Path:
    return method_root(dataset, method) / "runs" / f"{dataset.lower()}_{method.replace('-', '_')}_5splits"


def split_indices(args: argparse.Namespace) -> list[int]:
    return list(range(args.split_offset, args.split_offset + args.num_splits))


def missing_base_inputs(dataset: str, indices: list[int]) -> list[str]:
    cfg = DATASETS[dataset]
    missing: list[str] = []
    for key in ("split_dir", "label_file", "image_dir", "sentence_text_dir"):
        path = cfg[key]
        if not path.exists():
            missing.append(str(path))
    for split_idx in indices:
        path = cfg["split_dir"] / f"split_{split_idx}.csv"
        if not path.exists():
            missing.append(str(path))
    return missing


def hierarchy_graph_dir(dataset: str) -> Path:
    return OUTPUT_ROOT / "text_hierarchy_graphs_masked" / dataset


def concept_graph_dir(dataset: str, variant: str) -> Path:
    return OUTPUT_ROOT / "text_concept_graphs_ablation" / variant / dataset


def concept_manifest_path(dataset: str, variant: str, split_idx: int) -> Path:
    return (
        OUTPUT_ROOT
        / "manifests"
        / "ablation"
        / variant
        / f"{dataset.lower()}_concept_graph_manifest_split{split_idx}.csv"
    )


def build_concept_manifest(dataset: str, variant: str, split_idx: int, graph_dir: Path) -> Path:
    cfg = DATASETS[dataset]
    output_csv = concept_manifest_path(dataset, variant, split_idx)
    ensure_dir(output_csv.parent)
    args = [
        str(PYTHON),
        str(REPO_ROOT / "pathology_report_extraction" / "prepare_text_graph_manifest.py"),
        "--graph_dir",
        str(graph_dir),
        "--label_csv",
        str(cfg["label_file"]),
        "--split_csv",
        str(cfg["split_dir"] / f"split_{split_idx}.csv"),
        "--image_dir",
        str(cfg["image_dir"]),
        "--output_csv",
        str(output_csv),
    ]
    print(f"[{now()}] build manifest {dataset} {variant} split={split_idx}", flush=True)
    completed = subprocess.run(args, cwd=REPO_ROOT, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"prepare_text_graph_manifest failed for {dataset} split {split_idx}")
    return output_csv


def ensure_concept_manifests(dataset: str, variant: str, indices: list[int], graph_dir: Path) -> list[Path]:
    manifests: list[Path] = []
    for split_idx in indices:
        manifest = concept_manifest_path(dataset, variant, split_idx)
        if not manifest.exists():
            manifest = build_concept_manifest(dataset, variant, split_idx, graph_dir)
        manifests.append(manifest)
    return manifests


def base_payload(dataset: str, exp_dir: Path) -> dict[str, Any]:
    cfg = DATASETS[dataset]
    return {
        "seed": 23,
        "data": {
            "split_file": str(cfg["split_dir"] / "split_0.csv"),
            "label_file": str(cfg["label_file"]),
            "image_dir": str(cfg["image_dir"]),
            "text_dir": str(cfg["sentence_text_dir"]),
            "text_mode": "sentence_pt",
            "sentence_text_dir": str(cfg["sentence_text_dir"]),
            "graph_text_dir": "",
            "graph_manifest_csv": "",
            "text_graph_feature": "node_features",
            "text_use_graph_structure": False,
        },
        "graph": {
            "num_nodes_m": 64,
            "topk_k": 16,
            "kmeans_iters": 10,
            "max_patches_for_kmeans": 2048,
        },
        "model": {
            "feat_dim": 512,
            "mapper_hidden": 1024,
            "mapper_depth": 2,
            "mapper_dropout": 0.1,
            "attn_pool_hidden": 256,
            "attn_pool_dropout": 0.1,
            "moe_heads": 8,
            "moe_dropout": 0.1,
            "moe_tau_start": 1.0,
            "moe_tau_min": 0.1,
            "moe_tau_decay": 0.95,
            "moe_hard_start_epoch": 10,
            "classifier_hidden": 256,
            "classifier_dropout": 0.2,
            "text_graph_layers": 1,
            "text_graph_dropout": 0.05,
            "text_graph_use_next_edges": True,
            "text_graph_num_node_types": 0,
            "text_graph_num_base_relations": 0,
            "sentence_local_layers": 1,
            "sentence_local_heads": 8,
            "hier_readout_hidden": 256,
            "hier_readout_dropout": 0.1,
            "hier_readout_attention_init": -0.5,
            "use_section_title_embedding": False,
            "section_title_vocab": [
                "Document Body",
                "Diagnosis",
                "Final Diagnosis",
                "Gross Description",
                "Microscopic Description",
                "Comment",
                "Clinical Information",
                "Specimen Submitted",
                "Synoptic Report",
                "Procedure",
                "Ancillary Studies",
                "Patient History",
                "Summary of Sections",
                "Intraoperative Consultation",
            ],
            "text_dual_fusion_dropout": 0.1,
            "text_dual_graph_weight_max": 1.0,
        },
        "loss": {
            "warmup_epochs": 3,
            "align_decay_to": 0.3,
            "alpha_txt": 0.5,
            "alpha_concept": 0.0,
            "beta_node": 0.1,
            "gamma_topo": 0.1,
            "gamma_text_topology": 0.0,
            "dual_text_gate_reg_weight": 0.0,
            "dual_text_graph_weight_target": 0.1,
            "mmd_num_kernels": 3,
            "mmd_sigma_multipliers": [0.5, 1.0, 2.0],
            "mmd_unbiased": True,
            "mmd_clamp_nonneg": True,
        },
        "train": {
            "num_epochs": 60,
            "batch_size": 4,
            "num_workers": 0,
            "lr": 0.0001,
            "weight_decay": 0.01,
            "grad_clip": 1.0,
            "early_stop_patience": 10,
            "early_stop_min_delta": 0.0001,
        },
        "output": {
            "exp_dir": str(exp_dir / "split_0"),
            "save_best_by": "val_avg",
        },
    }


def write_config(dataset: str, method: str, variant: str, graph_manifest_template: str = "") -> Path:
    exp_dir = run_dir(dataset, method)
    payload = base_payload(dataset, exp_dir)
    cfg = DATASETS[dataset]

    if method == "sentence-only":
        pass
    elif method == "sentence-hierarchical-graph":
        graph_dir = hierarchy_graph_dir(dataset)
        payload["data"].update(
            {
                "text_dir": str(cfg["sentence_text_dir"]),
                "text_mode": "dual_text",
                "sentence_text_dir": str(cfg["sentence_text_dir"]),
                "graph_text_dir": str(graph_dir),
                "graph_manifest_csv": "",
                "text_use_graph_structure": True,
            }
        )
        payload["model"]["text_graph_num_node_types"] = 3
        payload["model"]["text_graph_num_base_relations"] = 2
    elif method == "sentence-hierarchical-graph-ontology":
        graph_dir = concept_graph_dir(dataset, variant)
        payload["data"].update(
            {
                "text_dir": str(cfg["sentence_text_dir"]),
                "text_mode": "dual_text",
                "sentence_text_dir": str(cfg["sentence_text_dir"]),
                "graph_text_dir": str(graph_dir),
                "graph_manifest_csv": graph_manifest_template.format(split_idx=0),
                "text_use_graph_structure": True,
            }
        )
        payload["model"]["text_graph_num_node_types"] = 4
        payload["model"]["text_graph_num_base_relations"] = 6
        payload["model"]["text_dual_graph_weight_max"] = 0.2
        payload["loss"]["alpha_concept"] = 0.2
        payload["loss"]["gamma_text_topology"] = 0.05
        payload["loss"]["dual_text_gate_reg_weight"] = 0.01
        payload["loss"]["dual_text_graph_weight_target"] = 0.1
    else:
        raise ValueError(f"Cannot write config for unsupported method: {method}")

    config_path = CONFIG_ROOT / f"{dataset.lower()}_{method.replace('-', '_')}_5splits.yaml"
    ensure_dir(config_path.parent)
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return config_path


def unavailable_reason(dataset: str, method: str, variant: str, indices: list[int]) -> str:
    missing = missing_base_inputs(dataset, indices)
    if missing:
        return "missing base inputs: " + "; ".join(missing)

    if method == "sentence-only":
        return ""
    if method == "sentence-ontology":
        return (
            "not implemented as a separate data product yet; current ontology data is represented "
            "as concept graphs and should be compared under sentence-hierarchical-graph-ontology"
        )
    if method == "sentence-hierarchical-graph":
        graph_dir = hierarchy_graph_dir(dataset)
        if not graph_dir.exists():
            return f"missing hierarchy graph directory: {graph_dir}"
        return ""
    if method == "sentence-hierarchical-graph-ontology":
        graph_dir = concept_graph_dir(dataset, variant)
        if not graph_dir.exists():
            return f"missing concept graph directory: {graph_dir}"
        return ""
    return f"unknown method: {method}"


def write_skipped_records(dataset: str, method: str, indices: list[int], reason: str) -> None:
    records_dir = method_root(dataset, method) / "records"
    ensure_dir(records_dir)
    rows = [
        {
            "dataset": dataset,
            "method": method,
            "run_name": "",
            "split_idx": split_idx,
            "status": "skipped",
            "acc": "",
            "auc": "",
            "source_type": "ordered_5split_runner",
            "source_path": "",
            "text_mode": "",
            "ontology_variant": "",
            "text_dir": "",
            "text_graph_feature": "",
            "text_use_graph_structure": "",
            "split_file": str(DATASETS[dataset]["split_dir"] / f"split_{split_idx}.csv"),
            "graph_manifest_csv": "",
            "exp_dir": "",
            "best_path": "",
            "train_size": "",
            "test_size": "",
            "fusion_gate_mean": "",
            "doc_attention_entropy_mean": "",
            "doc_attention_max_mean": "",
            "skip_reason": reason,
        }
        for split_idx in indices
    ]
    write_split_records(records_dir, rows)


def write_split_records(records_dir: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(records_dir)
    fields = [
        "dataset",
        "method",
        "run_name",
        "split_idx",
        "status",
        "acc",
        "auc",
        "source_type",
        "source_path",
        "text_mode",
        "ontology_variant",
        "text_dir",
        "text_graph_feature",
        "text_use_graph_structure",
        "split_file",
        "graph_manifest_csv",
        "exp_dir",
        "best_path",
        "train_size",
        "test_size",
        "fusion_gate_mean",
        "doc_attention_entropy_mean",
        "doc_attention_max_mean",
        "skip_reason",
    ]
    with (records_dir / "split_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (records_dir / "split_results.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_lines = [
        "# Split Results",
        "",
        "| split_idx | status | acc | auc | run_name | skip_reason |",
        "|---:|---|---:|---:|---|---|",
    ]
    if not rows:
        md_lines.append("|  | no rows |  |  |  |  |")
    for row in rows:
        md_lines.append(
            "| {split_idx} | {status} | {acc} | {auc} | {run_name} | {skip_reason} |".format(
                split_idx=row.get("split_idx", ""),
                status=row.get("status", ""),
                acc=row.get("acc", ""),
                auc=row.get("auc", ""),
                run_name=row.get("run_name", ""),
                skip_reason=row.get("skip_reason", ""),
            )
        )
    (records_dir / "split_results.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def run_task(task: Task, args: argparse.Namespace) -> None:
    cmd = [
        str(PYTHON),
        str(REPO_ROOT / "run_main_splits.py"),
        "--config",
        str(task.config_path),
        "--split_dir",
        str(DATASETS[task.dataset]["split_dir"]),
        "--output_root",
        str(task.run_dir),
        "--num_splits",
        str(args.num_splits),
        "--split_offset",
        str(args.split_offset),
        "--train_num_workers",
        str(args.train_num_workers),
    ]
    if task.graph_manifest_template:
        cmd.extend(["--graph_manifest_template", task.graph_manifest_template])
    if args.force_train:
        cmd.append("--force_rerun")

    ensure_dir(task.run_dir)
    log_path = task.run_dir / "ordered_5split_runner.log"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{now()}] START {task.dataset} {task.method}\n")
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        completed = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
        log.write(f"\n[{now()}] END exit={completed.returncode}\n")
    if completed.returncode != 0:
        raise RuntimeError(f"Training failed for {task.dataset} {task.method}. See {log_path}")


def load_summary_rows(task: Task, variant: str) -> list[dict[str, Any]]:
    summary_csv = task.run_dir / "summary.csv"
    if not summary_csv.exists():
        return []
    rows: list[dict[str, Any]] = []
    with summary_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "dataset": task.dataset,
                    "method": task.method,
                    "run_name": task.run_name,
                    "split_idx": row.get("split_idx", ""),
                    "status": row.get("status", ""),
                    "acc": row.get("acc", ""),
                    "auc": row.get("auc", ""),
                    "source_type": "summary.csv",
                    "source_path": str(summary_csv),
                    "text_mode": "dual_text" if task.method != "sentence-only" else "sentence_pt",
                    "ontology_variant": variant if task.method.endswith("ontology") else "",
                    "text_dir": "",
                    "text_graph_feature": "node_features",
                    "text_use_graph_structure": str(task.method != "sentence-only"),
                    "split_file": row.get("split_file", ""),
                    "graph_manifest_csv": row.get("graph_manifest_csv", ""),
                    "exp_dir": row.get("exp_dir", ""),
                    "best_path": "",
                    "train_size": "",
                    "test_size": "",
                    "fusion_gate_mean": "",
                    "doc_attention_entropy_mean": "",
                    "doc_attention_max_mean": "",
                    "skip_reason": "",
                }
            )
    return rows


def write_records_from_summary(task: Task, variant: str) -> None:
    write_split_records(task.records_dir, load_summary_rows(task, variant))


def write_overall_plan(plan_path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(plan_path.parent)
    plan_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def mean_std(values: list[float]) -> str:
    if not values:
        return ""
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.4f} +/- {std:.4f}"


def write_report(tasks: list[Task], skipped: list[dict[str, Any]], variant: str) -> None:
    report_dir = REPO_ROOT / "experiment_records" / "reports" / "ordered_5split"
    ensure_dir(report_dir)
    rows: list[dict[str, str]] = []
    for task in tasks:
        split_rows = load_summary_rows(task, variant)
        accs = [float(row["acc"]) for row in split_rows if str(row.get("acc", "")).strip()]
        aucs = [float(row["auc"]) for row in split_rows if str(row.get("auc", "")).strip()]
        rows.append(
            {
                "dataset": task.dataset,
                "method": task.method,
                "status": "complete" if split_rows else "missing",
                "splits": str(len(split_rows)),
                "acc": mean_std(accs),
                "auc": mean_std(aucs),
                "run_dir": str(task.run_dir),
            }
        )
    for item in skipped:
        rows.append(
            {
                "dataset": item["dataset"],
                "method": item["method"],
                "status": "skipped",
                "splits": "0",
                "acc": "",
                "auc": "",
                "run_dir": item["reason"],
            }
        )

    with (report_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "method", "status", "splits", "acc", "auc", "run_dir"])
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Ordered 5-Split Experiment Summary",
        "",
        f"Generated: {now()}",
        f"Ontology variant: `{variant}`",
        "",
        "| Dataset | Method | Status | Splits | ACC | AUC | Location / reason |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['status']} | {row['splits']} | "
            f"{row['acc']} | {row['auc']} | `{row['run_dir']}` |"
        )
    (report_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    indices = split_indices(args)
    tasks: list[Task] = []
    skipped: list[dict[str, Any]] = []

    for dataset in args.datasets:
        for method in args.methods:
            ensure_dir(method_root(dataset, method) / "runs")
            ensure_dir(method_root(dataset, method) / "records")
            reason = unavailable_reason(dataset, method, args.ontology_variant, indices)
            if reason:
                print(f"[{now()}] SKIP {dataset} {method}: {reason}", flush=True)
                if not args.dry_run:
                    write_skipped_records(dataset, method, indices, reason)
                skipped.append({"dataset": dataset, "method": method, "reason": reason})
                continue

            manifest_template = ""
            if method == "sentence-hierarchical-graph-ontology":
                graph_dir = concept_graph_dir(dataset, args.ontology_variant)
                if not args.dry_run:
                    ensure_concept_manifests(dataset, args.ontology_variant, indices, graph_dir)
                manifest_template = str(
                    OUTPUT_ROOT
                    / "manifests"
                    / "ablation"
                    / args.ontology_variant
                    / f"{dataset.lower()}_concept_graph_manifest_split{{split_idx}}.csv"
                )

            config_path = CONFIG_ROOT / f"{dataset.lower()}_{method.replace('-', '_')}_5splits.yaml"
            if not args.dry_run:
                config_path = write_config(dataset, method, args.ontology_variant, manifest_template)
            tasks.append(
                Task(
                    dataset=dataset,
                    method=method,
                    run_name=run_dir(dataset, method).name,
                    run_dir=run_dir(dataset, method),
                    config_path=config_path,
                    graph_manifest_template=manifest_template,
                )
            )

    plan = {
        "created_at": now(),
        "datasets": args.datasets,
        "methods": args.methods,
        "num_splits": args.num_splits,
        "split_offset": args.split_offset,
        "split_indices": indices,
        "ontology_variant": args.ontology_variant,
        "tasks": [
            {
                "dataset": task.dataset,
                "method": task.method,
                "run_dir": str(task.run_dir),
                "config_path": str(task.config_path),
                "graph_manifest_template": task.graph_manifest_template,
            }
            for task in tasks
        ],
        "skipped": skipped,
    }

    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
        return 0

    write_overall_plan(REPO_ROOT / "experiment_records" / "reports" / "ordered_5split" / "plan.json", plan)

    for task in tasks:
        print(f"[{now()}] START {task.dataset} {METHOD_LABELS[task.method]}", flush=True)
        run_task(task, args)
        write_records_from_summary(task, args.ontology_variant)
        write_report(tasks, skipped, args.ontology_variant)
        print(f"[{now()}] DONE  {task.dataset} {METHOD_LABELS[task.method]}", flush=True)

    write_report(tasks, skipped, args.ontology_variant)
    print(f"[{now()}] ordered 5-split experiments complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
