from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required to generate training configs.") from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(r"F:\Anaconda\envs\pytorch\python.exe")
ONTOLOGY_PROCESSED_DIR = Path(r"F:\Tasks\Ontologies\processed")
ONTOLOGY_ABLATION_DIR = ONTOLOGY_PROCESSED_DIR / "ablations"
OUTPUT_ROOT = REPO_ROOT / "pathology_report_extraction" / "Output"
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
METHOD_DIR = "sentence-hierarchical-graph-ontology"
LOG_ROOT = EXPERIMENTS_ROOT / "KIRC" / METHOD_DIR / "records" / "shared_dual_text_concept_graph_ablation_logs"
AUX_GRAPH_WEIGHT_MAX = 0.2
AUX_GRAPH_WEIGHT_TARGET = 0.1
EXPERIMENT_TAG = "auxgw20_residual_sectiontitle"

VARIANTS = {
    "ncit_only": ONTOLOGY_ABLATION_DIR / "ncit_only_ontology.json",
    "ncit_do": ONTOLOGY_ABLATION_DIR / "ncit_do_ontology.json",
    "ncit_snomed_mapped": ONTOLOGY_ABLATION_DIR / "ncit_snomed_mapped_ontology.json",
    "full_multi_ontology": ONTOLOGY_ABLATION_DIR / "full_multi_ontology_ontology.json",
}
DEFAULT_VARIANTS = ["ncit_do"]

DATASETS = {
    "KIRC": {
        "sentence_exports": OUTPUT_ROOT / "sentence_exports_masked" / "KIRC",
        "sentence_embeddings": OUTPUT_ROOT / "sentence_embeddings_conch_masked" / "KIRC",
        "label_csv": Path(r"F:\Tasks\Pathologic_Stage_Label\KIRC_pathologic_stage.csv"),
        "split_dir": Path(r"F:\Tasks\Split_Table\KIRC"),
        "image_dir": Path(r"F:\Tasks\WSI_extract_features\KIRC_WSI_extract_features"),
        "sentence_text_dir": Path(r"F:\Tasks\Text_Sentence_extract_features\KIRC_text"),
    },
    "BRCA": {
        "sentence_exports": OUTPUT_ROOT / "sentence_exports_masked" / "BRCA",
        "sentence_embeddings": OUTPUT_ROOT / "sentence_embeddings_conch_masked" / "BRCA",
        "label_csv": Path(r"F:\Tasks\Pathologic_Stage_Label\BRCA_pathologic_stage.csv"),
        "split_dir": Path(r"F:\Tasks\Split_Table\BRCA"),
        "image_dir": Path(r"F:\Tasks\WSI_extract_features\BRCA_WSI_extract_features"),
        "sentence_text_dir": Path(r"F:\Tasks\Text_Sentence_extract_features\BRCA_text"),
    },
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def experiment_name(dataset: str, variant: str) -> str:
    return f"{dataset.lower()}_dual_text_concept_graph_{variant}_{EXPERIMENT_TAG}_3splits_nw0"


def experiment_dir(dataset: str, variant: str) -> Path:
    return EXPERIMENTS_ROOT / dataset / METHOD_DIR / "runs" / experiment_name(dataset, variant)


def run_command(name: str, args: list[str], log_path: Path) -> None:
    ensure_dir(log_path.parent)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n\n[{timestamp}] START {name}\n")
        log.write(" ".join(args) + "\n\n")
        log.flush()
        completed = subprocess.run(
            args,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log.write(f"\n[{end_timestamp}] END {name} exit={completed.returncode}\n")
    if completed.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {completed.returncode}. See {log_path}")


def summary_success(path: Path, success_key: str, total_key: str, allow_skipped: bool = False) -> bool:
    if not path.exists():
        return False
    try:
        payload = load_json(path)
    except Exception:
        return False
    success_count = int(payload.get(success_key, -1))
    total_count = int(payload.get(total_key, -2))
    failure_count = int(payload.get("failure_count", 0))
    skipped_count = int(payload.get("skipped_count", 0))
    if failure_count != 0:
        return False
    if allow_skipped:
        return success_count + skipped_count == total_count
    return success_count == total_count


def build_ontology_bundles(variants: list[str]) -> None:
    run_command(
        "build_ontology_ablation_bundles",
        [
            str(PYTHON),
            str(REPO_ROOT / "pathology_report_extraction" / "build_ontology_ablation_bundles.py"),
            "--processed_dir",
            str(ONTOLOGY_PROCESSED_DIR),
            "--output_dir",
            str(ONTOLOGY_ABLATION_DIR),
            "--variants",
            *variants,
        ],
        LOG_ROOT / "build_ontology_ablation_bundles.log",
    )


def build_annotations(dataset: str, variant: str, force: bool) -> Path:
    cfg = DATASETS[dataset]
    output_dir = OUTPUT_ROOT / "concept_annotations_ablation" / variant / dataset
    summary_path = output_dir / "run_summary.json"
    if not force and summary_success(summary_path, "success_count", "total_sentence_exports"):
        return output_dir

    run_command(
        f"extract_concepts_{dataset}_{variant}",
        [
            str(PYTHON),
            str(REPO_ROOT / "pathology_report_extraction" / "extract_ontology_concepts.py"),
            "--input_dir",
            str(cfg["sentence_exports"]),
            "--output_dir",
            str(output_dir),
            "--ontology_path",
            str(VARIANTS[variant]),
            "--include_true_path",
        ],
        LOG_ROOT / variant / dataset / "extract_ontology_concepts.log",
    )
    return output_dir


def build_graphs(dataset: str, variant: str, concept_dir: Path, force: bool) -> Path:
    cfg = DATASETS[dataset]
    output_dir = OUTPUT_ROOT / "text_concept_graphs_ablation" / variant / dataset
    summary_path = output_dir / "run_summary.json"
    if not force and summary_success(summary_path, "success_count", "total_metadata_files", allow_skipped=True):
        return output_dir

    run_command(
        f"build_graphs_{dataset}_{variant}",
        [
            str(PYTHON),
            str(REPO_ROOT / "pathology_report_extraction" / "build_text_hierarchy_graphs.py"),
            "--input_dir",
            str(cfg["sentence_embeddings"]),
            "--output_dir",
            str(output_dir),
            "--concept_dir",
            str(concept_dir),
            "--attach_concepts",
        ],
        LOG_ROOT / variant / dataset / "build_text_hierarchy_graphs.log",
    )
    return output_dir


def build_manifest(dataset: str, variant: str, split_idx: int, graph_dir: Path) -> Path:
    cfg = DATASETS[dataset]
    output_csv = OUTPUT_ROOT / "manifests" / "ablation" / variant / f"{dataset.lower()}_concept_graph_manifest_split{split_idx}.csv"
    run_command(
        f"prepare_manifest_{dataset}_{variant}_split{split_idx}",
        [
            str(PYTHON),
            str(REPO_ROOT / "pathology_report_extraction" / "prepare_text_graph_manifest.py"),
            "--graph_dir",
            str(graph_dir),
            "--label_csv",
            str(cfg["label_csv"]),
            "--split_csv",
            str(cfg["split_dir"] / f"split_{split_idx}.csv"),
            "--image_dir",
            str(cfg["image_dir"]),
            "--output_csv",
            str(output_csv),
        ],
        LOG_ROOT / variant / dataset / f"prepare_manifest_split{split_idx}.log",
    )
    return output_csv


def write_train_config(dataset: str, variant: str, graph_dir: Path, manifest_template: str) -> Path:
    cfg = DATASETS[dataset]
    config_path = REPO_ROOT / "configs" / "generated" / f"{dataset.lower()}_{variant}_dual_text_concept_graph_{EXPERIMENT_TAG}.yaml"
    exp_dir = experiment_dir(dataset, variant)
    payload = {
        "seed": 23,
        "data": {
            "split_file": str(cfg["split_dir"] / "split_0.csv"),
            "label_file": str(cfg["label_csv"]),
            "image_dir": str(cfg["image_dir"]),
            "text_dir": str(graph_dir),
            "text_mode": "dual_text",
            "sentence_text_dir": str(cfg["sentence_text_dir"]),
            "graph_text_dir": str(graph_dir),
            "graph_manifest_csv": manifest_template.format(split_idx=0),
            "text_graph_feature": "node_features",
            "text_use_graph_structure": True,
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
            "text_graph_num_node_types": 4,
            "text_graph_num_base_relations": 6,
            "sentence_local_layers": 1,
            "sentence_local_heads": 8,
            "hier_readout_hidden": 256,
            "hier_readout_dropout": 0.1,
            "hier_readout_attention_init": -0.5,
            "use_section_title_embedding": True,
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
            "text_dual_graph_weight_max": AUX_GRAPH_WEIGHT_MAX,
            "text_dual_fusion_mode": "residual",
        },
        "loss": {
            "warmup_epochs": 3,
            "align_decay_to": 0.3,
            "alpha_txt": 0.5,
            "alpha_concept": 0.2,
            "beta_node": 0.1,
            "gamma_topo": 0.1,
            "gamma_text_topology": 0.05,
            "dual_text_gate_reg_weight": 0.01,
            "dual_text_graph_weight_target": AUX_GRAPH_WEIGHT_TARGET,
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
    ensure_dir(config_path.parent)
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return config_path


def train_variant(dataset: str, variant: str, config_path: Path, manifest_template: str, num_splits: int, force: bool) -> Path:
    cfg = DATASETS[dataset]
    output_root = experiment_dir(dataset, variant)
    args = [
        str(PYTHON),
        str(REPO_ROOT / "run_main_splits.py"),
        "--config",
        str(config_path),
        "--split_dir",
        str(cfg["split_dir"]),
        "--output_root",
        str(output_root),
        "--num_splits",
        str(num_splits),
        "--split_offset",
        "0",
        "--graph_manifest_template",
        manifest_template,
        "--train_num_workers",
        "0",
    ]
    if force:
        args.append("--force_rerun")
    run_command(
        f"train_{dataset}_{variant}",
        args,
        LOG_ROOT / variant / dataset / "run_main_splits.log",
    )
    return output_root


def aggregate_results(datasets: list[str], variants: list[str]) -> None:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for variant in variants:
            summary_path = experiment_dir(dataset, variant) / "summary.json"
            if not summary_path.exists():
                rows.append({"dataset": dataset, "variant": variant, "status": "missing"})
                continue
            payload = load_json(summary_path)
            rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "status": "complete",
                    "acc_mean": payload.get("acc_mean"),
                    "acc_std": payload.get("acc_std"),
                    "auc_mean": payload.get("auc_mean"),
                    "auc_std": payload.get("auc_std"),
                    "summary_json": str(summary_path),
                }
            )

    output_dir = REPO_ROOT / "experiment_records" / "dual_text_concept_graph_ablation"
    ensure_dir(output_dir)
    write_json(output_dir / "summary.json", {"rows": rows})

    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["dataset", "variant", "status", "acc_mean", "acc_std", "auc_mean", "auc_std", "summary_json"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# Dual Text + Concept Graph Ontology Ablation",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| Dataset | Variant | ACC mean +/- std | AUC mean +/- std | Summary |",
        "|---|---|---:|---:|---|",
    ]
    for row in rows:
        if row.get("status") != "complete":
            md_lines.append(f"| {row['dataset']} | {row['variant']} | missing | missing | |")
            continue
        md_lines.append(
            "| {dataset} | {variant} | {acc_mean:.4f} +/- {acc_std:.4f} | "
            "{auc_mean:.4f} +/- {auc_std:.4f} | `{summary_json}` |".format(
                dataset=row["dataset"],
                variant=row["variant"],
                acc_mean=float(row["acc_mean"]),
                acc_std=float(row["acc_std"]),
                auc_mean=float(row["auc_mean"]),
                auc_std=float(row["auc_std"]),
                summary_json=row["summary_json"],
            )
        )
    (output_dir / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dual_text + concept graph ontology ablations.")
    parser.add_argument("--datasets", nargs="+", default=["KIRC", "BRCA"], choices=sorted(DATASETS))
    parser.add_argument(
        "--variants",
        nargs="+",
        default=DEFAULT_VARIANTS,
        choices=sorted(VARIANTS),
        help="Default is ncit_do only. SNOMED/UMLS variants are retained as explicit legacy ablations.",
    )
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--force_preprocess", action="store_true")
    parser.add_argument("--force_train", action="store_true")
    parser.add_argument("--skip_preprocess", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(LOG_ROOT)
    build_ontology_bundles(args.variants)

    for dataset in args.datasets:
        for variant in args.variants:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] start {dataset} / {variant}", flush=True)
            if args.skip_preprocess:
                graph_dir = OUTPUT_ROOT / "text_concept_graphs_ablation" / variant / dataset
            else:
                concept_dir = build_annotations(dataset, variant, force=args.force_preprocess)
                graph_dir = build_graphs(dataset, variant, concept_dir=concept_dir, force=args.force_preprocess)
                for split_idx in range(args.num_splits):
                    build_manifest(dataset, variant, split_idx, graph_dir)

            manifest_template = str(
                OUTPUT_ROOT
                / "manifests"
                / "ablation"
                / variant
                / f"{dataset.lower()}_concept_graph_manifest_split{{split_idx}}.csv"
            )
            config_path = write_train_config(dataset, variant, graph_dir, manifest_template)
            if not args.skip_train:
                train_variant(
                    dataset=dataset,
                    variant=variant,
                    config_path=config_path,
                    manifest_template=manifest_template,
                    num_splits=args.num_splits,
                    force=args.force_train,
                )
            aggregate_results(args.datasets, args.variants)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] done {dataset} / {variant}", flush=True)

    aggregate_results(args.datasets, args.variants)


if __name__ == "__main__":
    main()
