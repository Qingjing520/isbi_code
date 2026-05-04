from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
from dataclasses import dataclass, replace
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
HIERARCHY_GRAPH_ROOT = Path(r"F:\Tasks\Pathology_Report_Hierarchy_Graphs")
HIERARCHY_GRAPH_TYPE_DIRS = {
    "basic": "basic_hierarchy",
    "ontology_concept": "ontology_concept_hierarchy",
    "stage_keyword_word": "stage_keyword_word_hierarchy",
    "stage_keyword_word_ontology": "stage_keyword_word_ontology_hierarchy",
}
CONFIG_ROOT = REPO_ROOT / "configs" / "generated" / "ordered_splits"
DEFAULT_ONTOLOGY_VARIANT = "ncit_do"
DEFAULT_HIERARCHY_GRAPH_WEIGHT_MAX = 0.1
DEFAULT_ONTOLOGY_GRAPH_WEIGHT_MAX = 0.1
DEFAULT_GRAPH_WEIGHT_TARGET = 0.0
DEFAULT_DUAL_TEXT_FUSION_MODE = "residual"
DEFAULT_TEXT_GRAPH_GATE_INIT = 0.05

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

LEGACY_RUN_PATTERNS = {
    ("LUSC", "sentence-only"): (
        r"^LUSC_(\d+)splits_sentence_only$",
        r"^LUSC_(\d+)splits_sentence_mode$",
    ),
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
    split_indices: list[int]
    completed_before: list[int]
    experiment_tag: str = ""
    graph_manifest_template: str = ""

    @property
    def records_dir(self) -> Path:
        return EXPERIMENTS_ROOT / self.dataset / self.method / "records"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run clean, ordered split experiments in the organized experiments tree."
    )
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_ORDER), choices=DATASET_ORDER)
    parser.add_argument("--methods", nargs="+", default=list(METHOD_ORDER), choices=METHOD_ORDER)
    parser.add_argument(
        "--num_splits",
        type=int,
        default=5,
        help="Number of new splits to run for each available dataset/method.",
    )
    parser.add_argument(
        "--split_offset",
        type=int,
        default=None,
        help=(
            "Optional explicit starting split. By default the runner continues from the "
            "next split after the latest completed split in the existing run folder."
        ),
    )
    parser.add_argument("--ontology_variant", type=str, default=DEFAULT_ONTOLOGY_VARIANT)
    parser.add_argument(
        "--hierarchy_ontology_graph_type",
        type=str,
        default=HIERARCHY_GRAPH_TYPE_DIRS["ontology_concept"],
        choices=[
            HIERARCHY_GRAPH_TYPE_DIRS["ontology_concept"],
            HIERARCHY_GRAPH_TYPE_DIRS["stage_keyword_word_ontology"],
        ],
        help=(
            "Graph library subfolder used by sentence-hierarchical-graph-ontology. "
            "Use stage_keyword_word_ontology_hierarchy for the cleaned word+ontology graph."
        ),
    )
    parser.add_argument(
        "--hierarchy_graph_weight_max",
        type=float,
        default=DEFAULT_HIERARCHY_GRAPH_WEIGHT_MAX,
        help="Maximum hierarchy graph residual weight in dual-text fusion.",
    )
    parser.add_argument(
        "--ontology_graph_weight_max",
        type=float,
        default=DEFAULT_ONTOLOGY_GRAPH_WEIGHT_MAX,
        help="Maximum ontology/concept graph residual weight in dual-text fusion.",
    )
    parser.add_argument(
        "--text_graph_gate_init",
        type=float,
        default=DEFAULT_TEXT_GRAPH_GATE_INIT,
        help="Initial bounded graph branch weight. Use 0.0 for identity sanity checks.",
    )
    parser.add_argument(
        "--ontology_evidence_only",
        action="store_true",
        help="Keep only staging/pathology evidence concepts for ontology graph branches.",
    )
    parser.add_argument(
        "--graph_weight_target",
        type=float,
        default=DEFAULT_GRAPH_WEIGHT_TARGET,
        help="Gate regularization target for the auxiliary graph branch.",
    )
    parser.add_argument(
        "--fusion_mode",
        type=str,
        default=DEFAULT_DUAL_TEXT_FUSION_MODE,
        choices=["convex", "residual"],
        help="dual_text fusion mode. residual keeps sentence features as the trunk.",
    )
    parser.set_defaults(use_section_title_embedding=True)
    parser.add_argument(
        "--use_section_title_embedding",
        dest="use_section_title_embedding",
        action="store_true",
        help="Use section title embeddings as section-role features.",
    )
    parser.add_argument(
        "--no_section_title_embedding",
        dest="use_section_title_embedding",
        action="store_false",
        help="Disable section title embeddings.",
    )
    parser.add_argument("--train_num_workers", type=int, default=0)
    parser.add_argument("--force_train", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--experiment_tag",
        type=str,
        default="",
        help="Optional tag inserted into run/config names, e.g. auxgw01_sectiontitle_residual.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def method_root(dataset: str, method: str) -> Path:
    return EXPERIMENTS_ROOT / dataset / method


def sanitize_tag(tag: str) -> str:
    tag = re.sub(r"[^a-zA-Z0-9_+-]+", "_", str(tag or "").strip())
    return tag.strip("_")


def run_dir(dataset: str, method: str, experiment_tag: str = "") -> Path:
    return run_dir_for_count(dataset, method, 0, experiment_tag=experiment_tag)


def run_prefix(dataset: str, method: str, experiment_tag: str = "") -> str:
    tag = sanitize_tag(experiment_tag)
    tag_part = f"{tag}_" if tag else ""
    return f"{dataset.lower()}_{method.replace('-', '_')}_{tag_part}"


def run_dir_for_count(dataset: str, method: str, count: int, experiment_tag: str = "") -> Path:
    count = max(0, int(count))
    return method_root(dataset, method) / "runs" / f"{run_prefix(dataset, method, experiment_tag=experiment_tag)}{count}splits"


def completed_split_indices(run_dir: Path) -> list[int]:
    indices: list[int] = []
    if not run_dir.exists():
        return indices
    for split_dir in run_dir.glob("split_*"):
        if not split_dir.is_dir():
            continue
        match = re.match(r"split_(\d+)$", split_dir.name)
        if not match:
            continue
        if (split_dir / "best_model.pt").exists() or (split_dir / "log.jsonl").exists():
            indices.append(int(match.group(1)))
    return sorted(set(indices))


def find_existing_run_dir(dataset: str, method: str, experiment_tag: str = "") -> Path | None:
    runs_root = method_root(dataset, method) / "runs"
    if not runs_root.exists():
        return None

    prefix = run_prefix(dataset, method, experiment_tag=experiment_tag)
    candidates: list[tuple[int, int, float, Path]] = []
    for child in runs_root.iterdir():
        if not child.is_dir() or child.name.startswith("_"):
            continue

        completed = completed_split_indices(child)
        if not completed:
            continue

        standard_match = re.match(rf"^{re.escape(prefix)}(\d+)splits$", child.name, flags=re.IGNORECASE)
        if standard_match:
            suffix_count = int(standard_match.group(1))
            candidates.append((1, max(suffix_count, len(completed)), child.stat().st_mtime, child))
            continue

        for pattern in LEGACY_RUN_PATTERNS.get((dataset, method), ()):
            legacy_match = re.match(pattern, child.name, flags=re.IGNORECASE)
            if legacy_match:
                suffix_count = int(legacy_match.group(1))
                candidates.append((1, max(suffix_count, len(completed)), child.stat().st_mtime, child))
                break

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return candidates[0][3]


def planned_run_dir(dataset: str, method: str, experiment_tag: str = "") -> Path:
    return find_existing_run_dir(dataset, method, experiment_tag=experiment_tag) or run_dir_for_count(
        dataset,
        method,
        0,
        experiment_tag=experiment_tag,
    )


def task_split_indices(run_dir_path: Path, args: argparse.Namespace) -> tuple[list[int], list[int]]:
    completed = completed_split_indices(run_dir_path)
    start = int(args.split_offset) if args.split_offset is not None else ((max(completed) + 1) if completed else 0)
    return list(range(start, start + args.num_splits)), completed


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
    return HIERARCHY_GRAPH_ROOT / dataset / HIERARCHY_GRAPH_TYPE_DIRS["basic"]


def sentence_ontology_graph_dir(dataset: str, variant: str) -> Path:
    return OUTPUT_ROOT / "text_sentence_ontology_graphs_ablation" / variant / dataset


def concept_graph_dir(
    dataset: str,
    variant: str,
    graph_type: str = HIERARCHY_GRAPH_TYPE_DIRS["ontology_concept"],
) -> Path:
    if variant == DEFAULT_ONTOLOGY_VARIANT:
        return HIERARCHY_GRAPH_ROOT / dataset / graph_type
    return OUTPUT_ROOT / "text_concept_graphs_ablation" / variant / dataset


def sentence_ontology_manifest_path(dataset: str, variant: str, split_idx: int) -> Path:
    return (
        OUTPUT_ROOT
        / "manifests"
        / "ablation"
        / variant
        / f"{dataset.lower()}_sentence_ontology_graph_manifest_split{split_idx}.csv"
    )


def concept_manifest_path(
    dataset: str,
    variant: str,
    split_idx: int,
    graph_type: str = HIERARCHY_GRAPH_TYPE_DIRS["ontology_concept"],
) -> Path:
    if graph_type == HIERARCHY_GRAPH_TYPE_DIRS["ontology_concept"]:
        return (
            OUTPUT_ROOT
            / "manifests"
            / "ablation"
            / variant
            / f"{dataset.lower()}_concept_graph_manifest_split{split_idx}.csv"
        )
    return (
        OUTPUT_ROOT
        / "manifests"
        / "graph_library"
        / graph_type
        / f"{dataset.lower()}_{graph_type}_manifest_split{split_idx}.csv"
    )


def build_graph_manifest(dataset: str, split_idx: int, graph_dir: Path, output_csv: Path) -> Path:
    cfg = DATASETS[dataset]
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
    print(f"[{now()}] build manifest {dataset} split={split_idx} output={output_csv.name}", flush=True)
    completed = subprocess.run(args, cwd=REPO_ROOT, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"prepare_text_graph_manifest failed for {dataset} split {split_idx}")
    return output_csv


def graph_manifest_matches_dir(manifest: Path, graph_dir: Path) -> bool:
    if not manifest.exists():
        return False
    try:
        graph_root = str(Path(graph_dir).resolve()).lower()
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                graph_pt = str(row.get("graph_pt") or "").strip()
                if not graph_pt:
                    continue
                graph_path = Path(graph_pt)
                return graph_path.exists() and str(graph_path.resolve()).lower().startswith(graph_root)
    except Exception:
        return False
    return False


def ensure_graph_manifests(
    dataset: str,
    indices: list[int],
    graph_dir: Path,
    manifest_path_fn,
) -> list[Path]:
    manifests: list[Path] = []
    for split_idx in indices:
        manifest = manifest_path_fn(dataset, split_idx)
        if not graph_manifest_matches_dir(manifest, graph_dir):
            manifest = build_graph_manifest(dataset, split_idx, graph_dir, manifest)
        manifests.append(manifest)
    return manifests


def base_payload(
    dataset: str,
    exp_dir: Path,
    *,
    use_section_title_embedding: bool = False,
    fusion_mode: str = "convex",
) -> dict[str, Any]:
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
            "use_section_title_embedding": bool(use_section_title_embedding),
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
            "text_dual_fusion_mode": str(fusion_mode),
            "text_dual_gate_init_bias": -6.0,
            "text_dual_apply_post_norm": False,
        },
        "text_graph": {
            "gate_init": DEFAULT_TEXT_GRAPH_GATE_INIT,
            "gate_max": 0.3,
        },
        "hierarchy": {
            "enabled": False,
            "use_section_edges": True,
            "use_sentence_similarity_edges": True,
            "sentence_topk": 5,
            "min_sentence_sim": 0.3,
            "edge_dropout": 0.1,
            "max_sentences_per_report": 128,
        },
        "ontology": {
            "enabled": False,
            "max_concepts_per_report": 30,
            "min_concept_confidence": 0.7,
            "remove_generic_concepts": True,
            "remove_leakage_concepts": True,
            "allowed_semantic_types": [],
            "edge_hop_limit": 1,
            "edge_dropout": 0.2,
            "evidence_only": False,
            "evidence_keywords": [
                "tumor size",
                "size",
                "invasion",
                "invasive",
                "lymph node",
                "lymph nodes",
                "lymphatic",
                "nodal",
                "margin",
                "margins",
                "metastasis",
                "metastatic",
                "anatomic extent",
                "extension",
                "extranodal",
                "capsule",
                "vascular invasion",
                "lymphovascular invasion",
                "perineural invasion",
            ],
            "evidence_deny_keywords": [
                "estrogen receptor",
                "progesterone receptor",
                "hormone receptor",
                "er",
                "pr",
                "her2",
                "immunohistochemistry",
                "procedure",
                "mastectomy",
                "biopsy",
                "carcinoma",
                "adenocarcinoma",
                "neoplasm",
            ],
            "generic_concept_denylist": [
                "Disease",
                "Neoplasm",
                "Patient",
                "Procedure",
                "Finding",
                "Clinical finding",
                "Body structure",
            ],
            "leakage_concept_denylist": [
                "stage",
                "staging",
                "pathologic stage",
                "TNM",
                "AJCC",
                "grade",
            ],
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
            "dual_text_graph_weight_target": 0.0,
            "mmd_num_kernels": 3,
            "mmd_sigma_multipliers": [0.5, 1.0, 2.0],
            "mmd_unbiased": True,
            "mmd_clamp_nonneg": True,
            "graph_aux_align_enabled": False,
            "graph_aux_align_weight": 0.01,
            "graph_aux_align_warmup_epochs": 8,
            "graph_aux_align_conf_threshold": 0.7,
            "graph_aux_align_decay_to": 0.2,
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


def write_config(
    dataset: str,
    method: str,
    variant: str,
    exp_dir: Path,
    graph_manifest_template: str = "",
    *,
    hierarchy_ontology_graph_type: str = HIERARCHY_GRAPH_TYPE_DIRS["ontology_concept"],
    hierarchy_graph_weight_max: float = DEFAULT_HIERARCHY_GRAPH_WEIGHT_MAX,
    ontology_graph_weight_max: float = DEFAULT_ONTOLOGY_GRAPH_WEIGHT_MAX,
    graph_weight_target: float = DEFAULT_GRAPH_WEIGHT_TARGET,
    text_graph_gate_init: float = DEFAULT_TEXT_GRAPH_GATE_INIT,
    ontology_evidence_only: bool = False,
    use_section_title_embedding: bool = True,
    fusion_mode: str = DEFAULT_DUAL_TEXT_FUSION_MODE,
    experiment_tag: str = "",
) -> Path:
    payload = base_payload(
        dataset,
        exp_dir,
        use_section_title_embedding=(method != "sentence-only" and use_section_title_embedding),
        fusion_mode=fusion_mode,
    )
    cfg = DATASETS[dataset]
    payload["text_graph"]["gate_init"] = float(text_graph_gate_init)
    payload["ontology"]["evidence_only"] = bool(ontology_evidence_only)

    if method == "sentence-only":
        pass
    elif method == "sentence-ontology":
        graph_dir = sentence_ontology_graph_dir(dataset, variant)
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
        payload["model"]["text_graph_num_node_types"] = 3
        payload["model"]["text_graph_num_base_relations"] = 5
        payload["model"]["text_dual_graph_weight_max"] = float(ontology_graph_weight_max)
        payload["text_graph"]["gate_max"] = float(ontology_graph_weight_max)
        payload["ontology"]["enabled"] = True
        payload["loss"]["alpha_concept"] = 0.0
        payload["loss"]["gamma_text_topology"] = 0.0
        payload["loss"]["dual_text_gate_reg_weight"] = 0.0
        payload["loss"]["dual_text_graph_weight_target"] = float(graph_weight_target)
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
        payload["model"]["text_dual_graph_weight_max"] = float(hierarchy_graph_weight_max)
        payload["text_graph"]["gate_max"] = float(hierarchy_graph_weight_max)
        payload["hierarchy"]["enabled"] = True
        payload["loss"]["gamma_text_topology"] = 0.0
        payload["loss"]["dual_text_gate_reg_weight"] = 0.0
        payload["loss"]["dual_text_graph_weight_target"] = float(graph_weight_target)
    elif method == "sentence-hierarchical-graph-ontology":
        graph_dir = concept_graph_dir(dataset, variant, graph_type=hierarchy_ontology_graph_type)
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
        payload["model"]["text_graph_num_node_types"] = (
            5
            if hierarchy_ontology_graph_type == HIERARCHY_GRAPH_TYPE_DIRS["stage_keyword_word_ontology"]
            else 4
        )
        payload["model"]["text_graph_num_base_relations"] = 6
        payload["model"]["text_dual_graph_weight_max"] = float(ontology_graph_weight_max)
        payload["text_graph"]["gate_max"] = float(ontology_graph_weight_max)
        payload["hierarchy"]["enabled"] = True
        payload["ontology"]["enabled"] = True
        payload["loss"]["alpha_concept"] = 0.0
        payload["loss"]["gamma_text_topology"] = 0.0
        payload["loss"]["dual_text_gate_reg_weight"] = 0.0
        payload["loss"]["dual_text_graph_weight_target"] = float(graph_weight_target)
    else:
        raise ValueError(f"Cannot write config for unsupported method: {method}")

    tag = sanitize_tag(experiment_tag)
    tag_part = f"_{tag}" if tag else ""
    config_path = CONFIG_ROOT / f"{dataset.lower()}_{method.replace('-', '_')}{tag_part}.yaml"
    ensure_dir(config_path.parent)
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return config_path


def unavailable_reason(
    dataset: str,
    method: str,
    variant: str,
    indices: list[int],
    hierarchy_ontology_graph_type: str = HIERARCHY_GRAPH_TYPE_DIRS["ontology_concept"],
) -> str:
    missing = missing_base_inputs(dataset, indices)
    if missing:
        return "missing base inputs: " + "; ".join(missing)

    if method == "sentence-only":
        return ""
    if method == "sentence-ontology":
        graph_dir = sentence_ontology_graph_dir(dataset, variant)
        if not graph_dir.exists():
            return f"missing sentence-ontology graph directory: {graph_dir}"
        return ""
    if method == "sentence-hierarchical-graph":
        graph_dir = hierarchy_graph_dir(dataset)
        if not graph_dir.exists():
            return f"missing hierarchy graph directory: {graph_dir}"
        return ""
    if method == "sentence-hierarchical-graph-ontology":
        graph_dir = concept_graph_dir(dataset, variant, graph_type=hierarchy_ontology_graph_type)
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
            "source_type": "ordered_split_runner",
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
    if not task.split_indices:
        return
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
        str(len(task.split_indices)),
        "--split_offset",
        str(task.split_indices[0]),
        "--train_num_workers",
        str(args.train_num_workers),
    ]
    if task.graph_manifest_template:
        cmd.extend(["--graph_manifest_template", task.graph_manifest_template])
    if args.force_train:
        cmd.append("--force_rerun")

    ensure_dir(task.run_dir)
    log_path = task.run_dir / "ordered_split_runner.log"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{now()}] START {task.dataset} {task.method}\n")
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        process = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        returncode = process.wait()
        log.write(f"\n[{now()}] END exit={returncode}\n")
    if returncode != 0:
        raise RuntimeError(f"Training failed for {task.dataset} {task.method}. See {log_path}")


def best_or_final_log_metrics(log_path: Path) -> tuple[str, str, str]:
    if not log_path.exists():
        return "", "", ""

    best_acc = ""
    best_auc = ""
    best_epoch = ""
    try:
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") == "final_evaluation":
                target = record.get("target", {})
                return str(target.get("acc", "")), str(target.get("auc", "")), "final_evaluation"
            target = record.get("target")
            if isinstance(target, dict) and "auc" in target:
                auc = float(target["auc"])
                if best_auc == "" or auc > float(best_auc):
                    best_acc = str(target.get("acc", ""))
                    best_auc = str(target.get("auc", ""))
                    best_epoch = str(record.get("epoch", ""))
    except Exception:
        return "", "", ""

    if best_auc != "":
        return best_acc, best_auc, f"from_log_best_epoch_{best_epoch}"
    return "", "", ""


def load_summary_rows(task: Task, variant: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_dir in sorted(task.run_dir.glob("split_*"), key=lambda path: int(path.name.split("_")[-1]) if path.name.split("_")[-1].isdigit() else 10**9):
        if not split_dir.is_dir():
            continue
        split_text = split_dir.name.split("_")[-1]
        if not split_text.isdigit():
            continue
        split_idx = int(split_text)
        acc, auc, status = best_or_final_log_metrics(split_dir / "log.jsonl")
        if not (split_dir / "best_model.pt").exists() and not auc:
            continue
        graph_manifest_csv = ""
        split_file = str(DATASETS[task.dataset]["split_dir"] / f"split_{split_idx}.csv")
        config_path = split_dir / "config.json"
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                data = config.get("data", {})
                graph_manifest_csv = str(data.get("graph_manifest_csv", ""))
                split_file = str(data.get("split_file", split_file))
            except Exception:
                pass
        rows.append(
            {
                "dataset": task.dataset,
                "method": task.method,
                "run_name": task.run_dir.name,
                "split_idx": str(split_idx),
                "status": status or "checkpoint_exists",
                "acc": acc,
                "auc": auc,
                "source_type": "log.jsonl",
                "source_path": str(split_dir / "log.jsonl"),
                "text_mode": "dual_text" if task.method != "sentence-only" else "sentence_pt",
                "ontology_variant": variant if task.method.endswith("ontology") else "",
                "text_dir": "",
                "text_graph_feature": "node_features",
                "text_use_graph_structure": str(task.method != "sentence-only"),
                "split_file": split_file,
                "graph_manifest_csv": graph_manifest_csv,
                "exp_dir": str(split_dir),
                "best_path": str(split_dir / "best_model.pt"),
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


def rename_run_dir_to_completed_count(task: Task) -> Task:
    completed = completed_split_indices(task.run_dir)
    if not completed:
        return task

    target = run_dir_for_count(
        task.dataset,
        task.method,
        len(completed),
        experiment_tag=task.experiment_tag,
    )
    if task.run_dir.resolve() == target.resolve():
        return task
    if target.exists():
        print(
            f"[{now()}] keep run folder name because target exists: {target}",
            flush=True,
        )
        return task

    ensure_dir(target.parent)
    task.run_dir.rename(target)
    print(f"[{now()}] renamed run folder: {task.run_dir.name} -> {target.name}", flush=True)
    return replace(task, run_name=target.name, run_dir=target)


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
    report_dir = REPO_ROOT / "experiment_records" / "reports" / "ordered_splits"
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
        "# Ordered Split Experiment Summary",
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


def serialize_tasks(tasks: list[Task]) -> list[dict[str, Any]]:
    return [
        {
            "dataset": task.dataset,
            "method": task.method,
            "run_dir": str(task.run_dir),
            "config_path": str(task.config_path),
            "graph_manifest_template": task.graph_manifest_template,
            "completed_before": task.completed_before,
            "requested_split_indices": task.split_indices,
            "experiment_tag": task.experiment_tag,
        }
        for task in tasks
    ]


def main() -> int:
    args = parse_args()
    tasks: list[Task] = []
    skipped: list[dict[str, Any]] = []

    for dataset in args.datasets:
        for method in args.methods:
            ensure_dir(method_root(dataset, method) / "runs")
            ensure_dir(method_root(dataset, method) / "records")
            current_run_dir = planned_run_dir(dataset, method, experiment_tag=args.experiment_tag)
            indices, completed_before = task_split_indices(current_run_dir, args)
            reason = unavailable_reason(
                dataset,
                method,
                args.ontology_variant,
                indices,
                hierarchy_ontology_graph_type=args.hierarchy_ontology_graph_type,
            )
            if reason:
                print(f"[{now()}] SKIP {dataset} {method}: {reason}", flush=True)
                if not args.dry_run:
                    write_skipped_records(dataset, method, indices, reason)
                skipped.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "reason": reason,
                        "requested_split_indices": indices,
                        "completed_before": completed_before,
                    }
                )
                continue

            manifest_template = ""
            if method == "sentence-ontology":
                graph_dir = sentence_ontology_graph_dir(dataset, args.ontology_variant)
                if not args.dry_run:
                    ensure_graph_manifests(
                        dataset=dataset,
                        indices=indices,
                        graph_dir=graph_dir,
                        manifest_path_fn=lambda dataset_name, split_idx, variant=args.ontology_variant: sentence_ontology_manifest_path(
                            dataset_name,
                            variant,
                            split_idx,
                        ),
                    )
                manifest_template = str(
                    OUTPUT_ROOT
                    / "manifests"
                    / "ablation"
                    / args.ontology_variant
                    / f"{dataset.lower()}_sentence_ontology_graph_manifest_split{{split_idx}}.csv"
                )
            elif method == "sentence-hierarchical-graph-ontology":
                graph_dir = concept_graph_dir(
                    dataset,
                    args.ontology_variant,
                    graph_type=args.hierarchy_ontology_graph_type,
                )
                if not args.dry_run:
                    ensure_graph_manifests(
                        dataset=dataset,
                        indices=indices,
                        graph_dir=graph_dir,
                        manifest_path_fn=lambda dataset_name, split_idx, variant=args.ontology_variant, graph_type=args.hierarchy_ontology_graph_type: concept_manifest_path(
                            dataset_name,
                            variant,
                            split_idx,
                            graph_type=graph_type,
                        ),
                    )
                manifest_template = str(
                    concept_manifest_path(
                        dataset,
                        args.ontology_variant,
                        split_idx="{split_idx}",  # type: ignore[arg-type]
                        graph_type=args.hierarchy_ontology_graph_type,
                    )
                )

            tag = sanitize_tag(args.experiment_tag)
            tag_part = f"_{tag}" if tag else ""
            config_path = CONFIG_ROOT / f"{dataset.lower()}_{method.replace('-', '_')}{tag_part}.yaml"
            if not args.dry_run:
                config_path = write_config(
                    dataset=dataset,
                    method=method,
                    variant=args.ontology_variant,
                    exp_dir=current_run_dir,
                    graph_manifest_template=manifest_template,
                    hierarchy_ontology_graph_type=args.hierarchy_ontology_graph_type,
                    hierarchy_graph_weight_max=args.hierarchy_graph_weight_max,
                    ontology_graph_weight_max=args.ontology_graph_weight_max,
                    graph_weight_target=args.graph_weight_target,
                    text_graph_gate_init=args.text_graph_gate_init,
                    ontology_evidence_only=args.ontology_evidence_only,
                    use_section_title_embedding=args.use_section_title_embedding,
                    fusion_mode=args.fusion_mode,
                    experiment_tag=args.experiment_tag,
                )
            tasks.append(
                Task(
                    dataset=dataset,
                    method=method,
                    run_name=current_run_dir.name,
                    run_dir=current_run_dir,
                    config_path=config_path,
                    split_indices=indices,
                    completed_before=completed_before,
                    experiment_tag=args.experiment_tag,
                    graph_manifest_template=manifest_template,
                )
            )

    plan = {
        "created_at": now(),
        "datasets": args.datasets,
        "methods": args.methods,
        "new_splits_per_task": args.num_splits,
        "split_offset": args.split_offset if args.split_offset is not None else "auto",
        "ontology_variant": args.ontology_variant,
        "hierarchy_ontology_graph_type": args.hierarchy_ontology_graph_type,
        "experiment_tag": sanitize_tag(args.experiment_tag),
        "fusion_mode": args.fusion_mode,
        "use_section_title_embedding": bool(args.use_section_title_embedding),
        "hierarchy_graph_weight_max": float(args.hierarchy_graph_weight_max),
        "ontology_graph_weight_max": float(args.ontology_graph_weight_max),
        "graph_weight_target": float(args.graph_weight_target),
        "tasks": serialize_tasks(tasks),
        "skipped": skipped,
    }

    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
        return 0

    write_overall_plan(REPO_ROOT / "experiment_records" / "reports" / "ordered_splits" / "plan.json", plan)

    for index, task in enumerate(tasks):
        print(f"[{now()}] START {task.dataset} {METHOD_LABELS[task.method]}", flush=True)
        run_task(task, args)
        task = rename_run_dir_to_completed_count(task)
        tasks[index] = task
        write_records_from_summary(task, args.ontology_variant)
        write_report(tasks, skipped, args.ontology_variant)
        plan["tasks"] = serialize_tasks(tasks)
        write_overall_plan(REPO_ROOT / "experiment_records" / "reports" / "ordered_splits" / "plan.json", plan)
        print(f"[{now()}] DONE  {task.dataset} {METHOD_LABELS[task.method]}", flush=True)

    write_report(tasks, skipped, args.ontology_variant)
    print(f"[{now()}] ordered split experiments complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
