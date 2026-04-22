from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"

MODE_DIRS = {
    "sentence-only": "Sentence-level representation only.",
    "sentence-ontology": "Sentence-level representation with medical knowledge augmentation.",
    "sentence-hierarchical-graph": "Sentence-level representation with document-structure augmentation.",
    "sentence-hierarchical-graph-ontology": "Sentence-level representation with joint structure and ontology augmentation.",
}

DATASETS = ("BRCA", "KIRC", "LUSC")


@dataclass(frozen=True)
class MovePlan:
    source: Path
    target: Path
    reason: str


def _safe_relative(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def _assert_inside_experiments(path: Path) -> None:
    resolved_root = EXPERIMENTS_ROOT.resolve()
    resolved_path = path.resolve()
    if resolved_path != resolved_root and resolved_root not in resolved_path.parents:
        raise ValueError(f"Refusing to operate outside experiments/: {path}")


def _dataset_from_name(name: str) -> str | None:
    lower = name.lower()
    for dataset in DATASETS:
        if lower.startswith(dataset.lower() + "_") or lower == dataset.lower():
            return dataset
    if lower.startswith("lusc"):
        return "LUSC"
    return None


def _mode_from_run_name(name: str) -> str | None:
    lower = name.lower()
    if "sentence_mode" in lower or "sentence_pt" in lower:
        return "sentence-only"
    if "dual_text_concept_graph" in lower:
        return "sentence-hierarchical-graph-ontology"
    if "concept_graph" in lower or "ontology" in lower or "_ncit_" in lower:
        return "sentence-hierarchical-graph-ontology"
    if "dual_text" in lower or "hierarchy_graph" in lower or "readout_v2" in lower:
        return "sentence-hierarchical-graph"
    return None


def _mode_from_child_name(name: str) -> str | None:
    lower = name.lower()
    if lower == "sentence_pt" or "sentence" in lower:
        return "sentence-only"
    if "concept" in lower or "ontology" in lower:
        return "sentence-hierarchical-graph-ontology"
    if "hierarchy" in lower or "graph" in lower:
        return "sentence-hierarchical-graph"
    return None


def _target_run_dir(dataset: str, mode: str, name: str) -> Path:
    return EXPERIMENTS_ROOT / dataset / mode / "runs" / name


def _target_record_dir(dataset: str, mode: str, name: str) -> Path:
    return EXPERIMENTS_ROOT / dataset / mode / "records" / name


def _sanitize_record_name(name: str) -> str:
    sanitized = name.replace("comparison", "mixed")
    sanitized = sanitized.replace("__", "_")
    return sanitized.strip("_")


def _unique_target(path: Path) -> Path:
    if not path.exists():
        return path
    for idx in range(2, 1000):
        candidate = path.with_name(f"{path.name}__{idx}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find a free target path for {path}")


def _is_already_organized(path: Path) -> bool:
    parts = path.relative_to(EXPERIMENTS_ROOT).parts
    return len(parts) >= 2 and parts[0] in DATASETS and parts[1] in MODE_DIRS


def ensure_layout() -> None:
    for dataset in DATASETS:
        for mode, description in MODE_DIRS.items():
            base = EXPERIMENTS_ROOT / dataset / mode
            (base / "runs").mkdir(parents=True, exist_ok=True)
            (base / "records").mkdir(parents=True, exist_ok=True)
            readme = base / "README.md"
            if not readme.exists():
                readme.write_text(
                    f"# {dataset} / {mode}\n\n{description}\n",
                    encoding="utf-8",
                )


def build_plan() -> tuple[list[MovePlan], list[str]]:
    plans: list[MovePlan] = []
    notes: list[str] = []

    for source in sorted(EXPERIMENTS_ROOT.iterdir(), key=lambda p: p.name.lower()):
        if not source.is_dir():
            continue
        if source.name in DATASETS or _is_already_organized(source):
            continue

        dataset = _dataset_from_name(source.name)
        if dataset is None:
            if source.name == "dual_text_concept_graph_ablation_logs":
                target = EXPERIMENTS_ROOT / "KIRC" / "sentence-hierarchical-graph-ontology" / "records" / "shared_dual_text_concept_graph_ablation_logs"
                plans.append(MovePlan(source=source, target=_unique_target(target), reason="shared ontology ablation logs"))
            else:
                notes.append(f"Unclassified: {_safe_relative(source)}")
            continue

        if "text_mode_comparison" in source.name.lower():
            record_target = _target_record_dir(
                dataset,
                "sentence-only",
                _sanitize_record_name(source.name),
            )
            for child in sorted(source.iterdir(), key=lambda p: p.name.lower()):
                if child.is_dir():
                    child_mode = _mode_from_child_name(child.name)
                    if child_mode is None:
                        notes.append(f"Unclassified mixed child: {_safe_relative(child)}")
                        continue
                    target_name = f"{_sanitize_record_name(source.name)}__{child.name}"
                    plans.append(
                        MovePlan(
                            source=child,
                            target=_unique_target(_target_run_dir(dataset, child_mode, target_name)),
                            reason="split mixed text-mode run into method bucket",
                        )
                    )
                else:
                    target = _unique_target(record_target / child.name)
                    plans.append(MovePlan(source=child, target=target, reason="mixed text-mode record"))
            plans.append(
                MovePlan(
                    source=source,
                    target=_unique_target(_target_record_dir(dataset, "sentence-only", f"{_sanitize_record_name(source.name)}__empty_container")),
                    reason="remove emptied mixed container",
                )
            )
            continue

        mode = _mode_from_run_name(source.name)
        if mode is None:
            notes.append(f"Unclassified: {_safe_relative(source)}")
            continue
        plans.append(
            MovePlan(
                source=source,
                target=_unique_target(_target_run_dir(dataset, mode, source.name)),
                reason=f"classified as {dataset}/{mode}",
            )
        )

    return plans, notes


def apply_plan(plans: list[MovePlan]) -> None:
    for plan in plans:
        _assert_inside_experiments(plan.source)
        _assert_inside_experiments(plan.target)
        if not plan.source.exists():
            continue
        plan.target.parent.mkdir(parents=True, exist_ok=True)
        if plan.source.is_dir() and not any(plan.source.iterdir()) and "empty_container" in plan.target.name:
            plan.source.rmdir()
            continue
        shutil.move(str(plan.source), str(plan.target))


def write_manifest(plans: list[MovePlan], notes: list[str]) -> None:
    manifest = {
        "layout": {
            dataset: {mode: str(EXPERIMENTS_ROOT / dataset / mode) for mode in MODE_DIRS}
            for dataset in DATASETS
        },
        "moves": [
            {
                "source": _safe_relative(plan.source),
                "target": _safe_relative(plan.target),
                "reason": plan.reason,
            }
            for plan in plans
        ],
        "notes": notes,
    }
    path = EXPERIMENTS_ROOT / "experiment_tree_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Organize experiments into dataset/method buckets.")
    parser.add_argument("--apply", action="store_true", help="Move files/directories. Without this, print a dry run.")
    args = parser.parse_args()

    ensure_layout()
    plans, notes = build_plan()

    print("Experiment organization plan:")
    for plan in plans:
        print(f"  MOVE {_safe_relative(plan.source)} -> {_safe_relative(plan.target)}  [{plan.reason}]")
    for note in notes:
        print(f"  NOTE {note}")

    if args.apply:
        apply_plan(plans)
        write_manifest(plans, notes)
        print("Applied experiment organization.")
    else:
        print("Dry run only. Re-run with --apply to move files.")


if __name__ == "__main__":
    main()
