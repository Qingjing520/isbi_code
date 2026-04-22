# Repository Cleanup Plan

This repository currently mixes three kinds of code:

- Core multimodal training code.
- Pathology-report preprocessing and ontology/concept-graph pipeline code.
- Experiment orchestration, analysis scripts, records, and runtime outputs.

The immediate cleanup goal is to make those boundaries explicit without breaking
ongoing experiments.

## Current Top-Level Roles

| Path | Current Role | Cleanup Status |
|---|---|---|
| `train.py` | Main training loop, model construction, eval, analysis hooks | Too large; should be split after current long run finishes |
| `main.py` | Single-split train/test CLI | Keep as stable public entry |
| `run_main_splits.py` | Multi-split training CLI | Keep as stable public entry |
| `run_dual_text_splits.py` | Older dual-text split runner | Candidate for archive after checking old experiment reproducibility |
| `analyze_*.py` | Experiment analysis scripts | Move to `tools/analysis/` later |
| `compare_brca_text_modes.py` | Historical comparison runner | Candidate for `tools/experiments/` |
| `auto_train_next.py` | Historical automation helper | Candidate for archive |
| `generate_split_tables.py` | Dataset split generation utility | Move to `tools/data/` later |
| `tmp_paper_extract.txt` | Temporary artifact | Candidate for removal/archive |
| `test.py` | Untracked Hello World scratch file | Ask before deleting |

## Experiment Output Layout

Runtime experiment outputs under `experiments/` should use this structure:

```text
experiments/
  BRCA/
    sentence-only/
    sentence-ontology/
    sentence-hierarchical-graph/
    sentence-hierarchical-graph-ontology/
  KIRC/
    sentence-only/
    sentence-ontology/
    sentence-hierarchical-graph/
    sentence-hierarchical-graph-ontology/
  LUSC/
    sentence-only/
    sentence-ontology/
    sentence-hierarchical-graph/
    sentence-hierarchical-graph-ontology/
```

Each method directory contains `runs/` for split outputs/checkpoints/configs/logs
and `records/` for generated split-level indexes only. The expected files in
`records/` are `split_results.csv`, `split_results.json`, and
`split_results.md`; historical comparison folders should be preserved under
`runs/_legacy_records/`. Do not create new top-level comparison folders under
`experiments/`; write cross-method summaries to `experiment_records/` instead.

## Target Layout

The clean target structure should look like this:

```text
configs/
datasets/
losses/
models/
utils/
pathology_report_extraction/
tools/
  analysis/
  data/
  experiments/
  ontology/
docs/
experiment_records/
experiments/                     # ignored runtime output
```

## Training Code Split

`train.py` should eventually be split into smaller modules:

| Future Module | Responsibility |
|---|---|
| `training/engine.py` | `train_one_split`, epoch loop, checkpointing |
| `training/eval.py` | `load_and_eval`, sample-level analysis |
| `training/text_encoding.py` | `encode_text_batch`, graph/text branch construction |
| `training/losses.py` | alignment/gate/concept MMD assembly |
| `training/factory.py` | mapper, pooling, graph encoder, fusion module construction |

Do this only after the current background ablation run finishes, because active
subprocesses import `train.py` directly.

## Pathology Pipeline Split

The `pathology_report_extraction/` package is coherent conceptually, but several
files are too large:

| Current File | Suggested Split |
|---|---|
| `build_project_ontology_resources.py` | ontology readers, crosswalk builders, bundle builders |
| `extract_ontology_concepts.py` | matching catalog, mention extraction, true-path expansion, export CLI |
| `build_text_hierarchy_graphs.py` | graph schema, graph construction, graph export CLI |
| `preprocess_pathology_reports.py` | PDF/text IO, cleaning orchestration, report filtering |

Suggested package folders:

```text
pathology_report_extraction/
  ontology/
  concepts/
  graphs/
  preprocessing/
  manifests/
```

## Safe Cleanup Now

These are safe while training is running:

- Add documentation and runbooks.
- Update `.gitignore` for generated configs and dynamic summaries.
- Add new helper scripts.
- Refactor code by adding helpers while preserving public function names.

## Defer Until Current Training Finishes

Do not do these while the background run is active:

- Move or rename `train.py`, `run_main_splits.py`, or imported model/dataset files.
- Change CLI argument names used by the running orchestration.
- Delete historical experiment scripts before checking if old records reference them.
- Move `pathology_report_extraction` scripts that current logs or commands may call.

## First Concrete Refactor Batch

Recommended first commit:

- `train.py`: keep public functions but extract text-module construction helpers.
- `tools/run_dual_text_concept_ablation.py`: central orchestration for ontology ablation.
- `tools/watch_dual_text_concept_ablation.py`: live watcher.
- `pathology_report_extraction/build_ontology_ablation_bundles.py`: ontology ablation resource builder.
- `tools/dual_text_concept_ablation.md`: runbook.
- `.gitignore`: ignore generated configs and dynamic ablation summaries.

Recommended second commit after the run finishes:

- Move old top-level analysis scripts into `tools/analysis/`.
- Move split/data utilities into `tools/data/`.
- Archive obsolete runners under `tools/archive/`.
- Replace the mojibake `README.md` with a short clean project overview.

## Notes

- Keep `experiment_records/` for stable, manually curated result summaries.
- Keep `experiments/` ignored; it is runtime output.
- Keep generated YAML under `configs/generated/` ignored; the orchestration script recreates it.
