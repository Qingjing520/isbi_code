# Tools

This directory contains project utilities that are not stable training entry
points.

## Layout

- `analysis/`: post-run analysis scripts.
- `data/`: data and split-table helpers.
- `experiments/`: historical or specialized experiment launchers.
- `archive/`: preserved old helpers that should not be used for new runs.

## Current Runners

- `run_ordered_5split_experiments.py`: ordered dataset/method runner.
- `watch_ordered_5split_experiments.py`: live watcher for ordered runs.
- `run_dual_text_concept_ablation.py`: ontology ablation runner.
- `watch_dual_text_concept_ablation.py`: ontology ablation watcher.
- `build_experiment_split_records.py`: split-level record builder.
- `build_ordered_method_comparison.py`: paper-facing comparison table builder.
