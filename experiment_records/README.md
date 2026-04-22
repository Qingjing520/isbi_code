# Experiment Records

This directory stores stable, lightweight experiment evidence that is safe to
version on GitHub.  It is for curated summaries and cross-run comparisons, not
for raw training artifacts.

## Directory Roles

```text
experiment_records/
  comparisons/
    BRCA/
    KIRC/
    cross_dataset/
  reports/
    BRCA/
    cross_dataset/
    ontology/
    repository/
```

- `comparisons/` contains small CSV/JSON/README bundles for paper-facing or
  cross-method comparisons.
- `reports/` contains final Markdown writeups, interpretation notes, and layout
  documentation.
- Raw checkpoints, per-epoch logs, configs, and split-level run outputs belong
  under `experiments/<dataset>/<method>/runs/`.
- Method-local split indexes belong under
  `experiments/<dataset>/<method>/records/split_results.*`.

## Current Records

- BRCA comparisons: `comparisons/BRCA/`
- KIRC comparisons: `comparisons/KIRC/`
- Cross-dataset comparisons and benefit analysis:
  `comparisons/cross_dataset/`
- Ontology/concept-graph reports: `reports/ontology/`
- Paper-ready text-mode reports: `reports/cross_dataset/`
- Experiment tree layout notes: `reports/repository/experiment_tree_layout.md`
