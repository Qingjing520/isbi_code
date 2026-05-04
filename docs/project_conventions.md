# Project Conventions

These conventions keep the project easy to resume, audit, and explain.

## Naming

- Use names that are unified, direct, and simple.
- Experiment run folders should make the dataset, method, and split count obvious:
  `DATASET_method_Nsplits`, for example `BRCA_sentence-only_150splits`.
- Use the canonical method names:
  `sentence-only`, `sentence-ontology`, `sentence-hierarchical-graph`,
  `sentence-hierarchical-graph-ontology`.
- Avoid one-off names such as `run_150splits`, vague tags, or mixed word order.

## Organization

- Keep files of the same type together.
- Training and experiment runners belong under `tools/experiments/` when they are
  not stable root entry points.
- Analysis scripts belong under `tools/analysis/`.
- Data preparation and feature extraction helpers belong under `tools/preprocessing/`
  or the relevant `pathology_report_extraction/` module.
- The repository root should stay small and contain only stable entry points,
  documentation, configs, and core packages.

## Temporary Files

- One-time scripts should be created in a temporary location or a clearly named
  scratch area, then deleted after the task is finished.
- Do not leave task-specific helper scripts in the root or `tools/` after they
  have served their purpose.
- If a temporary script proves reusable, rename it to a general-purpose name,
  move it into the appropriate tools subdirectory, and document its purpose.
- Temporary logs, smoke-test outputs, and scratch graph folders should be removed
  after verification unless they are needed as stable experiment records.

## Experiments

- Keep experiment outputs under:
  `experiments/DATASET/METHOD/runs` and `experiments/DATASET/METHOD/records`.
- New runs of the same dataset and method should continue from the next unfinished
  split in the same run family and update the completed split count.
- Stable summaries for comparison or reporting should be written to
  `experiment_records/reports/...`.
- Do not compare against ad-hoc folders when a canonical run folder exists.

## Data Inputs

- External reusable inputs should live under clearly named `F:\Tasks` directories,
  for example `F:\Tasks\Text_Sentence_extract_features` and
  `F:\Tasks\Pathology_Report_Hierarchy_Graphs`.
- Runtime output under `pathology_report_extraction/Output` should not be treated
  as the canonical long-term input location unless explicitly documented.

## Agent Workflow

- Before adding a file, check whether an existing module already owns that
  responsibility.
- Prefer extending general-purpose scripts over creating narrow one-off scripts.
- If a one-off script is unavoidable, remove it before finishing the task or
  promote it into a documented reusable tool.
- Keep project cleanup in mind during every code change, not only during explicit
  cleanup tasks.
