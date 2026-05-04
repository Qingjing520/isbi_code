# Codex Project Instructions

This file is the project-level memory for Codex/agent work in
`F:\Tasks\isbi_code`.

## Core Rules

- Keep naming unified, intuitive, and simple.
- Keep files and scripts grouped by purpose.
- Keep the repository root clean.
- Remove one-time temporary scripts and scratch outputs after use.
- Promote reusable helpers into stable, generally named tools instead of leaving
  task-specific scripts behind.

## Experiment Naming

- Use `DATASET_method_Nsplits` for run folders.
- Use uppercase dataset names: `BRCA`, `KIRC`, `LUSC`.
- Use canonical method names:
  `sentence-only`, `sentence-ontology`, `sentence-hierarchical-graph`,
  `sentence-hierarchical-graph-ontology`.
- Prefer clear tags only when they describe a reusable setting, for example
  `safe_aux_label_emb`.

## Directory Organization

- Stable root entry points may stay at the repository root.
- Experiment runners should live under `tools/experiments/` unless they are
  established root-level entry points.
- Analysis scripts should live under `tools/analysis/`.
- Data and feature preprocessing scripts should live under `tools/preprocessing/`
  or the relevant `pathology_report_extraction/` package.
- Stable documentation belongs under `docs/`.

## Temporary Script Policy

- Create one-off scripts only when necessary.
- Prefer temporary folders or scratch locations for throwaway scripts.
- Delete one-off scripts, smoke-test folders, and scratch outputs after the task
  is complete.
- If a script is useful beyond one task, rename it generically, move it into the
  correct tools/package directory, and mention it in project documentation.

## Current Project Context

- Python environment: `F:\Anaconda\envs\pytorch\python.exe`.
- Ontology root: `F:\Tasks\Ontologies\processed`.
- Canonical hierarchy graph input root:
  `F:\Tasks\Pathology_Report_Hierarchy_Graphs`.
- Default ontology mainline for current experiments is compact `NCIt + DO`.
