# Project Map

This is the current high-level map of `F:\Tasks\isbi_code`.

Project-wide naming, organization, and temporary-script cleanup rules are in
`docs/project_conventions.md`. Codex/agent instructions are mirrored in
the repository root `AGENTS.md`.

## Stable Training Surface

- `main.py`
  Single-config train/test entry point.

- `run_main_splits.py`
  Multi-split runner used by the current manifest-driven experiments.

- `train.py`
  Main training implementation. It is still the central integration point for
  WSI, sentence text, hierarchy graph text, and concept graph text.

- `configs/config.py`
  Dataclass-based YAML loader for training config.

- `configs/config.yaml`
  Main hand-edited training config.

## Core Training Packages

- `datasets/`
  Feature loading for WSI and text branches.

- `models/`
  Mapper, pooling, fusion, hierarchy text branch, text graph encoder, classifier.

- `losses/`
  MMD and related losses.

- `utils/`
  Graph feature construction, metrics, seed helpers.

## Pathology Report Pipeline

- `pathology_report_extraction/run_pipeline.py`
  Backward-compatible one-click entry point. The implementation lives in
  `pathology_report_extraction/pipeline/run_pipeline.py`.

- `pathology_report_extraction/pipeline/`
  Report preprocessing, section/sentence export, CONCH sentence embedding, and
  pipeline orchestration.

- `pathology_report_extraction/ontology/`
  Current NCIt+DO ontology bundle generation, ontology concept extraction,
  concept audits, and legacy explicit SNOMED/UMLS ablation support.

- `pathology_report_extraction/graphs/`
  Document/Section/Sentence hierarchy graphs, sentence-ontology auxiliary
  graphs, and training manifest generation.

- `pathology_report_extraction/labels/`
  Clinical XML label extraction and split-aligned label expansion.

- `pathology_report_extraction/visualization/`
  Unified hierarchy-graph visualization entry point for single-sample and
  two-dataset comparison figures.

- `pathology_report_extraction/common/`
  Shared default paths, output subdirectory names, ontology/CONCH roots, PDF,
  OCR, JSON, and text-cleaning helpers.

Root-level pathology scripts are thin compatibility wrappers so old commands
and running experiment orchestrators keep working while new code stays grouped
by purpose.

## Experiment Tools

- `tools/run_dual_text_concept_ablation.py`
  End-to-end dual-text concept-graph ontology ablation runner.

- `tools/watch_dual_text_concept_ablation.py`
  Live status view for that ablation.

- `docs/runbooks/dual_text_concept_ablation.md`
  Runbook.

- `tools/data/make_splits.py`
  General split generator. It supports single feature directories and
  BRCA/KIRC/LUSC standard roots, with optional label-stratified splits.

- `tools/build_ordered_method_comparison.py`
  Builds the paper-facing ordered method comparison table from split records
  and current run folders.

## Historical Scripts

These were moved out of the repository root to keep the main training surface
small.

- `tools/analysis/analyze_dual_text_benefit.py`
- `tools/analysis/analyze_dual_text_samples.py`
- `tools/archive/auto_train_next.py`
- `tools/experiments/compare_text_modes.py`
- `tools/experiments/run_dual_text_splits.py`

## Output And Records

- `experiments/`
  Ignored runtime outputs, checkpoints, logs. Organized by dataset and method:
  `BRCA|KIRC|LUSC / sentence-only|sentence-ontology|sentence-hierarchical-graph|sentence-hierarchical-graph-ontology / runs|records`.

- `pathology_report_extraction/Output/`
  Ignored preprocessing outputs and graph artifacts.

- `F:\Tasks\Pathology_Report_Hierarchy_Graphs`
  Shared, training-ready pathology report hierarchy graph inputs, organized as
  `BRCA|KIRC|LUSC / basic_hierarchy|ontology_concept_hierarchy|stage_keyword_word_hierarchy|stage_keyword_word_ontology_hierarchy`.

- `experiment_records/`
  Stable summaries and paper-facing experiment records.

## Known Cleanup Items

- Some archived historical scripts still contain old hard-coded defaults; prefer
  the current `run_main_splits.py`, ordered runners, and `tools/data/make_splits.py`
  for new work.
- Remove one-time temporary scripts and scratch outputs after use. Promote only
  reusable helpers into stable, documented tools.
