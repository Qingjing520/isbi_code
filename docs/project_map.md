# Project Map

This is the current high-level map of `F:\Tasks\isbi_code`.

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

- `pathology_report_extraction/preprocess_pathology_reports.py`
  Report text preprocessing.

- `pathology_report_extraction/export_sentence_views.py`
  Section/sentence export.

- `pathology_report_extraction/encode_sentence_exports_conch.py`
  Sentence embedding export.

- `pathology_report_extraction/extract_ontology_concepts.py`
  Ontology concept matching and true-path expansion.

- `pathology_report_extraction/build_text_hierarchy_graphs.py`
  Document/Section/Sentence/Concept graph construction.

- `pathology_report_extraction/prepare_text_graph_manifest.py`
  Slide-level manifest generation for training.

- `pathology_report_extraction/build_project_ontology_resources.py`
  NCIt/DO/SNOMED/UMLS resource builder.

- `pathology_report_extraction/build_ontology_ablation_bundles.py`
  Four-way ontology ablation bundle builder.

## Experiment Tools

- `tools/run_dual_text_concept_ablation.py`
  End-to-end dual-text concept-graph ontology ablation runner.

- `tools/watch_dual_text_concept_ablation.py`
  Live status view for that ablation.

- `tools/dual_text_concept_ablation.md`
  Runbook.

- `tools/make_splits.py`
  Split helper.

## Historical Scripts

These are useful but should eventually move under `tools/analysis/`,
`tools/experiments/`, or `tools/archive/`.

- `analyze_dual_text_benefit.py`
- `analyze_dual_text_samples.py`
- `auto_train_next.py`
- `compare_brca_text_modes.py`
- `generate_split_tables.py`
- `run_dual_text_splits.py`

## Output And Records

- `experiments/`
  Ignored runtime outputs, checkpoints, logs.

- `pathology_report_extraction/Output/`
  Ignored preprocessing outputs and graph artifacts.

- `experiment_records/`
  Stable summaries and paper-facing experiment records.

## Known Cleanup Items

- Root `README.md` has mojibake and should be replaced.
- Some pathology scripts contain UTF-8 BOM; harmless to Python execution, but annoying for static parsing.
- `test.py` is an untracked scratch file and should be deleted only if confirmed.
