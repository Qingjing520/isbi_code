# Pathology Report Extraction

This folder contains the current pathology-report text pipeline for TCGA BRCA and KIRC PDFs.

The active processing order is:

1. PDF preprocessing into `Document -> Section -> Sentence`
2. Sentence-view export
3. CONCH sentence encoding
4. `Document -> Section -> Sentence` hierarchy graph building
5. Optional text-graph manifest preparation for training

Core scripts:

- [run_pipeline.py](D:\Tasks\isbi_code\pathology_report_extraction\run_pipeline.py)
- [preprocess_pathology_reports.py](D:\Tasks\isbi_code\pathology_report_extraction\preprocess_pathology_reports.py)
- [export_sentence_views.py](D:\Tasks\isbi_code\pathology_report_extraction\export_sentence_views.py)
- [encode_sentence_exports_conch.py](D:\Tasks\isbi_code\pathology_report_extraction\encode_sentence_exports_conch.py)
- [build_text_hierarchy_graphs.py](D:\Tasks\isbi_code\pathology_report_extraction\build_text_hierarchy_graphs.py)
- [prepare_text_graph_manifest.py](D:\Tasks\isbi_code\pathology_report_extraction\prepare_text_graph_manifest.py)

Config files:

- [config](D:\Tasks\isbi_code\pathology_report_extraction\config)
- [pipeline.yaml](D:\Tasks\isbi_code\pathology_report_extraction\config\pipeline.yaml)
- [config\README.md](D:\Tasks\isbi_code\pathology_report_extraction\config\README.md)

## Dependencies

```powershell
D:\ProgrammeFiles\Anaconda\envs\Pytorch\python.exe -m pip install -r D:\Tasks\isbi_code\pathology_report_extraction\requirements.txt
```

Main packages:

- `PyMuPDF`
- `rapidocr_onnxruntime`
- `opencv-python`
- `Pillow`
- `numpy`
- `torch`
- `timm`
- `transformers`
- `PyYAML`

No external Tesseract installation is required.

## Default Paths

- Input PDF root:
  - `D:\Tasks\Pathology Report`
- Output root:
  - `D:\Tasks\isbi_code\pathology_report_extraction\Output`
- Master config:
  - `D:\Tasks\isbi_code\pathology_report_extraction\config\pipeline.yaml`

## Recommended One-Click Run

This is the recommended entry point for the full pipeline:

```powershell
D:\ProgrammeFiles\Anaconda\envs\Pytorch\python.exe D:\Tasks\isbi_code\pathology_report_extraction\run_pipeline.py --config "D:\Tasks\isbi_code\pathology_report_extraction\config\pipeline.yaml"
```

Quick debug run:

```powershell
D:\ProgrammeFiles\Anaconda\envs\Pytorch\python.exe D:\Tasks\isbi_code\pathology_report_extraction\run_pipeline.py --config "D:\Tasks\isbi_code\pathology_report_extraction\config\pipeline.yaml" --limit 10
```

Temporary mode override without editing YAML:

```powershell
D:\ProgrammeFiles\Anaconda\envs\Pytorch\python.exe D:\Tasks\isbi_code\pathology_report_extraction\run_pipeline.py --config "D:\Tasks\isbi_code\pathology_report_extraction\config\pipeline.yaml" --filter_mode full
```

Optional manifest preparation after graphs are ready:

```powershell
D:\ProgrammeFiles\Anaconda\envs\Pytorch\python.exe D:\Tasks\isbi_code\pathology_report_extraction\prepare_text_graph_manifest.py --config "D:\Tasks\isbi_code\pathology_report_extraction\config\pipeline.yaml"
```

## Single-YAML Design

All stage settings now live in one file:

- [pipeline.yaml](D:\Tasks\isbi_code\pathology_report_extraction\config\pipeline.yaml)

The YAML is ordered by processing sequence:

1. `defaults`
2. `preprocess`
3. `export_sentence_views`
4. `encode_sentence_exports_conch`
5. `build_text_hierarchy_graphs`
6. `prepare_text_graph_manifest`

Each stage has:

- `enabled`: whether the stage runs in the one-click pipeline
- stage-specific parameters
- `output_subdirs`: output folder names that switch automatically with `filter_mode`

Important mode switch:

- `preprocess.filter_mode`

Choices:

- `masked`
- `no_diagnosis`
- `no_diagnosis_masked`
- `full`

`masked` is the recommended default for the pathology stage-classification main experiment.

## Running Individual Stages

Each stage script still supports `--config`, and it can read the shared [pipeline.yaml](D:\Tasks\isbi_code\pathology_report_extraction\config\pipeline.yaml).

Preprocess only:

```powershell
D:\ProgrammeFiles\Anaconda\envs\Pytorch\python.exe D:\Tasks\isbi_code\pathology_report_extraction\preprocess_pathology_reports.py --config "D:\Tasks\isbi_code\pathology_report_extraction\config\pipeline.yaml"
```

Sentence export only:

```powershell
D:\ProgrammeFiles\Anaconda\envs\Pytorch\python.exe D:\Tasks\isbi_code\pathology_report_extraction\export_sentence_views.py --config "D:\Tasks\isbi_code\pathology_report_extraction\config\pipeline.yaml"
```

CONCH encoding only:

```powershell
D:\ProgrammeFiles\Anaconda\envs\Pytorch\python.exe D:\Tasks\isbi_code\pathology_report_extraction\encode_sentence_exports_conch.py --config "D:\Tasks\isbi_code\pathology_report_extraction\config\pipeline.yaml"
```

Graph build only:

```powershell
D:\ProgrammeFiles\Anaconda\envs\Pytorch\python.exe D:\Tasks\isbi_code\pathology_report_extraction\build_text_hierarchy_graphs.py --config "D:\Tasks\isbi_code\pathology_report_extraction\config\pipeline.yaml"
```

Manifest only:

```powershell
D:\ProgrammeFiles\Anaconda\envs\Pytorch\python.exe D:\Tasks\isbi_code\pathology_report_extraction\prepare_text_graph_manifest.py --config "D:\Tasks\isbi_code\pathology_report_extraction\config\pipeline.yaml"
```

## Output Layout

For the current default `masked` mode, the pipeline writes:

```text
D:\Tasks\isbi_code\pathology_report_extraction\Output
|-- pathology_report_preprocessed_masked
|   |-- run_summary.json
|   |-- preprocess.log
|   |-- BRCA\*.json
|   |-- KIRC\*.json
|-- sentence_exports_masked
|   |-- run_summary.json
|   |-- export.log
|   |-- BRCA\*.json / *.txt
|   |-- KIRC\*.json / *.txt
|-- sentence_embeddings_conch_masked
|   |-- run_summary.json
|   |-- encode.log
|   |-- BRCA\*.pt / *.json
|   |-- KIRC\*.pt / *.json
|-- text_hierarchy_graphs_masked
|   |-- run_summary.json
|   |-- graph.log
|   |-- BRCA\*.pt / *.json
|   |-- KIRC\*.pt / *.json
|-- manifests
|   |-- text_graph_manifest_masked.csv
|   |-- text_graph_manifest_masked_summary.json
|-- pipeline_run_summary_masked.json
```

If you switch `filter_mode`, the stage output subfolders switch automatically.

## File Mapping

For one report sample:

- `Output\pathology_report_preprocessed_<mode>\<report>.json`
  - structured text result
  - keeps `Document -> Section -> Sentence`
- `Output\sentence_exports_<mode>\<report>.json`
  - flattened sentence view before encoding
  - keeps `sentences`, `sentence_to_section`, and section spans
- `Output\sentence_embeddings_conch_<mode>\<report>.pt`
  - actual CONCH sentence embeddings for that report
  - one `.pt` corresponds to one report, not one sentence
  - shape is `(num_sentences, 512)`
- `Output\sentence_embeddings_conch_<mode>\<report>.json`
  - metadata for the same `.pt`
  - maps embedding rows back to sentences and sections
- `Output\text_hierarchy_graphs_<mode>\<report>.pt`
  - graph tensors for training
- `Output\text_hierarchy_graphs_<mode>\<report>.json`
  - readable node and edge metadata

Example:

- if a report has `65` sentences, its CONCH tensor shape is `(65, 512)`
- row `37` in the tensor is sentence `37`
- the companion metadata JSON tells you which section sentence `37` belongs to

## Training Prep

The next-stage helper script is:

- [prepare_text_graph_manifest.py](D:\Tasks\isbi_code\pathology_report_extraction\prepare_text_graph_manifest.py)

It matches case-level text graphs to slide-level labels and writes a manifest CSV for training.

The corresponding PyTorch dataset wrapper is:

- [text_graph_dataset.py](D:\Tasks\isbi_code\datasets\text_graph_dataset.py)

This dataset reads one row per sample from the manifest and loads:

- graph tensor dict from `graph_pt`
- classification label from `label`
- metadata such as `case_id`, `slide_id`, and `dataset`

## Workflow

See [workflow_overview.md](D:\Tasks\isbi_code\pathology_report_extraction\workflow_overview.md) for the PDF-to-graph overview.
