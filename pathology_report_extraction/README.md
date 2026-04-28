# Pathology Report Extraction

This directory contains the pathology-report preprocessing, ontology concept
extraction, graph building, and manifest preparation pipeline used by the
BRCA/KIRC/LUSC experiments.

## Main Pipeline

The usual processing order is:

1. Preprocess pathology reports into `Document -> Section -> Sentence`.
2. Export sentence views.
3. Extract ontology concepts with NCIt/SNOMED/UMLS/DO resources.
4. Encode sentences with CONCH.
5. Build text hierarchy graphs and optional concept-enhanced graphs.
6. Build text graph manifests for downstream training.

Core entry points:

- `run_pipeline.py`: one-click pipeline from the shared YAML config.
- `preprocess_pathology_reports.py`: report text extraction and cleaning.
- `export_sentence_views.py`: section/sentence export before encoding.
- `extract_ontology_concepts.py`: ontology mention matching and true-path expansion.
- `build_sentence_ontology_graphs.py`: lightweight sentence/concept auxiliary graphs.
- `build_text_hierarchy_graphs.py`: `Document -> Section -> Sentence` graph construction.
- `prepare_text_graph_manifest.py`: slide-level manifest generation for training.
- `prepare_stage_labels.py`: clinical XML label extraction and split-label expansion.
- `visualize_hierarchy_graphs.py`: single-graph or two-dataset hierarchy visualization.
- `pipeline_defaults.py`: shared paths, filter modes, and output subdirectory names.
- `pdf_utils.py`: shared filesystem, PDF, and JSON helpers.

Implementation layout:

- `pipeline/`: PDF preprocessing, sentence export, CONCH encoding, and full pipeline orchestration.
- `ontology/`: NCIt/DO/SNOMED/UMLS resource building, concept extraction, ablation bundles, and audits.
- `graphs/`: hierarchy graph, sentence-ontology graph, and training manifest builders.
- `labels/`: stage-label extraction and split-label helpers.
- `visualization/`: hierarchy graph plotting utilities.
- `common/`: shared path defaults, PDF/OCR/text-cleaning helpers.

The root-level `.py` files are compatibility entry points for existing commands
and long-running experiment scripts. New implementation work should go into the
purpose-specific subpackages above.

## Current Paths

- Python: `F:\Anaconda\envs\pytorch\python.exe`
- Report root: `F:\Tasks\Pathology Report`
- Project root: `F:\Tasks\isbi_code`
- Ontology root: `F:\Tasks\Ontologies\processed`
- Output root: `F:\Tasks\isbi_code\pathology_report_extraction\Output`
- Shared config: `F:\Tasks\isbi_code\pathology_report_extraction\config\pipeline.yaml`

## Common Commands

Full pipeline:

```powershell
F:\Anaconda\envs\pytorch\python.exe F:\Tasks\isbi_code\pathology_report_extraction\run_pipeline.py --config F:\Tasks\isbi_code\pathology_report_extraction\config\pipeline.yaml
```

Debug a small subset:

```powershell
F:\Anaconda\envs\pytorch\python.exe F:\Tasks\isbi_code\pathology_report_extraction\run_pipeline.py --config F:\Tasks\isbi_code\pathology_report_extraction\config\pipeline.yaml --limit 10
```

Run one stage:

```powershell
F:\Anaconda\envs\pytorch\python.exe F:\Tasks\isbi_code\pathology_report_extraction\extract_ontology_concepts.py --config F:\Tasks\isbi_code\pathology_report_extraction\config\pipeline.yaml
```

Build training manifests after graphs are ready:

```powershell
F:\Anaconda\envs\pytorch\python.exe F:\Tasks\isbi_code\pathology_report_extraction\prepare_text_graph_manifest.py --config F:\Tasks\isbi_code\pathology_report_extraction\config\pipeline.yaml
```

Visualize one hierarchy graph:

```powershell
F:\Anaconda\envs\pytorch\python.exe F:\Tasks\isbi_code\pathology_report_extraction\visualize_hierarchy_graphs.py single
```

Visualize a BRCA/KIRC comparison:

```powershell
F:\Anaconda\envs\pytorch\python.exe F:\Tasks\isbi_code\pathology_report_extraction\visualize_hierarchy_graphs.py compare
```

## Output Layout

The default `masked` run writes under `Output`:

```text
pathology_report_preprocessed_masked/
sentence_exports_masked/
concept_annotations_masked/
sentence_embeddings_conch_masked/
text_hierarchy_graphs_masked/
manifests/
```

Ontology ablation outputs use `concept_annotations_ablation/` and graph folders
keyed by ontology variant, such as `ncit_only`, `ncit_do`,
`ncit_snomed_mapped`, and `full_multi_ontology`.

## Notes

- Keep generated data under `Output/`; do not commit it.
- Use `config/pipeline.yaml` as the source of truth for stage parameters.
- For new experiments, prefer the ordered experiment tools under
  `F:\Tasks\isbi_code\tools`.
