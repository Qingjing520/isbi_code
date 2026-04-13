# Ontology / Concept-Graph Checkpoint (2026-04-13)

This checkpoint records the current NCIt-based pathology report pipeline status so the next session can resume directly without reconstructing the context.

## Current state

- Ontology choice in active pipeline: NCIt pathology subset
- Active ontology JSON:
  - `F:\Tasks\Ontologies\pocessed\ncit_pathology_subset_ontology.json`
- Pipeline config updated to use the pathology subset:
  - `F:\Tasks\isbi_code\pathology_report_extraction\config\pipeline.yaml`

## What was completed

### 1. NCIt filtering / subset tightening

- Added a pathology-focused NCIt subset builder in:
  - `F:\Tasks\isbi_code\pathology_report_extraction\build_project_ontology_resources.py`
- Added stronger concept filtering and faster phrase matching in:
  - `F:\Tasks\isbi_code\pathology_report_extraction\extract_ontology_concepts.py`
- Built the NCIt pathology subset with ancestor closure.

Current subset size:

- NCIt full project ontology: `211072` concepts
- NCIt pathology subset: `40379` concepts

### 2. KIRC full rerun

- Concept annotations rerun completed:
  - input: `F:\Tasks\isbi_code\pathology_report_extraction\Output\sentence_exports_masked\KIRC`
  - output: `F:\Tasks\isbi_code\pathology_report_extraction\Output\concept_annotations_masked\KIRC`
  - success: `538`
- Concept graph rebuild completed:
  - input embeddings: `F:\Tasks\isbi_code\pathology_report_extraction\Output\sentence_embeddings_conch_masked\KIRC`
  - output graphs: `F:\Tasks\isbi_code\pathology_report_extraction\Output\text_concept_graphs_masked\KIRC`
  - success: `538`
  - skipped: `0`

KIRC graph sanity check:

- graph files: `538`
- concept count min / max / avg: `7 / 114 / 64.77`
- zero-concept graphs: `0`

### 3. BRCA full rerun

- Concept annotations rerun completed:
  - input: `F:\Tasks\isbi_code\pathology_report_extraction\Output\sentence_exports_masked\BRCA`
  - output: `F:\Tasks\isbi_code\pathology_report_extraction\Output\concept_annotations_masked\BRCA`
  - success: `1105`
- Concept graph rebuild completed:
  - input embeddings: `F:\Tasks\isbi_code\pathology_report_extraction\Output\sentence_embeddings_conch_masked\BRCA`
  - output graphs: `F:\Tasks\isbi_code\pathology_report_extraction\Output\text_concept_graphs_masked\BRCA`
  - success: `1104`
  - skipped: `1`

BRCA graph sanity check:

- graph files: `1104`
- concept count min / max / avg: `5 / 165 / 71.55`
- zero-concept graphs: `0`

Skipped BRCA sample:

- case: `TCGA-A8-A07R.FD75D315-EED7-4350-A890-8FB94648E9FC`
- status: `skipped`
- reason: `empty_graph_after_cleanup`
- root cause:
  - the masked preprocessed report already became empty before graph building
  - `filter_audit.removed_sentence_count = 2`
  - `filter_audit.removed_sentence_reasons.stage_only = 2`
  - `filter_audit.emptied_section_titles = ["Document Body"]`

## Downstream manifest outputs

Concept-graph manifests were generated successfully and can be used directly for downstream dataset loading or auditing.

### KIRC concept-graph manifest

- CSV:
  - `F:\Tasks\isbi_code\pathology_report_extraction\Output\manifests\kirc_concept_graph_manifest_split0.csv`
- Summary:
  - `F:\Tasks\isbi_code\pathology_report_extraction\Output\manifests\kirc_concept_graph_manifest_split0_summary.json`
- Key stats:
  - matched rows: `499`
  - matched cases: `496`
  - image exists count: `499`
  - rows with concept level: `499`
  - total concept nodes across manifest rows: `32276`

### BRCA concept-graph manifest

- CSV:
  - `F:\Tasks\isbi_code\pathology_report_extraction\Output\manifests\brca_concept_graph_manifest_split149.csv`
- Summary:
  - `F:\Tasks\isbi_code\pathology_report_extraction\Output\manifests\brca_concept_graph_manifest_split149_summary.json`
- Key stats:
  - matched rows: `739`
  - matched cases: `739`
  - image exists count: `739`
  - rows with concept level: `739`
  - total concept nodes across manifest rows: `52979`

## Commands used for the successful rerun

### Build ontology resources

```powershell
F:\Anaconda\envs\pytorch\python.exe pathology_report_extraction\build_project_ontology_resources.py `
  --ontology_root F:\Tasks\Ontologies `
  --skip_do
```

### KIRC concept annotations

```powershell
F:\Anaconda\envs\pytorch\python.exe pathology_report_extraction\extract_ontology_concepts.py `
  --input_dir F:\Tasks\isbi_code\pathology_report_extraction\Output\sentence_exports_masked\KIRC `
  --output_dir F:\Tasks\isbi_code\pathology_report_extraction\Output\concept_annotations_masked\KIRC `
  --ontology_path F:\Tasks\Ontologies\pocessed\ncit_pathology_subset_ontology.json `
  --include_true_path
```

### KIRC concept graphs

```powershell
$env:CUDA_VISIBLE_DEVICES=''
F:\Anaconda\envs\pytorch\python.exe pathology_report_extraction\build_text_hierarchy_graphs.py `
  --input_dir F:\Tasks\isbi_code\pathology_report_extraction\Output\sentence_embeddings_conch_masked\KIRC `
  --output_dir F:\Tasks\isbi_code\pathology_report_extraction\Output\text_concept_graphs_masked\KIRC `
  --concept_dir F:\Tasks\isbi_code\pathology_report_extraction\Output\concept_annotations_masked\KIRC `
  --attach_concepts
```

### BRCA concept annotations

```powershell
F:\Anaconda\envs\pytorch\python.exe pathology_report_extraction\extract_ontology_concepts.py `
  --input_dir F:\Tasks\isbi_code\pathology_report_extraction\Output\sentence_exports_masked\BRCA `
  --output_dir F:\Tasks\isbi_code\pathology_report_extraction\Output\concept_annotations_masked\BRCA `
  --ontology_path F:\Tasks\Ontologies\pocessed\ncit_pathology_subset_ontology.json `
  --include_true_path
```

### BRCA concept graphs

```powershell
$env:CUDA_VISIBLE_DEVICES=''
F:\Anaconda\envs\pytorch\python.exe pathology_report_extraction\build_text_hierarchy_graphs.py `
  --input_dir F:\Tasks\isbi_code\pathology_report_extraction\Output\sentence_embeddings_conch_masked\BRCA `
  --output_dir F:\Tasks\isbi_code\pathology_report_extraction\Output\text_concept_graphs_masked\BRCA `
  --concept_dir F:\Tasks\isbi_code\pathology_report_extraction\Output\concept_annotations_masked\BRCA `
  --attach_concepts
```

## Recommended next step

Resume from one of these:

1. Inspect whether the single skipped BRCA sample should stay excluded or needs a preprocessing fix.
2. Spot-check a few KIRC and BRCA concept annotation JSON files for mapping quality after the new NCIt filters.
3. If the concept-graph outputs look good, connect the new graph directories into the downstream manifest / training run you want to use next.
