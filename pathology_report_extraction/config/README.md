# Pipeline Config

`pipeline.yaml` is the shared configuration file for the pathology report
pipeline. Stage-specific CLI arguments still override values from YAML.

## Stage Order

1. `defaults`
2. `preprocess`
3. `export_sentence_views`
4. `extract_ontology_concepts`
5. `encode_sentence_exports_conch`
6. `build_text_hierarchy_graphs`
7. `prepare_text_graph_manifest`

## Common Command

```powershell
F:\Anaconda\envs\pytorch\python.exe F:\Tasks\isbi_code\pathology_report_extraction\run_pipeline.py --config F:\Tasks\isbi_code\pathology_report_extraction\config\pipeline.yaml
```

Run a single stage with the same config:

```powershell
F:\Anaconda\envs\pytorch\python.exe F:\Tasks\isbi_code\pathology_report_extraction\<stage_script>.py --config F:\Tasks\isbi_code\pathology_report_extraction\config\pipeline.yaml
```
