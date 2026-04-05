# BRCA Text Mode Comparison (5 Splits, Node Features)

This record summarizes a BRCA comparison between:

- `sentence_pt`: the original sentence-level text feature mode
- `hierarchy_graph + node_features`: the hierarchical text mode using all document, section, and sentence nodes

## Setup

- Dataset: `BRCA`
- Splits: `D:\Tasks\Split_Table\BRCA\split_0.csv` to `split_4.csv`
- Label file: `D:\Tasks\Pathologic_Stage_Label\BRCA_pathologic_stage.csv`
- Image features: `D:\Tasks\WSI_extract_features\BRCA_WSI_extract_features`
- Sentence text features: `D:\Tasks\Text_Sentence_extract_features\BRCA_text`
- Hierarchy graph features: `D:\Tasks\isbi_code\pathology_report_extraction\Output\text_hierarchy_graphs_masked\BRCA`
- Hierarchy feature used for training: `node_features`

## Result Summary

`sentence_pt` remained slightly better than `hierarchy_graph + node_features` on BRCA when AUC is treated as the primary metric.

- `sentence_pt`
  - `ACC = 0.7385 ± 0.0380`
  - `AUC = 0.7444 ± 0.0212`
- `hierarchy_graph + node_features`
  - `ACC = 0.7541 ± 0.0175`
  - `AUC = 0.7296 ± 0.0498`

## Interpretation

- `node_features` is much stronger than the previous `section_features` hierarchy setting on BRCA.
- However, the current model still handles the original sentence-level text mode more reliably on AUC.
- The present hierarchical graph pathway is better viewed as a structured feature input rather than a graph-aware model.

## Files

- `comparison_results.csv`: per-split results
- `comparison_summary.json`: aggregated mean and standard deviation
