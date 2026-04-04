# BRCA Text Mode Comparison (5 Splits)

This record summarizes a BRCA comparison between:

- `sentence_pt`: the original sentence-level text feature mode
- `hierarchy_graph + section_features`: the current hierarchical text mode using section pooled features

## Setup

- Dataset: `BRCA`
- Splits: `D:\Tasks\Split_Table\BRCA\split_0.csv` to `split_4.csv`
- Label file: `D:\Tasks\Pathologic_Stage_Label\BRCA_pathologic_stage.csv`
- Image features: `D:\Tasks\WSI_extract_features\BRCA_WSI_extract_features`
- Sentence text features: `D:\Tasks\Text_Sentence_extract_features\BRCA_text`
- Hierarchy graph features: `D:\Tasks\isbi_code\pathology_report_extraction\Output\text_hierarchy_graphs_masked\BRCA`
- Hierarchy feature used for training: `section_features`

## Result Summary

`sentence_pt` outperformed `hierarchy_graph + section_features` on BRCA across 5 splits.

- `sentence_pt`
  - `ACC = 0.7376 ± 0.0409`
  - `AUC = 0.7489 ± 0.0235`
- `hierarchy_graph + section_features`
  - `ACC = 0.7034 ± 0.0389`
  - `AUC = 0.6521 ± 0.0300`

## Interpretation

- The current hierarchical text input is trainable and stable enough to run end-to-end.
- However, with the current model it still behaves as a hierarchical feature tensor rather than a graph-aware model input.
- For BRCA, the present implementation does **not** improve over the original sentence-level text mode.

## Files

- `comparison_results.csv`: per-split results
- `comparison_summary.json`: aggregated mean and standard deviation
