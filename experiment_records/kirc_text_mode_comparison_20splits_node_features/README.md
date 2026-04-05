# KIRC Text Mode Comparison (20 Splits, Node Features)

This record summarizes a KIRC comparison between:

- `sentence_pt`: the original sentence-level text feature mode
- `hierarchy_graph + node_features`: the hierarchical text mode using all document, section, and sentence nodes

## Setup

- Dataset: `KIRC`
- Splits: `D:\Tasks\Split_Table\KIRC\split_0.csv` to `split_19.csv`
- Label file: `D:\Tasks\Pathologic_Stage_Label\KIRC_pathologic_stage.csv`
- Image features: `D:\Tasks\WSI_extract_features\KIRC_WSI_extract_features`
- Sentence text features: `D:\Tasks\Text_Sentence_extract_features\KIRC_text`
- Hierarchy graph features: `D:\Tasks\isbi_code\pathology_report_extraction\Output\text_hierarchy_graphs_masked\KIRC`
- Hierarchy feature used for training: `node_features`

## Result Summary

`sentence_pt` outperformed `hierarchy_graph + node_features` on KIRC over 20 splits.

- `sentence_pt`
  - `ACC = 0.7824 ± 0.0446`
  - `AUC = 0.8658 ± 0.0209`
- `hierarchy_graph + node_features`
  - `ACC = 0.7468 ± 0.0568`
  - `AUC = 0.8374 ± 0.0346`

## Interpretation

- The hierarchical graph input is clearly trainable and reasonably competitive on KIRC.
- However, under the current model it still falls short of the original sentence-level text mode.
- This supports the view that the present implementation does not yet fully exploit graph structure, because the model still consumes feature tensors rather than explicit graph edges and node types.

## Files

- `comparison_results.csv`: per-split results
- `comparison_summary.json`: aggregated mean and standard deviation
