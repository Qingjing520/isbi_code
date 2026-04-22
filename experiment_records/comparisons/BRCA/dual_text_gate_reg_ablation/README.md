# BRCA dual_text gate-regularization ablation

This record summarizes the BRCA `dual_text_readout_v2 + gate_reg=0.01` run on aligned splits `0-9`.

## Setup

- Dataset: BRCA
- Splits: `D:\Tasks\Split_Table\BRCA\split_0.csv` to `split_9.csv`
- Label file: `D:\Tasks\Pathologic_Stage_Label\BRCA_pathologic_stage.csv`
- WSI features: `D:\Tasks\WSI_extract_features\BRCA_WSI_extract_features`
- Sentence text features: `D:\Tasks\Text_Sentence_extract_features\BRCA_text`
- Hierarchy graph features: `D:\Tasks\isbi_code\pathology_report_extraction\Output\text_hierarchy_graphs_masked\BRCA`
- Mode: `dual_text`
- Readout variant: `readout_v2`, with hierarchical attention mixed with mean-pooling fallback
- Gate regularization: `dual_text_gate_reg_weight = 0.01`, `dual_text_graph_weight_target = 0.2`

## 10-split aligned result

| Mode | ACC mean | AUC mean |
|---|---:|---:|
| sentence_pt, aligned previous record | 0.7561 | 0.7427 |
| dual_text v1, aligned previous record | 0.7366 | 0.7301 |
| dual_text readout_v2 + gate reg 0.01 | 0.7541 | 0.7404 |

Compared with the earlier no-regularization `dual_text`, gate regularization clearly recovers performance on BRCA. However, it still does not clearly outperform the `sentence_pt` baseline.

## Split-level observations

- Gate-reg `dual_text` beats `sentence_pt` on ACC in `5/10` splits.
- Gate-reg `dual_text` beats `sentence_pt` on AUC in `4/10` splits.
- Gate-reg `dual_text` beats previous `dual_text` on ACC in `5/10` splits.
- Gate-reg `dual_text` beats previous `dual_text` on AUC in `6/10` splits.

## Gate and attention analysis

For `dual_text_readout_v2 + gate reg 0.01`:

- Sentence-branch weight mean: `0.7688`
- Graph-branch weight mean: `0.2312`
- Document attention entropy mean: `0.5972`
- Maximum document attention weight mean: `0.6767`

Sample-level analysis over 1640 test appearances:

- Correct samples: graph-branch weight mean `0.2190`
- Wrong samples: graph-branch weight mean `0.1969`
- Most frequent top sections overall: `Document Body`, `Final Diagnosis`, `Procedure`, `Comment`, `Synoptic Report`

## Interpretation

The gate regularizer successfully increases graph-branch usage on BRCA and improves over the earlier dual_text run. But unlike KIRC, the improvement is not enough to beat sentence_pt. This suggests that the dual_text + hierarchy graph direction is promising but dataset-dependent.

A useful next step is to test a small BRCA gate-weight grid, especially a weaker regularizer such as `0.005`, because the current graph branch weight is already around `0.23`, slightly above the target `0.2`.

## Local artifacts

- Experiment summary: `D:\Tasks\isbi_code\experiments\brca_dual_text_readout_v2_gate001_10splits\comparison_summary.json`
- Per-split results: `D:\Tasks\isbi_code\experiments\brca_dual_text_readout_v2_gate001_10splits\comparison_results.csv`
- Sample-level details: `D:\Tasks\isbi_code\experiments\brca_dual_text_readout_v2_gate001_10splits\dual_text_sample_details.csv`
- Sample-level summary: `D:\Tasks\isbi_code\experiments\brca_dual_text_readout_v2_gate001_10splits\dual_text_sample_analysis_summary.json`
