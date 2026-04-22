# KIRC dual_text readout_v2 10-split result

This record summarizes the KIRC `dual_text_readout_v2` run on aligned splits `0-9`.

## Setup

- Dataset: KIRC
- Splits: `D:\Tasks\Split_Table\KIRC\split_0.csv` to `split_9.csv`
- Label file: `D:\Tasks\Pathologic_Stage_Label\KIRC_pathologic_stage.csv`
- WSI features: `D:\Tasks\WSI_extract_features\KIRC_WSI_extract_features`
- Sentence text features: `D:\Tasks\Text_Sentence_extract_features\KIRC_text`
- Hierarchy graph features: `D:\Tasks\isbi_code\pathology_report_extraction\Output\text_hierarchy_graphs_masked\KIRC`
- Mode: `dual_text`
- Readout variant: `readout_v2`, with hierarchical attention mixed with mean-pooling fallback.

## Result

| Mode | ACC mean ± std | AUC mean ± std |
|---|---:|---:|
| sentence_pt, aligned previous record | 0.7676 ± 0.0551 | 0.8650 ± 0.0271 |
| dual_text v1, aligned previous record | 0.7784 ± 0.0262 | 0.8670 ± 0.0220 |
| dual_text readout_v2 | 0.7842 ± 0.0192 | 0.8635 ± 0.0272 |

Compared with `sentence_pt`, `dual_text_readout_v2` improves ACC by about `+0.0165` but has a slightly lower AUC by about `-0.0015`. Compared with the previous `dual_text` run, v2 further improves ACC but slightly reduces AUC.

## Gate and attention analysis

For `dual_text_readout_v2`:

- `fusion_gate_mean = 0.8547`
- `fusion_gate` is the sentence-branch weight.
- Therefore the graph-branch weight is about `0.1453` on average.
- `doc_attention_entropy_mean = 0.7292`
- `doc_attention_max_mean = 0.5609`

Sample-level analysis over 1390 test appearances shows:

- Correct samples: graph-branch weight mean `0.1513`
- Wrong samples: graph-branch weight mean `0.1238`

This suggests the graph branch is used more on correctly classified cases, but the contribution is still relatively small.

## Interpretation

The v2 readout made `dual_text` more stable and slightly improved ACC, but it did not recover the AUC advantage. The next small ablation is to keep this readout fixed and add a mild gate regularization term to encourage a higher graph-branch contribution, then check whether KIRC AUC improves.

## Local artifacts

- Experiment summary: `D:\Tasks\isbi_code\experiments\kirc_dual_text_readout_v2_10splits\comparison_summary.json`
- Per-split results: `D:\Tasks\isbi_code\experiments\kirc_dual_text_readout_v2_10splits\comparison_results.csv`
- Sample-level details: `D:\Tasks\isbi_code\experiments\kirc_dual_text_readout_v2_10splits\dual_text_sample_details.csv`
- Sample-level summary: `D:\Tasks\isbi_code\experiments\kirc_dual_text_readout_v2_10splits\dual_text_sample_analysis_summary.json`
