# KIRC dual_text gate-regularization ablation

This record tracks a KIRC ablation after the `dual_text_readout_v2` update.

## Motivation

The `dual_text_readout_v2` run showed that the graph branch was used, but only as a small auxiliary signal:

- Sentence branch weight: about `0.8547`
- Graph branch weight: about `0.1453`
- ACC improved over `sentence_pt`, but AUC did not clearly improve.

To test whether the graph branch needs slightly more participation, we added an optional gate regularization term:

```text
loss_gate = mean(((1 - fusion_gate) - graph_weight_target)^2)
```

where `fusion_gate` is the sentence-branch weight and `1 - fusion_gate` is the graph-branch weight.

This regularizer is disabled by default. In this ablation we used:

- `dual_text_gate_reg_weight = 0.01`
- `dual_text_graph_weight_target = 0.2`

## Main 10-split result

This result uses KIRC `split_0` to `split_9`.

| Mode | ACC mean ± std | AUC mean ± std |
|---|---:|---:|
| sentence_pt, aligned previous record | 0.7676 ± 0.0551 | 0.8650 ± 0.0271 |
| dual_text v1, aligned previous record | 0.7784 ± 0.0262 | 0.8670 ± 0.0220 |
| dual_text readout_v2, no gate reg | 0.7842 ± 0.0192 | 0.8635 ± 0.0272 |
| dual_text readout_v2 + gate reg 0.01 | 0.7770 ± 0.0359 | 0.8758 ± 0.0246 |

## Gate and attention analysis for 10 splits

For `dual_text_readout_v2 + gate reg 0.01`:

- Sentence-branch weight mean: `0.8361`
- Graph-branch weight mean: `0.1639`
- Document attention entropy mean: `0.6133`
- Maximum document attention weight mean: `0.6480`

Sample-level analysis over 1390 test appearances:

- Correct samples: graph-branch weight mean `0.1648`
- Wrong samples: graph-branch weight mean `0.1610`
- Most frequent top sections overall: `Document Body`, `Diagnosis`, `Gross Description`, `Specimen Submitted`, `Clinical Information`

The gate regularizer raises graph-branch participation compared with no regularization (`0.1453 -> 0.1639`) and makes section attention sharper (`doc attention max 0.5609 -> 0.6480`). The main benefit is on AUC rather than ACC.

## Earlier 3-split pilot

The 3-split pilot gave the same positive signal:

| Mode | ACC mean | AUC mean |
|---|---:|---:|
| sentence_pt | 0.8129 | 0.8766 |
| dual_text readout_v2, no gate reg | 0.7746 | 0.8720 |
| dual_text readout_v2 + gate reg 0.01 | 0.8010 | 0.8861 |

## Interpretation

This ablation suggests that the hierarchy graph branch can help when it is encouraged to contribute mildly, rather than being left as a very small residual signal. The improvement is still moderate, but this is the strongest KIRC result so far for the hierarchy-enhanced text branch in terms of AUC.

The next useful checks are:

- Repeat the same gate setting on BRCA to see whether the effect is KIRC-specific.
- Try a tiny grid on KIRC, for example `gate_reg_weight = 0.005 / 0.01 / 0.02`, keeping `graph_weight_target = 0.2`.
- Inspect the confident wrong cases in `dual_text_sample_details.csv` to see whether attention is focusing on useful sections or leakage-prone/noisy sections.

## Local artifacts

10-split run:

- Experiment summary: `D:\Tasks\isbi_code\experiments\kirc_dual_text_readout_v2_gate001_10splits\comparison_summary.json`
- Per-split results: `D:\Tasks\isbi_code\experiments\kirc_dual_text_readout_v2_gate001_10splits\comparison_results.csv`
- Sample-level details: `D:\Tasks\isbi_code\experiments\kirc_dual_text_readout_v2_gate001_10splits\dual_text_sample_details.csv`
- Sample-level summary: `D:\Tasks\isbi_code\experiments\kirc_dual_text_readout_v2_gate001_10splits\dual_text_sample_analysis_summary.json`

