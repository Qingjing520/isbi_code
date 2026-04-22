# BRCA sentence_pt vs dual_text (splits 0-9)

This note compares `sentence_pt` and `dual_text` on the same BRCA `split_0~9`.

## Summary

- `sentence_pt`: ACC `0.7561 +/- 0.0319`, AUC `0.7427 +/- 0.0435`
- `dual_text`: ACC `0.7366 +/- 0.0593`, AUC `0.7301 +/- 0.0446`
- mean delta (`dual_text - sentence_pt`): ACC `-0.0195`, AUC `-0.0126`
- `dual_text` wins on ACC in `2/10` splits
- `dual_text` wins on AUC in `4/10` splits

## Gate/attention summary

- mean fusion gate: `0.4190`
- mean document-attention entropy: `0.5458`
- mean max document-attention weight: `0.6995`

Interpretation:
- The average gate stays around 0.42, so the model does not collapse fully to the sentence branch or the graph branch.
- The document-level attention remains fairly peaked (mean max weight about 0.70), which suggests the graph branch often relies on one dominant section.
- Despite that, the final performance still lags behind `sentence_pt`, so the current graph branch is not yet converting its structured focus into a robust classification gain.

## Per-split comparison

| split | sentence_acc | dual_acc | delta_acc | sentence_auc | dual_auc | delta_auc | gate_mean | doc_attn_entropy | doc_attn_max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.6878 | 0.7659 | +0.0780 | 0.7293 | 0.6907 | -0.0386 | 0.0671 | 0.5605 | 0.7080 |
| 1 | 0.7659 | 0.7463 | -0.0195 | 0.7707 | 0.7764 | +0.0057 | 0.4678 | 0.4399 | 0.7765 |
| 2 | 0.7268 | 0.7366 | +0.0098 | 0.7170 | 0.6798 | -0.0372 | 0.0654 | 0.6666 | 0.6378 |
| 3 | 0.7854 | 0.7854 | +0.0000 | 0.7516 | 0.7719 | +0.0203 | 0.0887 | 0.5199 | 0.7266 |
| 4 | 0.7268 | 0.5756 | -0.1512 | 0.7533 | 0.7251 | -0.0283 | 0.0041 | 0.6725 | 0.6314 |
| 5 | 0.7659 | 0.7659 | +0.0000 | 0.6346 | 0.6557 | +0.0211 | 0.9477 | 0.8509 | 0.4730 |
| 6 | 0.7707 | 0.7463 | -0.0244 | 0.7869 | 0.7253 | -0.0616 | 0.4432 | 0.2878 | 0.8589 |
| 7 | 0.7854 | 0.7561 | -0.0293 | 0.7545 | 0.7882 | +0.0337 | 0.7832 | 0.3041 | 0.8472 |
| 8 | 0.7659 | 0.7220 | -0.0439 | 0.7800 | 0.7609 | -0.0191 | 0.5855 | 0.3298 | 0.8477 |
| 9 | 0.7805 | 0.7659 | -0.0146 | 0.7492 | 0.7272 | -0.0220 | 0.7375 | 0.8261 | 0.4875 |

## Source files

- sentence baseline: `D:\Tasks\isbi_code\experiments\brca_text_mode_comparison_10splits_graph_structure\comparison_results.csv`
- dual_text runs: `D:\Tasks\isbi_code\experiments\brca_dual_text_10splits\comparison_results.csv`
- derived summary: `D:\Tasks\isbi_code\experiment_records\brca_sentence_vs_dual_text_10splits\comparison_summary.json`
- derived csv: `D:\Tasks\isbi_code\experiment_records\brca_sentence_vs_dual_text_10splits\comparison_results.csv`