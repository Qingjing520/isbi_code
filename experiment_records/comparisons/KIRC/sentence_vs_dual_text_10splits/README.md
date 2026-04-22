# KIRC sentence_pt vs dual_text (splits 0-9)

This note compares `sentence_pt` and `dual_text` on the same KIRC `split_0~9`, which is a fairer comparison than mixing results from different split ranges.

## Summary

- `sentence_pt`: ACC `0.7676 +/- 0.0551`, AUC `0.8650 +/- 0.0271`
- `dual_text`: ACC `0.7784 +/- 0.0262`, AUC `0.8670 +/- 0.0220`
- mean delta (`dual_text - sentence_pt`): ACC `+0.0108`, AUC `+0.0020`
- `dual_text` wins on ACC in `5/10` splits
- `dual_text` wins on AUC in `5/10` splits

## Interpretation

- `dual_text` shows a small positive trend on the aligned KIRC splits.
- The gain is modest rather than decisive, but it is no longer fair to say that `dual_text` is simply worse than `sentence_pt` on KIRC.
- At the current stage, `dual_text` is better viewed as a promising enhancement branch than as a fully established replacement for the sentence baseline.

## Per-split comparison

| split | sentence_acc | dual_acc | delta_acc | sentence_auc | dual_auc | delta_auc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.8273 | 0.7698 | -0.0576 | 0.9000 | 0.8699 | -0.0301 |
| 1 | 0.8058 | 0.7842 | -0.0216 | 0.8594 | 0.8699 | +0.0105 |
| 2 | 0.8058 | 0.8273 | +0.0216 | 0.8706 | 0.8692 | -0.0013 |
| 3 | 0.7914 | 0.7626 | -0.0288 | 0.8791 | 0.8962 | +0.0171 |
| 4 | 0.6978 | 0.7482 | +0.0504 | 0.8208 | 0.8464 | +0.0257 |
| 5 | 0.6475 | 0.7482 | +0.1007 | 0.8348 | 0.8260 | -0.0088 |
| 6 | 0.7986 | 0.7770 | -0.0216 | 0.8864 | 0.8497 | -0.0366 |
| 7 | 0.7698 | 0.7626 | -0.0072 | 0.8997 | 0.8993 | -0.0004 |
| 8 | 0.7698 | 0.8129 | +0.0432 | 0.8423 | 0.8730 | +0.0307 |
| 9 | 0.7626 | 0.7914 | +0.0288 | 0.8572 | 0.8703 | +0.0132 |

## Source files

- sentence baseline: `D:\Tasks\isbi_code\experiments\kirc_text_mode_comparison_20splits_node_features\comparison_results.csv`
- dual_text runs: `D:\Tasks\isbi_code\experiments\kirc_dual_text_3splits\comparison_results.csv`
- derived summary: `D:\Tasks\isbi_code\experiment_records\kirc_sentence_vs_dual_text_10splits\comparison_summary.json`
- derived csv: `D:\Tasks\isbi_code\experiment_records\kirc_sentence_vs_dual_text_10splits\comparison_results.csv`
