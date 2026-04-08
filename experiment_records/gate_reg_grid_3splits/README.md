# Gate regularization grid on aligned splits 0-2

This record summarizes a small `dual_text_readout_v2` gate-regularization grid on the same three splits, `split_0` to `split_2`, for KIRC and BRCA.

## Setup

- Text mode: `dual_text`
- Hierarchy branch: `readout_v2`, with hierarchical attention mixed with mean-pooling fallback
- Gate regularization target: `graph_weight_target = 0.2`
- Gate regularization weights tested: `0.005`, `0.01`, `0.02`
- Splits: aligned `split_0`, `split_1`, `split_2`
- Baselines included:
  - `sentence_pt`: original sentence-feature text input
  - `dual_text_v1`: earlier dual-text result without the readout_v2/gate-reg setting

## KIRC result

| Setting | ACC mean | AUC mean | Graph branch weight |
|---|---:|---:|---:|
| `sentence_pt` | 0.8129 | 0.8766 | - |
| `dual_text_v1` | 0.7938 | 0.8697 | - |
| `gate=0.005` | 0.8058 | 0.8825 | 0.1418 |
| `gate=0.01` | 0.8010 | 0.8861 | 0.1906 |
| `gate=0.02` | 0.7986 | 0.8724 | 0.0928 |

KIRC supports keeping `gate_reg_weight = 0.01` as the current best setting: it gives the highest AUC on this 3-split grid and is consistent with the stronger 10-split KIRC result already recorded under `kirc_dual_text_gate_reg_ablation`.

The `0.02` setting does not improve the KIRC result. It also unexpectedly lowers the observed graph-branch weight, suggesting that simply increasing the regularization coefficient does not guarantee more useful hierarchy usage.

## BRCA result

| Setting | ACC mean | AUC mean | Graph branch weight |
|---|---:|---:|---:|
| `sentence_pt` | 0.7268 | 0.7390 | - |
| `dual_text_v1` | 0.7496 | 0.7156 | - |
| `gate=0.005` | 0.6602 | 0.7182 | 0.3115 |
| `gate=0.01` | 0.7642 | 0.7221 | 0.1231 |
| `gate=0.02` | 0.6244 | 0.7403 | 0.3651 |

BRCA remains unstable. Although `gate=0.02` has a slightly higher mean AUC than `sentence_pt` on these three splits, it also has very low ACC and large variance. This should not be treated as a reliable improvement without more splits.

The `gate=0.01` setting gives the best BRCA ACC in this small grid, but it still trails `sentence_pt` on AUC. This matches the 10-split BRCA gate-ablation result: gate regularization can repair the earlier `dual_text` drop, but it does not yet clearly beat the original sentence baseline on BRCA.

## Interpretation

- KIRC is the dataset where hierarchy-enhanced `dual_text` currently looks most promising.
- For KIRC, `gate_reg_weight = 0.01` and `graph_weight_target = 0.2` remain the recommended setting.
- For BRCA, the grid does not justify stronger graph forcing. The graph branch is useful as an auxiliary signal, but increasing graph participation can be noisy and sometimes harmful.
- The next reasonable step is not to blindly increase the gate regularization. Instead, use `0.01` as the default KIRC setting and treat BRCA as a dataset-specific case where section-level noise or report-format variability may require a different strategy.

## Local artifacts

- KIRC `gate=0.005`: `D:\Tasks\isbi_code\experiments\kirc_dual_text_readout_v2_gate0005_3splits`
- KIRC `gate=0.01`: `D:\Tasks\isbi_code\experiments\kirc_dual_text_readout_v2_gate001_10splits`, using only `split_0-2` for this grid summary
- KIRC `gate=0.02`: `D:\Tasks\isbi_code\experiments\kirc_dual_text_readout_v2_gate002_3splits`
- BRCA `gate=0.005`: `D:\Tasks\isbi_code\experiments\brca_dual_text_readout_v2_gate0005_3splits`
- BRCA `gate=0.01`: `D:\Tasks\isbi_code\experiments\brca_dual_text_readout_v2_gate001_10splits`, using only `split_0-2` for this grid summary
- BRCA `gate=0.02`: `D:\Tasks\isbi_code\experiments\brca_dual_text_readout_v2_gate002_3splits`

See also:

- `grid_summary.csv`
- `grid_summary.json`
