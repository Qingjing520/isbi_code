# Dual Text + Concept Graph Ontology Ablation Final Report

Generated: 2026-04-22

Run completed at 2026-04-21 23:49 Asia/Shanghai. The ablation tested `dual_text` with raw sentence features preserved and concept graphs added as a structured graph branch. Each dataset used 3 splits across four ontology resource variants: `ncit_only`, `ncit_do`, `ncit_snomed_mapped`, and `full_multi_ontology`.

## Main Results

| Dataset | Variant | ACC mean +/- std | AUC mean +/- std | Delta AUC vs old dual_text | Graph weight at best | Graph weight final |
|---|---|---:|---:|---:|---:|---:|
| KIRC | `ncit_only` | 0.7794 +/- 0.0253 | 0.8634 +/- 0.0231 | -0.0126 | 0.0098 | 0.0093 |
| KIRC | `ncit_do` | 0.7842 +/- 0.0259 | 0.8742 +/- 0.0283 | -0.0018 | 0.0206 | 0.0490 |
| KIRC | `ncit_snomed_mapped` | 0.7938 +/- 0.0362 | 0.8730 +/- 0.0317 | -0.0029 | 0.0280 | 0.0488 |
| KIRC | `full_multi_ontology` | 0.7794 +/- 0.0220 | 0.8608 +/- 0.0164 | -0.0152 | 0.0314 | 0.0962 |
| BRCA | `ncit_only` | 0.7659 +/- 0.0000 | 0.6641 +/- 0.0883 | -0.0959 | 0.4211 | 0.7605 |
| BRCA | `ncit_do` | 0.6683 +/- 0.1162 | 0.6643 +/- 0.0903 | -0.0956 | 0.4368 | 0.6401 |
| BRCA | `ncit_snomed_mapped` | 0.7187 +/- 0.0775 | 0.6717 +/- 0.0791 | -0.0882 | 0.4227 | 0.5365 |
| BRCA | `full_multi_ontology` | 0.6374 +/- 0.1931 | 0.6594 +/- 0.0746 | -0.1005 | 0.2274 | 0.3553 |

Old dual-text baseline:

| Dataset | Old baseline | Old ACC mean +/- std | Old AUC mean +/- std |
|---|---|---:|---:|
| KIRC | `kirc_dual_text_section_title_gate001_3splits` | 0.7842 +/- 0.0190 | 0.8760 +/- 0.0194 |
| BRCA | `brca_dual_text_section_title_gate001_3splits` | 0.7089 +/- 0.0325 | 0.7599 +/- 0.0322 |

Standalone full multi-ontology concept graph baseline:

| Dataset | Standalone concept graph | AUC mean +/- std | Best dual_text + graph AUC | Delta AUC |
|---|---|---:|---:|---:|
| KIRC | `kirc_multi_ontology_concept_graph_main_3splits_nw0` | 0.8281 +/- 0.0201 | 0.8742 | +0.0461 |
| BRCA | `brca_multi_ontology_concept_graph_main_3splits_nw0` | 0.6314 +/- 0.0256 | 0.6717 | +0.0403 |

## Interpretation

KIRC is now almost back to the old raw-text dual baseline. The best KIRC variant is `ncit_do` at AUC 0.8742, only 0.0018 below the old dual-text baseline and clearly above the standalone concept-graph run. The learned fusion gate gives the graph branch very little weight at the best epochs, around 0.02 on average, so the model mostly trusts the sentence branch and uses the concept graph as a small supplement.

BRCA is different. The best ontology variant is `ncit_snomed_mapped` at AUC 0.6717, which improves over standalone concept graph but remains well below the old raw-text dual baseline. The graph branch is heavily used at BRCA best epochs, around 0.42 on average, so the weakness is probably not underuse of the graph branch. The more likely issue is noisy or task-misaligned BRCA graph signal, especially when full multi-ontology concepts are admitted.

The full multi-ontology setting is not the best setting for either dataset. It adds coverage, but current filtering/fusion is not selective enough to turn that coverage into better prediction.

## Recommended Next Step

Use `ncit_do` as the compact KIRC default and `ncit_snomed_mapped` as the BRCA concept-graph candidate, but keep the old raw-text dual baseline as the primary comparator. Do not promote `full_multi_ontology` as the default training setting yet.

The next useful experiment is a smaller fusion ablation:

| Experiment | Purpose |
|---|---|
| `dual_text` raw baseline rerun | Reconfirm current baseline under the same runner and current code. |
| `dual_text + ncit_do` for KIRC | Validate the near-tie with old KIRC dual-text baseline. |
| `dual_text + ncit_snomed_mapped` for BRCA | Test the least-bad BRCA ontology variant. |
| Gate regularization sweep | Try graph targets below 0.2 for BRCA or add a warmup where sentence branch dominates longer. |
| Concept noise audit for BRCA | Inspect high-weight BRCA graph cases where predictions flip incorrectly. |

Source summaries:

| Purpose | Path |
|---|---|
| KIRC `ncit_do` best summary | `F:\Tasks\isbi_code\experiments\kirc_dual_text_concept_graph_ncit_do_3splits_nw0\summary.json` |
| BRCA `ncit_snomed_mapped` best summary | `F:\Tasks\isbi_code\experiments\brca_dual_text_concept_graph_ncit_snomed_mapped_3splits_nw0\summary.json` |
| Ablation run log | `F:\Tasks\isbi_code\experiments\dual_text_concept_graph_ablation_logs\full_ablation.out.log` |
| Previous comparison table | `F:\Tasks\isbi_code\experiment_records\multi_ontology_concept_graph_baseline_comparison.md` |
