# Ordered Method Comparison

Generated: 2026-04-25 16:32:21

This table aligns the four text-side experiment classes. Rows marked `running`, `queued`, or `not-run` are excluded from official comparison metrics until their splits finish.

| Dataset | Method | Status | Official splits | ACC | AUC | Delta AUC vs sentence-only |
|---|---|---|---:|---:|---:|---:|
| BRCA | sentence-only | complete | 6 | 0.7577 +/- 0.0226 | 0.7458 +/- 0.0283 |  |
| BRCA | sentence + ontology | not-run | 0 |  |  |  |
| BRCA | sentence + hierarchical graph | complete | 5 | 0.7180 +/- 0.0863 | 0.6924 +/- 0.0720 | -0.0533 |
| BRCA | sentence + hierarchical graph + ontology | complete | 5 | 0.7727 +/- 0.0253 | 0.7310 +/- 0.0837 | -0.0148 |
| KIRC | sentence-only | complete | 5 | 0.7770 +/- 0.0274 | 0.8700 +/- 0.0319 |  |
| KIRC | sentence + ontology | not-run | 0 |  |  |  |
| KIRC | sentence + hierarchical graph | complete | 5 | 0.7640 +/- 0.0923 | 0.8853 +/- 0.0349 | +0.0153 |
| KIRC | sentence + hierarchical graph + ontology | complete | 5 | 0.7712 +/- 0.0280 | 0.8638 +/- 0.0332 | -0.0063 |
| LUSC | sentence-only | complete-with-note | 150 | 0.7854 +/- 0.0505 | 0.6865 +/- 0.0529 |  |
| LUSC | sentence + ontology | running | 0 |  |  |  |
| LUSC | sentence + hierarchical graph | queued | 0 |  |  |  |
| LUSC | sentence + hierarchical graph + ontology | queued | 0 |  |  |  |

## Sources And Notes

| Dataset | Method | Source | Notes |
|---|---|---|---|
| BRCA | sentence-only | `F:\Tasks\isbi_code\experiments\BRCA\sentence-only\runs\brca_sentence_only_6splits` |  |
| BRCA | sentence + ontology | `` | not run yet with the new sentence-ontology graph product |
| BRCA | sentence + hierarchical graph | `F:\Tasks\isbi_code\experiments\BRCA\sentence-hierarchical-graph\runs\brca_sentence_hierarchical_graph_5splits` |  |
| BRCA | sentence + hierarchical graph + ontology | `F:\Tasks\isbi_code\experiments\BRCA\sentence-hierarchical-graph-ontology\runs\brca_sentence_hierarchical_graph_ontology_5splits` |  |
| KIRC | sentence-only | `F:\Tasks\isbi_code\experiments\KIRC\sentence-only\runs\kirc_sentence_only_5splits` |  |
| KIRC | sentence + ontology | `` | not run yet with the new sentence-ontology graph product |
| KIRC | sentence + hierarchical graph | `F:\Tasks\isbi_code\experiments\KIRC\sentence-hierarchical-graph\runs\kirc_sentence_hierarchical_graph_5splits` |  |
| KIRC | sentence + hierarchical graph + ontology | `F:\Tasks\isbi_code\experiments\KIRC\sentence-hierarchical-graph-ontology\runs\kirc_sentence_hierarchical_graph_ontology_5splits` |  |
| LUSC | sentence-only | `F:\Tasks\isbi_code\experiments\LUSC\sentence-only\runs\LUSC_150splits_sentence_only` | historical 150-split baseline; features correspond to F:\Tasks\Text_Sentence_extract_features\LUSC_text after file migration; 149 splits have final_evaluation, 1 uses best logged epoch |
| LUSC | sentence + ontology | `F:\Tasks\isbi_code\experiments\LUSC\sentence-ontology\runs\lusc_sentence_ontology_0splits` | official comparison pending; 3 splits started, 0 finalized |
| LUSC | sentence + hierarchical graph | `F:\Tasks\isbi_code\experiments\LUSC\sentence-hierarchical-graph\runs\lusc_sentence_hierarchical_graph_0splits` | official comparison pending; 0 splits started, 0 finalized |
| LUSC | sentence + hierarchical graph + ontology | `F:\Tasks\isbi_code\experiments\LUSC\sentence-hierarchical-graph-ontology\runs\lusc_sentence_hierarchical_graph_ontology_0splits` | official comparison pending; 0 splits started, 0 finalized |
