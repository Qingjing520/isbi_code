# Ordered Split Experiment Summary

Generated: 2026-04-22 19:51:01
Ontology variant: `ncit_snomed_mapped`

| Dataset | Method | Status | Splits | ACC | AUC | Location / reason |
|---|---|---|---:|---:|---:|---|
| BRCA | sentence-only | complete | 6 | 0.7577 +/- 0.0226 | 0.7458 +/- 0.0283 | `F:\Tasks\isbi_code\experiments\BRCA\sentence-only\runs\brca_sentence_only_6splits` |
| BRCA | sentence-hierarchical-graph | complete | 5 | 0.7180 +/- 0.0863 | 0.6924 +/- 0.0720 | `F:\Tasks\isbi_code\experiments\BRCA\sentence-hierarchical-graph\runs\brca_sentence_hierarchical_graph_5splits` |
| BRCA | sentence-hierarchical-graph-ontology | complete | 5 | 0.7727 +/- 0.0253 | 0.7310 +/- 0.0837 | `F:\Tasks\isbi_code\experiments\BRCA\sentence-hierarchical-graph-ontology\runs\brca_sentence_hierarchical_graph_ontology_5splits` |
| KIRC | sentence-only | complete | 5 | 0.7770 +/- 0.0274 | 0.8700 +/- 0.0319 | `F:\Tasks\isbi_code\experiments\KIRC\sentence-only\runs\kirc_sentence_only_5splits` |
| KIRC | sentence-hierarchical-graph | complete | 5 | 0.7640 +/- 0.0923 | 0.8853 +/- 0.0349 | `F:\Tasks\isbi_code\experiments\KIRC\sentence-hierarchical-graph\runs\kirc_sentence_hierarchical_graph_5splits` |
| KIRC | sentence-hierarchical-graph-ontology | complete | 5 | 0.7712 +/- 0.0280 | 0.8638 +/- 0.0332 | `F:\Tasks\isbi_code\experiments\KIRC\sentence-hierarchical-graph-ontology\runs\kirc_sentence_hierarchical_graph_ontology_5splits` |
| BRCA | sentence-ontology | skipped | 0 |  |  | `not implemented as a separate data product yet; current ontology data is represented as concept graphs and should be compared under sentence-hierarchical-graph-ontology` |
| KIRC | sentence-ontology | skipped | 0 |  |  | `not implemented as a separate data product yet; current ontology data is represented as concept graphs and should be compared under sentence-hierarchical-graph-ontology` |
| LUSC | sentence-only | skipped | 0 |  |  | `missing base inputs: F:\Tasks\Split_Table\LUSC\split_150.csv; F:\Tasks\Split_Table\LUSC\split_151.csv; F:\Tasks\Split_Table\LUSC\split_152.csv; F:\Tasks\Split_Table\LUSC\split_153.csv; F:\Tasks\Split_Table\LUSC\split_154.csv` |
| LUSC | sentence-ontology | skipped | 0 |  |  | `not implemented as a separate data product yet; current ontology data is represented as concept graphs and should be compared under sentence-hierarchical-graph-ontology` |
| LUSC | sentence-hierarchical-graph | skipped | 0 |  |  | `missing hierarchy graph directory: F:\Tasks\isbi_code\pathology_report_extraction\Output\text_hierarchy_graphs_masked\LUSC` |
| LUSC | sentence-hierarchical-graph-ontology | skipped | 0 |  |  | `missing concept graph directory: F:\Tasks\isbi_code\pathology_report_extraction\Output\text_concept_graphs_ablation\ncit_snomed_mapped\LUSC` |
