# Multi-Ontology Concept Graph Baseline Comparison

Generated: 2026-04-21

Primary comparable baseline: the completed old 3-split `dual_text_section_title_gate001` runs. This is the cleanest paired comparison currently available for both KIRC and BRCA because the older KIRC `ncit_concept_graph` run only completed split0, and no BRCA `ncit_concept_graph` 3-split baseline was found in the workspace.

| Dataset | Old baseline | Old ACC mean +/- std | New multi-ontology concept graph | New ACC mean +/- std | Delta ACC | Old AUC mean +/- std | New AUC mean +/- std | Delta AUC |
|---|---|---:|---|---:|---:|---:|---:|---:|
| KIRC | `kirc_dual_text_section_title_gate001_3splits` | 0.7842 +/- 0.0190 | `kirc_multi_ontology_concept_graph_main_3splits_nw0` | 0.7626 +/- 0.0381 | -0.0216 | 0.8760 +/- 0.0194 | 0.8281 +/- 0.0201 | -0.0479 |
| BRCA | `brca_dual_text_section_title_gate001_3splits` | 0.7089 +/- 0.0325 | `brca_multi_ontology_concept_graph_main_3splits_nw0` | 0.5593 +/- 0.2566 | -0.1496 | 0.7599 +/- 0.0322 | 0.6314 +/- 0.0256 | -0.1285 |

Closest old NCIt-only concept-graph evidence currently available:

| Dataset | Old run | Split coverage | Old ACC | New same split ACC | Delta ACC | Old AUC | New same split AUC | Delta AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| KIRC | `kirc_ncit_concept_graph_main_3splits` | split0 only | 0.7842 | 0.7914 | +0.0072 | 0.8506 | 0.8449 | -0.0057 |
| BRCA | Not found | 0/3 | NA | NA | NA | NA | NA | NA |

Source files:

| Purpose | Path |
|---|---|
| KIRC new summary | `F:\Tasks\isbi_code\experiments\kirc_multi_ontology_concept_graph_main_3splits_nw0\summary.json` |
| BRCA new summary | `F:\Tasks\isbi_code\experiments\brca_multi_ontology_concept_graph_main_3splits_nw0\summary.json` |
| KIRC old comparable baseline | `F:\Tasks\isbi_code\experiments\kirc_dual_text_section_title_gate001_3splits\comparison_summary.json` |
| BRCA old comparable baseline | `F:\Tasks\isbi_code\experiments\brca_dual_text_section_title_gate001_3splits\comparison_summary.json` |
| KIRC old NCIt-only split0 log | `F:\Tasks\isbi_code\experiments\kirc_ncit_concept_graph_main_3splits\run_20260415_144913.out.log` |
