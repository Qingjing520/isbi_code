# Controlled Text-Graph Ablation Plan

Generated: 2026-04-26

## Design

The next ontology/hierarchy runs should keep the original sentence branch as the
main text representation. Hierarchy and ontology graphs are auxiliary signals
injected through bounded residual gates.

Default controlled settings:

| setting | value |
|---|---|
| dual-text fusion | `residual` |
| section title embedding | `true` |
| hierarchy graph max weight | `0.1`, `0.2`, `0.3` ablation |
| ontology graph max weight | `0.2` |
| graph weight regularization target | `0.1` |
| compact ontology variant | `ncit_do` |
| true-path max ancestor hops | `2` |
| concept co-occurrence edges | disabled for compact NCIt/DO graph rebuild |

## Current Resource Status

| resource | success | skipped | failed | datasets |
|---|---:|---:|---:|---|
| `concept_annotations_ablation/ncit_do` | 2147 | 0 | 0 | BRCA 1105, KIRC 538, LUSC 504 |
| `text_concept_graphs_ablation/ncit_do` | 2145 | 2 | 0 | BRCA 1104, KIRC 538, LUSC 503 |
| `text_sentence_ontology_graphs_ablation/ncit_do` | 2145 | 0 | 0 | BRCA 1104, KIRC 538, LUSC 503 |

Skipped graph samples:

| dataset | file | reason |
|---|---|---|
| BRCA | `TCGA-A8-A07R.FD75D315-EED7-4350-A890-8FB94648E9FC.PDF` | empty graph after cleanup |
| LUSC | `TCGA-63-5128.0883F7AA-5AA8-45D8-89B2-6279D2088C72.PDF` | empty graph after cleanup |

## Suggested Commands

Hierarchy auxiliary weight ablation:

```cmd
cd /d F:\Tasks\isbi_code
F:\Anaconda\envs\pytorch\python.exe tools\run_ordered_split_experiments.py --datasets BRCA KIRC LUSC --methods sentence-hierarchical-graph --num_splits 5 --ontology_variant ncit_do --fusion_mode residual --hierarchy_graph_weight_max 0.1 --experiment_tag auxgw01_sectiontitle_residual
F:\Anaconda\envs\pytorch\python.exe tools\run_ordered_split_experiments.py --datasets BRCA KIRC LUSC --methods sentence-hierarchical-graph --num_splits 5 --ontology_variant ncit_do --fusion_mode residual --hierarchy_graph_weight_max 0.2 --experiment_tag auxgw02_sectiontitle_residual
F:\Anaconda\envs\pytorch\python.exe tools\run_ordered_split_experiments.py --datasets BRCA KIRC LUSC --methods sentence-hierarchical-graph --num_splits 5 --ontology_variant ncit_do --fusion_mode residual --hierarchy_graph_weight_max 0.3 --experiment_tag auxgw03_sectiontitle_residual
```

Compact ontology and joint structure/ontology:

```cmd
cd /d F:\Tasks\isbi_code
F:\Anaconda\envs\pytorch\python.exe tools\run_ordered_split_experiments.py --datasets BRCA KIRC LUSC --methods sentence-ontology sentence-hierarchical-graph-ontology --num_splits 5 --ontology_variant ncit_do --fusion_mode residual --ontology_graph_weight_max 0.2 --experiment_tag compact_auxgw20_residual
```
