# Experiment Tree Layout

Generated: 2026-04-22

Runtime experiment outputs under `experiments/` are organized by dataset first,
then by representation type:

```text
experiments/
  BRCA/
    sentence-only/
      runs/
      records/
    sentence-ontology/
      runs/
      records/
    sentence-hierarchical-graph/
      runs/
      records/
    sentence-hierarchical-graph-ontology/
      runs/
      records/
  KIRC/
    ...
  LUSC/
    ...
```

Representation definitions:

| Directory | Meaning |
|---|---|
| `sentence-only` | Sentence-level representation only. |
| `sentence-ontology` | Sentence-level representation plus medical knowledge, without document-structure graph coupling. |
| `sentence-hierarchical-graph` | Sentence-level representation plus Document/Section/Sentence structure. |
| `sentence-hierarchical-graph-ontology` | Sentence-level representation plus document structure and ontology concepts/edges. |

Notes:

- Do not create new top-level comparison folders under `experiments/`.
- Put model outputs/checkpoints under `runs/`.
- Put logs, summaries, and lightweight run records under `records/`.
- Cross-method or paper-facing comparisons should live under `experiment_records/`.
- The local move manifest is `experiments/experiment_tree_manifest.json`.
