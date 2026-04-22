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
- Put only split-level result indexes under `records/`:
  `split_results.csv`, `split_results.json`, and `split_results.md`.
- Keep raw run summaries, logs, configs, checkpoints, and archived historical
  comparison files under `runs/`; old record folders are preserved under
  `runs/_legacy_records/`.
- Cross-method or paper-facing comparisons should live under `experiment_records/`.
- The local move manifest is `experiments/experiment_tree_manifest.json`.
- Rebuild method-level split indexes with
  `python tools/build_experiment_split_records.py --apply`.
