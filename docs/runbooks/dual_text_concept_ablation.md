# Dual Text Concept-Graph Ablation Runbook

This runbook documents the ontology ablation experiments. Generated configs and
experiment outputs are intentionally not source files.

## Scripts

- `pathology_report_extraction/build_ontology_ablation_bundles.py`
  Builds the current default `ncit_do` ontology resource. Legacy variants
  (`ncit_only`, `ncit_snomed_mapped`, `full_multi_ontology`) are available only
  when passed explicitly.

- `tools/run_dual_text_concept_ablation.py`
  Orchestrates concept annotations, concept graphs, manifests, training, and
  result aggregation.

- `tools/watch_dual_text_concept_ablation.py`
  Prints live progress for the background ablation run.

## Current Default Paths

- Python: `F:\Anaconda\envs\pytorch\python.exe`
- Repo: `F:\Tasks\isbi_code`
- Ontology processed root: `F:\Tasks\Ontologies\processed`
- Result summary: `experiment_records\dual_text_concept_graph_ablation\summary.md`
- Runtime logs: `experiments\KIRC\sentence-hierarchical-graph-ontology\records\shared_dual_text_concept_graph_ablation_logs`

## Monitor

From `cmd.exe`:

```cmd
F:\Anaconda\envs\pytorch\python.exe F:\Tasks\isbi_code\tools\watch_dual_text_concept_ablation.py
```

One-shot status:

```cmd
F:\Anaconda\envs\pytorch\python.exe F:\Tasks\isbi_code\tools\watch_dual_text_concept_ablation.py --once
```

## Re-run

Run the current KIRC/BRCA NCIt+DO concept-graph mainline:

```cmd
cd /d F:\Tasks\isbi_code
F:\Anaconda\envs\pytorch\python.exe tools\run_dual_text_concept_ablation.py --datasets KIRC BRCA --variants ncit_do --num_splits 3 --force_train
```

Run only preprocessing and manifest generation:

```cmd
F:\Anaconda\envs\pytorch\python.exe tools\run_dual_text_concept_ablation.py --skip_train
```

Run only training from existing graphs/manifests:

```cmd
F:\Anaconda\envs\pytorch\python.exe tools\run_dual_text_concept_ablation.py --skip_preprocess --force_train
```

## Notes

- `configs/generated/` is ignored by Git because orchestration recreates these YAML files.
- `experiments/` and `pathology_report_extraction/Output/` are runtime artifacts.
- Commit source changes separately from final result summaries.

## Completed 3-Split Result

The full KIRC/BRCA x four-ontology run completed on 2026-04-21. The stable report is:

```text
experiment_records\dual_text_concept_graph_ablation_final.md
```

Practical default for the next round:

- KIRC: prefer `ncit_do`; it was the best AUC variant and nearly tied the old raw-text dual baseline.
- BRCA: use `ncit_do` for the current compact ontology mainline; the older `ncit_snomed_mapped` result remains useful as a historical recall ablation, not as the default.
- Avoid promoting SNOMED/UMLS variants as defaults until concept noise and fusion behavior are audited.
- Keep concept graph auxiliary: generated ablation configs cap `text_dual_graph_weight_max` at `0.2` and use `dual_text_graph_weight_target: 0.1`, so the raw sentence branch remains the primary semantic carrier.
