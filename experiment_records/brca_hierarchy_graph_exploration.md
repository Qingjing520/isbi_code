# BRCA Hierarchy-Graph Exploration Notes

This note summarizes the current BRCA experiments that explored whether the new `Document -> Section -> Sentence` hierarchy graphs can outperform the original `sentence_pt` text representation.

## Goal

We wanted to turn the hierarchy graph branch into a meaningful improvement direction for BRCA stage classification, instead of only treating graph-derived tensors as another text sequence.

The exploration proceeded in four steps:

1. `hierarchy_graph + section_features`
2. `hierarchy_graph + node_features`
3. `hierarchy_graph + node_features + explicit graph structure`
4. Conservative graph encoder variants:
   - reduced graph depth with weak residual graph mixing
   - `parent-only` edges

## Experiment Summary

| Setting | Splits | sentence\_pt ACC / AUC | hierarchy\_graph ACC / AUC | Takeaway |
|---|---:|---|---|---|
| `section_features` | 5 | `0.7376 / 0.7489` | `0.7034 / 0.6521` | Too much information is lost after section pooling. |
| `node_features` | 5 | `0.7385 / 0.7444` | `0.7541 / 0.7296` | Best non-graph hierarchy variant so far; ACC is competitive, AUC still trails sentence baseline. |
| `node_features + explicit graph structure` | 10 | `0.7561 / 0.7427` | `0.5927 / 0.6571` | Unstable. Several splits collapse badly. |
| Conservative graph encoder | 3 | `0.7268 / 0.7390` | `0.7106 / 0.6273` | Stability improves, but AUC is still clearly behind. |
| Conservative `parent-only` graph | 3 | `0.7268 / 0.7390` | `0.6813 / 0.6298` | Removing `next` edges does not solve the gap. |

## Source Files

Tracked lightweight summaries:

- `section_features`: [brca_text_mode_comparison_5splits](D:\Tasks\isbi_code\experiment_records\brca_text_mode_comparison_5splits)
- `node_features`: [brca_text_mode_comparison_5splits_node_features](D:\Tasks\isbi_code\experiment_records\brca_text_mode_comparison_5splits_node_features)

Local raw experiment outputs:

- explicit graph structure, 10 splits:
  - [comparison_summary.json](D:\Tasks\isbi_code\experiments\BRCA\sentence-only\records\brca_text_mode_mixed_10splits_graph_structure\comparison_summary.json)
  - [comparison_results.csv](D:\Tasks\isbi_code\experiments\BRCA\sentence-only\records\brca_text_mode_mixed_10splits_graph_structure\comparison_results.csv)
- conservative graph encoder, 3 splits:
  - [comparison_summary.json](D:\Tasks\isbi_code\experiments\BRCA\sentence-only\records\brca_text_mode_mixed_3splits_graph_structure_conservative\comparison_summary.json)
  - [comparison_results.csv](D:\Tasks\isbi_code\experiments\BRCA\sentence-only\records\brca_text_mode_mixed_3splits_graph_structure_conservative\comparison_results.csv)
- conservative parent-only graph, 3 splits:
  - [comparison_summary.json](D:\Tasks\isbi_code\experiments\BRCA\sentence-only\records\brca_text_mode_mixed_3splits_graph_structure_parent_only\comparison_summary.json)
  - [comparison_results.csv](D:\Tasks\isbi_code\experiments\BRCA\sentence-only\records\brca_text_mode_mixed_3splits_graph_structure_parent_only\comparison_results.csv)

## What Failed in the First Explicit Graph Version

The first explicit graph-structure run was not uniformly poor; instead, a few splits failed badly and dragged the mean down.

Worst splits by ACC in the 10-split run:

- `split 7`: `ACC 0.2439`, `AUC 0.6921`
- `split 2`: `ACC 0.2829`, `AUC 0.6219`
- `split 4`: `ACC 0.4537`, `AUC 0.6748`
- `split 3`: `ACC 0.5415`, `AUC 0.6595`

Worst splits by AUC:

- `split 0`: `ACC 0.7122`, `AUC 0.5837`
- `split 2`: `ACC 0.2829`, `AUC 0.6219`
- `split 1`: `ACC 0.7659`, `AUC 0.6369`

This pattern suggests that the issue was not simply “all graph runs are weak.” Instead, the explicit graph encoder disturbed representation stability and calibration, especially on a subset of splits.

## What the Conservative Fix Achieved

To reduce instability, the graph encoder was changed to a more conservative regime:

- graph depth reduced from `2` to `1`
- dropout reduced from `0.10` to `0.05`
- graph updates applied as a small residual correction on top of the original node features
- document pooling biased toward the document node representation

This helped recover the worst collapses:

- `split 0`: `ACC 0.7122 -> 0.7610`, `AUC 0.5837 -> 0.6506`
- `split 2`: `ACC 0.2829 -> 0.6439`, `AUC 0.6219 -> 0.5977`

So the conservative variant clearly improved stability. However, its AUC still remained well below `sentence_pt`.

## Did Dropping `next` Edges Help?

No, not materially.

The `parent-only` variant produced:

- `ACC = 0.6813`
- `AUC = 0.6298`

Compared with the conservative `parent + next` variant:

- `ACC = 0.7106`
- `AUC = 0.6273`

This means removing `next` edges did not solve the main problem. The bottleneck is likely not the sequential edge type itself.

## Current Practical Conclusion

For BRCA, the current ranking is:

1. `sentence_pt` remains the strongest and safest main setting.
2. `hierarchy_graph + node_features` is the best hierarchy-based variant so far, but still does not beat the sentence baseline on AUC.
3. Explicit graph-structure encoding is not yet ready to replace the sentence baseline.

In other words, hierarchy graphs are already usable as an exploratory branch, but they are not yet a clear performance gain for BRCA under the current training design.

## Recommended Next Steps

If we continue improving the hierarchy-graph direction, the next changes should be more structural, not just small message-passing tweaks:

- try dual-path text fusion:
  - original `sentence_pt`
  - hierarchy-graph branch
- strengthen node-type-aware pooling instead of only relation updates
- consider text-only graph pretraining before multimodal fusion
- revisit how text alignment losses interact with graph-updated representations

Until then, BRCA main experiments should continue to use `sentence_pt`.
