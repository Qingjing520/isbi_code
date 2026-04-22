# BRCA and KIRC Text Mode Comparison Conclusion

This note summarizes the current comparison between the original sentence-level text input and the newer hierarchical text-graph input.

## Compared Modes

- `sentence_pt`
  - The original text branch setting used by the earlier pipeline.
  - Each case is represented by sentence-level CONCH text features.
- `hierarchy_graph + node_features`
  - The current hierarchical setting built from `Document -> Section -> Sentence`.
  - The model receives the concatenated document, section, and sentence node features.

## BRCA Result

Source:
- [BRCA node-feature summary](/D:/Tasks/isbi_code/experiment_records/brca_text_mode_comparison_5splits_node_features/comparison_summary.json)

Using 5 splits:

- `sentence_pt`
  - `ACC = 0.7385 ± 0.0380`
  - `AUC = 0.7444 ± 0.0212`
- `hierarchy_graph + node_features`
  - `ACC = 0.7541 ± 0.0175`
  - `AUC = 0.7296 ± 0.0498`

Interpretation:

- The hierarchical node feature version improved over the earlier BRCA `section_features` setting.
- Even so, `sentence_pt` still gives the better average AUC on BRCA.
- For BRCA, `sentence_pt` remains the safer main-experiment choice.

## KIRC Result

Source:
- [KIRC node-feature summary](/D:/Tasks/isbi_code/experiment_records/kirc_text_mode_comparison_20splits_node_features/comparison_summary.json)

Using 20 splits:

- `sentence_pt`
  - `ACC = 0.7824 ± 0.0446`
  - `AUC = 0.8658 ± 0.0209`
- `hierarchy_graph + node_features`
  - `ACC = 0.7468 ± 0.0568`
  - `AUC = 0.8374 ± 0.0346`

Interpretation:

- The hierarchical graph input is competitive, but it still does not surpass the original sentence-level mode.
- The gap on KIRC is clearer than on BRCA.

## Overall Conclusion

- The current hierarchical text graph pipeline is **usable and trainable**.
- However, under the **current model implementation**, it does **not yet outperform** the original `sentence_pt` text mode on either BRCA or KIRC.
- The likely reason is that the current training code still consumes `node_features` as a structured tensor sequence, but does not explicitly model:
  - `edge_index`
  - `edge_type`
  - `node_type`

## Practical Recommendation

- For the current main experiments, continue to use `sentence_pt`.
- Keep `hierarchy_graph + node_features` as the strongest hierarchical baseline so far.
- If hierarchical text is to become the main route later, the next step should be a model change that explicitly uses graph structure rather than only pooled node features.
