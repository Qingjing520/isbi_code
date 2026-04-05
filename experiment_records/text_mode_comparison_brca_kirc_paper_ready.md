# Text Mode Comparison: Paper-Ready Result Draft

This file rewrites the current BRCA/KIRC comparison into manuscript-style result paragraphs that can be adapted for a paper draft.

## Chinese Draft

为评估层次化病理报告表示在当前模型中的实际收益，我们比较了原始句子级文本输入 `sentence_pt` 与基于 `Document -> Section -> Sentence` 构建的层次图文本输入 `hierarchy_graph + node_features`。在 BRCA 数据集上，我们基于 5 组划分进行对比。结果显示，`sentence_pt` 的平均 AUC 为 `0.7444 +/- 0.0212`，平均 ACC 为 `0.7385 +/- 0.0380`；`hierarchy_graph + node_features` 的平均 AUC 为 `0.7296 +/- 0.0498`，平均 ACC 为 `0.7541 +/- 0.0175`。这说明层次图特征在 BRCA 上已经具备可训练性，并且在准确率上略有提升，但在更关键的 AUC 指标上仍略低于原始句子级方案。

在 KIRC 数据集上，我们进一步基于 20 组划分进行了更充分的比较。结果显示，`sentence_pt` 的平均 AUC 为 `0.8658 +/- 0.0209`，平均 ACC 为 `0.7824 +/- 0.0446`；`hierarchy_graph + node_features` 的平均 AUC 为 `0.8374 +/- 0.0346`，平均 ACC 为 `0.7468 +/- 0.0568`。与 BRCA 类似，层次图方案能够稳定完成训练并取得具有竞争力的结果，但整体性能仍未超过原始句子级文本模式，而且在 KIRC 上这一性能差距更为明显。

综合 BRCA 和 KIRC 的结果可以看出，当前层次图文本分支已经能够作为一个有效的结构化文本基线，但在现有模型框架下尚未优于原始 `sentence_pt` 输入。我们认为，造成这一现象的一个重要原因在于，当前训练流程虽然使用了层次图节点特征，但仍将其作为序列化特征张量输入模型，而未显式利用 `edge_index`、`edge_type` 和 `node_type` 等图结构信息。因此，现阶段更合理的实验结论是：`sentence_pt` 仍应作为主实验文本设置，而 `hierarchy_graph + node_features` 可作为最强的层次化文本基线；若后续希望进一步释放层次图表示的优势，则需要在模型层面显式引入图结构建模。

## English Draft

To assess whether hierarchical pathology-report representations provide practical benefits under the current model, we compared the original sentence-level text input (`sentence_pt`) with a hierarchical text-graph input (`hierarchy_graph + node_features`) constructed from the `Document -> Section -> Sentence` structure. On BRCA, we evaluated both settings over 5 splits. The original `sentence_pt` mode achieved an average AUC of `0.7444 +/- 0.0212` and an average ACC of `0.7385 +/- 0.0380`, whereas `hierarchy_graph + node_features` achieved an average AUC of `0.7296 +/- 0.0498` and an average ACC of `0.7541 +/- 0.0175`. These results indicate that the hierarchical graph features are trainable on BRCA and slightly improve accuracy, but they still underperform the original sentence-level setting on AUC.

We further conducted a larger comparison on KIRC using 20 splits. In this setting, `sentence_pt` achieved an average AUC of `0.8658 +/- 0.0209` and an average ACC of `0.7824 +/- 0.0446`, while `hierarchy_graph + node_features` achieved an average AUC of `0.8374 +/- 0.0346` and an average ACC of `0.7468 +/- 0.0568`. Similar to BRCA, the hierarchical graph setting was stable and competitive, but it did not surpass the original sentence-level mode. Notably, the performance gap was more evident on KIRC than on BRCA.

Taken together, these findings suggest that the current hierarchical text-graph pipeline is a usable and meaningful structured-text baseline, but it does not yet outperform the original `sentence_pt` representation under the present training framework. A likely reason is that the current model still consumes hierarchical node features as serialized feature tensors, without explicitly modeling graph structure via `edge_index`, `edge_type`, or `node_type`. Therefore, at the current stage, `sentence_pt` should remain the primary text setting for the main experiments, while `hierarchy_graph + node_features` should be retained as the strongest hierarchical baseline. Future improvements will likely require explicit graph-structured modeling rather than relying only on pooled node features.

## Source Summaries

- [BRCA node-feature summary](/D:/Tasks/isbi_code/experiment_records/brca_text_mode_comparison_5splits_node_features/comparison_summary.json)
- [KIRC node-feature summary](/D:/Tasks/isbi_code/experiment_records/kirc_text_mode_comparison_20splits_node_features/comparison_summary.json)
