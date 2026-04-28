# 病理报告文本结构与医学知识增强实验进展报告

生成时间：2026-04-28 13:48

## 一、研究目标

本阶段工作的目标是在原有病理报告 sentence-level 表示基础上，引入医学 ontology/concept graph 与 Document -> Section -> Sentence 层次结构，评估医学知识增强和文档结构增强是否能够提升 BRCA、KIRC、LUSC 三个癌症病理报告任务中的泛化性能。核心原则是保留原始 sentence 分支作为主语义分支，ontology 和 hierarchy 仅作为受控辅助分支。

## 二、采用的方法

1. 句子级表示（sentence-only）：沿用论文中的 sentence_pt 思路，将病理报告切分为句子，并使用 CONCH 文本编码得到句子特征；模型主要依赖原始文本语义。
2. 句子级表示 + 医学知识增强（sentence + ontology）：在 sentence 分支之外增加 concept graph 辅助分支。本轮 30-split 实验使用的是 NCIt+DO compact ontology，其中 NCIt 负责肿瘤相关核心标准化，DO 负责疾病层级补充。
3. 句子级表示 + 文档结构增强（sentence + hierarchical graph）：将病理报告组织为 Document、Section、Sentence 三层结构，并加入 section title/section role 信息，使模型获得句子所处上下文位置。
4. 句子级表示 + 结构/知识联合增强（sentence + hierarchical graph + ontology）：在层次图基础上进一步加入 concept 节点和 ontology 边，形成 Document -> Section -> Sentence -> Concept 的联合图。

融合方式采用 dual_text 残差门控：sentence 分支保持主导，graph 分支通过受限权重注入，当前 graph branch 最大权重约束为 0.2，并加入 gate regularization。

## 三、实验设置

- 数据集：BRCA、KIRC、LUSC。
- 评价指标：AUC 作为主要指标，ACC 作为辅助指标。
- ontology 版本：本轮主实验使用 `ncit_do`，即 NCIt+DO compact ontology。SNOMED CT 与 UMLS 已适配进资源构建流程，但未进入本轮 30-split 主结果。
- BRCA baseline：sentence-only 为本轮新跑 30 splits；增强模式均为本轮 30 splits。
- KIRC baseline：sentence-only 使用历史 run_150splits 的前 30 splits 做公平对齐；增强模式为本轮 30 splits。
- LUSC baseline：sentence-only 使用历史 LUSC_150splits_sentence_only 的前 30 splits 做公平对齐；增强模式为本轮 30 splits。
- 额外参考：LUSC 历史 sentence-only 150 splits: AUC 0.6865 ± 0.0529, ACC 0.7854 ± 0.0505。

## 四、总体实验结果

| 数据集 | sentence-only AUC | sentence+ontology AUC | sentence+hierarchy AUC | sentence+hierarchy+ontology AUC |
| --- | ---: | ---: | ---: | ---: |
| BRCA | 0.7564 ± 0.0352 | 0.7131 ± 0.0521 | 0.7394 ± 0.0432 | 0.7334 ± 0.0466 |
| KIRC | 0.8710 ± 0.0217 | 0.8770 ± 0.0261 | 0.8714 ± 0.0269 | 0.8700 ± 0.0252 |
| LUSC | 0.6979 ± 0.0522 | 0.6789 ± 0.0505 | 0.7098 ± 0.0416 | 0.6899 ± 0.0451 |

| 数据集 | sentence-only ACC | sentence+ontology ACC | sentence+hierarchy ACC | sentence+hierarchy+ontology ACC |
| --- | ---: | ---: | ---: | ---: |
| BRCA | 0.7462 ± 0.0575 | 0.6813 ± 0.1500 | 0.7122 ± 0.1214 | 0.7472 ± 0.0316 |
| KIRC | 0.7717 ± 0.0497 | 0.7748 ± 0.0613 | 0.7698 ± 0.0539 | 0.7779 ± 0.0443 |
| LUSC | 0.7855 ± 0.0456 | 0.7784 ± 0.1136 | 0.7856 ± 0.1020 | 0.7971 ± 0.0454 |

## 五、相对 sentence-only 的 paired delta

| 数据集 | 增强模式 | 共同 split 数 | ΔAUC | AUC 提升 split | ΔACC |
| --- | --- | ---: | ---: | ---: | ---: |
| BRCA | 句子级表示 + 医学知识增强（sentence + ontology） | 30 | -0.0432 ± 0.0564 | 8/30 | -0.0649 ± 0.1720 |
| BRCA | 句子级表示 + 文档结构增强（sentence + hierarchical graph） | 30 | -0.0169 ± 0.0477 | 12/30 | -0.0340 ± 0.1436 |
| BRCA | 句子级表示 + 结构/知识联合增强（sentence + hierarchical graph + ontology） | 30 | -0.0229 ± 0.0383 | 11/30 | +0.0010 ± 0.0546 |
| KIRC | 句子级表示 + 医学知识增强（sentence + ontology） | 30 | +0.0060 ± 0.0265 | 17/30 | +0.0031 ± 0.0609 |
| KIRC | 句子级表示 + 文档结构增强（sentence + hierarchical graph） | 30 | +0.0004 ± 0.0259 | 16/30 | -0.0019 ± 0.0685 |
| KIRC | 句子级表示 + 结构/知识联合增强（sentence + hierarchical graph + ontology） | 30 | -0.0010 ± 0.0227 | 15/30 | +0.0062 ± 0.0792 |
| LUSC | 句子级表示 + 医学知识增强（sentence + ontology） | 30 | -0.0190 ± 0.0624 | 13/30 | -0.0071 ± 0.1324 |
| LUSC | 句子级表示 + 文档结构增强（sentence + hierarchical graph） | 30 | +0.0120 ± 0.0560 | 15/30 | +0.0001 ± 0.1083 |
| LUSC | 句子级表示 + 结构/知识联合增强（sentence + hierarchical graph + ontology） | 30 | -0.0080 ± 0.0693 | 14/30 | +0.0116 ± 0.0648 |

## 六、分数据集结果分析

### 1. BRCA

BRCA 上 sentence-only 最强，AUC 为 0.7564 ± 0.0352。ontology 分支下降最明显，ΔAUC 为 -0.0432，只有 8/30 个 split 提升；hierarchy 分支下降较小，ΔAUC 为 -0.0169，12/30 个 split 提升；联合图也未超过 baseline。
这一现象说明当前 BRCA 中 NCIt+DO concept graph 没有稳定捕获分期相关判别信息。BRCA 报告中大量概念可能集中在肿瘤类型、解剖部位、常规病理描述上，这些概念并不一定直接对应 stage 标签；同时 concept 匹配与 true-path 祖先仍可能引入泛化节点，导致排序能力下降。

### 2. KIRC

KIRC 上 sentence+ontology 略高，AUC 为 0.8770 ± 0.0261，相对历史 sentence-only 前 30 splits 的 ΔAUC 为 +0.0060，17/30 个 split 提升。hierarchy 基本持平，joint 没有继续提升。
KIRC 与肾癌、透明细胞癌、肾脏解剖位置等 NCIt/DO 概念更直接相关，因此 ontology 有轻微帮助。但 gate 分析显示 KIRC 的 graph branch 平均权重很低，说明模型大多自动回避图分支，只在少量样本中利用知识补充，因此提升幅度有限。

### 3. LUSC

LUSC 上 hierarchy 是最有价值的增强，AUC 为 0.7098 ± 0.0416，相对 sentence-only 前 30 splits 的 ΔAUC 为 +0.0120。ontology 单独下降，joint 也下降，说明 ontology 噪声抵消了 hierarchy 的收益。
LUSC 报告可能更依赖 section role 和上下文组织，例如 diagnosis、clinical history、gross description、microscopic description 等区域对标签信息的贡献不同。层次图能帮助模型区分句子所在位置；但当前 ontology 仍偏泛化，加入后会稀释结构信号。

## 七、graph branch 使用情况

| 数据集 | 增强模式 | n | graph branch 平均权重 | 训练中 graph_weight 平均值 |
| --- | --- | ---: | ---: | ---: |
| BRCA | 句子级表示 + 医学知识增强（sentence + ontology） | 30 | 0.0855 | 0.0796 |
| BRCA | 句子级表示 + 文档结构增强（sentence + hierarchical graph） | 30 | 0.0730 | 0.0778 |
| BRCA | 句子级表示 + 结构/知识联合增强（sentence + hierarchical graph + ontology） | 30 | 0.0966 | 0.0997 |
| KIRC | 句子级表示 + 医学知识增强（sentence + ontology） | 30 | 0.0177 | 0.0195 |
| KIRC | 句子级表示 + 文档结构增强（sentence + hierarchical graph） | 30 | 0.0291 | 0.0327 |
| KIRC | 句子级表示 + 结构/知识联合增强（sentence + hierarchical graph + ontology） | 30 | 0.0129 | 0.0184 |
| LUSC | 句子级表示 + 医学知识增强（sentence + ontology） | 30 | 0.1125 | 0.1102 |
| LUSC | 句子级表示 + 文档结构增强（sentence + hierarchical graph） | 30 | 0.0996 | 0.0924 |
| LUSC | 句子级表示 + 结构/知识联合增强（sentence + hierarchical graph + ontology） | 30 | 0.1256 | 0.1197 |

从 gate 结果看，sentence 分支始终是主分支。KIRC 中 graph branch 权重最低，因此增强分支影响很小；BRCA 和 LUSC 中图分支权重更高，因此图分支质量会更直接影响最终结果。

## 八、结果原因总结

1. 当前 ontology 使用的是 NCIt+DO compact，并没有在主实验中引入 SNOMED CT/UMLS，因此目前只能评价 NCIt+DO 这一档，不能否定 SNOMED/UMLS 的潜在价值。
2. ontology 图的概念覆盖和标签判别目标之间仍有错位：概念能描述疾病和病理实体，但不一定能描述分期、侵犯、转移、淋巴结等真正与 stage 相关的证据。
3. hierarchy 的有效性依赖 section role。LUSC 对文档结构更敏感，BRCA 中层次聚合可能放大中性 section 或无关句子。
4. joint 模式没有加成，说明当前 ontology 与 hierarchy 不是简单互补关系；如果 ontology 噪声较强，它会抵消 hierarchy 提供的上下文收益。
5. residual gate 起到了保护作用，避免图分支完全覆盖 sentence 分支，但当图分支质量不足时，即使低权重也可能影响 AUC 排序。

## 九、后续改进建议

1. BRCA：暂时不要继续扩大 ontology。优先回到 sentence-only 与 section-title embedding，之后只保留 TNM、stage、grade、tumor size、lymph node、metastasis 等 typed compact concepts。
2. KIRC：可以保留 ontology 作为弱辅助，但需要做 NCIt-only、NCIt+DO、NCIt+SNOMED-mapped-to-NCIt 的小规模对照，确认轻微收益来自哪个资源。
3. LUSC：优先推进 hierarchy 方向，尤其是 section role 建模、section-title embedding、hierarchy graph auxiliary weight 的 0.05/0.10/0.20 ablation。
4. ontology 方向：不要直接 full multi-ontology。SNOMED CT 适合提高 mention recall，UMLS 适合跨术语对齐，NCIt 应继续作为肿瘤任务核心标准化，DO 只补疾病层级。
5. 图结构方向：增加 typed edge/weighted edge，降低泛化 ancestor 和高频概念的权重，并对 concept 节点使用 label embedding + mention sentence embedding 的融合表示。
6. 实验记录方向：统一把历史 sentence-only baseline 解析进 records，保证 BRCA/KIRC/LUSC 三个数据集都能直接进行 30-split paired comparison。

## 十、阶段性结论

本阶段实验表明，原始 sentence-level 文本语义仍是最可靠的基础。文档结构增强在 LUSC 上出现稳定潜力，KIRC 的 ontology 有轻微收益，而 BRCA 当前不适合直接引入 ontology 图。下一阶段应从“图分支替代/强融合”转向“受控辅助分支”，并把 ontology 从全量概念图收紧为 compact、typed、weighted、gated 的任务相关知识。
