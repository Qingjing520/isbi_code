我把这个项目整体读完了。它本质上是一个面向病理分期二分类的多模态训练工程：输入是图像 patch 特征 `.pt` 和文本特征 `.pt`，先做共享映射，再分别池化，之后用 3 个专家组成的 MoE 做融合，最后输出二分类结果；训练时还额外做源域/目标域的分布对齐。



\*\*项目结构\*\*

核心入口在 \[main.py](D:\\Tasks\\isbi\_code\\main.py#L28)，负责 `train/test` 两种模式。真正的训练逻辑在 \[train.py](D:\\Tasks\\isbi\_code\\train.py#L158)，配置解析在 \[configs/config.py](D:\\Tasks\\isbi\_code\\configs\\config.py)，当前配置文件是 \[configs/config.yaml](D:\\Tasks\\isbi\_code\\configs\\config.yaml)。数据读取在 \[datasets/feature\_dataset.py](D:\\Tasks\\isbi\_code\\datasets\\feature\_dataset.py#L8)，模型拆成 4 个文件：\[models/mapper.py](D:\\Tasks\\isbi\_code\\models\\mapper.py)、\[models/pooling.py](D:\\Tasks\\isbi\_code\\models\\pooling.py)、\[models/fusion.py](D:\\Tasks\\isbi\_code\\models\\fusion.py)、\[models/classifier.py](D:\\Tasks\\isbi\_code\\models\\classifier.py)。对齐损失在 \[losses/mmd\_loss.py](D:\\Tasks\\isbi\_code\\losses\\mmd\_loss.py)，图结构构造在 \[utils/graph\_utils.py](D:\\Tasks\\isbi\_code\\utils\\graph\_utils.py)。另外有两个辅助脚本：\[tools/make\_splits.py](D:\\Tasks\\isbi\_code\\tools\\make\_splits.py) 用来生成多个随机切分，\[auto\_train\_next.py](D:\\Tasks\\isbi\_code\\auto\_train\_next.py) 用来自动切到下一个 `split\_k.csv` 并继续训练。



\*\*训练流程\*\*

数据集类会从 `split\_x.csv` 里读取 `train/test` 列，把每个 `slide\_id` 映射到图像特征 `<slide\_id>.pt` 和文本特征 `<case\_id>.pt`，标签来自 `label\_csv` 里的 `case\_id + slide\_id + label` 三列，见 \[datasets/feature\_dataset.py](D:\\Tasks\\isbi\_code\\datasets\\feature\_dataset.py#L22)。训练时：

1\. 图像和文本序列都先过同一个 `SharedMapper`。

2\. 两路序列分别做 attention pooling，得到图像向量和文本向量。

3\. `MoEFusion` 在三种融合专家之间做路由：

&#x20;  - 拼接后自注意力

&#x20;  - 相加后自注意力

&#x20;  - 双向 cross-attention

4\. 分类损失只用源域训练集标签。

5\. 从 warmup 之后开始加入 3 个 MMD 对齐项：

&#x20;  - 文本均值特征对齐

&#x20;  - 图像节点特征对齐

&#x20;  - 图拓扑统计对齐



这个思路都集中在 \[train.py](D:\\Tasks\\isbi\_code\\train.py#L249) 到 \[train.py](D:\\Tasks\\isbi\_code\\train.py#L340)。图像图结构的做法是：先对 patch 特征做 K-means 得到固定数量节点，再按余弦相似度构图，再统计 1/2/3-hop 邻居数作为拓扑特征，见 \[utils/graph\_utils.py](D:\\Tasks\\isbi\_code\\utils\\graph\_utils.py)。



\*\*当前状态\*\*

当前配置指向 `split\_146`，数据路径是外部绝对路径，见 \[configs/config.yaml](D:\\Tasks\\isbi\_code\\configs\\config.yaml)。项目里已经有 147 个实验目录 `experiments/split\_0` 到 `experiments/split\_146`。最近这个 split 的日志显示前 3 轮是纯分类 warmup，第 4 轮开始加对齐损失；目前我看到的日志前几轮里，`epoch 4` 的目标域 `AUC` 到了约 `0.6737`，日志在 \[experiments/split\_146/log.jsonl](D:\\Tasks\\isbi\_code\\experiments\\split\_146\\log.jsonl)。



\*\*几个值得注意的点\*\*

\[train.py](D:\\Tasks\\isbi\_code\\train.py#L359) 直接在 `target\_loader` 上做每轮评估、早停和最佳模型选择，这从研究角度看相当于把测试集当验证集用了，会有信息泄漏风险。  

\[datasets/feature\_dataset.py](D:\\Tasks\\isbi\_code\\datasets\\feature\_dataset.py#L18) 虽然保留了 `val` 模式，但当前 dataloader 已经把 `val` 整段注释掉了。  

\[configs/config.yaml](D:\\Tasks\\isbi\_code\\configs\\config.yaml) 里 `output.save\_best\_by: val\_avg`，但训练代码里其实没有用这个字段，实际始终按 `target auc/acc` 选 best。  

\[main.py](D:\\Tasks\\isbi\_code\\main.py#L22) 默认 checkpoint 路径写的是 `experiments/split0/best\_model.pt`，而目录命名实际是 `split\_0`，默认值有小 bug。  

代码里有一些中文注释和打印出现乱码，像 \[train.py](D:\\Tasks\\isbi\_code\\train.py#L382) 和 \[auto\_train\_next.py](D:\\Tasks\\isbi\_code\\auto\_train\_next.py)，大概率是文件编码不一致。  

\[utils/seed.py](D:\\Tasks\\isbi\_code\\utils\\seed.py) 和 \[train.py](D:\\Tasks\\isbi\_code\\train.py#L22) 各有一份 `set\_seed`，目前是重复实现，训练里也没复用 `utils` 版本。



如果你愿意，我下一步可以继续帮你做两件事之一：  

1\. 画一张“从输入到 loss 的模型流程图”。  

2\. 基于这次阅读，给你出一份“项目重构/改进清单”。

