# ISBI Codebase

这是当前的主项目仓库，包含两条互相关联的工作线：

1. 多模态病理分期分类训练代码  
2. 病理报告文本预处理与层次图构建流程

当前目标是支持在 TCGA-BRCA、TCGA-KIRC、TCGA-LUSC 上进行病理分期二分类实验，并在文本侧同时兼容：

- 论文原始句子级文本特征
- 当前新增的 `Document -> Section -> Sentence` 层次图文本特征
- 可选的 `Document -> Section -> Sentence -> Concept` concept-enhanced 图文本特征

## 1. 项目结构

主要目录如下：

- [configs](D:\Tasks\isbi_code\configs)：训练配置与配置解析
- [datasets](D:\Tasks\isbi_code\datasets)：数据集读取逻辑
- [models](D:\Tasks\isbi_code\models)：映射、池化、融合、分类头
- [losses](D:\Tasks\isbi_code\losses)：MMD 等损失
- [utils](D:\Tasks\isbi_code\utils)：图构建、指标、随机种子等工具
- [tools](D:\Tasks\isbi_code\tools)：辅助脚本
- [diagrams](D:\Tasks\isbi_code\diagrams)：模型流程图
- [pathology_report_extraction](D:\Tasks\isbi_code\pathology_report_extraction)：病理报告文本处理全流程
- [main.py](D:\Tasks\isbi_code\main.py)：训练/测试入口
- [train.py](D:\Tasks\isbi_code\train.py)：核心训练逻辑

## 2. 当前训练输入

当前训练代码支持四种文本输入模式。

### `sentence_pt`

论文原始方式。  
`text_dir` 中每个文件通常是一个 `case_id.pt`，张量形状一般为：

```text
(句子数, 512)
```

典型目录：

- `D:\Tasks\text_extract_features\BRCA_text`
- `D:\Tasks\text_extract_features\KIRC_text`
- `D:\Tasks\text_extract_features\LUSC_text`

### `hierarchy_graph`

当前新增方式。  
`text_dir` 中每个文件是一个层次图 `.pt`，内部是字典，包含：

- `node_features`
- `sentence_features`
- `section_features`
- `document_feature`
- `edge_index`
- `edge_type`
- `node_type`

典型目录：

- [BRCA text hierarchy graphs](D:\Tasks\isbi_code\pathology_report_extraction\Output\text_hierarchy_graphs_masked\BRCA)
- [KIRC text hierarchy graphs](D:\Tasks\isbi_code\pathology_report_extraction\Output\text_hierarchy_graphs_masked\KIRC)

说明：

- 当前训练代码已经可以读取这些层次图 `.pt`
- 但目前主要是把选定的特征张量当作文本输入序列使用
- 还没有显式利用 `edge_index / edge_type` 做真正的图神经网络学习

### `concept_graph`

在 `hierarchy_graph` 基础上继续扩展 concept 节点和 ontology-aware 边。  
这一模式面向 `Document -> Section -> Sentence -> Concept` 的混合文本图，推荐与：

- `text_use_graph_structure: true`
- `model.text_graph_num_node_types: 0`（自动推断）
- `model.text_graph_num_base_relations: 0`（自动推断）

一起使用。

## 3. 标签与划分

当前正式训练标签统一采用与 LUSC 一致的格式：

```csv
case_id,slide_id,label
```

可直接用于训练的标签文件：

- [BRCA_pathologic_stage.csv](D:\Tasks\Pathologic_Stage_Label\BRCA_pathologic_stage.csv)
- [KIRC_pathologic_stage.csv](D:\Tasks\Pathologic_Stage_Label\KIRC_pathologic_stage.csv)
- [LUSC_pathologic_stage.csv](D:\Tasks\Pathologic_Stage_Label\LUSC_pathologic_stage.csv)

150 组 train/test 划分已生成在：

- [Split_Table\BRCA](D:\Tasks\Split_Table\BRCA)
- [Split_Table\KIRC](D:\Tasks\Split_Table\KIRC)
- [Split_Table\LUSC](D:\Tasks\Split_Table\LUSC)

每组划分只有两列：

```csv
idx,train,test
```

## 4. 配置方式

训练只需要修改一个 yaml：

- [config.yaml](D:\Tasks\isbi_code\configs\config.yaml)

常改字段主要是：

- `data.split_file`
- `data.label_file`
- `data.image_dir`
- `data.text_dir`
- `data.text_mode`
- `data.text_graph_feature`
- `output.exp_dir`

其中：

- `text_mode: sentence_pt` 表示使用论文原始句子文本
- `text_mode: hierarchy_graph` 表示使用当前层次图文本

当 `text_mode: hierarchy_graph` 时，可进一步切换：

- `node_features`
- `sentence_features`
- `section_features`
- `document_feature`

## 5. 训练命令

默认训练：

```powershell
D:\ProgrammeFiles\Anaconda\envs\Pytorch\python.exe D:\Tasks\isbi_code\main.py --config D:\Tasks\isbi_code\configs\config.yaml --mode train
```

仅测试已有 checkpoint：

```powershell
D:\ProgrammeFiles\Anaconda\envs\Pytorch\python.exe D:\Tasks\isbi_code\main.py --config D:\Tasks\isbi_code\configs\config.yaml --mode test --ckpt D:\Tasks\isbi_code\experiments\your_exp\best_model.pt
```

## 6. 病理报告文本流程

病理报告相关流程都在：

- [pathology_report_extraction](D:\Tasks\isbi_code\pathology_report_extraction)

这条线当前已经完成到：

```text
PDF -> 文本清洗 -> section/sentence 切分 -> masked 过滤
-> CONCH 句子特征 -> Document/Section/Sentence 三层层次图
```

推荐查看：

- [pathology_report_extraction\README.md](D:\Tasks\isbi_code\pathology_report_extraction\README.md)
- [pathology_report_extraction\workflow_overview.md](D:\Tasks\isbi_code\pathology_report_extraction\workflow_overview.md)
- [pathology_report_extraction\cmd.txt](D:\Tasks\isbi_code\pathology_report_extraction\cmd.txt)

## 7. 当前建议

如果你现在要继续做实验，推荐顺序是：

1. 在 [configs\config.yaml](D:\Tasks\isbi_code\configs\config.yaml) 中选择数据集、文本模式和特征类型
2. 指向对应的 `split_file` 和 `label_file`
3. 准备好对应数据集的 WSI `.pt` 特征目录
4. 先跑 baseline
5. 再继续做文本层次图相关消融

当前比较值得比较的文本设置：

- `sentence_pt`
- `hierarchy_graph + section_features`
- `hierarchy_graph + node_features`

## 8. 版本管理

本仓库已经接入 GitHub：

- [Qingjing520/isbi_code](https://github.com/Qingjing520/isbi_code)

日常更新命令：

```powershell
cd /d D:\Tasks\isbi_code
git status
git add .
git commit -m "说明这次修改内容"
git push
```

已忽略本地环境和大体积输出，例如：

- `.idea/`
- `.venv_pathology/`
- `experiments/`
- [pathology_report_extraction\Output](D:\Tasks\isbi_code\pathology_report_extraction\Output)

## 9. 备注

当前代码已经兼容“句子文本特征”和“层次图文本特征”两种输入。  
如果后续要真正发挥层次图 / concept graph 优势，下一步应考虑让训练代码显式使用：

- `edge_index`
- `edge_type`
- `node_type`

也就是从“层次化特征输入”进一步推进到“真正的图结构学习”。
