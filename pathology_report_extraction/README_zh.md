# 病理报告处理流程

这个目录负责 BRCA/KIRC/LUSC 的病理报告文本处理、ontology 概念抽取、层次图构建和训练 manifest 生成。

## 主流程

1. 将病理报告整理成 `Document -> Section -> Sentence`。
2. 导出 sentence view。
3. 使用 NCIt/SNOMED/UMLS/DO 抽取和标准化 ontology concepts。
4. 用 CONCH 编码句子特征。
5. 构建文本层次图，以及可选的 sentence/concept 辅助图。
6. 生成下游训练使用的 manifest。

常用入口：

- `run_pipeline.py`：按统一 YAML 一键跑流程。
- `preprocess_pathology_reports.py`：报告抽取和清洗。
- `export_sentence_views.py`：导出 section/sentence。
- `extract_ontology_concepts.py`：抽取 ontology concepts 和 true-path 祖先。
- `build_sentence_ontology_graphs.py`：构建轻量 sentence/concept 辅助图。
- `build_text_hierarchy_graphs.py`：构建 `Document -> Section -> Sentence` 层次图。
- `prepare_text_graph_manifest.py`：生成训练 manifest。
- `prepare_stage_labels.py`：从 clinical XML 抽标签，或把 case label 展开到 split slide。
- `visualize_hierarchy_graphs.py`：可视化单个层次图或两个数据集对比。
- `pipeline_defaults.py`：统一维护默认路径、filter mode 和输出子目录名。
- `pdf_utils.py`：统一维护文件夹、PDF 和 JSON 写入工具。

## 当前路径

- Python：`F:\Anaconda\envs\pytorch\python.exe`
- 报告目录：`F:\Tasks\Pathology Report`
- 项目目录：`F:\Tasks\isbi_code`
- Ontology 目录：`F:\Tasks\Ontologies\processed`
- 输出目录：`F:\Tasks\isbi_code\pathology_report_extraction\Output`
- 配置文件：`F:\Tasks\isbi_code\pathology_report_extraction\config\pipeline.yaml`

## 常用命令

完整流程：

```powershell
F:\Anaconda\envs\pytorch\python.exe F:\Tasks\isbi_code\pathology_report_extraction\run_pipeline.py --config F:\Tasks\isbi_code\pathology_report_extraction\config\pipeline.yaml
```

小样本调试：

```powershell
F:\Anaconda\envs\pytorch\python.exe F:\Tasks\isbi_code\pathology_report_extraction\run_pipeline.py --config F:\Tasks\isbi_code\pathology_report_extraction\config\pipeline.yaml --limit 10
```

只跑 concept 抽取：

```powershell
F:\Anaconda\envs\pytorch\python.exe F:\Tasks\isbi_code\pathology_report_extraction\extract_ontology_concepts.py --config F:\Tasks\isbi_code\pathology_report_extraction\config\pipeline.yaml
```

图构建完成后生成训练 manifest：

```powershell
F:\Anaconda\envs\pytorch\python.exe F:\Tasks\isbi_code\pathology_report_extraction\prepare_text_graph_manifest.py --config F:\Tasks\isbi_code\pathology_report_extraction\config\pipeline.yaml
```

可视化一个层次图：

```powershell
F:\Anaconda\envs\pytorch\python.exe F:\Tasks\isbi_code\pathology_report_extraction\visualize_hierarchy_graphs.py single
```

可视化 BRCA/KIRC 对比：

```powershell
F:\Anaconda\envs\pytorch\python.exe F:\Tasks\isbi_code\pathology_report_extraction\visualize_hierarchy_graphs.py compare
```

## 输出结构

默认 `masked` 模式会写入：

```text
Output/
  pathology_report_preprocessed_masked/
  sentence_exports_masked/
  concept_annotations_masked/
  sentence_embeddings_conch_masked/
  text_hierarchy_graphs_masked/
  manifests/
```

ontology 消融相关结果会写入 `concept_annotations_ablation/` 以及对应 graph 输出目录，例如 `ncit_only`、`ncit_do`、`ncit_snomed_mapped`、`full_multi_ontology`。

生成数据都放在 `Output/`，不要提交到 git；新的实验调度优先使用 `F:\Tasks\isbi_code\tools` 下面的工具。
