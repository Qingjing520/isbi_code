# 病理报告文本处理流程

本目录保存当前用于 TCGA BRCA 和 KIRC 病理报告 PDF 的正式文本流程。

当前处理顺序是：

1. PDF 预处理为 `Document -> Section -> Sentence`
2. 导出句子视图
3. 可选：抽取 ontology / concept annotation
4. 用 CONCH 做句子编码
5. 构建 `Document -> Section -> Sentence` 三层文本图
6. 可选：整理训练用文本图 manifest

核心脚本：

- [run_pipeline.py](D:\Tasks\isbi_code\pathology_report_extraction\run_pipeline.py)
- [preprocess_pathology_reports.py](D:\Tasks\isbi_code\pathology_report_extraction\preprocess_pathology_reports.py)
- [export_sentence_views.py](D:\Tasks\isbi_code\pathology_report_extraction\export_sentence_views.py)
- [extract_ontology_concepts.py](D:\Tasks\isbi_code\pathology_report_extraction\extract_ontology_concepts.py)
- [encode_sentence_exports_conch.py](D:\Tasks\isbi_code\pathology_report_extraction\encode_sentence_exports_conch.py)
- [build_text_hierarchy_graphs.py](D:\Tasks\isbi_code\pathology_report_extraction\build_text_hierarchy_graphs.py)
- [prepare_text_graph_manifest.py](D:\Tasks\isbi_code\pathology_report_extraction\prepare_text_graph_manifest.py)

配置文件：

- [config](D:\Tasks\isbi_code\pathology_report_extraction\config)
- [pipeline.yaml](D:\Tasks\isbi_code\pathology_report_extraction\config\pipeline.yaml)
- [config\README.md](D:\Tasks\isbi_code\pathology_report_extraction\config\README.md)

## 依赖安装

```powershell
D:\ProgrammeFiles\Anaconda\envs\Pytorch\python.exe -m pip install -r D:\Tasks\isbi_code\pathology_report_extraction\requirements.txt
```

主要依赖：

- `PyMuPDF`
- `rapidocr_onnxruntime`
- `opencv-python`
- `Pillow`
- `numpy`
- `torch`
- `timm`
- `transformers`
- `PyYAML`

当前流程不需要额外安装 Tesseract。

## 默认路径

- 输入 PDF 根目录：
  - `D:\Tasks\Pathology Report`
- 输出根目录：
  - `D:\Tasks\isbi_code\pathology_report_extraction\Output`
- 总控配置文件：
  - `D:\Tasks\isbi_code\pathology_report_extraction\config\pipeline.yaml`

## 推荐的一键运行方式

完整跑通整条流程，直接执行：

```powershell
D:\ProgrammeFiles\Anaconda\envs\Pytorch\python.exe D:\Tasks\isbi_code\pathology_report_extraction\run_pipeline.py --config "D:\Tasks\isbi_code\pathology_report_extraction\config\pipeline.yaml"
```

调试小样本：

```powershell
D:\ProgrammeFiles\Anaconda\envs\Pytorch\python.exe D:\Tasks\isbi_code\pathology_report_extraction\run_pipeline.py --config "D:\Tasks\isbi_code\pathology_report_extraction\config\pipeline.yaml" --limit 10
```

临时切换模式而不改 yaml：

```powershell
D:\ProgrammeFiles\Anaconda\envs\Pytorch\python.exe D:\Tasks\isbi_code\pathology_report_extraction\run_pipeline.py --config "D:\Tasks\isbi_code\pathology_report_extraction\config\pipeline.yaml" --filter_mode full
```

如果图已经构建好，想单独生成训练清单：

```powershell
D:\ProgrammeFiles\Anaconda\envs\Pytorch\python.exe D:\Tasks\isbi_code\pathology_report_extraction\prepare_text_graph_manifest.py --config "D:\Tasks\isbi_code\pathology_report_extraction\config\pipeline.yaml"
```

## 单 YAML 设计

现在所有阶段参数都放在一个文件里：

- [pipeline.yaml](D:\Tasks\isbi_code\pathology_report_extraction\config\pipeline.yaml)

文件内部已经按处理顺序排列：

1. `defaults`
2. `preprocess`
3. `export_sentence_views`
4. `extract_ontology_concepts`
5. `encode_sentence_exports_conch`
6. `build_text_hierarchy_graphs`
7. `prepare_text_graph_manifest`

每个阶段都支持：

- `enabled`
- 阶段专属参数
- `output_subdirs`

只要改这里的：

- `preprocess.filter_mode`

就可以在下面几种模式之间切换：

- `masked`
- `no_diagnosis`
- `no_diagnosis_masked`
- `full`

当前主实验推荐默认模式是：

- `masked`

## 单独运行某一步

这些脚本仍然支持 `--config`，并且可以直接读取同一个 [pipeline.yaml](D:\Tasks\isbi_code\pathology_report_extraction\config\pipeline.yaml)。

只跑预处理：

```powershell
D:\ProgrammeFiles\Anaconda\envs\Pytorch\python.exe D:\Tasks\isbi_code\pathology_report_extraction\preprocess_pathology_reports.py --config "D:\Tasks\isbi_code\pathology_report_extraction\config\pipeline.yaml"
```

只跑句子导出：

```powershell
D:\ProgrammeFiles\Anaconda\envs\Pytorch\python.exe D:\Tasks\isbi_code\pathology_report_extraction\export_sentence_views.py --config "D:\Tasks\isbi_code\pathology_report_extraction\config\pipeline.yaml"
```

只跑 concept annotation 抽取：

```powershell
D:\ProgrammeFiles\Anaconda\envs\Pytorch\python.exe D:\Tasks\isbi_code\pathology_report_extraction\extract_ontology_concepts.py --config "D:\Tasks\isbi_code\pathology_report_extraction\config\pipeline.yaml"
```

只跑 ontology audit：

```powershell
D:\ProgrammeFiles\Anaconda\envs\Pytorch\python.exe D:\Tasks\isbi_code\pathology_report_extraction\audit_ontology_concepts.py `
  --cohort_name KIRC `
  --annotation_dir "D:\Tasks\isbi_code\pathology_report_extraction\Output\concept_annotations_masked\KIRC" `
  --graph_dir "D:\Tasks\isbi_code\pathology_report_extraction\Output\text_concept_graphs_masked\KIRC" `
  --cohort_name BRCA `
  --annotation_dir "D:\Tasks\isbi_code\pathology_report_extraction\Output\concept_annotations_masked\BRCA" `
  --graph_dir "D:\Tasks\isbi_code\pathology_report_extraction\Output\text_concept_graphs_masked\BRCA" `
  --output_json "D:\Tasks\isbi_code\pathology_report_extraction\Output\ontology_audits\ncit_pathology_subset_masked_brca_kirc_summary.json"
```

只跑 CONCH 编码：

```powershell
D:\ProgrammeFiles\Anaconda\envs\Pytorch\python.exe D:\Tasks\isbi_code\pathology_report_extraction\encode_sentence_exports_conch.py --config "D:\Tasks\isbi_code\pathology_report_extraction\config\pipeline.yaml"
```

只跑层次图构建：

```powershell
D:\ProgrammeFiles\Anaconda\envs\Pytorch\python.exe D:\Tasks\isbi_code\pathology_report_extraction\build_text_hierarchy_graphs.py --config "D:\Tasks\isbi_code\pathology_report_extraction\config\pipeline.yaml"
```

只跑 manifest 生成：

```powershell
D:\ProgrammeFiles\Anaconda\envs\Pytorch\python.exe D:\Tasks\isbi_code\pathology_report_extraction\prepare_text_graph_manifest.py --config "D:\Tasks\isbi_code\pathology_report_extraction\config\pipeline.yaml"
```

## 输出目录结构

默认 `masked` 模式下，输出结构是：

```text
D:\Tasks\isbi_code\pathology_report_extraction\Output
|-- pathology_report_preprocessed_masked
|   |-- run_summary.json
|   |-- preprocess.log
|   |-- BRCA\*.json
|   |-- KIRC\*.json
|-- sentence_exports_masked
|   |-- run_summary.json
|   |-- export.log
|   |-- BRCA\*.json / *.txt
|   |-- KIRC\*.json / *.txt
|-- concept_annotations_masked
|   |-- run_summary.json
|   |-- concepts.log
|   |-- BRCA\*.json
|   |-- KIRC\*.json
|-- sentence_embeddings_conch_masked
|   |-- run_summary.json
|   |-- encode.log
|   |-- BRCA\*.pt / *.json
|   |-- KIRC\*.pt / *.json
|-- text_hierarchy_graphs_masked
|   |-- run_summary.json
|   |-- graph.log
|   |-- BRCA\*.pt / *.json
|   |-- KIRC\*.pt / *.json
|-- manifests
|   |-- text_graph_manifest_masked.csv
|   |-- text_graph_manifest_masked_summary.json
|-- pipeline_run_summary_masked.json
```

如果切换 `filter_mode`，各阶段子目录会自动切换到对应模式。

## 文件对应关系

对同一份病理报告样本：

- `Output\pathology_report_preprocessed_<mode>\<report>.json`
  - 结构化文本结果
  - 保留 `Document -> Section -> Sentence`
- `Output\sentence_exports_<mode>\<report>.json`
  - 编码前的扁平句子视图
  - 保留 `sentences`、`sentence_to_section`、section 句子范围
- `Output\sentence_embeddings_conch_<mode>\<report>.pt`
  - 该报告全部句子的 CONCH 特征
  - 一个 `.pt` 对应一整份报告，不是一个句子一个 `.pt`
  - 形状是 `(句子数, 512)`
- `Output\sentence_embeddings_conch_<mode>\<report>.json`
  - 上面同名 `.pt` 的结构说明
  - 用于把第 `i` 行特征映射回句子和 section
- `Output\concept_annotations_<mode>\<report>.json`
  - 轻量 ontology / concept 标注结果
  - 包含 direct mention、true-path 扩展概念和 concept-concept ontology 边
- `Output\ontology_audits\*.json`
  - cohort 级别 ontology 审计结果
  - 可用于看覆盖率、top concepts 和 BRCA/KIRC 的概念差异
- `Output\text_hierarchy_graphs_<mode>\<report>.pt`
  - 训练用图张量
  - `attach_concepts: true` 时会升级成 concept-enhanced graph
- `Output\text_hierarchy_graphs_<mode>\<report>.json`
  - 可读的节点和边说明

示例：

- 如果一份报告有 `65` 个句子，那么对应 CONCH `.pt` 形状就是 `(65, 512)`
- 第 `37` 行表示第 `37` 个句子的特征
- 同名 `.json` 会告诉你第 `37` 个句子属于哪个 section

## 训练前准备

用于生成训练清单的脚本：

- [prepare_text_graph_manifest.py](D:\Tasks\isbi_code\pathology_report_extraction\prepare_text_graph_manifest.py)

它会把：

- 文本图输出
- 分期标签 `case_id / slide_id / label`

整理成训练用 `manifest.csv`。

对应的数据集封装在：

- [text_graph_dataset.py](D:\Tasks\isbi_code\datasets\text_graph_dataset.py)

这个 `Dataset` 会按 manifest 中每一行加载：

- `graph_pt` 里的图张量
- `label`
- `case_id / slide_id / dataset` 等元数据

## 流程概览

完整流程图见 [workflow_overview.md](D:\Tasks\isbi_code\pathology_report_extraction\workflow_overview.md)。
