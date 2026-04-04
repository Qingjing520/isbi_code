# 配置说明

当前 `config` 目录只保留一个总控配置文件：

- [pipeline.yaml](D:\Tasks\isbi_code\pathology_report_extraction\config\pipeline.yaml)

这个文件已经按处理顺序组织好：

1. `defaults`
2. `preprocess`
3. `export_sentence_views`
4. `encode_sentence_exports_conch`
5. `build_text_hierarchy_graphs`
6. `prepare_text_graph_manifest`

各部分含义如下：

- `defaults`
  - 放全局共享的输入输出根目录
  - 可设置全局 `limit`
- `preprocess`
  - 负责 PDF 到三层 JSON 的预处理
  - 在这里切换 `filter_mode`
- `export_sentence_views`
  - 负责把三层 JSON 导出成句子视图
- `encode_sentence_exports_conch`
  - 负责用 CONCH 编码句子
- `build_text_hierarchy_graphs`
  - 负责把句子特征进一步构造成三层文本图
- `prepare_text_graph_manifest`
  - 负责把文本图和标签整理成训练 manifest

推荐的一键命令：

```powershell
D:\ProgrammeFiles\Anaconda\envs\Pytorch\python.exe D:\Tasks\isbi_code\pathology_report_extraction\run_pipeline.py --config "D:\Tasks\isbi_code\pathology_report_extraction\config\pipeline.yaml"
```

如果只想跑某一步，也可以继续用同一个 yaml：

```powershell
python <script>.py --config "D:\Tasks\isbi_code\pathology_report_extraction\config\pipeline.yaml"
```

命令行参数仍然会覆盖 yaml 里的同名配置。
