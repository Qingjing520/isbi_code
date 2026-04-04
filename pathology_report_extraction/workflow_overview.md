# 病理报告文本到层次图的当前流程

## 一句话概括

当前已经完成到：

`原始 PDF -> 文本提取/OCR -> 清洗 -> Section 切分 -> Sentence 切分 -> masked 过滤 -> 三层 JSON -> 句子导出 -> CONCH 句子特征 -> 第一版三层文本图`

后续还没做的是：

`图上的训练与对齐 -> section/doc pooling 训练接入 -> section-level MMD -> text-topology MMD -> 与 WSI 融合`

## 当前完整流程线

```mermaid
flowchart TD
    A["原始病理报告 PDF<br/>D:\\Tasks\\Pathology Report"] --> B["PDF 文本提取<br/>原生文本优先 + 低文本页 OCR fallback"]
    B --> C["文本清洗<br/>去页眉页脚、断行合并、噪声清理"]
    C --> D["Section 切分<br/>Gross / Microscopic / Comment ..."]
    D --> E["Sentence 切分"]
    E --> F["句子级过滤<br/>masked: keep / mask / drop"]
    F --> G["三层 JSON 输出<br/>Document -> Section -> Sentence"]
    G --> H["句子视图导出<br/>保留 sentence_to_section 与 section spans"]
    H --> I["Sentence 用 CONCH 编码<br/>得到 512 维句子特征"]
    I --> J["构建三层文本图<br/>Document / Section / Sentence"]
    J --> K["添加两类边<br/>parent + next"]
    K --> L["Section 特征<br/>对本 section 子句子 mean pooling"]
    L --> M["Document 特征<br/>对 section 特征 mean pooling"]
    M --> N["后续训练扩展<br/>section-level MMD + text-topology MMD"]
    N --> O["与 WSI 分支融合"]
    O --> P["早期 vs 晚期分期二分类<br/>Stage I/II vs Stage III/IV"]

    style G fill:#dff3e3,stroke:#2f855a,stroke-width:2px
    style H fill:#dff3e3,stroke:#2f855a,stroke-width:2px
    style I fill:#dff3e3,stroke:#2f855a,stroke-width:2px
    style J fill:#dff3e3,stroke:#2f855a,stroke-width:2px
```

## 当前已完成的目录

- 三层预处理 JSON：
  - `D:\Tasks\isbi_code\pathology_report_extraction\Output\pathology_report_preprocessed_masked`
- 句子导出结果：
  - `D:\Tasks\isbi_code\pathology_report_extraction\Output\sentence_exports_masked`
- 句子级 CONCH 特征：
  - `D:\Tasks\isbi_code\pathology_report_extraction\Output\sentence_embeddings_conch_masked`
- 第一版三层文本图：
  - `D:\Tasks\isbi_code\pathology_report_extraction\Output\text_hierarchy_graphs_masked`

## 当前图设计

第一版图刻意保持简单：

1. 节点只有三层
   - `Document`
   - `Section`
   - `Sentence`
2. 边只保留两类
   - `parent`
   - `next`
3. 特征来源保持和论文兼容
   - `Sentence`：直接使用 CONCH 512 维文本特征
   - `Section`：对子句子特征做 mean pooling
   - `Document`：对 section 特征做 mean pooling

## 下一步

最自然的下一步是：

1. 读取 `Output\text_hierarchy_graphs_masked`
2. 把 `Document / Section / Sentence` 三层表示接入文本图分支
3. 在文本图上补：
   - `section-level MMD`
   - `text-topology MMD`
4. 再与 WSI 分支融合
