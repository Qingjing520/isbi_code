# Pathology Report Workflow

The current report pipeline turns raw pathology reports into sentence features,
hierarchy graphs, ontology concept graphs, and training manifests.

```mermaid
flowchart TD
    A["Raw pathology reports<br/>F:\\Tasks\\Pathology Report"] --> B["Preprocess<br/>text/OCR, cleaning, section split"]
    B --> C["Sentence export<br/>sentence_to_section + section spans"]
    C --> D["Ontology concept extraction<br/>NCIt + DO"]
    C --> E["CONCH sentence encoding<br/>512-d sentence features"]
    D --> F["Sentence + concept auxiliary graphs"]
    E --> G["Report hierarchy graph library<br/>basic / ontology concept / word-node variants"]
    F --> H["Training manifests"]
    G --> H
    H --> I["Downstream multimodal training"]
```

## Main Outputs

- `Output/pathology_report_preprocessed_masked/`
- `Output/sentence_exports_masked/`
- `Output/concept_annotations_masked/`
- `Output/sentence_embeddings_conch_masked/`
- `F:\Tasks\Pathology_Report_Hierarchy_Graphs\<BRCA|KIRC|LUSC>\basic_hierarchy`
- `F:\Tasks\Pathology_Report_Hierarchy_Graphs\<BRCA|KIRC|LUSC>\ontology_concept_hierarchy`
- `F:\Tasks\Pathology_Report_Hierarchy_Graphs\<BRCA|KIRC|LUSC>\stage_keyword_word_hierarchy`
- `F:\Tasks\Pathology_Report_Hierarchy_Graphs\<BRCA|KIRC|LUSC>\stage_keyword_word_ontology_hierarchy`
- `Output/manifests/`

Ontology ablation outputs live under `Output/concept_annotations_ablation/` and
the corresponding graph folders.

## Current Modeling Use

The training side treats the original sentence feature branch as the primary
text signal. Hierarchy graphs and ontology concept graphs are auxiliary
enhancements, not replacements for the sentence-only baseline.
