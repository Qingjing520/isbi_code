from __future__ import annotations

"""Shared defaults for the pathology report pipeline.

Keep paths and stage output folder names here so individual stage scripts do
not drift apart over time.
"""

from pathlib import Path


# Keep historical defaults anchored at pathology_report_extraction/, even
# though this module now lives in pathology_report_extraction/common/.
PIPELINE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PIPELINE_DIR.parent
TASKS_ROOT = Path(r"F:\Tasks")

DEFAULT_PIPELINE_CONFIG = PIPELINE_DIR / "config" / "pipeline.yaml"
DEFAULT_FILTER_MODE = "masked"

DEFAULT_INPUT_DIR = TASKS_ROOT / "Pathology Report"
DEFAULT_OUTPUT_ROOT = PIPELINE_DIR / "Output"
DEFAULT_LABEL_CSV = TASKS_ROOT / "pathologic_stage.csv"

DEFAULT_ONTOLOGY_ROOT = TASKS_ROOT / "Ontologies"
DEFAULT_ONTOLOGY_RAW_ROOT = DEFAULT_ONTOLOGY_ROOT / "raw"
DEFAULT_ONTOLOGY_PROCESSED_DIR = DEFAULT_ONTOLOGY_ROOT / "processed"
DEFAULT_ONTOLOGY_ABLATION_DIR = DEFAULT_ONTOLOGY_PROCESSED_DIR / "ablations"

DEFAULT_CONCH_REPO_DIR = TASKS_ROOT / "CLAM-master"
DEFAULT_CONCH_CKPT_PATH = DEFAULT_CONCH_REPO_DIR / "ckpts" / "pytorch_model.bin"

PREPROCESS_OUTPUT_SUBDIRS = {
    "masked": "pathology_report_preprocessed_masked",
    "no_diagnosis": "pathology_report_preprocessed_no_diagnosis",
    "no_diagnosis_masked": "pathology_report_preprocessed_no_diagnosis_masked",
    "full": "pathology_report_preprocessed_full",
}

SENTENCE_EXPORT_OUTPUT_SUBDIRS = {
    "masked": "sentence_exports_masked",
    "no_diagnosis": "sentence_exports_no_diagnosis",
    "no_diagnosis_masked": "sentence_exports_no_diagnosis_masked",
    "full": "sentence_exports_full",
}

CONCEPT_OUTPUT_SUBDIRS = {
    "masked": "concept_annotations_masked",
    "no_diagnosis": "concept_annotations_no_diagnosis",
    "no_diagnosis_masked": "concept_annotations_no_diagnosis_masked",
    "full": "concept_annotations_full",
}

CONCH_OUTPUT_SUBDIRS = {
    "masked": "sentence_embeddings_conch_masked",
    "no_diagnosis": "sentence_embeddings_conch_no_diagnosis",
    "no_diagnosis_masked": "sentence_embeddings_conch_no_diagnosis_masked",
    "full": "sentence_embeddings_conch_full",
}

GRAPH_OUTPUT_SUBDIRS = {
    "masked": "text_hierarchy_graphs_masked",
    "no_diagnosis": "text_hierarchy_graphs_no_diagnosis",
    "no_diagnosis_masked": "text_hierarchy_graphs_no_diagnosis_masked",
    "full": "text_hierarchy_graphs_full",
}

CONCEPT_GRAPH_OUTPUT_SUBDIRS = {
    "masked": "text_concept_graphs_masked",
    "no_diagnosis": "text_concept_graphs_no_diagnosis",
    "no_diagnosis_masked": "text_concept_graphs_no_diagnosis_masked",
    "full": "text_concept_graphs_full",
}

SENTENCE_ONTOLOGY_GRAPH_OUTPUT_SUBDIRS = {
    "masked": "text_sentence_ontology_graphs_masked",
    "no_diagnosis": "text_sentence_ontology_graphs_no_diagnosis",
    "no_diagnosis_masked": "text_sentence_ontology_graphs_no_diagnosis_masked",
    "full": "text_sentence_ontology_graphs_full",
}

DEFAULT_MANIFEST_OUTPUT_SUBDIR = "manifests"
