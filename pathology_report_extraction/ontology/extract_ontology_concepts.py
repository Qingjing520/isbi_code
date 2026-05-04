from __future__ import annotations

"""Extract lightweight ontology concept annotations from sentence-export JSON files.

This stage is designed as a practical bridge between the existing sentence-view
pipeline and later concept-graph construction. It intentionally keeps
dependencies light so the project can start running concept-aware experiments
without requiring a heavyweight ontology linker at training time.

Supported ontology file shape:

1. {"concepts": {"C001": {"name": "...", "synonyms": ["..."], "parents": ["C000"], "ic": 2.3}}}
2. {"concepts": [{"id": "C001", "name": "...", "synonyms": [...], "parents": [...], "ic": 2.3}]}
3. [{"id": "C001", "name": "...", "synonyms": [...], "parents": [...], "ic": 2.3}]

When no ontology file is provided, the script falls back to a small built-in
pathology concept set so the concept-graph workflow can still be exercised end
to end.
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

from pathology_report_extraction.common.pdf_utils import ensure_dir, write_json
from pathology_report_extraction.common.pipeline_defaults import CONCEPT_OUTPUT_SUBDIRS, DEFAULT_OUTPUT_ROOT, SENTENCE_EXPORT_OUTPUT_SUBDIRS
from pathology_report_extraction.config.config import get_bool, get_path, get_stage_config, get_value, load_yaml_config


LOGGER = logging.getLogger("extract_ontology_concepts")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
DOMAIN_CONCEPT_NAME_CUES = (
    "cancer",
    "carcinoma",
    "tumor",
    "tumour",
    "neoplasm",
    "renal",
    "kidney",
    "breast",
    "ductal",
    "lobular",
    "clear cell",
    "papillary",
    "chromophobe",
    "oncocyt",
    "metasta",
    "invasion",
    "invasive",
    "necrosis",
    "margin",
    "lymph",
    "node",
    "nodal",
    "ureter",
    "adrenal",
    "pelvis",
    "vein",
    "capsule",
    "nephrectomy",
    "mastectomy",
    "resection",
    "biopsy",
    "histolog",
    "nuclear",
    "grade",
    "stage",
    "tnm",
    "receptor",
    "estrogen",
    "progesterone",
    "her2",
    "er positive",
    "pr positive",
    "sarcomatoid",
    "rhabdoid",
    "lesion",
    "mass",
)
CONCEPT_NAME_EXCLUDE_CUES = (
    "sample",
    "terminology",
    "question",
    "indicator",
    "dose form",
    "pathway",
    "criteria",
    "measurement",
    "unit",
    "report",
    "protocol",
    "allele",
    "gene",
    "protein",
    "murine",
    "mouse",
    "domain",
    "scale",
    "drug component",
    "molecular mass",
)
GENERIC_TERM_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "m",
    "n",
    "no",
    "not",
    "of",
    "on",
    "or",
    "r",
    "s",
    "t",
    "the",
    "to",
    "with",
    "within",
    "up",
    "age",
    "sex",
    "final",
    "report",
    "received",
    "submitted",
    "identified",
    "labeled",
    "labelled",
    "patient",
    "section",
    "sectioning",
    "block",
    "representative",
    "gross",
    "location",
    "laboratory",
    "accession",
    "mask",
    "mass",
    "capsule",
    "present",
    "pathologic",
    "left",
    "right",
    "multiple",
    "other",
    "spleen",
    "uninvolved",
}
PATHOLOGY_SINGLE_TOKEN_WHITELIST = {
    "breast",
    "kidney",
    "renal",
    "tumor",
    "tumour",
    "cancer",
    "margin",
    "margins",
    "necrosis",
    "invasion",
    "metastasis",
    "metastatic",
    "ureter",
    "pelvis",
    "adrenal",
    "vein",
    "papillary",
    "lobular",
    "ductal",
    "vascular",
    "lymph",
    "nodal",
    "node",
}
PATHOLOGY_SINGLE_TOKEN_ABBREVIATIONS = {
    "ccrcc",
    "chrcc",
    "dcis",
    "her2",
    "idc",
    "ilc",
    "lcis",
    "prcc",
    "rcc",
    "tnm",
}
TRUE_PATH_ANCESTOR_EXACT_EXCLUDE_NAMES = {
    "action",
    "activity",
    "abdominal aorta branch",
    "ablation therapy",
    "ablative endocrine surgery",
    "adverse event classification",
    "anatomic site",
    "anatomic pathology procedure",
    "anatomic structure",
    "anatomical structure",
    "anatomic structure, system, or substance",
    "anatomical structure, system, or substance",
    "anatomy qualifier",
    "assessment",
    "antibody",
    "biological process",
    "biological sciences",
    "biology",
    "biospecimen type",
    "biopsy procedure",
    "biopsy procedure by anatomic location",
    "blood vessel",
    "body part",
    "body cavity",
    "cancer biology",
    "cancer histology",
    "breast disorder",
    "breast cancer diagnostic or therapeutic procedure",
    "breast cancer therapeutic procedure",
    "breast surgery",
    "cancer diagnostic or therapeutic procedure",
    "cancer therapeutic procedure",
    "cell",
    "classification",
    "classification of event",
    "clinical or research assessment answer",
    "clinical evaluation",
    "clinical intervention or procedure",
    "clinical or research activity",
    "clinical course of disease",
    "clinical test result",
    "clavien-dindo classification",
    "conceptual entity",
    "cptac responses",
    "diagnostic procedure",
    "diagnostic or prognostic factor",
    "disease progression",
    "disease grade qualifier",
    "disease involvement site",
    "disease morphology qualifier",
    "disease qualifier",
    "disease, disorder or finding",
    "disease or disorder",
    "disease response",
    "disorder by site",
    "endocrine gland",
    "epithelial cell",
    "focality",
    "finding",
    "finding by site or system",
    "general qualifier",
    "gene product",
    "gland",
    "glandular cell",
    "histological procedure",
    "histopathology result",
    "intellectual property",
    "immunoprotein",
    "information",
    "kidney part",
    "laboratory procedure",
    "laboratory test result",
    "laparoscopic surgery",
    "lesion size",
    "ligand binding protein",
    "lipid",
    "lymph node biopsy",
    "mass",
    "medical examination assessment",
    "medical status",
    "microanatomic structure",
    "morphologic finding",
    "negative test result",
    "needle biopsy",
    "neoplasm by special category",
    "non-neoplastic breast disorder",
    "non-neoplastic disorder",
    "number",
    "other anatomic concept",
    "occupation or discipline",
    "organic chemical",
    "organ",
    "organ capsule",
    "organ state",
    "pathologic process",
    "pathology result",
    "physiology-regulatory factor",
    "progestogen",
    "progressive disease",
    "progressive neoplastic disease",
    "protein",
    "protein, organized by function",
    "property or attribute",
    "qualifier",
    "ring compound",
    "sample type",
    "size",
    "spatial anatomic qualifier",
    "spatial qualifier",
    "specimen anatomic site",
    "special histology staining method",
    "status",
    "staining method",
    "steroid compound",
    "steroid hormone",
    "substance",
    "surgical procedure",
    "surgical procedure by site or system",
    "temporal qualifier",
    "therapeutic procedure",
    "renal tissue",
    "tissue",
    "tumor event information",
    "tumor-associated process",
    "tumor progression",
    "type",
    "urologic surgical procedure",
    "urogenital surgical procedure",
    "urinary system finding",
    "urinary system part",
    "vein",
    "artery",
    "drug, food, chemical or biomedical material",
    "drug or chemical by structure",
    "heterocyclic compound",
    "hormone",
    "cancer progression",
}
TRUE_PATH_ANCESTOR_EXCLUDE_CUES = (
    "anatomic qualifier",
    "anatomic site",
    "anatomic concept",
    "anatomic pathology procedure",
    "assessment",
    "by anatomic site",
    "biopsy procedure",
    "biospecimen",
    "chemical",
    "classification",
    "diagnostic or therapeutic procedure",
    "diagnostic procedure",
    "clinical intervention or procedure",
    "clinical or research activity",
    "clinical or research assessment",
    "clinical evaluation",
    "disease, disorder or finding",
    "disease or disorder",
    "disease morphology qualifier",
    "disease grade qualifier",
    "disease qualifier",
    "disease progression",
    "examination",
    "finding by",
    "generic qualifier",
    "hormone",
    "laboratory procedure",
    "laboratory test result",
    "clinical test result",
    "anatomic structure, system, or substance",
    "non-neoplastic disorder",
    "organized by function",
    "status",
    "specimen anatomic",
    "steroid",
    "staining method",
    "surgical procedure",
    "system part",
)

DEFAULT_INPUT_DIR = DEFAULT_OUTPUT_ROOT / SENTENCE_EXPORT_OUTPUT_SUBDIRS["masked"]
DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / CONCEPT_OUTPUT_SUBDIRS["masked"]

DEFAULT_ONTOLOGY_CONCEPTS: list[dict[str, Any]] = [
    {"id": "ONT:NEOPLASM", "name": "Neoplasm", "synonyms": ["neoplasm", "tumor", "tumour"], "parents": [], "ic": 1.1},
    {"id": "ONT:CARCINOMA", "name": "Carcinoma", "synonyms": ["carcinoma"], "parents": ["ONT:NEOPLASM"], "ic": 1.7},
    {
        "id": "ONT:ADENOCARCINOMA",
        "name": "Adenocarcinoma",
        "synonyms": ["adenocarcinoma"],
        "parents": ["ONT:CARCINOMA"],
        "ic": 2.2,
    },
    {
        "id": "ONT:SQUAMOUS_CELL_CARCINOMA",
        "name": "Squamous Cell Carcinoma",
        "synonyms": ["squamous cell carcinoma", "squamous carcinoma"],
        "parents": ["ONT:CARCINOMA"],
        "ic": 2.4,
    },
    {
        "id": "ONT:INVASIVE_DUCTAL_CARCINOMA",
        "name": "Invasive Ductal Carcinoma",
        "synonyms": ["invasive ductal carcinoma", "ductal carcinoma"],
        "parents": ["ONT:CARCINOMA"],
        "ic": 2.6,
    },
    {
        "id": "ONT:CLEAR_CELL_RENAL_CELL_CARCINOMA",
        "name": "Clear Cell Renal Cell Carcinoma",
        "synonyms": ["clear cell renal cell carcinoma", "clear cell renal carcinoma", "clear cell rcc", "clear cell carcinoma"],
        "parents": ["ONT:CARCINOMA"],
        "ic": 2.8,
    },
    {
        "id": "ONT:PAPILLARY_RENAL_CELL_CARCINOMA",
        "name": "Papillary Renal Cell Carcinoma",
        "synonyms": ["papillary renal cell carcinoma", "papillary renal carcinoma", "papillary rcc"],
        "parents": ["ONT:CARCINOMA"],
        "ic": 2.8,
    },
    {
        "id": "ONT:METASTASIS",
        "name": "Metastasis",
        "synonyms": ["metastasis", "metastatic", "metastases"],
        "parents": ["ONT:NEOPLASM"],
        "ic": 2.1,
    },
    {
        "id": "ONT:NECROSIS",
        "name": "Necrosis",
        "synonyms": ["necrosis", "necrotic"],
        "parents": ["ONT:NEOPLASM"],
        "ic": 1.9,
    },
    {
        "id": "ONT:LYMPHOVASCULAR_INVASION",
        "name": "Lymphovascular Invasion",
        "synonyms": ["lymphovascular invasion", "vascular invasion", "lymphatic invasion"],
        "parents": ["ONT:NEOPLASM"],
        "ic": 2.5,
    },
    {
        "id": "ONT:MARGIN_POSITIVE",
        "name": "Positive Margin",
        "synonyms": ["positive margin", "margin involved", "involved margin"],
        "parents": ["ONT:NEOPLASM"],
        "ic": 2.3,
    },
    {
        "id": "ONT:LYMPH_NODE_INVOLVEMENT",
        "name": "Lymph Node Involvement",
        "synonyms": ["lymph node involvement", "lymph node metastasis", "nodal metastasis", "positive lymph node"],
        "parents": ["ONT:METASTASIS"],
        "ic": 2.7,
    },
]


def iter_sentence_export_jsons(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*.json")
        if path.is_file() and path.name not in {"run_summary.json"}
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()


def _normalize_phrase(text: str) -> str:
    text = _normalize_text(text)
    return text.strip(" ,.;:()[]{}<>-_/\\")


def _compile_term_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term)
    escaped = escaped.replace(r"\ ", r"\s+")
    return re.compile(rf"(?<!\w){escaped}(?!\w)", flags=re.IGNORECASE)


def _synonym_term(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("term", "text", "label", "value"):
            token = str(value.get(key) or "").strip()
            if token:
                return token
        return ""
    return str(value or "").strip()


def _coerce_concept_record(concept_id: str, record: dict[str, Any], default_ic: float) -> dict[str, Any]:
    name = str(record.get("name") or concept_id).strip()
    source_terminology = str(record.get("source_terminology") or "").strip()
    canonical_source = str(record.get("canonical_source") or source_terminology).strip()
    synonyms = list(record.get("synonyms") or [])
    synonym_records = list(record.get("synonym_records") or [])
    normalized_synonyms = []
    seen_synonyms: set[str] = set()
    normalized_synonym_records: list[dict[str, Any]] = []
    seen_synonym_records: set[tuple[str, str, str, str, str]] = set()

    for item in synonym_records:
        term = _synonym_term(item)
        normalized = _normalize_phrase(term)
        if not normalized:
            continue
        if normalized not in seen_synonyms:
            seen_synonyms.add(normalized)
            normalized_synonyms.append(str(term).strip())

        if not isinstance(item, dict) or not term:
            continue
        item_source_terminology = str(item.get("source_terminology") or source_terminology).strip()
        item_source_id = str(item.get("source_id") or concept_id).strip()
        item_origin = str(item.get("origin") or "ontology_synonym").strip()
        item_umls_cui = str(item.get("umls_cui") or "").strip()
        key = (normalized, item_source_terminology, item_source_id, item_origin, item_umls_cui)
        if key in seen_synonym_records:
            continue
        seen_synonym_records.add(key)
        normalized_synonym_records.append(
            {
                "term": term,
                "source_terminology": item_source_terminology,
                "source_id": item_source_id,
                "origin": item_origin,
                **({"umls_cui": item_umls_cui} if item_umls_cui else {}),
            }
        )

    for term in [name, *synonyms]:
        token = _synonym_term(term)
        normalized = _normalize_phrase(token)
        if not normalized or normalized in seen_synonyms:
            continue
        seen_synonyms.add(normalized)
        normalized_synonyms.append(token)
        key = (normalized, source_terminology, concept_id, "ontology_synonym", "")
        if key in seen_synonym_records:
            continue
        seen_synonym_records.add(key)
        normalized_synonym_records.append(
            {
                "term": token,
                "source_terminology": source_terminology,
                "source_id": concept_id,
                "origin": "ontology_synonym",
            }
        )

    parents = [str(parent).strip() for parent in record.get("parents", []) if str(parent).strip()]
    coerced = {
        "id": concept_id,
        "name": name,
        "synonyms": normalized_synonyms,
        "synonym_records": normalized_synonym_records,
        "parents": parents,
        "ic": float(record.get("ic", default_ic)),
        "xrefs": dict(record.get("xrefs", {}) or {}),
        "source_terminology": source_terminology,
        "canonical_source": canonical_source,
        "aligned_sources": list(record.get("aligned_sources", []) or []),
        "match_enabled": bool(record.get("match_enabled", True)),
        "definition": str(record.get("definition") or ""),
        "normalization_role": str(record.get("normalization_role") or ""),
    }
    for key, value in record.items():
        if key not in coerced:
            coerced[key] = value
    return coerced


def load_ontology_concepts(ontology_path: Path | None, default_ic: float) -> dict[str, dict[str, Any]]:
    if ontology_path is None:
        return {
            item["id"]: _coerce_concept_record(item["id"], item, default_ic)
            for item in DEFAULT_ONTOLOGY_CONCEPTS
        }

    payload = load_json(ontology_path)
    concepts_block = payload.get("concepts", payload) if isinstance(payload, dict) else payload

    normalized: dict[str, dict[str, Any]] = {}
    if isinstance(concepts_block, dict):
        for concept_id, record in concepts_block.items():
            if not isinstance(record, dict):
                continue
            normalized[str(concept_id).strip()] = _coerce_concept_record(str(concept_id).strip(), record, default_ic)
        return normalized

    if isinstance(concepts_block, list):
        for item in concepts_block:
            if not isinstance(item, dict):
                continue
            concept_id = str(item.get("id") or item.get("concept_id") or "").strip()
            if not concept_id:
                continue
            normalized[concept_id] = _coerce_concept_record(concept_id, item, default_ic)
        return normalized

    raise ValueError(f"Unsupported ontology payload in {ontology_path}")


def _normalized_tokens(text: str) -> tuple[str, ...]:
    return tuple(TOKEN_PATTERN.findall(_normalize_text(text)))


def _tokenize_with_spans(text: str) -> list[tuple[str, int, int]]:
    return [(match.group(0), int(match.start()), int(match.end())) for match in TOKEN_PATTERN.finditer(text)]


def _concept_is_domain_relevant(concept_name: str) -> bool:
    normalized_name = _normalize_text(concept_name)
    if any(cue in normalized_name for cue in CONCEPT_NAME_EXCLUDE_CUES):
        return False
    return any(cue in normalized_name for cue in DOMAIN_CONCEPT_NAME_CUES)


def _term_is_domain_relevant(term_tokens: tuple[str, ...]) -> bool:
    joined = " ".join(term_tokens)
    return any(cue in joined for cue in DOMAIN_CONCEPT_NAME_CUES)


def _is_usable_synonym(term: str, concept_name: str) -> bool:
    term_tokens = _normalized_tokens(term)
    if not term_tokens:
        return False
    concept_name_tokens = _normalized_tokens(concept_name)
    if all(token.isdigit() for token in term_tokens):
        return False
    if all(token in GENERIC_TERM_STOPWORDS for token in term_tokens):
        return False

    concept_relevant = _concept_is_domain_relevant(concept_name)
    term_relevant = _term_is_domain_relevant(term_tokens)
    if not concept_relevant and not term_relevant:
        return False

    if len(term_tokens) == 1:
        token = term_tokens[0]
        if token in PATHOLOGY_SINGLE_TOKEN_ABBREVIATIONS:
            return concept_relevant
        if token in PATHOLOGY_SINGLE_TOKEN_WHITELIST:
            return len(concept_name_tokens) == 1
        if token in GENERIC_TERM_STOPWORDS:
            return False
        if len(token) < 4:
            return False
        if len(concept_name_tokens) > 1:
            return False
        return concept_relevant

    return True


def _concept_match_priority(phrase_tokens: tuple[str, ...], concept_name: str) -> tuple[int, int, int, int, str]:
    normalized_name = _normalize_text(concept_name)
    phrase_text = " ".join(phrase_tokens)
    exact_name_match = 0 if normalized_name == phrase_text else 1
    phrase_in_name = 0 if phrase_text in normalized_name else 1
    blocked_name = 1 if any(cue in normalized_name for cue in CONCEPT_NAME_EXCLUDE_CUES) else 0
    token_count = len(_normalized_tokens(concept_name))
    return (blocked_name, exact_name_match, phrase_in_name, token_count, normalized_name)


def _concept_source_priority(concept: dict[str, Any]) -> int:
    source = str(concept.get("canonical_source") or concept.get("source_terminology") or "").strip().lower()
    if source == "ncit":
        return 0
    if source == "do":
        return 1
    if source in {"snomedct", "snomed ct"}:
        return 2
    return 9


def compile_match_catalog(ontology: dict[str, dict[str, Any]]) -> dict[str, Any]:
    phrase_index: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    term_lengths: set[int] = set()
    normalized_term_count = 0
    retained_concept_count = 0

    for concept_id, concept in ontology.items():
        if not bool(concept.get("match_enabled", True)):
            continue
        concept_name = concept["name"]
        if not _concept_is_domain_relevant(concept_name):
            continue
        retained_concept_count += 1
        synonym_records = list(concept.get("synonym_records", []) or [])
        if not synonym_records:
            synonym_records = [{"term": term, "source_terminology": concept.get("source_terminology", ""), "source_id": concept_id, "origin": "ontology_synonym"} for term in concept.get("synonyms", []) or []]
        if not synonym_records:
            continue
        seen_phrases: set[tuple[str, ...]] = set()
        for synonym_record in synonym_records:
            synonym = _synonym_term(synonym_record)
            if not _is_usable_synonym(synonym, concept_name):
                continue
            phrase_tokens = _normalized_tokens(synonym)
            if not phrase_tokens or phrase_tokens in seen_phrases:
                continue
            seen_phrases.add(phrase_tokens)
            normalized_term_count += 1
            term_lengths.add(len(phrase_tokens))
            phrase_index.setdefault(phrase_tokens, []).append(
                {
                    "concept_id": concept_id,
                    "concept_name": concept_name,
                    "parents": list(concept.get("parents", []) or []),
                    "ic": float(concept.get("ic", 0.0)),
                    "matched_term": synonym,
                    "matched_source_terminology": str(synonym_record.get("source_terminology") or concept.get("source_terminology") or ""),
                    "matched_source_id": str(synonym_record.get("source_id") or concept_id),
                    "matched_origin": str(synonym_record.get("origin") or "ontology_synonym"),
                    "source_priority": _concept_source_priority(concept),
                }
            )

    for phrase_tokens, matches in phrase_index.items():
        matches.sort(
            key=lambda item: (
                *_concept_match_priority(phrase_tokens, item["concept_name"]),
                item["source_priority"],
                item["concept_id"],
            )
        )

    return {
        "phrase_index": phrase_index,
        "term_lengths": sorted(term_lengths, reverse=True),
        "normalized_term_count": normalized_term_count,
        "retained_concept_count": retained_concept_count,
    }


def compute_depths(ontology: dict[str, dict[str, Any]]) -> dict[str, int]:
    memo: dict[str, int] = {}
    visiting: set[str] = set()

    def depth(concept_id: str) -> int:
        if concept_id in memo:
            return memo[concept_id]
        if concept_id in visiting:
            return 0
        visiting.add(concept_id)
        parents = list(ontology.get(concept_id, {}).get("parents", []) or [])
        if not parents:
            result = 0
        else:
            result = 1 + max(depth(parent) for parent in parents if parent in ontology)
        visiting.discard(concept_id)
        memo[concept_id] = result
        return result

    for concept_id in ontology:
        depth(concept_id)
    return memo


def _is_generic_true_path_ancestor(concept_name: str) -> bool:
    normalized_name = _normalize_text(concept_name)
    if not normalized_name:
        return True
    if normalized_name in TRUE_PATH_ANCESTOR_EXACT_EXCLUDE_NAMES:
        return True
    return any(cue in normalized_name for cue in TRUE_PATH_ANCESTOR_EXCLUDE_CUES)


def _should_retain_true_path_ancestor(
    concept_id: str,
    ontology: dict[str, dict[str, Any]],
) -> bool:
    concept_record = ontology.get(concept_id, {})
    concept_name = str(concept_record.get("name") or concept_id)
    return not _is_generic_true_path_ancestor(concept_name)


def _find_sentence_mentions(
    sentence_text: str,
    sentence_index: int,
    section_index: int,
    section_title: str,
    catalog: dict[str, Any],
) -> list[dict[str, Any]]:
    normalized_text = _normalize_text(sentence_text)
    if not normalized_text:
        return []

    tokens = _tokenize_with_spans(normalized_text)
    if not tokens:
        return []

    phrase_index = catalog["phrase_index"]
    term_lengths = catalog["term_lengths"]
    mentions: list[dict[str, Any]] = []
    seen_mentions: set[tuple[int, int, str]] = set()
    for start_idx in range(len(tokens)):
        for length in term_lengths:
            end_idx = start_idx + length
            if end_idx > len(tokens):
                continue
            phrase_tokens = tuple(token for token, _, _ in tokens[start_idx:end_idx])
            candidates = phrase_index.get(phrase_tokens)
            if not candidates:
                continue

            start_char = tokens[start_idx][1]
            end_char = tokens[end_idx - 1][2]
            mention_text = normalized_text[start_char:end_char]
            best_item = candidates[0]
            mention_key = (start_char, end_char, best_item["concept_id"])
            if mention_key in seen_mentions:
                continue
            seen_mentions.add(mention_key)
            mentions.append(
                {
                    "sentence_index": sentence_index,
                    "section_index": section_index,
                    "section_title": section_title,
                    "concept_id": best_item["concept_id"],
                    "concept_name": best_item["concept_name"],
                    "matched_term": best_item["matched_term"],
                    "matched_source_terminology": best_item["matched_source_terminology"],
                    "matched_source_id": best_item["matched_source_id"],
                    "matched_origin": best_item["matched_origin"],
                    "mention_text": mention_text,
                    "start_char": start_char,
                    "end_char": end_char,
                    "ic": float(best_item["ic"]),
                }
            )
    mentions.sort(key=lambda item: (item["start_char"], -(item["end_char"] - item["start_char"]), item["concept_id"]))

    filtered: list[dict[str, Any]] = []
    for mention in mentions:
        overlaps_existing = False
        for kept in filtered:
            overlaps_existing = not (
                mention["end_char"] <= kept["start_char"] or mention["start_char"] >= kept["end_char"]
            )
            if overlaps_existing:
                break
        if not overlaps_existing:
            filtered.append(mention)
    return filtered


def _get_or_create_concept_summary(
    concept_index: dict[str, dict[str, Any]],
    ontology: dict[str, dict[str, Any]],
    depths: dict[str, int],
    concept_id: str,
) -> dict[str, Any]:
    existing = concept_index.get(concept_id)
    if existing is not None:
        return existing

    ontology_record = ontology.get(concept_id, {})
    entry = {
        "concept_id": concept_id,
        "concept_name": str(ontology_record.get("name") or concept_id),
        "parents": list(ontology_record.get("parents", []) or []),
        "ic": float(ontology_record.get("ic", 0.0)),
        "depth": int(depths.get(concept_id, 0)),
        "xrefs": dict(ontology_record.get("xrefs", {}) or {}),
        "source_terminology": str(ontology_record.get("source_terminology") or ""),
        "canonical_source": str(ontology_record.get("canonical_source") or ontology_record.get("source_terminology") or ""),
        "aligned_sources": list(ontology_record.get("aligned_sources", []) or []),
        "match_enabled": bool(ontology_record.get("match_enabled", True)),
        "definition": str(ontology_record.get("definition") or ""),
        "normalization_role": str(ontology_record.get("normalization_role") or ""),
        "sentence_indices": set(),
        "section_indices": set(),
        "matched_terms": set(),
        "direct_mention_count": 0,
        "is_direct": False,
        "ancestor_of": set(),
    }
    concept_index[concept_id] = entry
    return entry


def _expand_true_path(
    concept_index: dict[str, dict[str, Any]],
    ontology: dict[str, dict[str, Any]],
    depths: dict[str, int],
    max_ancestor_hops: int | None = 2,
) -> None:
    for concept_id, direct_entry in list(concept_index.items()):
        if not direct_entry["is_direct"]:
            continue
        queue = [(parent_id, 1) for parent_id in list(direct_entry.get("parents", []) or [])]
        visited: set[str] = set()
        while queue:
            parent_id, hop = queue.pop(0)
            if parent_id in visited or parent_id not in ontology:
                continue
            visited.add(parent_id)
            if max_ancestor_hops is not None and hop > int(max_ancestor_hops):
                continue
            queue.extend((next_parent_id, hop + 1) for next_parent_id in list(ontology.get(parent_id, {}).get("parents", []) or []))
            if not _should_retain_true_path_ancestor(parent_id, ontology):
                continue
            parent_entry = _get_or_create_concept_summary(concept_index, ontology, depths, parent_id)
            parent_entry["sentence_indices"].update(direct_entry["sentence_indices"])
            parent_entry["section_indices"].update(direct_entry["section_indices"])
            parent_entry["ancestor_of"].add(concept_id)


def build_concept_annotation(
    sentence_view: dict[str, Any],
    ontology: dict[str, dict[str, Any]],
    depths: dict[str, int],
    catalog: dict[str, Any],
    include_true_path: bool,
    true_path_max_ancestor_hops: int | None,
    ontology_source: str,
) -> dict[str, Any]:
    mentions: list[dict[str, Any]] = []
    concept_index: dict[str, dict[str, Any]] = {}

    sentence_records = list(sentence_view.get("sentence_records", []) or [])
    for record in sentence_records:
        sentence_index = int(record.get("sentence_index", 0))
        section_index = int(record.get("section_index", 0))
        section_title = str(record.get("section_title", "Document Body"))
        text = str(record.get("text", ""))

        sentence_mentions = _find_sentence_mentions(
            sentence_text=text,
            sentence_index=sentence_index,
            section_index=section_index,
            section_title=section_title,
            catalog=catalog,
        )
        mentions.extend(sentence_mentions)
        for mention in sentence_mentions:
            entry = _get_or_create_concept_summary(concept_index, ontology, depths, mention["concept_id"])
            entry["sentence_indices"].add(sentence_index)
            entry["section_indices"].add(section_index)
            entry["matched_terms"].add(str(mention["matched_term"]))
            entry["direct_mention_count"] += 1
            entry["is_direct"] = True

    if include_true_path:
        _expand_true_path(
            concept_index,
            ontology,
            depths,
            max_ancestor_hops=true_path_max_ancestor_hops,
        )

    concept_edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str]] = set()
    for concept_id, entry in concept_index.items():
        for parent_id in entry.get("parents", []):
            if parent_id not in concept_index:
                continue
            key = (concept_id, parent_id, "ontology")
            if key in seen_edges:
                continue
            seen_edges.add(key)
            concept_edges.append(
                {
                    "source_concept_id": concept_id,
                    "target_concept_id": parent_id,
                    "edge_type": "ontology",
                }
            )

    concepts: list[dict[str, Any]] = []
    for concept_id, entry in concept_index.items():
        sentence_indices = sorted(int(index) for index in entry["sentence_indices"])
        section_indices = sorted(int(index) for index in entry["section_indices"])
        matched_terms = sorted(str(term) for term in entry["matched_terms"])
        ancestor_of = sorted(str(desc) for desc in entry["ancestor_of"])
        concepts.append(
            {
                "concept_id": concept_id,
                "concept_name": entry["concept_name"],
                "parents": list(entry.get("parents", [])),
                "depth": int(entry["depth"]),
                "ic": float(entry["ic"]),
                "xrefs": dict(entry.get("xrefs", {}) or {}),
                "source_terminology": entry.get("source_terminology", ""),
                "canonical_source": entry.get("canonical_source", ""),
                "aligned_sources": list(entry.get("aligned_sources", []) or []),
                "match_enabled": bool(entry.get("match_enabled", True)),
                "definition": entry.get("definition", ""),
                "normalization_role": entry.get("normalization_role", ""),
                "sentence_indices": sentence_indices,
                "section_indices": section_indices,
                "matched_terms": matched_terms,
                "direct_mention_count": int(entry["direct_mention_count"]),
                "is_direct": bool(entry["is_direct"]),
                "is_ancestor_only": (not bool(entry["is_direct"])),
                "ancestor_of": ancestor_of,
            }
        )

    concepts.sort(key=lambda item: (item["depth"], item["concept_name"].lower(), item["concept_id"]))

    return {
        "document_id": sentence_view.get("document_id"),
        "file_name": sentence_view.get("file_name"),
        "dataset": sentence_view.get("dataset"),
        "filter_mode": sentence_view.get("filter_mode"),
        "source_sentence_view_json": sentence_view.get("source_json"),
        "matching_method": "keyword_synonym_filtered",
        "include_true_path": bool(include_true_path),
        "true_path_max_ancestor_hops": true_path_max_ancestor_hops,
        "true_path_ancestor_filter": (
            "generic_superclasses_pruned_v5_limited_hops" if include_true_path else "disabled"
        ),
        "ontology_source": ontology_source,
        "sentence_count": int(sentence_view.get("sentence_count", 0)),
        "section_count": int(sentence_view.get("section_count", 0)),
        "mention_count": len(mentions),
        "direct_concept_count": sum(1 for concept in concepts if concept["is_direct"]),
        "concept_count": len(concepts),
        "mentions": mentions,
        "concepts": concepts,
        "concept_edges": concept_edges,
    }


def export_document(
    sentence_view_json_path: Path,
    input_root: Path,
    output_root: Path,
    ontology: dict[str, dict[str, Any]],
    depths: dict[str, int],
    catalog: dict[str, Any],
    include_true_path: bool,
    true_path_max_ancestor_hops: int | None,
    ontology_source: str,
) -> dict[str, Any]:
    sentence_view = load_json(sentence_view_json_path)
    annotation = build_concept_annotation(
        sentence_view=sentence_view,
        ontology=ontology,
        depths=depths,
        catalog=catalog,
        include_true_path=include_true_path,
        true_path_max_ancestor_hops=true_path_max_ancestor_hops,
        ontology_source=ontology_source,
    )

    relative_path = sentence_view_json_path.relative_to(input_root)
    output_json_path = output_root / relative_path
    ensure_dir(output_json_path.parent)
    write_json(output_json_path, annotation)

    return {
        "document_id": annotation.get("document_id"),
        "file_name": annotation.get("file_name"),
        "dataset": annotation.get("dataset"),
        "source_sentence_view_json": str(sentence_view_json_path),
        "output_json": str(output_json_path),
        "sentence_count": annotation["sentence_count"],
        "concept_count": annotation["concept_count"],
        "direct_concept_count": annotation["direct_concept_count"],
        "mention_count": annotation["mention_count"],
        "status": "success",
    }


def process_all_documents(
    input_dir: Path,
    output_dir: Path,
    ontology_path: Path | None,
    include_true_path: bool,
    true_path_max_ancestor_hops: int | None,
    default_ic: float,
    limit: int | None,
) -> dict[str, Any]:
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    ensure_dir(output_root)

    log_path = output_root / "concepts.log"
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    ontology = load_ontology_concepts(ontology_path=ontology_path, default_ic=default_ic)
    depths = compute_depths(ontology)
    catalog = compile_match_catalog(ontology)
    ontology_source = str(ontology_path) if ontology_path is not None else "built_in"

    sentence_export_paths = iter_sentence_export_jsons(input_root)
    if limit is not None:
        sentence_export_paths = sentence_export_paths[:limit]

    LOGGER.info(
        "Loaded ontology with %s concepts | phrase_index=%s | normalized_terms=%s",
        len(ontology),
        len(catalog["phrase_index"]),
        catalog["normalized_term_count"],
    )
    LOGGER.info("Retained %s domain-relevant concepts for matching", catalog["retained_concept_count"])
    LOGGER.info("Discovered %s sentence-export JSON files under %s", len(sentence_export_paths), input_root)

    summaries: list[dict[str, Any]] = []
    for index, sentence_view_json_path in enumerate(sentence_export_paths, start=1):
        LOGGER.info("[%s/%s] Extracting concepts for %s", index, len(sentence_export_paths), sentence_view_json_path)
        try:
            summary = export_document(
                sentence_view_json_path=sentence_view_json_path,
                input_root=input_root,
                output_root=output_root,
                ontology=ontology,
                depths=depths,
                catalog=catalog,
                include_true_path=include_true_path,
                true_path_max_ancestor_hops=true_path_max_ancestor_hops,
                ontology_source=ontology_source,
            )
        except Exception as exc:
            LOGGER.exception("Failed to extract concepts for %s", sentence_view_json_path)
            summary = {
                "document_id": sentence_view_json_path.stem,
                "file_name": sentence_view_json_path.name,
                "dataset": None,
                "source_sentence_view_json": str(sentence_view_json_path),
                "output_json": None,
                "sentence_count": 0,
                "concept_count": 0,
                "direct_concept_count": 0,
                "mention_count": 0,
                "status": "failed",
                "error": str(exc),
            }
        summaries.append(summary)

    successes = [item for item in summaries if item["status"] == "success"]
    failures = [item for item in summaries if item["status"] == "failed"]

    dataset_counts: dict[str, int] = {}
    total_mentions = 0
    total_concepts = 0
    for item in successes:
        dataset = item.get("dataset") or "UNKNOWN"
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        total_mentions += int(item.get("mention_count", 0))
        total_concepts += int(item.get("concept_count", 0))

    run_summary = {
        "input_dir": str(input_root),
        "output_dir": str(output_root),
        "ontology_path": str(ontology_path) if ontology_path is not None else None,
        "ontology_source": ontology_source,
        "ontology_concept_count": len(ontology),
        "include_true_path": bool(include_true_path),
        "true_path_max_ancestor_hops": true_path_max_ancestor_hops,
        "true_path_ancestor_filter": (
            "generic_superclasses_pruned_v5_limited_hops" if include_true_path else "disabled"
        ),
        "default_ic": float(default_ic),
        "total_sentence_exports": len(sentence_export_paths),
        "success_count": len(successes),
        "failure_count": len(failures),
        "datasets": dataset_counts,
        "total_mentions": total_mentions,
        "total_concepts": total_concepts,
        "avg_concepts_per_document": (float(total_concepts) / len(successes)) if successes else 0.0,
        "files": summaries,
    }
    write_json(output_root / "run_summary.json", run_summary)
    LOGGER.info(
        "Completed concept extraction | total=%s | success=%s | failed=%s",
        len(sentence_export_paths),
        len(successes),
        len(failures),
    )
    return run_summary


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=None, help="Optional YAML config preset.")
    pre_args, _ = pre_parser.parse_known_args()
    raw_config, config_path = load_yaml_config(pre_args.config)
    stage_block = (
        raw_config.get("extract_ontology_concepts")
        if isinstance(raw_config.get("extract_ontology_concepts"), dict)
        else {}
    )
    config = get_stage_config(raw_config, "extract_ontology_concepts")
    defaults_block = raw_config.get("defaults") if isinstance(raw_config.get("defaults"), dict) else {}
    preprocess_config = get_stage_config(raw_config, "preprocess")
    filter_mode = str(get_value(preprocess_config, "filter_mode", "masked"))
    output_root = get_path(defaults_block, "output_root", DEFAULT_OUTPUT_ROOT, config_path)

    input_default = get_path(stage_block, "input_dir", DEFAULT_INPUT_DIR, config_path)
    output_default = get_path(stage_block, "output_dir", DEFAULT_OUTPUT_DIR, config_path)
    if stage_block:
        if stage_block.get("input_dir") in (None, ""):
            export_output_subdirs = dict(SENTENCE_EXPORT_OUTPUT_SUBDIRS)
            export_output_subdirs.update(
                get_value(get_stage_config(raw_config, "export_sentence_views"), "output_subdirs", {}) or {}
            )
            input_default = output_root / export_output_subdirs.get(filter_mode, SENTENCE_EXPORT_OUTPUT_SUBDIRS["masked"])
        if stage_block.get("output_dir") in (None, ""):
            concept_output_subdirs = dict(CONCEPT_OUTPUT_SUBDIRS)
            concept_output_subdirs.update(get_value(config, "output_subdirs", {}) or {})
            output_default = output_root / concept_output_subdirs.get(filter_mode, CONCEPT_OUTPUT_SUBDIRS["masked"])

    ontology_default = None
    ontology_raw = get_value(config, "ontology_path", None)
    if ontology_raw not in (None, ""):
        ontology_default = get_path(config, "ontology_path", Path(str(ontology_raw)), config_path)

    parser = argparse.ArgumentParser(
        description="Extract lightweight ontology concept annotations from sentence exports."
    )
    parser.add_argument("--config", type=Path, default=pre_args.config, help="Optional YAML config preset.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=input_default,
        help="Directory containing sentence-export JSON files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=output_default,
        help="Directory for concept annotation outputs.",
    )
    parser.add_argument(
        "--ontology_path",
        type=Path,
        default=ontology_default,
        help="Optional ontology JSON file. If omitted, a small built-in pathology ontology is used.",
    )
    parser.add_argument(
        "--include_true_path",
        action="store_true",
        default=get_bool(config, "include_true_path", True),
        help="Expand direct concepts to their ontology ancestors before exporting.",
    )
    true_path_hops_raw = get_value(config, "true_path_max_ancestor_hops", 2)
    parser.add_argument(
        "--true_path_max_ancestor_hops",
        type=int,
        default=None if true_path_hops_raw in (None, "") else int(true_path_hops_raw),
        help="Maximum true-path ancestor hops to retain. Use 0 to keep direct concepts only.",
    )
    parser.add_argument(
        "--default_ic",
        type=float,
        default=float(get_value(config, "default_ic", 1.0)),
        help="Fallback information content assigned to ontology concepts when missing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None if get_value(config, "limit", None) is None else int(get_value(config, "limit", None)),
        help="Optional cap on number of sentence-export files to process.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = process_all_documents(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        ontology_path=args.ontology_path,
        include_true_path=args.include_true_path,
        true_path_max_ancestor_hops=args.true_path_max_ancestor_hops,
        default_ic=args.default_ic,
        limit=args.limit,
    )
    print(
        "Concept extraction finished | "
        f"total={summary['total_sentence_exports']} | "
        f"success={summary['success_count']} | "
        f"failed={summary['failure_count']}"
    )


if __name__ == "__main__":
    main()
