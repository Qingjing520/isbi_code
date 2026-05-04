from __future__ import annotations

"""Build compact Document -> Section -> Sentence -> Word graphs.

The word layer only keeps staging/pathology-relevant keywords. This is meant to
reduce neutral clinical text noise before the graph branch sees the report.
The original sentence embeddings are still used for retained sentence nodes.
Word/concept nodes can use real CONCH text embeddings via ``--keyword_embedding_pt``
and ``--concept_embedding_pt``; otherwise the builder falls back to the previous
sentence-context/hash features.

With ``--attach_concepts``, NCIt+DO concept nodes are attached only to the
retained staging-relevant sentences.
"""

import argparse
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from pathology_report_extraction.common.pdf_utils import ensure_dir, write_json
from pathology_report_extraction.common.pipeline_defaults import (
    CONCH_OUTPUT_SUBDIRS,
    DEFAULT_HIERARCHY_GRAPH_ROOT,
    DEFAULT_OUTPUT_ROOT,
    HIERARCHY_GRAPH_TYPE_DIRS,
)
from pathology_report_extraction.graphs.build_text_hierarchy_graphs import (
    FEATURE_DIM,
    clean_payload,
    iter_metadata_jsons,
    load_optional_concept_annotation,
    load_json,
    mean_pool,
    validate_payload,
)


LOGGER = logging.getLogger("build_stage_keyword_hierarchy_graphs")

DEFAULT_INPUT_DIR = DEFAULT_OUTPUT_ROOT / CONCH_OUTPUT_SUBDIRS["masked"]
DEFAULT_OUTPUT_DIR = DEFAULT_HIERARCHY_GRAPH_ROOT
DEFAULT_CONCEPT_DIR = DEFAULT_OUTPUT_ROOT / "concept_annotations_ablation" / "ncit_do"
DATASET_DIR_NAMES = {"BRCA", "KIRC", "LUSC"}

NODE_TYPE_TO_ID = {"document": 0, "section": 1, "sentence": 2, "word": 3}
EDGE_TYPE_TO_ID = {"parent": 0, "next": 1}
EDGE_TYPE_WEIGHTS = {"parent": 1.0, "next": 0.25}
ONTOLOGY_NODE_TYPE_TO_ID = {"document": 0, "section": 1, "sentence": 2, "word": 3, "concept": 4}
ONTOLOGY_EDGE_TYPE_TO_ID = {
    "parent": 0,
    "next": 1,
    "mention": 2,
    "ontology": 3,
    "same_sentence": 4,
    "same_section": 5,
}
ONTOLOGY_EDGE_TYPE_WEIGHTS = {
    **EDGE_TYPE_WEIGHTS,
    "mention": 1.0,
    "ontology": 0.7,
    "same_sentence": 0.25,
    "same_section": 0.1,
}


def resolve_library_relative_path(relative_path: Path, attach_concepts: bool = False) -> Path:
    parts = relative_path.parts
    graph_type_key = "stage_keyword_word_ontology" if attach_concepts else "stage_keyword_word"
    graph_type_dir = HIERARCHY_GRAPH_TYPE_DIRS[graph_type_key]
    if parts and parts[0].upper() in DATASET_DIR_NAMES:
        return Path(parts[0]) / graph_type_dir / Path(*parts[1:])
    return Path(graph_type_dir) / relative_path


@dataclass(frozen=True)
class KeywordRule:
    canonical: str
    category: str
    pattern: re.Pattern[str]
    weight: float = 1.0


def _rule(canonical: str, category: str, pattern: str, weight: float = 1.0) -> KeywordRule:
    return KeywordRule(
        canonical=canonical,
        category=category,
        pattern=re.compile(pattern, flags=re.IGNORECASE),
        weight=float(weight),
    )


KEYWORD_RULES: tuple[KeywordRule, ...] = (
    _rule("tnm_code", "tnm", r"\b[ypra]?[tnm](?:is|x|[0-4][a-d]?)(?:\([a-z+]+\))?\b", 1.4),
    _rule("ajcc_stage", "stage", r"\b(?:ajcc\s*)?stage\s*(?:0|i{1,3}|iv|[1-4])\s*[abc]?\b", 1.5),
    _rule("pathologic_stage", "stage", r"\bpatholog(?:ic|ical)?\s+stage\b", 1.5),
    _rule("tnm", "stage", r"\btnm\b", 1.3),
    _rule("tumor", "tumor", r"\btumou?r\b", 1.0),
    _rule("carcinoma", "tumor", r"\bcarcinoma\b", 1.0),
    _rule("adenocarcinoma", "tumor", r"\badenocarcinoma\b", 1.0),
    _rule("squamous_cell_carcinoma", "tumor", r"\bsquamous\s+cell\s+carcinoma\b", 1.1),
    _rule("clear_cell", "tumor", r"\bclear\s+cell\b", 1.0),
    _rule("invasive", "invasion", r"\binvas(?:ive|ion|ion[s]?)\b", 1.2),
    _rule("microinvasion", "invasion", r"\bmicroinvas(?:ion|ive)\b", 1.2),
    _rule("lymphovascular_invasion", "invasion", r"\blymphovascular\s+invasion\b", 1.5),
    _rule("vascular_invasion", "invasion", r"\bvascular\s+invasion\b", 1.3),
    _rule("perineural_invasion", "invasion", r"\bperineural\s+invasion\b", 1.2),
    _rule("pleural_invasion", "invasion", r"\b(?:visceral\s+)?pleural\s+invasion\b", 1.3),
    _rule("extension", "invasion", r"\bextension\b", 1.0),
    _rule("extracapsular_extension", "invasion", r"\bextracapsular\s+extension\b", 1.3),
    _rule("metastasis", "metastasis", r"\bmetasta(?:sis|tic|ses|sized|size)\b", 1.4),
    _rule("lymph_node", "node", r"\blymph\s+node[s]?\b", 1.4),
    _rule("sentinel_node", "node", r"\bsentinel\s+(?:lymph\s+)?node[s]?\b", 1.3),
    _rule("nodal", "node", r"\bnodal\b", 1.2),
    _rule("extranodal_extension", "node", r"\bextranodal\s+extension\b", 1.3),
    _rule("margin", "margin", r"\bmargin[s]?\b", 1.1),
    _rule("positive", "status", r"\bpositive\b", 0.9),
    _rule("negative", "status", r"\bnegative\b", 0.9),
    _rule("involved", "status", r"\binvolved\b", 1.0),
    _rule("uninvolved", "status", r"\buninvolved\b", 1.0),
    _rule("grade", "grade", r"\b(?:histologic\s+|nuclear\s+|nottingham\s+|fuhrman\s+)?grade\s*(?:[1-4]|i{1,3}|iv)?\b", 1.2),
    _rule("differentiation", "grade", r"\b(?:well|moderately|poorly)\s+differentiated\b", 1.0),
    _rule("size", "size", r"\b(?:size|dimension|diameter|greatest\s+dimension)\b", 1.0),
    _rule("measurement", "size", r"\b\d+(?:\.\d+)?\s*(?:cm|mm)\b", 1.0),
    _rule("depth", "size", r"\bdepth\b", 0.9),
    _rule("necrosis", "aggressive_feature", r"\bnecrosis\b", 0.9),
    _rule("sarcomatoid", "aggressive_feature", r"\bsarcomatoid\b", 1.1),
    _rule("rhabdoid", "aggressive_feature", r"\brhabdoid\b", 1.1),
    _rule("renal_vein", "anatomic_extent", r"\brenal\s+vein\b", 1.2),
    _rule("sinus", "anatomic_extent", r"\b(?:renal\s+)?sinus\b", 1.1),
    _rule("perirenal_fat", "anatomic_extent", r"\bperirenal\s+fat\b", 1.1),
    _rule("adrenal", "anatomic_extent", r"\badrenal\b", 0.9),
    _rule("chest_wall", "anatomic_extent", r"\bchest\s+wall\b", 1.1),
    _rule("skin", "anatomic_extent", r"\bskin\b", 0.8),
    _rule("nipple", "anatomic_extent", r"\bnipple\b", 0.8),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean pathology report hierarchy graphs by keeping staging-related "
            "keywords and adding word nodes: Document -> Section -> Sentence -> Word. "
            "Optionally attach ontology concept nodes on the retained sentences."
        )
    )
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--concept_dir", type=Path, default=DEFAULT_CONCEPT_DIR)
    parser.add_argument("--datasets", nargs="+", default=None, help="Optional dataset filter, e.g. BRCA KIRC LUSC.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--attach_concepts", action="store_true", help="Attach NCIt+DO concept nodes on retained sentences.")
    parser.add_argument(
        "--disable_concept_cooccurrence_edges",
        action="store_true",
        help="Skip same-sentence and same-section concept co-occurrence edges.",
    )
    parser.add_argument(
        "--keep_all_sentences",
        action="store_true",
        help="Keep sentence nodes even when no staging keyword is found. Word nodes are still keyword-only.",
    )
    parser.add_argument(
        "--lexical_weight",
        type=float,
        default=0.35,
        help="Weight for deterministic lexical hash features in word node embeddings.",
    )
    parser.add_argument(
        "--keyword_embedding_pt",
        type=Path,
        default=None,
        help="Optional torch-saved dict mapping keyword labels to CONCH text embeddings.",
    )
    parser.add_argument(
        "--concept_embedding_pt",
        type=Path,
        default=None,
        help="Optional torch-saved dict mapping concept ids/names to CONCH text embeddings.",
    )
    parser.add_argument(
        "--concept_label_weight",
        type=float,
        default=0.7,
        help="Blend weight for concept label embeddings when --concept_embedding_pt is provided.",
    )
    parser.add_argument(
        "--min_keywords_per_doc",
        type=int,
        default=1,
        help="Skip documents with fewer keyword nodes than this value.",
    )
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def normalize_keyword_text(text: str) -> str:
    cleaned = re.sub(r"\s+", "_", str(text).strip().lower())
    cleaned = re.sub(r"[^a-z0-9_()+-]+", "", cleaned)
    return cleaned.strip("_") or "keyword"


def find_stage_keywords(text: str) -> list[dict]:
    mentions: list[dict] = []
    seen: set[tuple[str, int, int]] = set()
    for rule in KEYWORD_RULES:
        for match in rule.pattern.finditer(text or ""):
            matched = match.group(0)
            canonical = rule.canonical
            if canonical == "tnm_code":
                canonical = normalize_keyword_text(matched)
            elif canonical == "ajcc_stage":
                canonical = normalize_keyword_text(matched)
            elif canonical == "measurement":
                canonical = normalize_keyword_text(matched)
            key = (canonical, int(match.start()), int(match.end()))
            if key in seen:
                continue
            seen.add(key)
            mentions.append(
                {
                    "keyword": canonical,
                    "matched_text": matched,
                    "category": rule.category,
                    "start": int(match.start()),
                    "end": int(match.end()),
                    "weight": float(rule.weight),
                }
            )
    mentions.sort(key=lambda item: (int(item["start"]), int(item["end"]), str(item["keyword"])))
    return mentions


def deduplicate_sentence_keywords(mentions: Iterable[dict]) -> list[dict]:
    by_keyword: dict[str, dict] = {}
    for mention in mentions:
        keyword = str(mention["keyword"])
        if keyword not in by_keyword:
            item = dict(mention)
            item["mention_count"] = 1
            item["spans"] = [[int(mention["start"]), int(mention["end"])]]
            by_keyword[keyword] = item
        else:
            item = by_keyword[keyword]
            item["mention_count"] = int(item["mention_count"]) + 1
            item["weight"] = max(float(item["weight"]), float(mention["weight"]))
            item["spans"].append([int(mention["start"]), int(mention["end"])])
    return sorted(by_keyword.values(), key=lambda item: (str(item["category"]), str(item["keyword"])))


def keyword_hash_embedding(keyword: str, dim: int = FEATURE_DIM) -> torch.Tensor:
    digest = hashlib.blake2b(keyword.encode("utf-8"), digest_size=8).digest()
    seed = int.from_bytes(digest, byteorder="little", signed=False) % (2**31 - 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    vector = torch.randn(dim, generator=generator, dtype=torch.float32)
    return F.normalize(vector, dim=0)


def normalize_embedding_key(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("_", " ").strip().lower())


def load_embedding_map(path: Path | None) -> dict[str, torch.Tensor]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Embedding map not found: {path}")
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "embeddings" in payload and isinstance(payload["embeddings"], dict):
        payload = payload["embeddings"]
    if not isinstance(payload, dict):
        raise TypeError(f"Expected embedding map dict in {path}, got {type(payload)!r}")

    embeddings: dict[str, torch.Tensor] = {}
    for key, value in payload.items():
        tensor = torch.as_tensor(value, dtype=torch.float32).view(-1)
        if tensor.numel() != FEATURE_DIM:
            continue
        embeddings[str(key)] = tensor
        embeddings[normalize_embedding_key(str(key))] = tensor
    return embeddings


def lookup_embedding(embedding_map: dict[str, torch.Tensor], *keys: str) -> torch.Tensor | None:
    for key in keys:
        if not key:
            continue
        for candidate in (str(key), normalize_embedding_key(str(key))):
            value = embedding_map.get(candidate)
            if value is not None:
                return value.float()
    return None


def word_feature(
    sentence_feature: torch.Tensor,
    keyword: str,
    lexical_weight: float,
    keyword_embedding_map: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    lexical_weight = max(0.0, min(1.0, float(lexical_weight)))
    context = sentence_feature.float()
    context_norm = context.norm().clamp_min(1.0)
    lexical = lookup_embedding(keyword_embedding_map or {}, keyword)
    if lexical is None:
        lexical = keyword_hash_embedding(keyword, dim=int(context.shape[0])) * context_norm
    else:
        lexical = F.normalize(lexical, dim=0) * context_norm
    return ((1.0 - lexical_weight) * context + lexical_weight * lexical).float()


def remap_concept_annotation_to_retained_sentences(
    concept_annotation: dict | None,
    old_to_new_sentence: dict[int, int],
    new_sentence_to_section: dict[int, int],
) -> dict | None:
    if not concept_annotation:
        return None

    kept_concepts: list[dict] = []
    kept_concept_ids: set[str] = set()
    for concept in concept_annotation.get("concepts", []) or []:
        remapped_sentence_indices = sorted(
            {
                int(old_to_new_sentence[int(index)])
                for index in concept.get("sentence_indices", []) or []
                if int(index) in old_to_new_sentence
            }
        )
        if not remapped_sentence_indices:
            continue
        item = dict(concept)
        concept_id = str(item.get("concept_id", ""))
        if not concept_id:
            continue
        item["sentence_indices"] = remapped_sentence_indices
        kept_concepts.append(item)
        kept_concept_ids.add(concept_id)

    kept_mentions: list[dict] = []
    for mention in concept_annotation.get("mentions", []) or []:
        concept_id = str(mention.get("concept_id", ""))
        try:
            old_sentence_index = int(mention.get("sentence_index", -1))
        except (TypeError, ValueError):
            continue
        if concept_id not in kept_concept_ids or old_sentence_index not in old_to_new_sentence:
            continue
        new_sentence_index = int(old_to_new_sentence[old_sentence_index])
        item = dict(mention)
        item["sentence_index"] = new_sentence_index
        item["original_sentence_index"] = old_sentence_index
        item["section_index"] = int(new_sentence_to_section.get(new_sentence_index, -1))
        kept_mentions.append(item)

    kept_edges: list[dict] = []
    for edge in concept_annotation.get("concept_edges", []) or []:
        source_id = str(edge.get("source_concept_id", ""))
        target_id = str(edge.get("target_concept_id", ""))
        if source_id in kept_concept_ids and target_id in kept_concept_ids:
            kept_edges.append(dict(edge))

    if not kept_concepts:
        return None

    direct_concept_count = sum(1 for concept in kept_concepts if bool(concept.get("is_direct", False)))
    return {
        **concept_annotation,
        "concepts": kept_concepts,
        "mentions": kept_mentions,
        "concept_edges": kept_edges,
        "concept_count": len(kept_concepts),
        "direct_concept_count": int(direct_concept_count),
        "mention_count": len(kept_mentions),
    }


def build_selected_concept_features(
    concept_annotation: dict | None,
    selected_sentence_features: torch.Tensor,
    concept_embedding_map: dict[str, torch.Tensor] | None = None,
    concept_label_weight: float = 0.7,
) -> tuple[torch.Tensor, list[str], torch.Tensor, torch.Tensor, torch.Tensor, list[list[int]]]:
    if not concept_annotation:
        return (
            torch.empty((0, FEATURE_DIM), dtype=torch.float32),
            [],
            torch.empty((0,), dtype=torch.float32),
            torch.empty((0,), dtype=torch.float32),
            torch.empty((0,), dtype=torch.float32),
            [],
        )

    features: list[torch.Tensor] = []
    concept_ids: list[str] = []
    concept_ic: list[float] = []
    concept_depth: list[float] = []
    concept_direct_mentions: list[float] = []
    concept_sentence_links: list[list[int]] = []
    sentence_count = int(selected_sentence_features.shape[0])

    for concept in concept_annotation.get("concepts", []) or []:
        sentence_indices = sorted(
            {
                int(index)
                for index in concept.get("sentence_indices", []) or []
                if 0 <= int(index) < sentence_count
            }
        )
        if not sentence_indices:
            continue
        context_feature = mean_pool(selected_sentence_features[sentence_indices]).float()
        concept_id = str(concept.get("concept_id", ""))
        concept_name = str(concept.get("concept_name", concept_id))
        label_feature = lookup_embedding(concept_embedding_map or {}, concept_id, concept_name)
        if label_feature is not None:
            weight = max(0.0, min(1.0, float(concept_label_weight)))
            label_feature = F.normalize(label_feature, dim=0) * context_feature.norm().clamp_min(1.0)
            features.append(((1.0 - weight) * context_feature + weight * label_feature).float())
        else:
            features.append(context_feature)
        concept_ids.append(concept_id)
        concept_ic.append(float(concept.get("ic", 0.0)))
        concept_depth.append(float(concept.get("depth", 0.0)))
        concept_direct_mentions.append(float(concept.get("direct_mention_count", 0.0)))
        concept_sentence_links.append(sentence_indices)

    if not features:
        return (
            torch.empty((0, FEATURE_DIM), dtype=torch.float32),
            [],
            torch.empty((0,), dtype=torch.float32),
            torch.empty((0,), dtype=torch.float32),
            torch.empty((0,), dtype=torch.float32),
            [],
        )

    return (
        torch.stack(features, dim=0).float(),
        concept_ids,
        torch.tensor(concept_ic, dtype=torch.float32),
        torch.tensor(concept_depth, dtype=torch.float32),
        torch.tensor(concept_direct_mentions, dtype=torch.float32),
        concept_sentence_links,
    )


def selected_sections_and_sentences(
    payload: dict,
    sentence_keywords: dict[int, list[dict]],
    keep_all_sentences: bool,
) -> tuple[list[dict], list[dict], dict[int, int], dict[int, int]]:
    selected_sentence_records: list[dict] = []
    old_to_new_sentence: dict[int, int] = {}

    raw_records = sorted(payload.get("sentence_records", []), key=lambda item: int(item.get("sentence_index", 0)))
    for record in raw_records:
        old_sentence_index = int(record["sentence_index"])
        if not keep_all_sentences and not sentence_keywords.get(old_sentence_index):
            continue
        new_record = dict(record)
        new_index = len(selected_sentence_records)
        old_to_new_sentence[old_sentence_index] = new_index
        new_record["original_sentence_index"] = old_sentence_index
        new_record["sentence_index"] = new_index
        keywords = sentence_keywords.get(old_sentence_index, [])
        new_record["stage_keywords"] = [item["keyword"] for item in keywords]
        new_record["stage_keyword_text"] = " ".join(item["keyword"] for item in keywords)
        selected_sentence_records.append(new_record)

    old_to_new_section: dict[int, int] = {}
    selected_sections: list[dict] = []
    for section in payload.get("sections", []):
        old_section_index = int(section["section_index"])
        section_sentence_old_indices = [
            int(record["original_sentence_index"])
            for record in selected_sentence_records
            if int(record["section_index"]) == old_section_index
        ]
        if not section_sentence_old_indices:
            continue
        new_section_index = len(selected_sections)
        old_to_new_section[old_section_index] = new_section_index
        new_sentence_indices = [old_to_new_sentence[idx] for idx in section_sentence_old_indices]
        selected_sections.append(
            {
                "section_index": new_section_index,
                "original_section_index": old_section_index,
                "section_title": section["section_title"],
                "sentence_start": min(new_sentence_indices),
                "sentence_end": max(new_sentence_indices) + 1,
                "sentence_count": len(new_sentence_indices),
            }
        )

    for record in selected_sentence_records:
        old_section_index = int(record["section_index"])
        new_section_index = old_to_new_section[old_section_index]
        section_start = int(selected_sections[new_section_index]["sentence_start"])
        record["original_section_index"] = old_section_index
        record["section_index"] = new_section_index
        record["section_sentence_index"] = int(record["sentence_index"]) - section_start

    return selected_sections, selected_sentence_records, old_to_new_sentence, old_to_new_section


def build_keyword_graph_payload(
    payload: dict,
    sentence_features: torch.Tensor,
    lexical_weight: float,
    keep_all_sentences: bool,
    min_keywords_per_doc: int,
    concept_annotation: dict | None = None,
    attach_concepts: bool = False,
    add_concept_cooccurrence_edges: bool = True,
    keyword_embedding_map: dict[str, torch.Tensor] | None = None,
    concept_embedding_map: dict[str, torch.Tensor] | None = None,
    concept_label_weight: float = 0.7,
) -> tuple[dict | None, dict | None, str]:
    validate_payload(payload, sentence_features)

    sentence_keywords: dict[int, list[dict]] = {}
    raw_mentions = 0
    for record in payload.get("sentence_records", []):
        sentence_index = int(record["sentence_index"])
        mentions = find_stage_keywords(str(record.get("text", "")))
        raw_mentions += len(mentions)
        sentence_keywords[sentence_index] = deduplicate_sentence_keywords(mentions)

    keyword_count = sum(len(items) for items in sentence_keywords.values())
    if keyword_count < int(min_keywords_per_doc):
        return None, None, "no_stage_keywords"

    sections, sentence_records, old_to_new_sentence, _old_to_new_section = selected_sections_and_sentences(
        payload=payload,
        sentence_keywords=sentence_keywords,
        keep_all_sentences=keep_all_sentences,
    )
    if not sentence_records:
        return None, None, "no_keyword_sentences"

    selected_old_sentence_indices = [int(record["original_sentence_index"]) for record in sentence_records]
    selected_sentence_features = sentence_features[selected_old_sentence_indices].float()
    new_sentence_to_section = {
        int(record["sentence_index"]): int(record["section_index"]) for record in sentence_records
    }
    remapped_concept_annotation = (
        remap_concept_annotation_to_retained_sentences(
            concept_annotation=concept_annotation,
            old_to_new_sentence=old_to_new_sentence,
            new_sentence_to_section=new_sentence_to_section,
        )
        if attach_concepts
        else None
    )
    concept_features, concept_ids, concept_ic, concept_depth, concept_direct_mentions, concept_sentence_links = (
        build_selected_concept_features(
            concept_annotation=remapped_concept_annotation,
            selected_sentence_features=selected_sentence_features,
            concept_embedding_map=concept_embedding_map,
            concept_label_weight=concept_label_weight,
        )
    )

    section_features: list[torch.Tensor] = []
    for section in sections:
        start = int(section["sentence_start"])
        end = int(section["sentence_end"])
        section_features.append(mean_pool(selected_sentence_features[start:end]))
    section_feature_tensor = torch.stack(section_features, dim=0).float() if section_features else torch.empty((0, FEATURE_DIM))
    document_feature = mean_pool(section_feature_tensor if section_feature_tensor.numel() else selected_sentence_features).float()

    nodes: list[dict] = [
        {
            "node_index": 0,
            "node_id": "doc_0",
            "node_type": "document",
            "level": 0,
            "title": payload.get("document_id") or payload.get("file_name") or "document",
        }
    ]
    section_node_indices: list[int] = []
    sentence_node_indices: list[int] = []
    word_node_indices: list[int] = []
    concept_node_indices: dict[str, int] = {}
    word_features: list[torch.Tensor] = []

    for section in sections:
        node_index = len(nodes)
        section_node_indices.append(node_index)
        nodes.append(
            {
                "node_index": node_index,
                "node_id": f"sec_{section['section_index']}",
                "node_type": "section",
                "level": 1,
                **section,
            }
        )

    for record in sentence_records:
        node_index = len(nodes)
        sentence_node_indices.append(node_index)
        nodes.append(
            {
                "node_index": node_index,
                "node_id": f"sent_{record['sentence_index']}",
                "node_type": "sentence",
                "level": 2,
                "sentence_index": int(record["sentence_index"]),
                "original_sentence_index": int(record["original_sentence_index"]),
                "section_index": int(record["section_index"]),
                "original_section_index": int(record.get("original_section_index", record["section_index"])),
                "section_title": record["section_title"],
                "section_sentence_index": int(record["section_sentence_index"]),
                "text": record.get("text", ""),
                "stage_keyword_text": record.get("stage_keyword_text", ""),
                "stage_keywords": record.get("stage_keywords", []),
            }
        )

    sentence_to_word_indices: dict[int, list[int]] = {}
    for record in sentence_records:
        old_sentence_index = int(record["original_sentence_index"])
        new_sentence_index = int(record["sentence_index"])
        parent_sentence_feature = selected_sentence_features[new_sentence_index]
        for keyword_item in sentence_keywords.get(old_sentence_index, []):
            node_index = len(nodes)
            word_node_indices.append(node_index)
            sentence_to_word_indices.setdefault(new_sentence_index, []).append(node_index)
            keyword = str(keyword_item["keyword"])
            word_features.append(
                word_feature(
                    parent_sentence_feature,
                    keyword=keyword,
                    lexical_weight=lexical_weight,
                    keyword_embedding_map=keyword_embedding_map,
                )
            )
            nodes.append(
                {
                    "node_index": node_index,
                    "node_id": f"word_{new_sentence_index}_{len(sentence_to_word_indices[new_sentence_index]) - 1}_{keyword}",
                    "node_type": "word",
                    "level": 3,
                    "keyword": keyword,
                    "category": keyword_item["category"],
                    "matched_text": keyword_item["matched_text"],
                    "mention_count": int(keyword_item["mention_count"]),
                    "spans": keyword_item["spans"],
                    "weight": float(keyword_item["weight"]),
                    "sentence_index": new_sentence_index,
                    "original_sentence_index": old_sentence_index,
                    "section_index": int(record["section_index"]),
                }
            )

    if attach_concepts and remapped_concept_annotation:
        for concept in remapped_concept_annotation.get("concepts", []) or []:
            concept_id = str(concept.get("concept_id", ""))
            if not concept_id or concept_id in concept_node_indices:
                continue
            node_index = len(nodes)
            concept_node_indices[concept_id] = node_index
            nodes.append(
                {
                    "node_index": node_index,
                    "node_id": f"concept_{concept_id}",
                    "node_type": "concept",
                    "level": 3,
                    "concept_id": concept_id,
                    "concept_name": str(concept.get("concept_name", concept_id)),
                    "depth": int(concept.get("depth", 0)),
                    "ic": float(concept.get("ic", 0.0)),
                    "direct_mention_count": int(concept.get("direct_mention_count", 0)),
                    "is_direct": bool(concept.get("is_direct", False)),
                    "is_ancestor_only": bool(concept.get("is_ancestor_only", False)),
                    "sentence_indices": list(concept.get("sentence_indices", []) or []),
                }
            )

    edges: list[dict] = []
    for section_index, section_node_index in enumerate(section_node_indices):
        edges.append(
            {
                "source_index": 0,
                "target_index": section_node_index,
                "edge_type": "parent",
                "edge_weight": 1.0,
                "source_type": "document",
                "target_type": "section",
                "section_index": section_index,
            }
        )

    sentence_to_section: list[int] = []
    for record in sentence_records:
        sentence_index = int(record["sentence_index"])
        section_index = int(record["section_index"])
        sentence_to_section.append(section_index)
        edges.append(
            {
                "source_index": section_node_indices[section_index],
                "target_index": sentence_node_indices[sentence_index],
                "edge_type": "parent",
                "edge_weight": 1.0,
                "source_type": "section",
                "target_type": "sentence",
                "section_index": section_index,
                "sentence_index": sentence_index,
            }
        )

    for sentence_index, word_indices in sentence_to_word_indices.items():
        for order, word_node_index in enumerate(word_indices):
            word_node = nodes[word_node_index]
            edges.append(
                {
                    "source_index": sentence_node_indices[sentence_index],
                    "target_index": word_node_index,
                    "edge_type": "parent",
                    "edge_weight": float(word_node.get("weight", 1.0)),
                    "source_type": "sentence",
                    "target_type": "word",
                    "sentence_index": sentence_index,
                    "word_order": order,
                    "keyword": word_node.get("keyword"),
                }
            )
        for left, right in zip(word_indices, word_indices[1:]):
            edges.append(
                {
                    "source_index": left,
                    "target_index": right,
                    "edge_type": "next",
                    "edge_weight": 0.1,
                    "source_type": "word",
                    "target_type": "word",
                    "sentence_index": sentence_index,
                }
            )

    for left, right in zip(section_node_indices, section_node_indices[1:]):
        edges.append(
            {
                "source_index": left,
                "target_index": right,
                "edge_type": "next",
                "edge_weight": 0.25,
                "source_type": "section",
                "target_type": "section",
            }
        )
    for left, right in zip(sentence_node_indices, sentence_node_indices[1:]):
        edges.append(
            {
                "source_index": left,
                "target_index": right,
                "edge_type": "next",
                "edge_weight": 0.25,
                "source_type": "sentence",
                "target_type": "sentence",
            }
            )

    if attach_concepts and remapped_concept_annotation and concept_node_indices:
        mention_edge_keys: set[tuple[int, int, str]] = set()
        sentence_to_concepts: dict[int, set[str]] = {}
        section_to_concepts: dict[int, set[str]] = {}

        for mention in remapped_concept_annotation.get("mentions", []) or []:
            concept_id = str(mention.get("concept_id", ""))
            sentence_index = int(mention.get("sentence_index", -1))
            section_index = int(mention.get("section_index", -1))
            if concept_id not in concept_node_indices or not (0 <= sentence_index < len(sentence_node_indices)):
                continue

            source_index = sentence_node_indices[sentence_index]
            target_index = concept_node_indices[concept_id]
            edge_key = (source_index, target_index, "mention")
            if edge_key not in mention_edge_keys:
                mention_edge_keys.add(edge_key)
                edges.append(
                    {
                        "source_index": source_index,
                        "target_index": target_index,
                        "edge_type": "mention",
                        "edge_weight": ONTOLOGY_EDGE_TYPE_WEIGHTS["mention"],
                        "source_type": "sentence",
                        "target_type": "concept",
                        "sentence_index": sentence_index,
                        "original_sentence_index": mention.get("original_sentence_index"),
                        "section_index": section_index,
                        "concept_id": concept_id,
                    }
                )

            sentence_to_concepts.setdefault(sentence_index, set()).add(concept_id)
            if section_index >= 0:
                section_to_concepts.setdefault(section_index, set()).add(concept_id)

        ontology_edge_keys: set[tuple[int, int, str]] = set()
        for concept_edge in remapped_concept_annotation.get("concept_edges", []) or []:
            source_concept_id = str(concept_edge.get("source_concept_id", ""))
            target_concept_id = str(concept_edge.get("target_concept_id", ""))
            if source_concept_id not in concept_node_indices or target_concept_id not in concept_node_indices:
                continue
            edge_key = (
                concept_node_indices[source_concept_id],
                concept_node_indices[target_concept_id],
                "ontology",
            )
            if edge_key in ontology_edge_keys:
                continue
            ontology_edge_keys.add(edge_key)
            edges.append(
                {
                    "source_index": concept_node_indices[source_concept_id],
                    "target_index": concept_node_indices[target_concept_id],
                    "edge_type": "ontology",
                    "edge_weight": ONTOLOGY_EDGE_TYPE_WEIGHTS["ontology"],
                    "source_type": "concept",
                    "target_type": "concept",
                    "source_concept_id": source_concept_id,
                    "target_concept_id": target_concept_id,
                }
            )

        if add_concept_cooccurrence_edges:
            cooccurrence_edge_keys: set[tuple[int, int, str]] = set()
            for sentence_index, concept_set in sentence_to_concepts.items():
                sorted_ids = sorted(concept_set)
                for left_pos in range(len(sorted_ids)):
                    for right_pos in range(left_pos + 1, len(sorted_ids)):
                        left_id = sorted_ids[left_pos]
                        right_id = sorted_ids[right_pos]
                        edge_key = (
                            concept_node_indices[left_id],
                            concept_node_indices[right_id],
                            "same_sentence",
                        )
                        if edge_key in cooccurrence_edge_keys:
                            continue
                        cooccurrence_edge_keys.add(edge_key)
                        edges.append(
                            {
                                "source_index": concept_node_indices[left_id],
                                "target_index": concept_node_indices[right_id],
                                "edge_type": "same_sentence",
                                "edge_weight": ONTOLOGY_EDGE_TYPE_WEIGHTS["same_sentence"],
                                "source_type": "concept",
                                "target_type": "concept",
                                "sentence_index": sentence_index,
                            }
                        )

            for section_index, concept_set in section_to_concepts.items():
                sorted_ids = sorted(concept_set)
                for left_pos in range(len(sorted_ids)):
                    for right_pos in range(left_pos + 1, len(sorted_ids)):
                        left_id = sorted_ids[left_pos]
                        right_id = sorted_ids[right_pos]
                        edge_key = (
                            concept_node_indices[left_id],
                            concept_node_indices[right_id],
                            "same_section",
                        )
                        if edge_key in cooccurrence_edge_keys:
                            continue
                        cooccurrence_edge_keys.add(edge_key)
                        edges.append(
                            {
                                "source_index": concept_node_indices[left_id],
                                "target_index": concept_node_indices[right_id],
                                "edge_type": "same_section",
                                "edge_weight": ONTOLOGY_EDGE_TYPE_WEIGHTS["same_section"],
                                "source_type": "concept",
                                "target_type": "concept",
                                "section_index": section_index,
                            }
                        )

    edge_type_mapping = ONTOLOGY_EDGE_TYPE_TO_ID if attach_concepts else EDGE_TYPE_TO_ID
    node_type_mapping = ONTOLOGY_NODE_TYPE_TO_ID if attach_concepts else NODE_TYPE_TO_ID
    edge_type_weights = ONTOLOGY_EDGE_TYPE_WEIGHTS if attach_concepts else EDGE_TYPE_WEIGHTS
    edge_index = torch.tensor(
        [[int(edge["source_index"]) for edge in edges], [int(edge["target_index"]) for edge in edges]],
        dtype=torch.long,
    )
    edge_type = torch.tensor([edge_type_mapping[str(edge["edge_type"])] for edge in edges], dtype=torch.long)
    edge_weight = torch.tensor(
        [float(edge.get("edge_weight", edge_type_weights.get(str(edge["edge_type"]), 1.0))) for edge in edges],
        dtype=torch.float32,
    )

    word_feature_tensor = torch.stack(word_features, dim=0).float() if word_features else torch.empty((0, FEATURE_DIM))
    node_feature_pieces = [document_feature.view(1, -1), section_feature_tensor, selected_sentence_features]
    if word_feature_tensor.numel() > 0:
        node_feature_pieces.append(word_feature_tensor)
    if attach_concepts and concept_features.numel() > 0:
        node_feature_pieces.append(concept_features)
    node_features = torch.cat(node_feature_pieces, dim=0).float()
    node_type = torch.tensor(
        [node_type_mapping["document"]]
        + [node_type_mapping["section"]] * len(section_node_indices)
        + [node_type_mapping["sentence"]] * len(sentence_node_indices)
        + [node_type_mapping["word"]] * len(word_node_indices)
        + ([node_type_mapping["concept"]] * len(concept_node_indices) if attach_concepts else []),
        dtype=torch.long,
    )

    section_spans = torch.tensor(
        [[int(section["sentence_start"]), int(section["sentence_end"])] for section in sections],
        dtype=torch.long,
    )

    tensor_payload = {
        "document_feature": document_feature.float(),
        "section_features": section_feature_tensor.float(),
        "sentence_features": selected_sentence_features.float(),
        "word_features": word_feature_tensor.float(),
        "concept_features": concept_features.float(),
        "concept_ids": concept_ids,
        "concept_ic": concept_ic,
        "concept_depth": concept_depth,
        "concept_direct_mentions": concept_direct_mentions,
        "concept_sentence_links": concept_sentence_links,
        "node_features": node_features.float(),
        "node_type": node_type,
        "edge_index": edge_index,
        "edge_type": edge_type,
        "edge_weight": edge_weight,
        "sentence_to_section": torch.tensor(sentence_to_section, dtype=torch.long),
        "section_spans": section_spans,
        "node_type_mapping": node_type_mapping,
        "edge_type_mapping": edge_type_mapping,
        "original_sentence_indices": torch.tensor(selected_old_sentence_indices, dtype=torch.long),
    }

    category_counts: dict[str, int] = {}
    for node_index in word_node_indices:
        category = str(nodes[node_index].get("category", "keyword"))
        category_counts[category] = category_counts.get(category, 0) + 1

    json_payload = {
        "document_id": payload.get("document_id"),
        "file_name": payload.get("file_name"),
        "dataset": payload.get("dataset"),
        "filter_mode": payload.get("filter_mode"),
        "source_pdf": payload.get("source_pdf"),
        "source_sentence_view_json": payload.get("source_sentence_view_json"),
        "source_preprocessed_json": payload.get("source_preprocessed_json") or payload.get("source_sentence_json"),
        "source_concept_json": remapped_concept_annotation.get("source_json") if isinstance(remapped_concept_annotation, dict) else None,
        "embedding_path": payload.get("embedding_path"),
        "pooling": "mean",
        "feature_dim": FEATURE_DIM,
        "graph_variant": "stage_keyword_ontology_hierarchy_graph" if attach_concepts else "stage_keyword_hierarchy_graph",
        "graph_cleanup": {
            **payload.get("graph_cleanup", {}),
            "keep_all_sentences": bool(keep_all_sentences),
            "raw_keyword_mentions": int(raw_mentions),
            "keyword_node_count": int(len(word_node_indices)),
            "selected_sentence_count": int(len(sentence_records)),
            "original_sentence_count": int(payload.get("sentence_count", 0)),
            "keyword_embedding_source": "conch_label_embedding" if keyword_embedding_map else "hash_fallback",
            "concept_embedding_source": "conch_label_embedding" if concept_embedding_map else "sentence_mean_fallback",
            "concept_label_weight": float(concept_label_weight),
        },
        "node_type_mapping": node_type_mapping,
        "edge_type_mapping": edge_type_mapping,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "node_counts": {
            "document": 1,
            "section": len(section_node_indices),
            "sentence": len(sentence_node_indices),
            "word": len(word_node_indices),
            "concept": len(concept_node_indices),
        },
        "concept_count": len(concept_node_indices),
        "concept_annotation_summary": {
            "mention_count": int(remapped_concept_annotation.get("mention_count", 0)) if remapped_concept_annotation else 0,
            "direct_concept_count": int(remapped_concept_annotation.get("direct_concept_count", 0)) if remapped_concept_annotation else 0,
        },
        "keyword_category_counts": category_counts,
        "nodes": nodes,
        "edges": edges,
    }
    return tensor_payload, json_payload, ""


def build_graph_for_document(
    metadata_json_path: Path,
    input_root: Path,
    output_root: Path,
    lexical_weight: float,
    keep_all_sentences: bool,
    min_keywords_per_doc: int,
    concept_dir: Path | None = None,
    attach_concepts: bool = False,
    add_concept_cooccurrence_edges: bool = True,
    keyword_embedding_map: dict[str, torch.Tensor] | None = None,
    concept_embedding_map: dict[str, torch.Tensor] | None = None,
    concept_label_weight: float = 0.7,
) -> dict:
    payload = clean_payload(load_json(metadata_json_path), metadata_json_path=metadata_json_path, input_root=input_root)
    relative_path = metadata_json_path.relative_to(input_root)
    if Path(output_root).resolve() == DEFAULT_HIERARCHY_GRAPH_ROOT.resolve():
        relative_path = resolve_library_relative_path(relative_path, attach_concepts=attach_concepts)
    output_json_path = output_root / relative_path
    output_pt_path = output_root / relative_path.with_suffix(".pt")
    ensure_dir(output_json_path.parent)

    embedding_path = metadata_json_path.with_suffix(".pt")
    if not embedding_path.exists():
        embedding_path = Path(payload["embedding_path"])
    payload["embedding_path"] = str(embedding_path)
    sentence_features = torch.load(embedding_path, map_location="cpu")
    concept_annotation, concept_json_path = load_optional_concept_annotation(
        metadata_json_path=metadata_json_path,
        input_root=input_root,
        concept_dir=concept_dir if attach_concepts else None,
    )
    if concept_annotation is not None:
        concept_annotation["source_json"] = str(concept_json_path or "")

    tensor_payload, json_payload, skip_reason = build_keyword_graph_payload(
        payload=payload,
        sentence_features=sentence_features,
        lexical_weight=lexical_weight,
        keep_all_sentences=keep_all_sentences,
        min_keywords_per_doc=min_keywords_per_doc,
        concept_annotation=concept_annotation,
        attach_concepts=attach_concepts,
        add_concept_cooccurrence_edges=add_concept_cooccurrence_edges,
        keyword_embedding_map=keyword_embedding_map,
        concept_embedding_map=concept_embedding_map,
        concept_label_weight=concept_label_weight,
    )
    if tensor_payload is None or json_payload is None:
        if output_json_path.exists():
            output_json_path.unlink()
        if output_pt_path.exists():
            output_pt_path.unlink()
        return {
            "document_id": payload.get("document_id"),
            "file_name": payload.get("file_name"),
            "dataset": payload.get("dataset"),
            "source_embedding_json": str(metadata_json_path),
            "source_embedding_pt": str(embedding_path),
            "output_graph_json": None,
            "output_graph_pt": None,
            "node_count": 0,
            "edge_count": 0,
            "sentence_count": int(payload.get("sentence_count", 0)),
            "section_count": int(payload.get("section_count", 0)),
            "keyword_count": 0,
            "concept_count": 0,
            "status": "skipped",
            "skip_reason": skip_reason,
        }

    torch.save(tensor_payload, output_pt_path)
    json_payload["graph_tensor_path"] = str(output_pt_path)
    write_json(output_json_path, json_payload)
    return {
        "document_id": payload.get("document_id"),
        "file_name": payload.get("file_name"),
        "dataset": payload.get("dataset"),
        "source_embedding_json": str(metadata_json_path),
        "source_embedding_pt": str(embedding_path),
        "output_graph_json": str(output_json_path),
        "output_graph_pt": str(output_pt_path),
        "node_count": int(json_payload["node_count"]),
        "edge_count": int(json_payload["edge_count"]),
        "sentence_count": int(json_payload["node_counts"]["sentence"]),
        "section_count": int(json_payload["node_counts"]["section"]),
        "keyword_count": int(json_payload["node_counts"]["word"]),
        "concept_count": int(json_payload["node_counts"].get("concept", 0)),
        "status": "success",
        "skip_reason": "",
    }


def dataset_allowed(path: Path, input_root: Path, datasets: set[str] | None) -> bool:
    if not datasets:
        return True
    try:
        relative = path.relative_to(input_root)
    except ValueError:
        return True
    if not relative.parts:
        return False
    return relative.parts[0].upper() in datasets


def process_all_documents(
    input_dir: Path,
    output_dir: Path,
    datasets: list[str] | None,
    lexical_weight: float,
    keep_all_sentences: bool,
    min_keywords_per_doc: int,
    concept_dir: Path | None,
    attach_concepts: bool,
    add_concept_cooccurrence_edges: bool,
    keyword_embedding_map: dict[str, torch.Tensor] | None,
    concept_embedding_map: dict[str, torch.Tensor] | None,
    concept_label_weight: float,
    limit: int | None,
) -> dict:
    input_root = input_dir.resolve()
    output_root = output_dir.resolve()
    ensure_dir(output_root)
    allowed_datasets = {item.upper() for item in datasets} if datasets else None
    metadata_paths = [
        path for path in iter_metadata_jsons(input_root) if dataset_allowed(path, input_root, allowed_datasets)
    ]
    if limit is not None:
        metadata_paths = metadata_paths[: int(limit)]

    rows: list[dict] = []
    for index, metadata_json_path in enumerate(metadata_paths, start=1):
        LOGGER.info("[%s/%s] Building keyword graph for %s", index, len(metadata_paths), metadata_json_path)
        try:
            row = build_graph_for_document(
                metadata_json_path=metadata_json_path,
                input_root=input_root,
                output_root=output_root,
                lexical_weight=lexical_weight,
                keep_all_sentences=keep_all_sentences,
                min_keywords_per_doc=min_keywords_per_doc,
                concept_dir=concept_dir,
                attach_concepts=attach_concepts,
                add_concept_cooccurrence_edges=add_concept_cooccurrence_edges,
                keyword_embedding_map=keyword_embedding_map,
                concept_embedding_map=concept_embedding_map,
                concept_label_weight=concept_label_weight,
            )
        except Exception as exc:  # pragma: no cover - logged in batch runs
            LOGGER.exception("Failed to build keyword graph for %s", metadata_json_path)
            row = {
                "document_id": metadata_json_path.stem,
                "file_name": metadata_json_path.name,
                "dataset": metadata_json_path.parent.name,
                "source_embedding_json": str(metadata_json_path),
                "output_graph_json": None,
                "output_graph_pt": None,
                "node_count": 0,
                "edge_count": 0,
                "sentence_count": 0,
                "section_count": 0,
                "keyword_count": 0,
                "concept_count": 0,
                "status": "failed",
                "skip_reason": str(exc),
            }
        rows.append(row)

    success_rows = [row for row in rows if row.get("status") == "success"]
    skipped_rows = [row for row in rows if row.get("status") == "skipped"]
    failed_rows = [row for row in rows if row.get("status") == "failed"]
    summary = {
        "input_dir": str(input_root),
        "output_dir": str(output_root),
        "datasets": sorted(allowed_datasets) if allowed_datasets else "all",
        "total_metadata_files": len(metadata_paths),
        "success_count": len(success_rows),
        "skipped_count": len(skipped_rows),
        "failure_count": len(failed_rows),
        "total_keyword_nodes": int(sum(int(row.get("keyword_count", 0)) for row in success_rows)),
        "total_concept_nodes": int(sum(int(row.get("concept_count", 0)) for row in success_rows)),
        "total_sentence_nodes": int(sum(int(row.get("sentence_count", 0)) for row in success_rows)),
        "keep_all_sentences": bool(keep_all_sentences),
        "attach_concepts": bool(attach_concepts),
        "concept_dir": str(concept_dir) if concept_dir is not None else None,
        "add_concept_cooccurrence_edges": bool(add_concept_cooccurrence_edges),
        "lexical_weight": float(lexical_weight),
        "keyword_embedding_count": len(keyword_embedding_map or {}),
        "concept_embedding_count": len(concept_embedding_map or {}),
        "concept_label_weight": float(concept_label_weight),
        "min_keywords_per_doc": int(min_keywords_per_doc),
        "rows": rows,
    }
    write_json(output_root / "run_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper()))
    keyword_embedding_map = load_embedding_map(args.keyword_embedding_pt)
    concept_embedding_map = load_embedding_map(args.concept_embedding_pt)
    summary = process_all_documents(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        datasets=args.datasets,
        lexical_weight=args.lexical_weight,
        keep_all_sentences=args.keep_all_sentences,
        min_keywords_per_doc=args.min_keywords_per_doc,
        concept_dir=args.concept_dir,
        attach_concepts=args.attach_concepts,
        add_concept_cooccurrence_edges=not args.disable_concept_cooccurrence_edges,
        keyword_embedding_map=keyword_embedding_map,
        concept_embedding_map=concept_embedding_map,
        concept_label_weight=args.concept_label_weight,
        limit=args.limit,
    )
    print(
        "Built stage-keyword graphs for "
        f"{summary['total_metadata_files']} files | success={summary['success_count']} | "
        f"skipped={summary['skipped_count']} | failed={summary['failure_count']} | "
        f"keyword_nodes={summary['total_keyword_nodes']} | concept_nodes={summary['total_concept_nodes']}"
    )


if __name__ == "__main__":
    main()
