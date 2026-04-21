from __future__ import annotations

"""Build project ontology resources for concept extraction and concept graphs.

This script now supports a multi-ontology oncology bundle with the following
roles:

- NCIt: core canonical oncology concepts for normalization
- SNOMED CT: broader lexical coverage for mention finding
- UMLS: alignment layer across terminologies via shared CUIs
- DO: disease hierarchy supplementation

The emitted bundle remains compatible with `extract_ontology_concepts.py`.
"""

import argparse
import json
import logging
import math
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET


LOGGER = logging.getLogger("build_project_ontology_resources")

DEFAULT_ONTOLOGY_ROOT = Path(r"F:\Tasks\Ontologies")
DEFAULT_RAW_ROOT = DEFAULT_ONTOLOGY_ROOT / "raw"
DEFAULT_OUTPUT_ROOT = DEFAULT_ONTOLOGY_ROOT / "processed"

RDF_ABOUT = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about"
RDF_RESOURCE = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource"
OWL_CLASS = "{http://www.w3.org/2002/07/owl#}Class"
OWL_ONTOLOGY = "{http://www.w3.org/2002/07/owl#}Ontology"
RDFS_LABEL = "{http://www.w3.org/2000/01/rdf-schema#}label"
RDFS_SUBCLASS = "{http://www.w3.org/2000/01/rdf-schema#}subClassOf"
OWL_VERSION_INFO = "{http://www.w3.org/2002/07/owl#}versionInfo"
NCIT_NS = "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#"
DOID_PREFIX = "http://purl.obolibrary.org/obo/"
OBOINOWL_NS = "{http://www.geneontology.org/formats/oboInOwl#}"
OBO_NS = "{http://purl.obolibrary.org/obo/}"

SNOMED_FSN_TYPE_ID = "900000000000003001"
SNOMED_SYNONYM_TYPE_ID = "900000000000013009"

NCIT_PATHOLOGY_INCLUDE_CUES = (
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
    "sarcomatoid",
    "rhabdoid",
)
NCIT_PATHOLOGY_EXCLUDE_CUES = (
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
SNOMED_ALLOWED_SEMANTIC_TAGS = {
    "disorder",
    "finding",
    "morphologic abnormality",
    "procedure",
}


def ensure_dir(path: Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _text(elem: ET.Element | None) -> str:
    return str(elem.text or "").strip() if elem is not None else ""


def _normalize_text(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _matches_pathology_text(value: str) -> bool:
    normalized = _normalize_text(value)
    if not normalized:
        return False
    if any(cue in normalized for cue in NCIT_PATHOLOGY_EXCLUDE_CUES):
        return False
    return any(cue in normalized for cue in NCIT_PATHOLOGY_INCLUDE_CUES)


def _safe_add_synonym(bucket: list[str], value: str) -> None:
    token = str(value or "").strip()
    if token and token not in bucket:
        bucket.append(token)


def _dedupe_sorted(values: Iterable[str]) -> list[str]:
    return sorted({str(value).strip() for value in values if str(value).strip()}, key=str.lower)


def _resource_to_fragment(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if "#" in text:
        return text.rsplit("#", 1)[-1]
    if "/" in text:
        return text.rsplit("/", 1)[-1]
    return text


def _append_xref(record: dict[str, Any], key: str, value: str) -> None:
    token = str(value or "").strip()
    if not token:
        return
    xrefs = record.setdefault("xrefs", {})
    xrefs.setdefault(key, [])
    if token not in xrefs[key]:
        xrefs[key].append(token)


def _parse_ncit_xref(text: str, xrefs: dict[str, list[str]]) -> None:
    token = str(text or "").strip()
    if token.startswith("C") and token[1:].isdigit():
        xrefs.setdefault("umls", [])
        if token not in xrefs["umls"]:
            xrefs["umls"].append(token)


def _parse_do_xref(text: str, xrefs: dict[str, list[str]]) -> None:
    token = str(text or "").strip()
    if not token or ":" not in token:
        return
    prefix, rest = token.split(":", 1)
    prefix = prefix.strip()
    rest = rest.strip()
    if not rest:
        return

    if prefix == "NCI":
        key = "ncit"
        value = f"NCIT:{rest}"
    elif prefix.startswith("SNOMEDCT"):
        key = "snomed_ct"
        value = rest
    elif prefix == "UMLS_CUI":
        key = "umls"
        value = rest
    elif prefix == "MESH":
        key = "mesh"
        value = rest
    else:
        key = prefix.lower()
        value = rest

    xrefs.setdefault(key, [])
    if value not in xrefs[key]:
        xrefs[key].append(value)


def _clone_xrefs(payload: dict[str, Any]) -> dict[str, list[str]]:
    source = payload.get("xrefs", {}) if isinstance(payload, dict) else {}
    return {
        str(key): _dedupe_sorted(values)
        for key, values in source.items()
        if isinstance(values, list)
    }


def _iter_ncit_class_elements(ncit_zip_path: Path) -> Iterable[ET.Element]:
    with zipfile.ZipFile(ncit_zip_path) as zf:
        owl_entries = [entry for entry in zf.namelist() if entry.lower().endswith(".owl")]
        if not owl_entries:
            raise FileNotFoundError(f"No OWL entry found in zip: {ncit_zip_path}")
        with zf.open(owl_entries[0], "r") as handle:
            context = ET.iterparse(handle, events=("end",))
            for _event, elem in context:
                if elem.tag == OWL_CLASS:
                    yield elem
                    elem.clear()


def _parse_ncit_version(ncit_zip_path: Path) -> str | None:
    with zipfile.ZipFile(ncit_zip_path) as zf:
        owl_entries = [entry for entry in zf.namelist() if entry.lower().endswith(".owl")]
        if not owl_entries:
            return None
        with zf.open(owl_entries[0], "r") as handle:
            context = ET.iterparse(handle, events=("end",))
            for _event, elem in context:
                if elem.tag == OWL_VERSION_INFO:
                    version = _text(elem)
                    elem.clear()
                    return version or None
                if elem.tag == OWL_ONTOLOGY:
                    elem.clear()
    return None


def build_ncit_resource(ncit_zip_path: Path) -> dict[str, Any]:
    LOGGER.info("Parsing NCIt from %s", ncit_zip_path)
    concepts: dict[str, dict[str, Any]] = {}

    for class_elem in _iter_ncit_class_elements(ncit_zip_path):
        about = class_elem.attrib.get(RDF_ABOUT, "")
        code = ""
        label = ""
        parents: list[str] = []
        synonyms: list[str] = []
        definition = ""
        xrefs: dict[str, list[str]] = {}

        for child in class_elem:
            tag = _local_name(child.tag)
            if tag == "NHC0":
                code = _text(child)
            elif child.tag == RDFS_LABEL:
                label = _text(child)
            elif child.tag == RDFS_SUBCLASS:
                parent = _resource_to_fragment(child.attrib.get(RDF_RESOURCE, ""))
                if parent:
                    parents.append(parent)
            elif tag in {"P90", "P108"}:
                _safe_add_synonym(synonyms, _text(child))
            elif tag == "P97" and not definition:
                definition = _text(child)
            elif tag == "P207":
                _parse_ncit_xref(_text(child), xrefs)

        if not code:
            code = _resource_to_fragment(about)
        if not code.startswith("C") or not code[1:].isdigit() or not label:
            continue

        _safe_add_synonym(synonyms, label)
        concepts[f"NCIT:{code}"] = {
            "name": label,
            "synonyms": _dedupe_sorted(synonyms),
            "parents": _dedupe_sorted(f"NCIT:{parent}" for parent in parents if parent),
            "definition": definition,
            "xrefs": xrefs,
            "source_terminology": "NCIt",
            "canonical_source": "NCIt",
            "match_enabled": True,
        }

    return {
        "ontology_name": "NCIt",
        "ontology_version": _parse_ncit_version(ncit_zip_path),
        "concepts": concepts,
    }


def _iter_do_class_elements(do_owl_path: Path) -> Iterable[ET.Element]:
    with do_owl_path.open("rb") as handle:
        context = ET.iterparse(handle, events=("end",))
        for _event, elem in context:
            if elem.tag == OWL_CLASS:
                yield elem
                elem.clear()


def _parse_do_version(do_owl_path: Path) -> str | None:
    with do_owl_path.open("rb") as handle:
        context = ET.iterparse(handle, events=("end",))
        for _event, elem in context:
            if elem.tag == OWL_VERSION_INFO:
                version = _text(elem)
                elem.clear()
                return version or None
            if elem.tag == OWL_ONTOLOGY:
                elem.clear()
    return None


def _normalize_doid_id(raw: str) -> str:
    token = str(raw or "").strip()
    if token.startswith("DOID:"):
        return token
    if token.startswith("DOID_"):
        return token.replace("DOID_", "DOID:", 1)
    if token.startswith(DOID_PREFIX):
        frag = token.rsplit("/", 1)[-1]
        if frag.startswith("DOID_"):
            return frag.replace("DOID_", "DOID:", 1)
    return token


def build_do_resource(do_owl_path: Path) -> dict[str, Any]:
    LOGGER.info("Parsing DO from %s", do_owl_path)
    concepts: dict[str, dict[str, Any]] = {}

    for class_elem in _iter_do_class_elements(do_owl_path):
        about = class_elem.attrib.get(RDF_ABOUT, "")
        code = ""
        label = ""
        parents: list[str] = []
        synonyms: list[str] = []
        definition = ""
        xrefs: dict[str, list[str]] = {}

        for child in class_elem:
            if child.tag == RDFS_LABEL:
                label = _text(child)
            elif child.tag == RDFS_SUBCLASS:
                parent = _normalize_doid_id(child.attrib.get(RDF_RESOURCE, ""))
                if parent:
                    parents.append(parent)
            elif child.tag == f"{OBOINOWL_NS}id":
                code = _normalize_doid_id(_text(child))
            elif child.tag in {
                f"{OBOINOWL_NS}hasExactSynonym",
                f"{OBOINOWL_NS}hasRelatedSynonym",
                f"{OBOINOWL_NS}hasBroadSynonym",
                f"{OBOINOWL_NS}hasNarrowSynonym",
            }:
                _safe_add_synonym(synonyms, _text(child))
            elif child.tag == f"{OBO_NS}IAO_0000115" and not definition:
                definition = _text(child)
            elif child.tag == f"{OBOINOWL_NS}hasDbXref":
                _parse_do_xref(_text(child), xrefs)

        if not code:
            code = _normalize_doid_id(_resource_to_fragment(about))
        if not code.startswith("DOID:") or not label:
            continue

        _safe_add_synonym(synonyms, label)
        concepts[code] = {
            "name": label,
            "synonyms": _dedupe_sorted(synonyms),
            "parents": _dedupe_sorted(parent for parent in parents if parent.startswith("DOID:")),
            "definition": definition,
            "xrefs": xrefs,
            "source_terminology": "DO",
            "canonical_source": "DO",
            "match_enabled": False,
        }

    return {
        "ontology_name": "DO",
        "ontology_version": _parse_do_version(do_owl_path),
        "concepts": concepts,
    }


def _compute_depths(concepts: dict[str, dict[str, Any]]) -> dict[str, int]:
    memo: dict[str, int] = {}
    visiting: set[str] = set()

    def depth(concept_id: str) -> int:
        if concept_id in memo:
            return memo[concept_id]
        if concept_id in visiting:
            return 0
        visiting.add(concept_id)
        parents = [parent for parent in concepts.get(concept_id, {}).get("parents", []) if parent in concepts]
        result = 0 if not parents else 1 + max(depth(parent) for parent in parents)
        visiting.discard(concept_id)
        memo[concept_id] = result
        return result

    for concept_id in concepts:
        depth(concept_id)
    return memo


def _compute_descendant_counts(concepts: dict[str, dict[str, Any]]) -> dict[str, int]:
    children: dict[str, list[str]] = defaultdict(list)
    for concept_id, record in concepts.items():
        for parent in record.get("parents", []):
            if parent in concepts:
                children[parent].append(concept_id)

    memo: dict[str, set[str]] = {}
    visiting: set[str] = set()

    def descendants(concept_id: str) -> set[str]:
        if concept_id in memo:
            return memo[concept_id]
        if concept_id in visiting:
            return set()
        visiting.add(concept_id)
        acc: set[str] = set()
        for child in children.get(concept_id, []):
            acc.add(child)
            acc.update(descendants(child))
        visiting.discard(concept_id)
        memo[concept_id] = acc
        return acc

    return {concept_id: len(descendants(concept_id)) for concept_id in concepts}


def _normalize_synonym_records(record: dict[str, Any]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, str, str]] = set()
    normalized: list[dict[str, Any]] = []
    raw_records = record.get("synonym_records", []) or []
    for item in raw_records:
        if not isinstance(item, dict):
            continue
        term = str(item.get("term") or "").strip()
        if not term:
            continue
        source_terminology = str(item.get("source_terminology") or "").strip()
        source_id = str(item.get("source_id") or "").strip()
        origin = str(item.get("origin") or "").strip()
        umls_cui = str(item.get("umls_cui") or "").strip()
        key = (_normalize_text(term), source_terminology, source_id, origin, umls_cui)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(
            {
                "term": term,
                "source_terminology": source_terminology,
                "source_id": source_id,
                "origin": origin,
                **({"umls_cui": umls_cui} if umls_cui else {}),
            }
        )
    normalized.sort(
        key=lambda item: (
            _normalize_text(item.get("term", "")),
            item.get("source_terminology", ""),
            item.get("source_id", ""),
            item.get("origin", ""),
        )
    )
    return normalized


def finalize_resource(resource: dict[str, Any]) -> dict[str, Any]:
    concepts = resource["concepts"]
    total = len(concepts)
    depths = _compute_depths(concepts)
    descendant_counts = _compute_descendant_counts(concepts)

    for concept_id, record in concepts.items():
        depth = int(depths.get(concept_id, 0))
        descendant_count = int(descendant_counts.get(concept_id, 0))
        ic = math.log((total + 1.0) / (descendant_count + 1.0))
        synonym_records = _normalize_synonym_records(record)
        lexical_terms = [str(term).strip() for term in record.get("synonyms", []) or [] if str(term).strip()]
        lexical_terms.extend(item["term"] for item in synonym_records)

        record["depth"] = depth
        record["descendant_count"] = descendant_count
        record["ic"] = round(float(ic), 6)
        record["synonyms"] = _dedupe_sorted(lexical_terms)
        record["parents"] = _dedupe_sorted(record.get("parents", []) or [])
        record["xrefs"] = {
            str(key): _dedupe_sorted(values)
            for key, values in (record.get("xrefs", {}) or {}).items()
            if isinstance(values, list)
        }
        if synonym_records:
            record["synonym_records"] = synonym_records
        if "aligned_sources" in record:
            record["aligned_sources"] = _dedupe_sorted(record.get("aligned_sources", []) or [])
        if "match_enabled" not in record:
            record["match_enabled"] = True

    resource["concept_count"] = total
    resource["max_depth"] = max(depths.values()) if depths else 0
    resource["match_enabled_concept_count"] = sum(1 for item in concepts.values() if item.get("match_enabled", True))
    return resource


def _matches_pathology_subset(record: dict[str, Any]) -> bool:
    concept_name = _normalize_text(record.get("name", ""))
    if not concept_name:
        return False
    if any(cue in concept_name for cue in NCIT_PATHOLOGY_EXCLUDE_CUES):
        return False
    if any(cue in concept_name for cue in NCIT_PATHOLOGY_INCLUDE_CUES):
        return True

    for synonym in record.get("synonyms", []) or []:
        if _matches_pathology_text(str(synonym)):
            return True
    return False


def _collect_ancestor_closure(concepts: dict[str, dict[str, Any]], seed_ids: set[str]) -> set[str]:
    keep_ids = set(seed_ids)
    queue = list(seed_ids)
    while queue:
        concept_id = queue.pop()
        parents = concepts.get(concept_id, {}).get("parents", []) or []
        for parent_id in parents:
            if parent_id in concepts and parent_id not in keep_ids:
                keep_ids.add(parent_id)
                queue.append(parent_id)
    return keep_ids


def build_ncit_pathology_subset(ncit_resource: dict[str, Any]) -> dict[str, Any]:
    concepts = ncit_resource["concepts"]
    seed_ids = {concept_id for concept_id, record in concepts.items() if _matches_pathology_subset(record)}
    keep_ids = _collect_ancestor_closure(concepts, seed_ids)

    subset_concepts: dict[str, dict[str, Any]] = {}
    for concept_id in sorted(keep_ids):
        record = concepts[concept_id]
        subset_concepts[concept_id] = {
            "name": record.get("name", concept_id),
            "synonyms": list(record.get("synonyms", []) or []),
            "parents": [parent for parent in record.get("parents", []) or [] if parent in keep_ids],
            "definition": record.get("definition", ""),
            "xrefs": _clone_xrefs(record),
            "source_terminology": record.get("source_terminology", "NCIt"),
            "canonical_source": "NCIt",
            "match_enabled": True,
        }

    subset_resource = {
        "ontology_name": "NCIt Pathology Subset",
        "ontology_version": ncit_resource.get("ontology_version"),
        "source_ontology_name": ncit_resource.get("ontology_name", "NCIt"),
        "subset_strategy": {
            "type": "cue_filter_plus_ancestor_closure",
            "seed_concept_count": len(seed_ids),
            "include_cues": list(NCIT_PATHOLOGY_INCLUDE_CUES),
            "exclude_cues": list(NCIT_PATHOLOGY_EXCLUDE_CUES),
        },
        "concepts": subset_concepts,
    }
    return finalize_resource(subset_resource)


def build_crosswalks(
    ncit_resource: dict[str, Any] | None,
    do_resource: dict[str, Any] | None,
) -> dict[str, Any]:
    crosswalk: dict[str, Any] = {
        "do_to_ncit": {},
        "ncit_to_doid": {},
        "snomed_to_ncit": {},
        "ncit_to_snomed": {},
        "snomed_to_doid": {},
        "doid_to_snomed": {},
    }
    if do_resource is None:
        return crosswalk

    for doid, record in do_resource["concepts"].items():
        xrefs = record.get("xrefs", {})
        ncit_ids = _dedupe_sorted(xrefs.get("ncit", []) or [])
        if ncit_ids:
            crosswalk["do_to_ncit"][doid] = ncit_ids
            for ncit_id in ncit_ids:
                crosswalk["ncit_to_doid"].setdefault(ncit_id, [])
                if doid not in crosswalk["ncit_to_doid"][ncit_id]:
                    crosswalk["ncit_to_doid"][ncit_id].append(doid)

        for raw_snomed_id in xrefs.get("snomed_ct", []) or []:
            full_snomed_id = f"SNOMEDCT:{raw_snomed_id}"
            crosswalk["snomed_to_doid"].setdefault(full_snomed_id, [])
            if doid not in crosswalk["snomed_to_doid"][full_snomed_id]:
                crosswalk["snomed_to_doid"][full_snomed_id].append(doid)
            crosswalk["doid_to_snomed"].setdefault(doid, [])
            if full_snomed_id not in crosswalk["doid_to_snomed"][doid]:
                crosswalk["doid_to_snomed"][doid].append(full_snomed_id)

    for key in ("ncit_to_doid", "snomed_to_doid", "doid_to_snomed"):
        for concept_id in list(crosswalk[key].keys()):
            crosswalk[key][concept_id].sort()
    return crosswalk


def collect_relevant_umls_cuis(*resources: dict[str, Any] | None) -> set[str]:
    cuis: set[str] = set()
    for resource in resources:
        if resource is None:
            continue
        for record in resource.get("concepts", {}).values():
            for cui in record.get("xrefs", {}).get("umls", []) or []:
                token = str(cui).strip()
                if token:
                    cuis.add(token)
    return cuis


def parse_targeted_umls_alignment(
    mrconso_path: Path,
    relevant_cuis: set[str],
) -> dict[str, Any]:
    LOGGER.info("Parsing targeted UMLS alignment from %s | relevant_cuis=%s", mrconso_path, len(relevant_cuis))
    alignment: dict[str, Any] = {
        "relevant_cui_count": len(relevant_cuis),
        "matched_cui_count": 0,
        "cui_to_ncit": {},
        "cui_to_snomed": {},
        "cui_to_doid": {},
    }
    if not relevant_cuis:
        return alignment

    matched_cuis: set[str] = set()
    with mrconso_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_index, line in enumerate(handle, start=1):
            parts = line.rstrip("\n").split("|")
            if len(parts) < 15:
                continue
            cui = parts[0].strip()
            if cui not in relevant_cuis:
                continue
            lat = parts[1].strip()
            if lat != "ENG":
                continue

            sab = parts[11].strip()
            code = parts[13].strip()
            matched_cuis.add(cui)

            if sab in {"NCI", "NCIT"} and code.startswith("C") and code[1:].isdigit():
                alignment["cui_to_ncit"].setdefault(cui, set()).add(f"NCIT:{code}")
            elif sab.startswith("SNOMEDCT") and code.isdigit():
                alignment["cui_to_snomed"].setdefault(cui, set()).add(code)
            elif sab == "DOID":
                doid = _normalize_doid_id(code)
                if doid.startswith("DOID:"):
                    alignment["cui_to_doid"].setdefault(cui, set()).add(doid)

            if line_index % 2_000_000 == 0:
                LOGGER.info("UMLS scan progress | lines=%s | matched_cuis=%s", line_index, len(matched_cuis))

    alignment["matched_cui_count"] = len(matched_cuis)
    for key in ("cui_to_ncit", "cui_to_snomed", "cui_to_doid"):
        alignment[key] = {
            cui: sorted(values)
            for cui, values in alignment[key].items()
            if values
        }
    return alignment


def _find_first_path(root: Path, patterns: list[str], expect_dir: bool = False) -> Path | None:
    for pattern in patterns:
        matches = sorted(root.rglob(pattern))
        if matches:
            for match in matches:
                if expect_dir and match.is_dir():
                    return match
                if not expect_dir and match.is_file():
                    return match
    return None


def _discover_ncit_zip(raw_root: Path) -> Path:
    path = _find_first_path(raw_root / "NCIt", ["*.OWL.zip"])
    if path is None:
        raise FileNotFoundError(f"Could not locate NCIt OWL zip under {raw_root}")
    return path


def _discover_do_owl(raw_root: Path) -> Path:
    path = _find_first_path(raw_root / "DO", ["doid.owl"])
    if path is None:
        raise FileNotFoundError(f"Could not locate doid.owl under {raw_root}")
    return path


def _discover_umls_mrconso(raw_root: Path) -> Path:
    path = _find_first_path(raw_root, ["MRCONSO.RRF"])
    if path is None:
        raise FileNotFoundError(f"Could not locate MRCONSO.RRF under {raw_root}")
    return path


def _discover_snomed_root(raw_root: Path) -> Path:
    concept_file = _find_first_path(raw_root, ["sct2_Concept_Snapshot_*.txt"])
    if concept_file is None:
        raise FileNotFoundError(f"Could not locate SNOMED CT concept snapshot under {raw_root}")
    return concept_file.parents[2]


def _split_snomed_fsn(term: str) -> tuple[str, str]:
    text = str(term or "").strip()
    match = re.match(r"^(?P<label>.+?)\s+\((?P<tag>[^()]+)\)$", text)
    if not match:
        return text, ""
    return match.group("label").strip(), match.group("tag").strip().lower()


def _load_active_snomed_concepts(concept_path: Path) -> set[str]:
    LOGGER.info("Loading active SNOMED concepts from %s", concept_path)
    active_ids: set[str] = set()
    with concept_path.open("r", encoding="utf-8", errors="ignore") as handle:
        next(handle, None)
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5 or parts[2] != "1":
                continue
            concept_id = parts[0].strip()
            if concept_id:
                active_ids.add(concept_id)
    return active_ids


def _scan_snomed_descriptions(
    description_path: Path,
    active_ids: set[str],
    external_seed_ids: set[str],
) -> tuple[dict[str, list[dict[str, str]]], dict[str, str]]:
    LOGGER.info("Scanning SNOMED descriptions from %s", description_path)
    collected: dict[str, list[dict[str, str]]] = defaultdict(list)
    semantic_tags: dict[str, str] = {}

    with description_path.open("r", encoding="utf-8", errors="ignore") as handle:
        next(handle, None)
        for line_index, line in enumerate(handle, start=1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8 or parts[2] != "1":
                continue
            concept_id = parts[4].strip()
            if concept_id not in active_ids:
                continue
            if parts[5].strip().lower() != "en":
                continue

            type_id = parts[6].strip()
            raw_term = parts[7].strip()
            if not raw_term:
                continue

            semantic_tag = semantic_tags.get(concept_id, "")
            term = raw_term
            if type_id == SNOMED_FSN_TYPE_ID:
                term, semantic_tag = _split_snomed_fsn(raw_term)
                if semantic_tag:
                    semantic_tags[concept_id] = semantic_tag

            if concept_id not in external_seed_ids and not _matches_pathology_text(term):
                continue

            collected[concept_id].append(
                {
                    "term": term,
                    "type": "fsn" if type_id == SNOMED_FSN_TYPE_ID else "synonym",
                    "semantic_tag": semantic_tag,
                }
            )

            if line_index % 1_000_000 == 0:
                LOGGER.info("SNOMED description scan progress | lines=%s | kept_concepts=%s", line_index, len(collected))

    return collected, semantic_tags


def build_snomed_resource(
    snomed_root: Path,
    do_resource: dict[str, Any] | None,
    umls_alignment: dict[str, Any],
) -> dict[str, Any]:
    concept_path = _find_first_path(snomed_root, ["sct2_Concept_Snapshot_*.txt"])
    description_path = _find_first_path(snomed_root, ["sct2_Description_Snapshot-en_*.txt"])
    if concept_path is None or description_path is None:
        raise FileNotFoundError(f"Missing SNOMED CT snapshot files under {snomed_root}")

    do_snomed_to_doids: dict[str, set[str]] = defaultdict(set)
    if do_resource is not None:
        for doid, record in do_resource.get("concepts", {}).items():
            for raw_snomed_id in record.get("xrefs", {}).get("snomed_ct", []) or []:
                token = str(raw_snomed_id).strip()
                if token:
                    do_snomed_to_doids[token].add(doid)

    umls_snomed_ids = {
        snomed_id
        for snomed_ids in umls_alignment.get("cui_to_snomed", {}).values()
        for snomed_id in snomed_ids
    }
    external_seed_ids = set(do_snomed_to_doids.keys()) | umls_snomed_ids

    active_ids = _load_active_snomed_concepts(concept_path)
    collected, semantic_tags = _scan_snomed_descriptions(
        description_path=description_path,
        active_ids=active_ids,
        external_seed_ids=external_seed_ids,
    )

    snomed_to_cuis: dict[str, set[str]] = defaultdict(set)
    for cui, snomed_ids in umls_alignment.get("cui_to_snomed", {}).items():
        for snomed_id in snomed_ids:
            snomed_to_cuis[snomed_id].add(cui)
            for doid in umls_alignment.get("cui_to_doid", {}).get(cui, []) or []:
                do_snomed_to_doids[snomed_id].add(doid)

    concepts: dict[str, dict[str, Any]] = {}
    for snomed_id, entries in collected.items():
        semantic_tag = semantic_tags.get(snomed_id, "")
        if semantic_tag and semantic_tag not in SNOMED_ALLOWED_SEMANTIC_TAGS:
            continue

        lexical_terms = _dedupe_sorted(item["term"] for item in entries)
        if not lexical_terms:
            continue
        label = next((item["term"] for item in entries if item["type"] == "fsn"), lexical_terms[0])
        concept_id = f"SNOMEDCT:{snomed_id}"

        synonym_records = [
            {
                "term": item["term"],
                "source_terminology": "SNOMEDCT",
                "source_id": concept_id,
                "origin": "snomed_description",
                **({"umls_cui": next(iter(sorted(snomed_to_cuis.get(snomed_id, []))))} if snomed_to_cuis.get(snomed_id) else {}),
            }
            for item in entries
        ]

        parents = sorted(do_snomed_to_doids.get(snomed_id, []))
        concepts[concept_id] = {
            "name": label,
            "synonyms": lexical_terms,
            "synonym_records": synonym_records,
            "parents": parents,
            "definition": "",
            "xrefs": {
                **({"umls": sorted(snomed_to_cuis.get(snomed_id, []))} if snomed_to_cuis.get(snomed_id) else {}),
                **({"doid": sorted(do_snomed_to_doids.get(snomed_id, []))} if do_snomed_to_doids.get(snomed_id) else {}),
            },
            "source_terminology": "SNOMEDCT",
            "canonical_source": "SNOMEDCT",
            "aligned_sources": ["SNOMEDCT"] + (["UMLS"] if snomed_to_cuis.get(snomed_id) else []),
            "semantic_tag": semantic_tag,
            "match_enabled": True,
            "normalization_role": "mention_expansion",
        }

    resource = {
        "ontology_name": "SNOMED CT Pathology Subset",
        "ontology_version": snomed_root.name,
        "subset_strategy": {
            "type": "umls_or_do_seed_plus_pathology_description_filter",
            "external_seed_count": len(external_seed_ids),
            "allowed_semantic_tags": sorted(SNOMED_ALLOWED_SEMANTIC_TAGS),
        },
        "concepts": concepts,
    }
    return finalize_resource(resource)


def _merge_xrefs(target: dict[str, Any], source: dict[str, list[str]]) -> None:
    for key, values in source.items():
        for value in values:
            _append_xref(target, key, value)


def _append_synonym_record(
    target: dict[str, Any],
    term: str,
    source_terminology: str,
    source_id: str,
    origin: str,
    umls_cui: str = "",
) -> None:
    token = str(term or "").strip()
    if not token:
        return
    target.setdefault("synonyms", [])
    if token not in target["synonyms"]:
        target["synonyms"].append(token)
    target.setdefault("synonym_records", [])
    target["synonym_records"].append(
        {
            "term": token,
            "source_terminology": source_terminology,
            "source_id": source_id,
            "origin": origin,
            **({"umls_cui": umls_cui} if umls_cui else {}),
        }
    )


def _copy_lexicalizations(
    target: dict[str, Any],
    source_id: str,
    source_record: dict[str, Any],
    source_terminology: str,
    origin: str,
) -> None:
    seen_terms: set[str] = set()
    for term in [source_record.get("name", ""), *(source_record.get("synonyms", []) or [])]:
        token = str(term or "").strip()
        if not token:
            continue
        normalized = _normalize_text(token)
        if normalized in seen_terms:
            continue
        seen_terms.add(normalized)
        _append_synonym_record(
            target=target,
            term=token,
            source_terminology=source_terminology,
            source_id=source_id,
            origin=origin,
        )


def _augment_crosswalks_with_umls(
    crosswalk: dict[str, Any],
    ncit_resource: dict[str, Any] | None,
    umls_alignment: dict[str, Any],
) -> None:
    if ncit_resource is None:
        return

    ncit_by_cui: dict[str, set[str]] = defaultdict(set)
    for ncit_id, record in ncit_resource.get("concepts", {}).items():
        for cui in record.get("xrefs", {}).get("umls", []) or []:
            token = str(cui).strip()
            if token:
                ncit_by_cui[token].add(ncit_id)

    for cui, snomed_ids in umls_alignment.get("cui_to_snomed", {}).items():
        mapped_ncit_ids = sorted(ncit_by_cui.get(cui, set()))
        if not mapped_ncit_ids:
            continue
        for raw_snomed_id in snomed_ids:
            full_snomed_id = f"SNOMEDCT:{raw_snomed_id}"
            crosswalk["snomed_to_ncit"].setdefault(full_snomed_id, [])
            for ncit_id in mapped_ncit_ids:
                if ncit_id not in crosswalk["snomed_to_ncit"][full_snomed_id]:
                    crosswalk["snomed_to_ncit"][full_snomed_id].append(ncit_id)
                crosswalk["ncit_to_snomed"].setdefault(ncit_id, [])
                if full_snomed_id not in crosswalk["ncit_to_snomed"][ncit_id]:
                    crosswalk["ncit_to_snomed"][ncit_id].append(full_snomed_id)

    for key in ("snomed_to_ncit", "ncit_to_snomed"):
        for concept_id in list(crosswalk[key].keys()):
            crosswalk[key][concept_id].sort()


def _augment_crosswalks_with_snomed_resource(
    crosswalk: dict[str, Any],
    snomed_resource: dict[str, Any] | None,
) -> None:
    if snomed_resource is None:
        return
    for snomed_id, record in snomed_resource.get("concepts", {}).items():
        for doid in record.get("xrefs", {}).get("doid", []) or []:
            crosswalk["snomed_to_doid"].setdefault(snomed_id, [])
            if doid not in crosswalk["snomed_to_doid"][snomed_id]:
                crosswalk["snomed_to_doid"][snomed_id].append(doid)
            crosswalk["doid_to_snomed"].setdefault(doid, [])
            if snomed_id not in crosswalk["doid_to_snomed"][doid]:
                crosswalk["doid_to_snomed"][doid].append(snomed_id)

    for key in ("snomed_to_doid", "doid_to_snomed"):
        for concept_id in list(crosswalk[key].keys()):
            crosswalk[key][concept_id].sort()


def build_oncology_multi_ontology_bundle(
    ncit_resource: dict[str, Any],
    do_resource: dict[str, Any] | None,
    snomed_resource: dict[str, Any] | None,
    crosswalk: dict[str, Any],
) -> dict[str, Any]:
    bundle_concepts: dict[str, dict[str, Any]] = {}
    do_seed_ids: set[str] = set()

    for ncit_id in ncit_resource.get("concepts", {}):
        do_seed_ids.update(crosswalk.get("ncit_to_doid", {}).get(ncit_id, []) or [])
    if snomed_resource is not None:
        for snomed_id, record in snomed_resource.get("concepts", {}).items():
            do_seed_ids.update(record.get("xrefs", {}).get("doid", []) or [])

    do_keep_ids: set[str] = set()
    if do_resource is not None and do_seed_ids:
        do_keep_ids = _collect_ancestor_closure(do_resource["concepts"], do_seed_ids)
        for doid in sorted(do_keep_ids):
            source = do_resource["concepts"][doid]
            record = {
                "name": source.get("name", doid),
                "synonyms": [],
                "parents": [parent for parent in source.get("parents", []) or [] if parent in do_keep_ids],
                "definition": source.get("definition", ""),
                "xrefs": _clone_xrefs(source),
                "source_terminology": "DO",
                "canonical_source": "DO",
                "aligned_sources": ["DO"],
                "match_enabled": False,
                "normalization_role": "disease_hierarchy",
            }
            _copy_lexicalizations(record, doid, source, "DO", "do_hierarchy")
            bundle_concepts[doid] = record

    for ncit_id, source in ncit_resource.get("concepts", {}).items():
        do_parents = [doid for doid in crosswalk.get("ncit_to_doid", {}).get(ncit_id, []) or [] if doid in do_keep_ids]
        record = {
            "name": source.get("name", ncit_id),
            "synonyms": [],
            "parents": _dedupe_sorted([*(source.get("parents", []) or []), *do_parents]),
            "definition": source.get("definition", ""),
            "xrefs": _clone_xrefs(source),
            "source_terminology": "NCIt",
            "canonical_source": "NCIt",
            "aligned_sources": ["NCIt"],
            "match_enabled": True,
            "normalization_role": "core_normalization",
        }
        _copy_lexicalizations(record, ncit_id, source, "NCIt", "ncit")
        for doid in do_parents:
            _append_xref(record, "doid", doid)
            if do_resource is not None and doid in do_resource["concepts"]:
                _copy_lexicalizations(record, doid, do_resource["concepts"][doid], "DO", "do_aligned")
                if "DO" not in record["aligned_sources"]:
                    record["aligned_sources"].append("DO")
        bundle_concepts[ncit_id] = record

    merged_snomed_count = 0
    standalone_snomed_count = 0
    if snomed_resource is not None:
        for snomed_id, source in snomed_resource.get("concepts", {}).items():
            mapped_ncit_ids = [ncit_id for ncit_id in crosswalk.get("snomed_to_ncit", {}).get(snomed_id, []) or [] if ncit_id in bundle_concepts]
            if mapped_ncit_ids:
                merged_snomed_count += 1
                for ncit_id in mapped_ncit_ids:
                    target = bundle_concepts[ncit_id]
                    _append_xref(target, "snomed_ct", snomed_id.replace("SNOMEDCT:", "", 1))
                    _merge_xrefs(target, source.get("xrefs", {}) or {})
                    if "SNOMEDCT" not in target["aligned_sources"]:
                        target["aligned_sources"].append("SNOMEDCT")
                    if source.get("xrefs", {}).get("umls"):
                        if "UMLS" not in target["aligned_sources"]:
                            target["aligned_sources"].append("UMLS")
                    for synonym_record in source.get("synonym_records", []) or []:
                        _append_synonym_record(
                            target=target,
                            term=str(synonym_record.get("term") or ""),
                            source_terminology="SNOMEDCT",
                            source_id=snomed_id,
                            origin="snomed_aligned",
                            umls_cui=str(synonym_record.get("umls_cui") or ""),
                        )
                continue

            standalone_snomed_count += 1
            parents = [parent for parent in source.get("parents", []) or [] if parent in bundle_concepts]
            record = {
                "name": source.get("name", snomed_id),
                "synonyms": [],
                "parents": parents,
                "definition": source.get("definition", ""),
                "xrefs": _clone_xrefs(source),
                "source_terminology": "SNOMEDCT",
                "canonical_source": "SNOMEDCT",
                "aligned_sources": list(source.get("aligned_sources", []) or ["SNOMEDCT"]),
                "match_enabled": True,
                "normalization_role": "mention_expansion",
                "semantic_tag": source.get("semantic_tag", ""),
            }
            for synonym_record in source.get("synonym_records", []) or []:
                _append_synonym_record(
                    target=record,
                    term=str(synonym_record.get("term") or ""),
                    source_terminology="SNOMEDCT",
                    source_id=snomed_id,
                    origin=str(synonym_record.get("origin") or "snomed_description"),
                    umls_cui=str(synonym_record.get("umls_cui") or ""),
                )
            bundle_concepts[snomed_id] = record

    bundle = {
        "ontology_name": "Oncology Multi Ontology Bundle",
        "ontology_version": {
            "ncit": ncit_resource.get("ontology_version"),
            "do": do_resource.get("ontology_version") if do_resource is not None else None,
            "snomed_ct": snomed_resource.get("ontology_version") if snomed_resource is not None else None,
        },
        "source_resources": {
            "ncit": ncit_resource.get("ontology_name"),
            "do": do_resource.get("ontology_name") if do_resource is not None else None,
            "snomed_ct": snomed_resource.get("ontology_name") if snomed_resource is not None else None,
        },
        "bundle_strategy": {
            "canonical_priority": ["NCIt", "DO", "SNOMEDCT"],
            "lexical_coverage_sources": ["SNOMEDCT", "NCIt", "DO"],
            "match_resolution_priority": ["NCIt", "DO", "SNOMEDCT"],
            "roles": {
                "NCIt": "core_normalization",
                "SNOMEDCT": "mention_expansion",
                "UMLS": "cross_terminology_alignment",
                "DO": "disease_hierarchy",
            },
            "merged_snomed_concepts": merged_snomed_count,
            "standalone_snomed_concepts": standalone_snomed_count,
            "do_hierarchy_nodes": len(do_keep_ids),
        },
        "concepts": bundle_concepts,
    }
    return finalize_resource(bundle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert NCIt / DO / SNOMED CT / UMLS ontology downloads into project JSON resources."
    )
    parser.add_argument("--ontology_root", type=Path, default=DEFAULT_ONTOLOGY_ROOT)
    parser.add_argument("--raw_root", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--ncit_zip", type=Path, default=None)
    parser.add_argument("--do_owl", type=Path, default=None)
    parser.add_argument("--snomed_root", type=Path, default=None)
    parser.add_argument("--umls_mrconso", type=Path, default=None)
    parser.add_argument("--skip_ncit", action="store_true")
    parser.add_argument("--skip_do", action="store_true")
    parser.add_argument("--skip_snomed", action="store_true")
    parser.add_argument("--skip_umls", action="store_true")
    parser.add_argument("--skip_bundle", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ontology_root = Path(args.ontology_root)
    raw_root = Path(args.raw_root) if args.raw_root is not None else ontology_root / "raw"
    output_dir = Path(args.output_dir) if args.output_dir is not None else ontology_root / "processed"
    ensure_dir(output_dir)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    ncit_zip = Path(args.ncit_zip) if args.ncit_zip is not None else _discover_ncit_zip(raw_root)
    do_owl = Path(args.do_owl) if args.do_owl is not None else _discover_do_owl(raw_root)
    snomed_root = Path(args.snomed_root) if args.snomed_root is not None else _discover_snomed_root(raw_root)
    umls_mrconso = Path(args.umls_mrconso) if args.umls_mrconso is not None else _discover_umls_mrconso(raw_root)

    ncit_resource = None
    ncit_pathology_subset = None
    do_resource = None
    umls_alignment = {
        "relevant_cui_count": 0,
        "matched_cui_count": 0,
        "cui_to_ncit": {},
        "cui_to_snomed": {},
        "cui_to_doid": {},
    }
    snomed_resource = None

    if not args.skip_ncit:
        if not ncit_zip.exists():
            raise FileNotFoundError(f"NCIt zip not found: {ncit_zip}")
        ncit_resource = finalize_resource(build_ncit_resource(ncit_zip))
        ncit_pathology_subset = build_ncit_pathology_subset(ncit_resource)
        write_json(output_dir / "ncit_project_ontology.json", ncit_resource)
        write_json(output_dir / "ncit_pathology_subset_ontology.json", ncit_pathology_subset)
        LOGGER.info("Wrote NCIt resource with %s concepts", ncit_resource["concept_count"])
        LOGGER.info("Wrote NCIt pathology subset with %s concepts", ncit_pathology_subset["concept_count"])

    if not args.skip_do:
        if not do_owl.exists():
            raise FileNotFoundError(f"DO owl not found: {do_owl}")
        do_resource = finalize_resource(build_do_resource(do_owl))
        write_json(output_dir / "do_project_ontology.json", do_resource)
        LOGGER.info("Wrote DO resource with %s concepts", do_resource["concept_count"])

    relevant_cuis = collect_relevant_umls_cuis(ncit_pathology_subset, do_resource)
    if not args.skip_umls:
        if not umls_mrconso.exists():
            raise FileNotFoundError(f"UMLS MRCONSO not found: {umls_mrconso}")
        umls_alignment = parse_targeted_umls_alignment(umls_mrconso, relevant_cuis)
        write_json(output_dir / "umls_targeted_alignment.json", umls_alignment)

    if not args.skip_snomed:
        if not snomed_root.exists():
            raise FileNotFoundError(f"SNOMED CT root not found: {snomed_root}")
        snomed_resource = build_snomed_resource(
            snomed_root=snomed_root,
            do_resource=do_resource,
            umls_alignment=umls_alignment,
        )
        write_json(output_dir / "snomed_pathology_subset_ontology.json", snomed_resource)
        LOGGER.info("Wrote SNOMED CT subset with %s concepts", snomed_resource["concept_count"])

    crosswalk = build_crosswalks(ncit_resource=ncit_pathology_subset, do_resource=do_resource)
    _augment_crosswalks_with_umls(crosswalk, ncit_pathology_subset, umls_alignment)
    _augment_crosswalks_with_snomed_resource(crosswalk, snomed_resource)
    write_json(output_dir / "ontology_crosswalk_summary.json", crosswalk)

    bundle_resource = None
    if not args.skip_bundle and ncit_pathology_subset is not None:
        bundle_resource = build_oncology_multi_ontology_bundle(
            ncit_resource=ncit_pathology_subset,
            do_resource=do_resource,
            snomed_resource=snomed_resource,
            crosswalk=crosswalk,
        )
        write_json(output_dir / "oncology_multi_ontology_bundle.json", bundle_resource)
        LOGGER.info("Wrote oncology multi-ontology bundle with %s concepts", bundle_resource["concept_count"])

    summary = {
        "ontology_root": str(ontology_root),
        "raw_root": str(raw_root),
        "output_dir": str(output_dir),
        "ncit_zip": str(ncit_zip) if not args.skip_ncit else None,
        "do_owl": str(do_owl) if not args.skip_do else None,
        "snomed_root": str(snomed_root) if not args.skip_snomed else None,
        "umls_mrconso": str(umls_mrconso) if not args.skip_umls else None,
        "outputs": {
            "ncit_project_ontology_json": str(output_dir / "ncit_project_ontology.json") if ncit_resource is not None else None,
            "ncit_pathology_subset_ontology_json": str(output_dir / "ncit_pathology_subset_ontology.json") if ncit_pathology_subset is not None else None,
            "do_project_ontology_json": str(output_dir / "do_project_ontology.json") if do_resource is not None else None,
            "umls_targeted_alignment_json": str(output_dir / "umls_targeted_alignment.json") if not args.skip_umls else None,
            "snomed_pathology_subset_ontology_json": str(output_dir / "snomed_pathology_subset_ontology.json") if snomed_resource is not None else None,
            "oncology_multi_ontology_bundle_json": str(output_dir / "oncology_multi_ontology_bundle.json") if bundle_resource is not None else None,
            "ontology_crosswalk_summary_json": str(output_dir / "ontology_crosswalk_summary.json"),
        },
        "counts": {
            "ncit_concepts": int(ncit_resource["concept_count"]) if ncit_resource is not None else 0,
            "ncit_pathology_subset_concepts": int(ncit_pathology_subset["concept_count"]) if ncit_pathology_subset is not None else 0,
            "do_concepts": int(do_resource["concept_count"]) if do_resource is not None else 0,
            "umls_relevant_cuis": len(relevant_cuis),
            "umls_matched_cuis": int(umls_alignment.get("matched_cui_count", 0)),
            "snomed_subset_concepts": int(snomed_resource["concept_count"]) if snomed_resource is not None else 0,
            "bundle_concepts": int(bundle_resource["concept_count"]) if bundle_resource is not None else 0,
            "bundle_match_enabled_concepts": int(bundle_resource.get("match_enabled_concept_count", 0)) if bundle_resource is not None else 0,
        },
    }
    write_json(output_dir / "ontology_build_summary.json", summary)
    print(
        "Ontology resource build finished | "
        f"ncit={summary['counts']['ncit_concepts']} | "
        f"do={summary['counts']['do_concepts']} | "
        f"snomed={summary['counts']['snomed_subset_concepts']} | "
        f"bundle={summary['counts']['bundle_concepts']} | "
        f"output={output_dir}"
    )


if __name__ == "__main__":
    main()
