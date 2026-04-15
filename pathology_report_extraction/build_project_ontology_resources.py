from __future__ import annotations

"""Convert downloaded ontology files into the lightweight JSON schema used by this project.

Current support:
- NCIt OWL zip exported from EVS
- Disease Ontology OWL

The generated JSON is compatible with `extract_ontology_concepts.py` and keeps
extra metadata such as xrefs, definitions, and proxy information content scores.
"""

import argparse
import json
import logging
import math
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET


LOGGER = logging.getLogger("build_project_ontology_resources")

DEFAULT_ONTOLOGY_ROOT = Path(r"F:\Tasks\Ontologies")
DEFAULT_RAW_ROOT = DEFAULT_ONTOLOGY_ROOT / "raw"
DEFAULT_OUTPUT_ROOT = DEFAULT_ONTOLOGY_ROOT / "pocessed"

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


def _safe_add_synonym(bucket: list[str], value: str) -> None:
    value = str(value or "").strip()
    if not value:
        return
    if value not in bucket:
        bucket.append(value)


def _normalize_text(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _resource_to_fragment(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if "#" in text:
        return text.rsplit("#", 1)[-1]
    if "/" in text:
        return text.rsplit("/", 1)[-1]
    return text


def _parse_ncit_xref(text: str, xrefs: dict[str, list[str]]) -> None:
    token = str(text or "").strip()
    if not token:
        return
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
        if not code or not code.startswith("C") or not code[1:].isdigit():
            continue
        if not label:
            continue

        _safe_add_synonym(synonyms, label)
        concepts[f"NCIT:{code}"] = {
            "name": label,
            "synonyms": sorted(synonyms, key=str.lower),
            "parents": sorted({f"NCIT:{parent}" for parent in parents if parent}),
            "definition": definition,
            "xrefs": xrefs,
            "source_terminology": "NCIt",
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
            "synonyms": sorted(synonyms, key=str.lower),
            "parents": sorted({parent for parent in parents if parent.startswith("DOID:")}),
            "definition": definition,
            "xrefs": xrefs,
            "source_terminology": "DO",
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


def finalize_resource(resource: dict[str, Any]) -> dict[str, Any]:
    concepts = resource["concepts"]
    total = len(concepts)
    depths = _compute_depths(concepts)
    descendant_counts = _compute_descendant_counts(concepts)

    for concept_id, record in concepts.items():
        depth = int(depths.get(concept_id, 0))
        descendant_count = int(descendant_counts.get(concept_id, 0))
        ic = math.log((total + 1.0) / (descendant_count + 1.0))
        record["depth"] = depth
        record["descendant_count"] = descendant_count
        record["ic"] = round(float(ic), 6)
        record["synonyms"] = sorted({str(term).strip() for term in record.get("synonyms", []) if str(term).strip()}, key=str.lower)
        record["parents"] = sorted({str(parent).strip() for parent in record.get("parents", []) if str(parent).strip()})

    resource["concept_count"] = total
    resource["max_depth"] = max(depths.values()) if depths else 0
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
        normalized_synonym = _normalize_text(synonym)
        if not normalized_synonym:
            continue
        if any(cue in normalized_synonym for cue in NCIT_PATHOLOGY_INCLUDE_CUES):
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
    seed_ids = {
        concept_id
        for concept_id, record in concepts.items()
        if _matches_pathology_subset(record)
    }
    keep_ids = _collect_ancestor_closure(concepts, seed_ids)

    subset_concepts: dict[str, dict[str, Any]] = {}
    for concept_id in sorted(keep_ids):
        record = concepts[concept_id]
        subset_concepts[concept_id] = {
            "name": record.get("name", concept_id),
            "synonyms": list(record.get("synonyms", []) or []),
            "parents": [parent for parent in record.get("parents", []) or [] if parent in keep_ids],
            "definition": record.get("definition", ""),
            "xrefs": dict(record.get("xrefs", {}) or {}),
            "source_terminology": record.get("source_terminology", "NCIt"),
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
    }
    if do_resource is None:
        return crosswalk

    for doid, record in do_resource["concepts"].items():
        xrefs = record.get("xrefs", {})
        ncit_ids = sorted(set(xrefs.get("ncit", []) or []))
        if ncit_ids:
            crosswalk["do_to_ncit"][doid] = ncit_ids
            for ncit_id in ncit_ids:
                crosswalk["ncit_to_doid"].setdefault(ncit_id, [])
                if doid not in crosswalk["ncit_to_doid"][ncit_id]:
                    crosswalk["ncit_to_doid"][ncit_id].append(doid)

    for ncit_id in list(crosswalk["ncit_to_doid"].keys()):
        crosswalk["ncit_to_doid"][ncit_id].sort()
    return crosswalk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert NCIt / DO ontology downloads into project JSON resources."
    )
    parser.add_argument(
        "--ontology_root",
        type=Path,
        default=DEFAULT_ONTOLOGY_ROOT,
        help="Base ontology directory containing raw/ and processed output directories.",
    )
    parser.add_argument(
        "--raw_root",
        type=Path,
        default=None,
        help="Optional explicit raw ontology root. Defaults to <ontology_root>/raw.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Optional explicit output directory. Defaults to <ontology_root>/pocessed.",
    )
    parser.add_argument(
        "--ncit_zip",
        type=Path,
        default=None,
        help="Optional explicit NCIt OWL zip path.",
    )
    parser.add_argument(
        "--do_owl",
        type=Path,
        default=None,
        help="Optional explicit Disease Ontology OWL path.",
    )
    parser.add_argument(
        "--skip_ncit",
        action="store_true",
        help="Skip NCIt conversion.",
    )
    parser.add_argument(
        "--skip_do",
        action="store_true",
        help="Skip DO conversion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ontology_root = Path(args.ontology_root)
    raw_root = Path(args.raw_root) if args.raw_root is not None else ontology_root / "raw"
    output_dir = Path(args.output_dir) if args.output_dir is not None else ontology_root / "pocessed"
    ensure_dir(output_dir)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    ncit_zip = Path(args.ncit_zip) if args.ncit_zip is not None else raw_root / "NCIt" / "Thesaurus_26.03e.OWL.zip"
    do_owl = Path(args.do_owl) if args.do_owl is not None else raw_root / "DO" / "doid.owl"

    ncit_resource = None
    do_resource = None

    if not args.skip_ncit:
        if not ncit_zip.exists():
            raise FileNotFoundError(f"NCIt zip not found: {ncit_zip}")
        ncit_resource = finalize_resource(build_ncit_resource(ncit_zip))
        write_json(output_dir / "ncit_project_ontology.json", ncit_resource)
        ncit_pathology_subset = build_ncit_pathology_subset(ncit_resource)
        write_json(output_dir / "ncit_pathology_subset_ontology.json", ncit_pathology_subset)
        LOGGER.info("Wrote NCIt resource with %s concepts", ncit_resource["concept_count"])
        LOGGER.info(
            "Wrote NCIt pathology subset with %s concepts",
            ncit_pathology_subset["concept_count"],
        )

    if not args.skip_do:
        if not do_owl.exists():
            raise FileNotFoundError(f"DO owl not found: {do_owl}")
        do_resource = finalize_resource(build_do_resource(do_owl))
        write_json(output_dir / "do_project_ontology.json", do_resource)
        LOGGER.info("Wrote DO resource with %s concepts", do_resource["concept_count"])

    crosswalk = build_crosswalks(ncit_resource=ncit_resource, do_resource=do_resource)
    write_json(output_dir / "ontology_crosswalk_summary.json", crosswalk)

    summary = {
        "ontology_root": str(ontology_root),
        "raw_root": str(raw_root),
        "output_dir": str(output_dir),
        "ncit_zip": str(ncit_zip) if not args.skip_ncit else None,
        "do_owl": str(do_owl) if not args.skip_do else None,
        "outputs": {
            "ncit_project_ontology_json": str(output_dir / "ncit_project_ontology.json") if ncit_resource is not None else None,
            "ncit_pathology_subset_ontology_json": str(output_dir / "ncit_pathology_subset_ontology.json") if ncit_resource is not None else None,
            "do_project_ontology_json": str(output_dir / "do_project_ontology.json") if do_resource is not None else None,
            "ontology_crosswalk_summary_json": str(output_dir / "ontology_crosswalk_summary.json"),
        },
        "counts": {
            "ncit_concepts": int(ncit_resource["concept_count"]) if ncit_resource is not None else 0,
            "ncit_pathology_subset_concepts": int(ncit_pathology_subset["concept_count"]) if ncit_resource is not None else 0,
            "do_concepts": int(do_resource["concept_count"]) if do_resource is not None else 0,
            "do_to_ncit_xrefs": len(crosswalk["do_to_ncit"]),
        },
    }
    write_json(output_dir / "ontology_build_summary.json", summary)
    print(
        "Ontology resource build finished | "
        f"ncit={summary['counts']['ncit_concepts']} | "
        f"do={summary['counts']['do_concepts']} | "
        f"output={output_dir}"
    )


if __name__ == "__main__":
    main()
