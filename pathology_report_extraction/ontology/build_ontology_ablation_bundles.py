from __future__ import annotations

"""Create compact ontology bundles for concept-graph ablation experiments."""

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from pathology_report_extraction.common.pdf_utils import write_json
from pathology_report_extraction.common.pipeline_defaults import DEFAULT_ONTOLOGY_ABLATION_DIR, DEFAULT_ONTOLOGY_PROCESSED_DIR
from pathology_report_extraction.ontology.build_project_ontology_resources import (
    _append_synonym_record,
    _append_xref,
    _clone_xrefs,
    _copy_lexicalizations,
    _merge_xrefs,
    build_crosswalks,
    build_oncology_multi_ontology_bundle,
    finalize_resource,
)


DEFAULT_PROCESSED_DIR = DEFAULT_ONTOLOGY_PROCESSED_DIR
DEFAULT_OUTPUT_DIR = DEFAULT_ONTOLOGY_ABLATION_DIR


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _normalise_resource_metadata(resource: dict[str, Any], name: str, strategy: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(resource)
    payload["ontology_name"] = name
    payload["ablation_strategy"] = strategy
    return finalize_resource(payload)


def build_ncit_snomed_mapped_bundle(
    ncit_resource: dict[str, Any],
    snomed_resource: dict[str, Any],
    crosswalk: dict[str, Any],
) -> dict[str, Any]:
    concepts: dict[str, dict[str, Any]] = {}

    for ncit_id, source in ncit_resource.get("concepts", {}).items():
        record = {
            "name": source.get("name", ncit_id),
            "synonyms": [],
            "parents": list(source.get("parents", []) or []),
            "definition": source.get("definition", ""),
            "xrefs": _clone_xrefs(source),
            "source_terminology": "NCIt",
            "canonical_source": "NCIt",
            "aligned_sources": ["NCIt"],
            "match_enabled": True,
            "normalization_role": "core_normalization",
        }
        _copy_lexicalizations(record, ncit_id, source, "NCIt", "ncit")
        concepts[ncit_id] = record

    merged_snomed_count = 0
    for snomed_id, source in snomed_resource.get("concepts", {}).items():
        mapped_ncit_ids = [
            ncit_id
            for ncit_id in crosswalk.get("snomed_to_ncit", {}).get(snomed_id, []) or []
            if ncit_id in concepts
        ]
        if not mapped_ncit_ids:
            continue

        merged_snomed_count += 1
        for ncit_id in mapped_ncit_ids:
            target = concepts[ncit_id]
            _append_xref(target, "snomed_ct", snomed_id.replace("SNOMEDCT:", "", 1))
            _merge_xrefs(target, source.get("xrefs", {}) or {})
            if "SNOMEDCT" not in target["aligned_sources"]:
                target["aligned_sources"].append("SNOMEDCT")
            if source.get("xrefs", {}).get("umls") and "UMLS" not in target["aligned_sources"]:
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

    bundle = {
        "ontology_name": "NCIt + SNOMED CT mapped-to-NCIt Bundle",
        "ontology_version": {
            "ncit": ncit_resource.get("ontology_version"),
            "snomed_ct": snomed_resource.get("ontology_version"),
        },
        "bundle_strategy": {
            "type": "ncit_core_with_snomed_lexicalizations_only_when_aligned_to_ncit",
            "canonical_priority": ["NCIt"],
            "lexical_coverage_sources": ["NCIt", "SNOMEDCT"],
            "standalone_snomed_concepts": 0,
            "merged_snomed_concepts": merged_snomed_count,
        },
        "concepts": concepts,
    }
    return finalize_resource(bundle)


def build_ablation_bundles(processed_dir: Path, output_dir: Path) -> dict[str, Any]:
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)

    ncit = load_json(processed_dir / "ncit_pathology_subset_ontology.json")
    do = load_json(processed_dir / "do_project_ontology.json")
    snomed = load_json(processed_dir / "snomed_pathology_subset_ontology.json")
    crosswalk = load_json(processed_dir / "ontology_crosswalk_summary.json")

    # Ensure the NCIt/DO crosswalk is populated even if an older summary file is supplied.
    base_crosswalk = build_crosswalks(ncit_resource=ncit, do_resource=do)
    for key, values in crosswalk.items():
        if isinstance(values, dict):
            base_crosswalk.setdefault(key, {})
            base_crosswalk[key].update(values)
    crosswalk = base_crosswalk

    variants = {
        "ncit_only": _normalise_resource_metadata(
            ncit,
            "NCIt-only Pathology Subset",
            {"type": "ncit_only", "match_sources": ["NCIt"]},
        ),
        "ncit_do": build_oncology_multi_ontology_bundle(
            ncit_resource=ncit,
            do_resource=do,
            snomed_resource=None,
            crosswalk=crosswalk,
        ),
        "ncit_snomed_mapped": build_ncit_snomed_mapped_bundle(
            ncit_resource=ncit,
            snomed_resource=snomed,
            crosswalk=crosswalk,
        ),
        "full_multi_ontology": load_json(processed_dir / "oncology_multi_ontology_bundle.json"),
    }
    variants["ncit_do"]["ontology_name"] = "NCIt + DO Bundle"
    variants["ncit_do"]["ablation_strategy"] = {
        "type": "ncit_core_with_do_hierarchy",
        "match_sources": ["NCIt"],
        "hierarchy_sources": ["NCIt", "DO"],
    }
    variants["full_multi_ontology"]["ablation_strategy"] = {
        "type": "ncit_do_snomed_umls_full",
        "match_sources": ["NCIt", "SNOMEDCT"],
        "hierarchy_sources": ["NCIt", "DO", "SNOMEDCT"],
    }

    summary: dict[str, Any] = {
        "processed_dir": str(processed_dir),
        "output_dir": str(output_dir),
        "variants": {},
    }
    for variant_name, resource in variants.items():
        output_path = output_dir / f"{variant_name}_ontology.json"
        write_json(output_path, resource)
        summary["variants"][variant_name] = {
            "path": str(output_path),
            "concept_count": int(resource.get("concept_count", len(resource.get("concepts", {})))),
            "match_enabled_concept_count": int(resource.get("match_enabled_concept_count", 0)),
        }

    write_json(output_dir / "ontology_ablation_bundles_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ontology ablation bundles for NCIt/DO/SNOMED experiments.")
    parser.add_argument("--processed_dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_ablation_bundles(args.processed_dir, args.output_dir)
    for name, record in summary["variants"].items():
        print(
            f"{name}: concepts={record['concept_count']} | "
            f"match_enabled={record['match_enabled_concept_count']} | path={record['path']}"
        )


if __name__ == "__main__":
    main()
