from __future__ import annotations

"""Export sentence-level views from preprocessed pathology-report JSON files.

Dependencies:
    pip install PyMuPDF rapidocr_onnxruntime opencv-python Pillow numpy

This script reads existing Document -> Section -> Sentence JSON files and
writes a sentence-export directory without modifying the original structured
JSON outputs. The export keeps section alignment metadata so the sentence list
can be encoded by CONCH later while preserving the section boundaries needed
for hierarchical graphs.
"""

import argparse
import json
import logging
from pathlib import Path

from config.config import get_bool, get_path, get_stage_config, get_value, load_yaml_config


LOGGER = logging.getLogger("export_sentence_views")

DEFAULT_OUTPUT_ROOT = Path(r"D:\Tasks\isbi_code\pathology_report_extraction\Output")
DEFAULT_INPUT_DIR = DEFAULT_OUTPUT_ROOT / "pathology_report_preprocessed_masked"
DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "sentence_exports_masked"
DEFAULT_PREPROCESS_OUTPUT_SUBDIRS = {
    "masked": "pathology_report_preprocessed_masked",
    "no_diagnosis": "pathology_report_preprocessed_no_diagnosis",
    "no_diagnosis_masked": "pathology_report_preprocessed_no_diagnosis_masked",
    "full": "pathology_report_preprocessed_full",
}
DEFAULT_SENTENCE_EXPORT_OUTPUT_SUBDIRS = {
    "masked": "sentence_exports_masked",
    "no_diagnosis": "sentence_exports_no_diagnosis",
    "no_diagnosis_masked": "sentence_exports_no_diagnosis_masked",
    "full": "sentence_exports_full",
}


def ensure_dir(path: Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def iter_document_jsons(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*.json")
        if path.is_file() and path.name not in {"run_summary.json"}
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_sentence_view(payload: dict, source_json_path: Path) -> dict:
    sections = payload.get("sections", []) or []

    exported_sections: list[dict] = []
    sentences: list[str] = []
    sentence_to_section: list[int] = []
    sentence_records: list[dict] = []

    for section_index, section in enumerate(sections):
        section_title = str(section.get("section_title", "Document Body"))
        raw_sentences = section.get("sentences", []) or []
        start = len(sentences)

        for local_index, sentence in enumerate(raw_sentences):
            text = str(sentence).strip()
            if not text:
                continue
            global_index = len(sentences)
            sentences.append(text)
            sentence_to_section.append(section_index)
            sentence_records.append(
                {
                    "sentence_index": global_index,
                    "section_index": section_index,
                    "section_title": section_title,
                    "section_sentence_index": local_index,
                    "text": text,
                }
            )

        end = len(sentences)
        exported_sections.append(
            {
                "section_index": section_index,
                "section_title": section_title,
                "sentence_start": start,
                "sentence_end": end,
                "sentence_count": end - start,
            }
        )

    return {
        "document_id": payload.get("document_id"),
        "file_name": payload.get("file_name"),
        "source_json": str(source_json_path),
        "source_pdf": payload.get("source_path"),
        "dataset": payload.get("dataset"),
        "filter_mode": payload.get("filter_mode"),
        "page_count": payload.get("page_count"),
        "section_count": len(exported_sections),
        "sentence_count": len(sentences),
        "sections": exported_sections,
        "sentences": sentences,
        "sentence_to_section": sentence_to_section,
        "sentence_records": sentence_records,
    }


def format_sentence_txt(sentence_view: dict) -> str:
    lines: list[str] = []
    lines.append(f"document_id: {sentence_view.get('document_id', '')}")
    lines.append(f"file_name: {sentence_view.get('file_name', '')}")
    lines.append(f"dataset: {sentence_view.get('dataset', '')}")
    lines.append(f"filter_mode: {sentence_view.get('filter_mode', '')}")
    lines.append("")

    records = sentence_view.get("sentence_records", [])
    current_section = None
    for record in records:
        section_title = record["section_title"]
        if section_title != current_section:
            if lines and lines[-1] != "":
                lines.append("")
            lines.append(f"[{section_title}]")
            current_section = section_title
        lines.append(f"{record['sentence_index']:04d}\t{record['text']}")

    if not records:
        lines.append("[EMPTY]")
    return "\n".join(lines).strip() + "\n"


def export_document(
    json_path: Path,
    input_root: Path,
    output_root: Path,
    write_txt_copy: bool,
) -> dict:
    payload = load_json(json_path)
    sentence_view = build_sentence_view(payload, source_json_path=json_path)

    relative_path = json_path.relative_to(input_root)
    output_json_path = output_root / relative_path
    ensure_dir(output_json_path.parent)
    write_json(output_json_path, sentence_view)

    output_txt_path = None
    if write_txt_copy:
        output_txt_path = output_json_path.with_suffix(".txt")
        write_text(output_txt_path, format_sentence_txt(sentence_view))

    return {
        "document_id": sentence_view.get("document_id"),
        "file_name": sentence_view.get("file_name"),
        "dataset": sentence_view.get("dataset"),
        "source_json": str(json_path),
        "output_json": str(output_json_path),
        "output_txt": str(output_txt_path) if output_txt_path else None,
        "section_count": sentence_view["section_count"],
        "sentence_count": sentence_view["sentence_count"],
        "status": "success",
    }


def process_all_documents(
    input_dir: Path,
    output_dir: Path,
    limit: int | None,
    write_txt_copy: bool,
) -> dict:
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    ensure_dir(output_root)

    log_path = output_root / "export.log"
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

    json_paths = iter_document_jsons(input_root)
    if limit is not None:
        json_paths = json_paths[:limit]

    LOGGER.info("Discovered %s document JSON files under %s", len(json_paths), input_root)

    summaries: list[dict] = []
    for index, json_path in enumerate(json_paths, start=1):
        LOGGER.info("[%s/%s] Exporting %s", index, len(json_paths), json_path)
        try:
            summary = export_document(
                json_path=json_path,
                input_root=input_root,
                output_root=output_root,
                write_txt_copy=write_txt_copy,
            )
        except Exception as exc:
            LOGGER.exception("Failed to export %s", json_path)
            summary = {
                "document_id": json_path.stem,
                "file_name": json_path.name,
                "dataset": None,
                "source_json": str(json_path),
                "output_json": None,
                "output_txt": None,
                "section_count": 0,
                "sentence_count": 0,
                "status": "failed",
                "error": str(exc),
            }
        summaries.append(summary)

    successes = [item for item in summaries if item["status"] == "success"]
    failures = [item for item in summaries if item["status"] == "failed"]
    empty_sentence_docs = sum(1 for item in successes if item["sentence_count"] == 0)

    dataset_counts: dict[str, int] = {}
    for item in successes:
        dataset = item.get("dataset") or "UNKNOWN"
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1

    run_summary = {
        "input_dir": str(input_root),
        "output_dir": str(output_root),
        "write_txt_copy": write_txt_copy,
        "total_json_files": len(json_paths),
        "success_count": len(successes),
        "failure_count": len(failures),
        "empty_sentence_docs": empty_sentence_docs,
        "datasets": dataset_counts,
        "files": summaries,
    }
    write_json(output_root / "run_summary.json", run_summary)
    LOGGER.info(
        "Completed sentence export | total=%s | success=%s | failed=%s | empty_sentence_docs=%s",
        len(json_paths),
        len(successes),
        len(failures),
        empty_sentence_docs,
    )
    return run_summary


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=None, help="Optional YAML config preset.")
    pre_args, _ = pre_parser.parse_known_args()
    raw_config, config_path = load_yaml_config(pre_args.config)
    stage_block = raw_config.get("export_sentence_views") if isinstance(raw_config.get("export_sentence_views"), dict) else {}
    config = get_stage_config(raw_config, "export_sentence_views")
    defaults_block = raw_config.get("defaults") if isinstance(raw_config.get("defaults"), dict) else {}
    preprocess_config = get_stage_config(raw_config, "preprocess")
    filter_mode = str(get_value(preprocess_config, "filter_mode", "masked"))
    output_root = get_path(defaults_block, "output_root", DEFAULT_OUTPUT_ROOT, config_path)

    input_default = get_path(stage_block, "input_dir", DEFAULT_INPUT_DIR, config_path)
    output_default = get_path(stage_block, "output_dir", DEFAULT_OUTPUT_DIR, config_path)

    if stage_block:
        if stage_block.get("input_dir") in (None, ""):
            preprocess_output_subdirs = dict(DEFAULT_PREPROCESS_OUTPUT_SUBDIRS)
            preprocess_output_subdirs.update(get_value(preprocess_config, "output_subdirs", {}) or {})
            input_default = output_root / preprocess_output_subdirs.get(filter_mode, DEFAULT_PREPROCESS_OUTPUT_SUBDIRS["masked"])
        if stage_block.get("output_dir") in (None, ""):
            export_output_subdirs = dict(DEFAULT_SENTENCE_EXPORT_OUTPUT_SUBDIRS)
            export_output_subdirs.update(get_value(config, "output_subdirs", {}) or {})
            output_default = output_root / export_output_subdirs.get(filter_mode, DEFAULT_SENTENCE_EXPORT_OUTPUT_SUBDIRS["masked"])

    parser = argparse.ArgumentParser(
        description="Export flattened sentence views from Document -> Section -> Sentence pathology-report JSON files."
    )
    parser.add_argument("--config", type=Path, default=pre_args.config, help="Optional YAML config preset.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=input_default,
        help="Directory containing preprocessed JSON files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=output_default,
        help="Directory for sentence-view exports.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None if get_value(config, "limit", None) is None else int(get_value(config, "limit", None)),
        help="Optional cap on number of documents to export.",
    )
    parser.set_defaults(write_txt=get_bool(config, "write_txt", False))
    write_group = parser.add_mutually_exclusive_group()
    write_group.add_argument(
        "--write_txt",
        dest="write_txt",
        action="store_true",
        help="Also write a human-readable .txt copy for each exported document.",
    )
    write_group.add_argument(
        "--no_write_txt",
        dest="write_txt",
        action="store_false",
        help="Disable .txt copies even if enabled in the YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = process_all_documents(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        write_txt_copy=args.write_txt,
    )
    print(
        f"Exported {summary['total_json_files']} JSON files | success={summary['success_count']} | "
        f"failed={summary['failure_count']} | empty_sentence_docs={summary['empty_sentence_docs']}"
    )


if __name__ == "__main__":
    main()
