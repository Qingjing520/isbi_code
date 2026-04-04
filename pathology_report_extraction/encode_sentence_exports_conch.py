from __future__ import annotations

"""Encode exported pathology-report sentences with the CONCH text encoder.

Dependencies:
    pip install torch timm transformers

Expected input:
    sentence-export JSON files created by export_sentence_views.py

Expected output:
    One .pt file per document containing sentence-level CONCH embeddings and
    one metadata .json file preserving section alignment for downstream graph
    construction.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from config.config import get_path, get_stage_config, get_value, load_yaml_config
from pdf_utils import ensure_dir, write_json


LOGGER = logging.getLogger("encode_sentence_exports_conch")

DEFAULT_OUTPUT_ROOT = Path(r"D:\Tasks\isbi_code\pathology_report_extraction\Output")
DEFAULT_INPUT_DIR = DEFAULT_OUTPUT_ROOT / "sentence_exports_masked"
DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "sentence_embeddings_conch_masked"
DEFAULT_CONCH_REPO_DIR = Path(r"D:\Tasks\CLAM-master")
DEFAULT_CONCH_CKPT_PATH = Path(r"D:\Tasks\CLAM-master\ckpts\pytorch_model.bin")
DEFAULT_SENTENCE_EXPORT_OUTPUT_SUBDIRS = {
    "masked": "sentence_exports_masked",
    "no_diagnosis": "sentence_exports_no_diagnosis",
    "no_diagnosis_masked": "sentence_exports_no_diagnosis_masked",
    "full": "sentence_exports_full",
}
DEFAULT_CONCH_OUTPUT_SUBDIRS = {
    "masked": "sentence_embeddings_conch_masked",
    "no_diagnosis": "sentence_embeddings_conch_no_diagnosis",
    "no_diagnosis_masked": "sentence_embeddings_conch_no_diagnosis_masked",
    "full": "sentence_embeddings_conch_full",
}


def iter_sentence_jsons(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*.json")
        if path.is_file() and path.name not in {"run_summary.json"}
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def setup_conch_imports(conch_repo_dir: Path):
    repo_dir = str(conch_repo_dir)
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer  # type: ignore

    return create_model_from_pretrained, get_tokenizer


def choose_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def tokenize_texts_compatible(tokenizer, texts: list[str]) -> torch.Tensor:
    """Mirror CONCH tokenization while staying compatible with newer transformers."""
    encoded = tokenizer(
        texts,
        max_length=127,
        add_special_tokens=True,
        return_token_type_ids=False,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return F.pad(encoded["input_ids"], (0, 1), value=tokenizer.pad_token_id)


def load_conch_model(conch_repo_dir: Path, checkpoint_path: Path, device: str):
    create_model_from_pretrained, get_tokenizer = setup_conch_imports(conch_repo_dir)
    model, _ = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=str(checkpoint_path))
    model = model.to(device)
    model.eval()
    tokenizer = get_tokenizer()
    return model, tokenizer


def encode_sentences(model, tokenizer, sentences: list[str], batch_size: int, device: str) -> torch.Tensor:
    if not sentences:
        return torch.empty((0, 512), dtype=torch.float32)

    all_outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(sentences), batch_size):
            batch = sentences[start : start + batch_size]
            tokens = tokenize_texts_compatible(tokenizer, batch).to(device)
            features = model.encode_text(tokens)
            all_outputs.append(features.detach().cpu().float())

    return torch.cat(all_outputs, dim=0)


def build_embedding_metadata(payload: dict, output_pt_path: Path, source_sentence_view_json: Path) -> dict:
    return {
        "document_id": payload.get("document_id"),
        "file_name": payload.get("file_name"),
        "dataset": payload.get("dataset"),
        "filter_mode": payload.get("filter_mode"),
        "source_pdf": payload.get("source_pdf"),
        "source_sentence_view_json": str(source_sentence_view_json),
        "source_preprocessed_json": payload.get("source_json"),
        "embedding_path": str(output_pt_path),
        "section_count": payload.get("section_count", 0),
        "sentence_count": payload.get("sentence_count", 0),
        "sections": payload.get("sections", []),
        "sentence_to_section": payload.get("sentence_to_section", []),
        "sentence_records": payload.get("sentence_records", []),
    }


def encode_document(
    json_path: Path,
    input_root: Path,
    output_root: Path,
    model,
    tokenizer,
    batch_size: int,
    device: str,
) -> dict:
    payload = load_json(json_path)
    sentences = [str(sentence).strip() for sentence in payload.get("sentences", []) if str(sentence).strip()]

    relative_path = json_path.relative_to(input_root)
    output_pt_path = output_root / relative_path.with_suffix(".pt")
    output_meta_path = output_root / relative_path
    ensure_dir(output_pt_path.parent)

    embeddings = encode_sentences(model, tokenizer, sentences, batch_size=batch_size, device=device)
    torch.save(embeddings, output_pt_path)

    metadata = build_embedding_metadata(
        payload,
        output_pt_path=output_pt_path,
        source_sentence_view_json=json_path,
    )
    write_json(output_meta_path, metadata)

    return {
        "document_id": payload.get("document_id"),
        "file_name": payload.get("file_name"),
        "dataset": payload.get("dataset"),
        "source_json": str(json_path),
        "output_pt": str(output_pt_path),
        "output_meta_json": str(output_meta_path),
        "sentence_count": len(sentences),
        "embedding_shape": list(embeddings.shape),
        "status": "success",
    }


def process_all_documents(
    input_dir: Path,
    output_dir: Path,
    conch_repo_dir: Path,
    checkpoint_path: Path,
    batch_size: int,
    device_arg: str,
    limit: int | None,
) -> dict:
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    ensure_dir(output_root)

    log_path = output_root / "encode.log"
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

    device = choose_device(device_arg)
    LOGGER.info("Using device: %s", device)
    LOGGER.info("Loading CONCH from repo=%s checkpoint=%s", conch_repo_dir, checkpoint_path)
    model, tokenizer = load_conch_model(conch_repo_dir=conch_repo_dir, checkpoint_path=checkpoint_path, device=device)

    json_paths = iter_sentence_jsons(input_root)
    if limit is not None:
        json_paths = json_paths[:limit]

    LOGGER.info("Discovered %s sentence-export JSON files under %s", len(json_paths), input_root)
    summaries: list[dict] = []

    for index, json_path in enumerate(json_paths, start=1):
        LOGGER.info("[%s/%s] Encoding %s", index, len(json_paths), json_path)
        try:
            summary = encode_document(
                json_path=json_path,
                input_root=input_root,
                output_root=output_root,
                model=model,
                tokenizer=tokenizer,
                batch_size=batch_size,
                device=device,
            )
        except Exception as exc:
            LOGGER.exception("Failed to encode %s", json_path)
            summary = {
                "document_id": json_path.stem,
                "file_name": json_path.name,
                "dataset": None,
                "source_json": str(json_path),
                "output_pt": None,
                "output_meta_json": None,
                "sentence_count": 0,
                "embedding_shape": [0, 512],
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
        "conch_repo_dir": str(conch_repo_dir),
        "checkpoint_path": str(checkpoint_path),
        "device": device,
        "batch_size": batch_size,
        "total_json_files": len(json_paths),
        "success_count": len(successes),
        "failure_count": len(failures),
        "empty_sentence_docs": empty_sentence_docs,
        "datasets": dataset_counts,
        "files": summaries,
    }
    write_json(output_root / "run_summary.json", run_summary)
    LOGGER.info(
        "Completed CONCH encoding | total=%s | success=%s | failed=%s | empty_sentence_docs=%s",
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
    stage_block = raw_config.get("encode_sentence_exports_conch") if isinstance(raw_config.get("encode_sentence_exports_conch"), dict) else {}
    config = get_stage_config(raw_config, "encode_sentence_exports_conch")
    defaults_block = raw_config.get("defaults") if isinstance(raw_config.get("defaults"), dict) else {}
    preprocess_config = get_stage_config(raw_config, "preprocess")
    filter_mode = str(get_value(preprocess_config, "filter_mode", "masked"))
    output_root = get_path(defaults_block, "output_root", DEFAULT_OUTPUT_ROOT, config_path)

    input_default = get_path(stage_block, "input_dir", DEFAULT_INPUT_DIR, config_path)
    output_default = get_path(stage_block, "output_dir", DEFAULT_OUTPUT_DIR, config_path)

    if stage_block:
        if stage_block.get("input_dir") in (None, ""):
            sentence_output_subdirs = dict(DEFAULT_SENTENCE_EXPORT_OUTPUT_SUBDIRS)
            sentence_output_subdirs.update(get_value(get_stage_config(raw_config, "export_sentence_views"), "output_subdirs", {}) or {})
            input_default = output_root / sentence_output_subdirs.get(filter_mode, DEFAULT_SENTENCE_EXPORT_OUTPUT_SUBDIRS["masked"])
        if stage_block.get("output_dir") in (None, ""):
            conch_output_subdirs = dict(DEFAULT_CONCH_OUTPUT_SUBDIRS)
            conch_output_subdirs.update(get_value(config, "output_subdirs", {}) or {})
            output_default = output_root / conch_output_subdirs.get(filter_mode, DEFAULT_CONCH_OUTPUT_SUBDIRS["masked"])

    parser = argparse.ArgumentParser(description="Encode exported pathology-report sentences with CONCH.")
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
        help="Directory for CONCH embedding outputs.",
    )
    parser.add_argument(
        "--conch_repo_dir",
        type=Path,
        default=get_path(config, "conch_repo_dir", DEFAULT_CONCH_REPO_DIR, config_path),
        help="Repository root that contains the local conch package.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=get_path(config, "checkpoint_path", DEFAULT_CONCH_CKPT_PATH, config_path),
        help="Path to the CONCH checkpoint file.",
    )
    parser.add_argument("--batch_size", type=int, default=int(get_value(config, "batch_size", 64)), help="Sentence batch size for CONCH encoding.")
    parser.add_argument("--device", type=str, default=str(get_value(config, "device", "auto")), help='Encoding device: "auto", "cuda", or "cpu".')
    parser.add_argument(
        "--limit",
        type=int,
        default=None if get_value(config, "limit", None) is None else int(get_value(config, "limit", None)),
        help="Optional cap on number of documents to encode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = process_all_documents(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        conch_repo_dir=args.conch_repo_dir,
        checkpoint_path=args.checkpoint_path,
        batch_size=args.batch_size,
        device_arg=args.device,
        limit=args.limit,
    )
    print(
        f"Encoded {summary['total_json_files']} JSON files | success={summary['success_count']} | "
        f"failed={summary['failure_count']} | empty_sentence_docs={summary['empty_sentence_docs']} | "
        f"device={summary['device']}"
    )


if __name__ == "__main__":
    main()
