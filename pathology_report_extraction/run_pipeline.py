from __future__ import annotations

"""One-click runner for the pathology-report text pipeline.

This script reads a single pipeline YAML and executes the processing stages
in order:

1. PDF preprocessing
2. Sentence-view export
3. CONCH sentence encoding
4. Text hierarchy graph construction
"""

import argparse
from pathlib import Path
from typing import Any

from build_text_hierarchy_graphs import (
    process_all_documents as build_graphs,
)
from config.config import get_bool, get_path, get_stage_config, get_value, load_yaml_config
from encode_sentence_exports_conch import (
    DEFAULT_CONCH_CKPT_PATH,
    DEFAULT_CONCH_REPO_DIR,
    process_all_documents as encode_sentence_exports,
)
from export_sentence_views import process_all_documents as export_sentence_views
from prepare_text_graph_manifest import (
    DEFAULT_LABEL_CSV,
    prepare_manifest,
)
from pdf_utils import write_json
from preprocess_pathology_reports import (
    DEFAULT_FILTER_MODE,
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PREPROCESS_OUTPUT_SUBDIRS,
    process_all_pdfs,
)


DEFAULT_PIPELINE_CONFIG = Path(__file__).resolve().parent / "config" / "pipeline.yaml"

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

DEFAULT_GRAPH_OUTPUT_SUBDIRS = {
    "masked": "text_hierarchy_graphs_masked",
    "no_diagnosis": "text_hierarchy_graphs_no_diagnosis",
    "no_diagnosis_masked": "text_hierarchy_graphs_no_diagnosis_masked",
    "full": "text_hierarchy_graphs_full",
}

DEFAULT_MANIFEST_OUTPUT_SUBDIR = "manifests"


def _resolve_output_dir(
    stage_block: dict[str, Any],
    stage_config: dict[str, Any],
    config_path: Path | None,
    output_root: Path,
    filter_mode: str,
    default_subdirs: dict[str, str],
) -> Path:
    if stage_block.get("output_dir") not in (None, ""):
        return get_path(stage_block, "output_dir", output_root / default_subdirs[DEFAULT_FILTER_MODE], config_path)

    output_subdirs = dict(default_subdirs)
    output_subdirs.update(get_value(stage_config, "output_subdirs", {}) or {})
    subdir = output_subdirs.get(filter_mode, default_subdirs.get(DEFAULT_FILTER_MODE, next(iter(default_subdirs.values()))))
    return output_root / subdir


def _pipeline_summary_path(output_root: Path, filter_mode: str) -> Path:
    return output_root / f"pipeline_run_summary_{filter_mode}.json"


def _load_pipeline_config(config_path_arg: Path | None) -> tuple[dict[str, Any], Path | None]:
    return load_yaml_config(config_path_arg or DEFAULT_PIPELINE_CONFIG)


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=DEFAULT_PIPELINE_CONFIG, help="Single pipeline YAML config.")
    pre_args, _ = pre_parser.parse_known_args()
    raw_config, _ = _load_pipeline_config(pre_args.config)

    preprocess_cfg = get_stage_config(raw_config, "preprocess")

    parser = argparse.ArgumentParser(description="Run the full pathology-report processing pipeline from one YAML config.")
    parser.add_argument("--config", type=Path, default=pre_args.config, help="Single pipeline YAML config.")
    parser.add_argument(
        "--filter_mode",
        choices=["full", "no_diagnosis", "masked", "no_diagnosis_masked"],
        default=str(get_value(preprocess_cfg, "filter_mode", DEFAULT_FILTER_MODE)),
        help="Override the preprocess filtering mode for this pipeline run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None if get_value(preprocess_cfg, "limit", None) is None else int(get_value(preprocess_cfg, "limit", None)),
        help="Optional cap on how many PDFs/documents to process in each stage.",
    )
    return parser.parse_args()


def run_pipeline(config_path_arg: Path | None, filter_mode_override: str | None, limit_override: int | None) -> dict[str, Any]:
    raw_config, config_path = _load_pipeline_config(config_path_arg)

    defaults_block = raw_config.get("defaults") if isinstance(raw_config.get("defaults"), dict) else {}
    output_root = get_path(defaults_block, "output_root", DEFAULT_OUTPUT_ROOT, config_path)
    global_input_dir = get_path(defaults_block, "input_dir", DEFAULT_INPUT_DIR, config_path)

    preprocess_block = raw_config.get("preprocess") if isinstance(raw_config.get("preprocess"), dict) else {}
    export_block = raw_config.get("export_sentence_views") if isinstance(raw_config.get("export_sentence_views"), dict) else {}
    encode_block = raw_config.get("encode_sentence_exports_conch") if isinstance(raw_config.get("encode_sentence_exports_conch"), dict) else {}
    graph_block = raw_config.get("build_text_hierarchy_graphs") if isinstance(raw_config.get("build_text_hierarchy_graphs"), dict) else {}
    manifest_block = raw_config.get("prepare_text_graph_manifest") if isinstance(raw_config.get("prepare_text_graph_manifest"), dict) else {}

    preprocess_cfg = get_stage_config(raw_config, "preprocess")
    export_cfg = get_stage_config(raw_config, "export_sentence_views")
    encode_cfg = get_stage_config(raw_config, "encode_sentence_exports_conch")
    graph_cfg = get_stage_config(raw_config, "build_text_hierarchy_graphs")
    manifest_cfg = get_stage_config(raw_config, "prepare_text_graph_manifest")

    filter_mode = filter_mode_override or str(get_value(preprocess_cfg, "filter_mode", DEFAULT_FILTER_MODE))
    limit = limit_override if limit_override is not None else get_value(preprocess_cfg, "limit", None)
    if limit is not None:
        limit = int(limit)

    preprocess_input_dir = get_path(preprocess_block, "input_dir", global_input_dir, config_path)
    preprocess_output_dir = _resolve_output_dir(
        stage_block=preprocess_block,
        stage_config=preprocess_cfg,
        config_path=config_path,
        output_root=output_root,
        filter_mode=filter_mode,
        default_subdirs=DEFAULT_PREPROCESS_OUTPUT_SUBDIRS,
    )

    export_input_dir = get_path(export_block, "input_dir", preprocess_output_dir, config_path)
    export_output_dir = _resolve_output_dir(
        stage_block=export_block,
        stage_config=export_cfg,
        config_path=config_path,
        output_root=output_root,
        filter_mode=filter_mode,
        default_subdirs=DEFAULT_SENTENCE_EXPORT_OUTPUT_SUBDIRS,
    )

    encode_input_dir = get_path(encode_block, "input_dir", export_output_dir, config_path)
    encode_output_dir = _resolve_output_dir(
        stage_block=encode_block,
        stage_config=encode_cfg,
        config_path=config_path,
        output_root=output_root,
        filter_mode=filter_mode,
        default_subdirs=DEFAULT_CONCH_OUTPUT_SUBDIRS,
    )

    graph_input_dir = get_path(graph_block, "input_dir", encode_output_dir, config_path)
    graph_output_dir = _resolve_output_dir(
        stage_block=graph_block,
        stage_config=graph_cfg,
        config_path=config_path,
        output_root=output_root,
        filter_mode=filter_mode,
        default_subdirs=DEFAULT_GRAPH_OUTPUT_SUBDIRS,
    )

    manifest_output_dir = get_path(manifest_block, "output_dir", output_root / DEFAULT_MANIFEST_OUTPUT_SUBDIR, config_path)
    if manifest_block and manifest_block.get("output_dir") in (None, ""):
        manifest_output_dir = output_root / str(get_value(manifest_cfg, "output_subdir", DEFAULT_MANIFEST_OUTPUT_SUBDIR))

    pipeline_summary: dict[str, Any] = {
        "config_path": str(config_path) if config_path is not None else None,
        "filter_mode": filter_mode,
        "limit": limit,
        "output_root": str(output_root),
        "stages": {},
    }

    if get_bool(preprocess_cfg, "enabled", True):
        summary = process_all_pdfs(
            input_dir=preprocess_input_dir,
            output_dir=preprocess_output_dir,
            native_char_threshold=int(get_value(preprocess_cfg, "native_char_threshold", 40)),
            ocr_zoom=float(get_value(preprocess_cfg, "ocr_zoom", 2.0)),
            limit=limit,
            filter_mode=filter_mode,
        )
        pipeline_summary["stages"]["preprocess"] = {
            "enabled": True,
            "input_dir": str(preprocess_input_dir),
            "output_dir": str(preprocess_output_dir),
            "summary_path": str(preprocess_output_dir / "run_summary.json"),
            "result": summary,
        }
    else:
        pipeline_summary["stages"]["preprocess"] = {
            "enabled": False,
            "input_dir": str(preprocess_input_dir),
            "output_dir": str(preprocess_output_dir),
        }

    if get_bool(export_cfg, "enabled", True):
        summary = export_sentence_views(
            input_dir=export_input_dir,
            output_dir=export_output_dir,
            limit=limit if limit is not None else (None if get_value(export_cfg, "limit", None) is None else int(get_value(export_cfg, "limit", None))),
            write_txt_copy=get_bool(export_cfg, "write_txt", False),
        )
        pipeline_summary["stages"]["export_sentence_views"] = {
            "enabled": True,
            "input_dir": str(export_input_dir),
            "output_dir": str(export_output_dir),
            "summary_path": str(export_output_dir / "run_summary.json"),
            "result": summary,
        }
    else:
        pipeline_summary["stages"]["export_sentence_views"] = {
            "enabled": False,
            "input_dir": str(export_input_dir),
            "output_dir": str(export_output_dir),
        }

    if get_bool(encode_cfg, "enabled", True):
        summary = encode_sentence_exports(
            input_dir=encode_input_dir,
            output_dir=encode_output_dir,
            conch_repo_dir=get_path(encode_cfg, "conch_repo_dir", DEFAULT_CONCH_REPO_DIR, config_path),
            checkpoint_path=get_path(encode_cfg, "checkpoint_path", DEFAULT_CONCH_CKPT_PATH, config_path),
            batch_size=int(get_value(encode_cfg, "batch_size", 64)),
            device_arg=str(get_value(encode_cfg, "device", "auto")),
            limit=limit if limit is not None else (None if get_value(encode_cfg, "limit", None) is None else int(get_value(encode_cfg, "limit", None))),
        )
        pipeline_summary["stages"]["encode_sentence_exports_conch"] = {
            "enabled": True,
            "input_dir": str(encode_input_dir),
            "output_dir": str(encode_output_dir),
            "summary_path": str(encode_output_dir / "run_summary.json"),
            "result": summary,
        }
    else:
        pipeline_summary["stages"]["encode_sentence_exports_conch"] = {
            "enabled": False,
            "input_dir": str(encode_input_dir),
            "output_dir": str(encode_output_dir),
        }

    if get_bool(graph_cfg, "enabled", True):
        summary = build_graphs(
            input_dir=graph_input_dir,
            output_dir=graph_output_dir,
            limit=limit if limit is not None else (None if get_value(graph_cfg, "limit", None) is None else int(get_value(graph_cfg, "limit", None))),
        )
        pipeline_summary["stages"]["build_text_hierarchy_graphs"] = {
            "enabled": True,
            "input_dir": str(graph_input_dir),
            "output_dir": str(graph_output_dir),
            "summary_path": str(graph_output_dir / "run_summary.json"),
            "result": summary,
        }
    else:
        pipeline_summary["stages"]["build_text_hierarchy_graphs"] = {
            "enabled": False,
            "input_dir": str(graph_input_dir),
            "output_dir": str(graph_output_dir),
        }

    if get_bool(manifest_cfg, "enabled", False):
        manifest_output_name = get_value(manifest_cfg, "output_name", None)
        if manifest_output_name in (None, ""):
            manifest_output_csv = manifest_output_dir / f"text_graph_manifest_{filter_mode}.csv"
        else:
            manifest_output_csv = manifest_output_dir / str(manifest_output_name)

        summary = prepare_manifest(
            graph_dir=get_path(manifest_block, "graph_dir", graph_output_dir, config_path),
            label_csv=get_path(manifest_block, "label_csv", DEFAULT_LABEL_CSV, config_path),
            output_csv=manifest_output_csv,
            split_csv=None if get_value(manifest_block, "split_csv", None) is None else get_path(manifest_block, "split_csv", DEFAULT_LABEL_CSV, config_path),
            image_dir=None if get_value(manifest_block, "image_dir", None) is None else get_path(manifest_block, "image_dir", output_root, config_path),
        )
        pipeline_summary["stages"]["prepare_text_graph_manifest"] = {
            "enabled": True,
            "graph_dir": str(get_path(manifest_block, "graph_dir", graph_output_dir, config_path)),
            "output_csv": summary["output_csv"],
            "summary_path": summary["output_summary_json"],
            "result": summary,
        }
    else:
        pipeline_summary["stages"]["prepare_text_graph_manifest"] = {
            "enabled": False,
            "graph_dir": str(get_path(manifest_block, "graph_dir", graph_output_dir, config_path)),
            "output_dir": str(manifest_output_dir),
        }

    summary_path = _pipeline_summary_path(output_root, filter_mode)
    write_json(summary_path, pipeline_summary)
    pipeline_summary["summary_path"] = str(summary_path)
    return pipeline_summary


def main() -> None:
    args = parse_args()
    summary = run_pipeline(
        config_path_arg=args.config,
        filter_mode_override=args.filter_mode,
        limit_override=args.limit,
    )
    print(
        "Pipeline finished | "
        f"filter_mode={summary['filter_mode']} | "
        f"summary={summary['summary_path']}"
    )


if __name__ == "__main__":
    main()
