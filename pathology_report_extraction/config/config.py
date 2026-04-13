from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:
    yaml = None


PIPELINE_STAGE_NAMES = {
    "preprocess",
    "export_sentence_views",
    "extract_ontology_concepts",
    "encode_sentence_exports_conch",
    "build_text_hierarchy_graphs",
    "prepare_text_graph_manifest",
}


def _require_yaml() -> None:
    if yaml is None:
        raise ImportError("PyYAML is required. Install with: pip install PyYAML")


def load_yaml_config(path: str | Path | None) -> tuple[dict[str, Any], Path | None]:
    if path is None:
        return {}, None

    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = config_path.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    _require_yaml()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML content must be a mapping: {config_path}")
    return data, config_path


def get_stage_config(config: dict[str, Any], stage_name: str) -> dict[str, Any]:
    if stage_name not in PIPELINE_STAGE_NAMES:
        raise ValueError(f"Unsupported pipeline stage: {stage_name}")

    stage_block = config.get(stage_name)
    defaults_block = config.get("defaults")

    if isinstance(stage_block, dict):
        merged: dict[str, Any] = {}
        if isinstance(defaults_block, dict):
            merged.update(defaults_block)
        merged.update(stage_block)
        return merged

    return config


def get_value(config: dict[str, Any], key: str, default: Any) -> Any:
    value = config.get(key)
    return default if value is None else value


def get_path(config: dict[str, Any], key: str, default: Path, config_path: Path | None) -> Path:
    value = config.get(key)
    if value in (None, ""):
        return Path(default)

    path = Path(str(value)).expanduser()
    if not path.is_absolute() and config_path is not None:
        path = (config_path.parent / path).resolve()
    return path


def get_bool(config: dict[str, Any], key: str, default: bool) -> bool:
    value = config.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)
