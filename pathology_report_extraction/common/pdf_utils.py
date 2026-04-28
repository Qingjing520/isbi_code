from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import fitz
import numpy as np


LOGGER = logging.getLogger(__name__)

CASE_ID_RE = re.compile(r"(TCGA-[A-Z0-9-]+)", re.IGNORECASE)
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\u2028\u2029]")


@dataclass
class PageExtraction:
    page_number: int
    strategy: str
    native_text: str
    final_text: str
    native_char_count: int
    final_char_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def iter_pdf_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("*.PDF") if path.is_file())


def resolve_dataset_input_dir(input_dir: Path, dataset_token: str) -> Path:
    dataset_token = dataset_token.upper()
    if input_dir.name.upper() == dataset_token:
        return input_dir

    direct_candidates = sorted(
        child for child in input_dir.iterdir() if child.is_dir() and dataset_token in child.name.upper()
    )
    if direct_candidates:
        return direct_candidates[0]

    recursive_candidates = sorted(
        child for child in input_dir.rglob("*") if child.is_dir() and dataset_token in child.name.upper()
    )
    if recursive_candidates:
        return recursive_candidates[0]

    raise FileNotFoundError(f"Could not locate a dataset directory matching '{dataset_token}' under {input_dir}.")


def extract_case_id(pdf_path: Path) -> str:
    match = CASE_ID_RE.search(pdf_path.stem)
    if match:
        return match.group(1).upper()
    return pdf_path.stem.upper()


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def render_page(page: fitz.Page, zoom: float = 2.0) -> np.ndarray:
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def extract_native_text_from_page(page: fitz.Page) -> str:
    blocks = page.get_text("blocks")
    sorted_blocks = sorted(blocks, key=lambda item: (round(item[1], 1), round(item[0], 1)))
    pieces: list[str] = []
    for block in sorted_blocks:
        text = block[4]
        if text and text.strip():
            pieces.append(text.strip())
    return "\n".join(pieces).strip()


def native_text_quality(text: str) -> dict[str, float]:
    stripped = text.strip()
    if not stripped:
        return {
            "char_count": 0,
            "alpha_count": 0,
            "digit_count": 0,
            "line_count": 0,
            "alpha_ratio": 0.0,
            "score": 0.0,
        }

    char_count = len(stripped)
    alpha_count = sum(ch.isalpha() for ch in stripped)
    digit_count = sum(ch.isdigit() for ch in stripped)
    line_count = sum(1 for line in stripped.splitlines() if line.strip())
    alpha_ratio = alpha_count / max(char_count, 1)
    barcode_like_lines = sum(
        1
        for line in stripped.splitlines()
        if len(line.strip()) >= 18 and re.fullmatch(r"[I1l| ]+", line.strip())
    )
    score = char_count + alpha_count * 0.5 + line_count * 5 - barcode_like_lines * 120 - digit_count * 0.05
    return {
        "char_count": float(char_count),
        "alpha_count": float(alpha_count),
        "digit_count": float(digit_count),
        "line_count": float(line_count),
        "alpha_ratio": alpha_ratio,
        "score": score,
    }


def is_low_text_page(text: str, min_chars: int = 80, min_alpha_ratio: float = 0.2) -> bool:
    metrics = native_text_quality(text)
    return metrics["char_count"] < min_chars or metrics["alpha_ratio"] < min_alpha_ratio


def flatten_pages(pages: Sequence[PageExtraction], use_final_text: bool = True) -> str:
    key = "final_text" if use_final_text else "native_text"
    chunks = [getattr(page, key).strip() for page in pages if getattr(page, key).strip()]
    return "\n\n".join(chunks).strip()


def sanitize_json_string(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = CONTROL_CHAR_RE.sub(" ", value)
    value = value.encode("utf-8", "replace").decode("utf-8", "replace")
    return value


def sanitize_json_payload(value):
    if isinstance(value, dict):
        return {str(key): sanitize_json_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json_payload(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_json_payload(item) for item in value]
    if isinstance(value, str):
        return sanitize_json_string(value)
    return value


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    clean_payload = sanitize_json_payload(payload)
    path.write_text(json.dumps(clean_payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def make_output_stem(pdf_path: Path) -> str:
    return sanitize_filename(pdf_path.stem)


def summarize_failures(failures: Sequence[dict]) -> dict:
    return {
        "failure_count": len(failures),
        "failures": list(failures),
    }


def log_pdf_exception(pdf_path: Path, exc: Exception) -> None:
    LOGGER.exception("Failed to process %s: %s", pdf_path, exc)
