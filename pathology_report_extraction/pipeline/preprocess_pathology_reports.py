from __future__ import annotations

"""Pathology report preprocessing pipeline.

Dependencies:
    pip install PyMuPDF rapidocr_onnxruntime opencv-python Pillow numpy

This script recursively scans pathology-report PDFs and writes a uniform
three-level JSON structure:

    Document -> Section -> Sentence

The default filtering mode is "masked", which keeps pathology evidence
sentences while deleting or masking explicit stage and TNM leakage before
sentence encoding.
"""

import argparse
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Sequence

import fitz

from pathology_report_extraction.common.ocr_utils import ocr_page
from pathology_report_extraction.common.pdf_utils import ensure_dir, extract_case_id, extract_native_text_from_page, write_json
from pathology_report_extraction.common.pipeline_defaults import DEFAULT_FILTER_MODE, DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_ROOT, PREPROCESS_OUTPUT_SUBDIRS
from pathology_report_extraction.common.text_cleaning import clean_lines, normalize_line, normalize_text, split_sentences
from pathology_report_extraction.config.config import get_path, get_stage_config, get_value, load_yaml_config


LOGGER = logging.getLogger("preprocess_pathology_reports")

DEFAULT_SECTION_TITLE = "Document Body"
DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / PREPROCESS_OUTPUT_SUBDIRS[DEFAULT_FILTER_MODE]


COMMON_SECTION_RULES: list[tuple[str, Sequence[re.Pattern[str]]]] = [
    (
        "Clinical Information",
        [
            re.compile(r"^CLINICAL INFORMATION\b[:]?", re.IGNORECASE),
            re.compile(r"^CLINICAL HISTORY\b[:]?", re.IGNORECASE),
            re.compile(r"^PATIENT HISTORY\b[:]?", re.IGNORECASE),
            re.compile(r"^HISTORY\b[:]?$", re.IGNORECASE),
            re.compile(r"^CLINICAL DIAGNOSIS\s*&\s*HISTORY\b[:]?", re.IGNORECASE),
            re.compile(r"^CLINICAL HISTORY/DIAGNOSIS\b[:]?", re.IGNORECASE),
        ],
    ),
    ("Procedure", [re.compile(r"^PROCEDURE\b[:]?", re.IGNORECASE)]),
    (
        "Specimen Submitted",
        [
            re.compile(r"^SPECIMENS? SUBMITTED\b[:]?", re.IGNORECASE),
            re.compile(r"^SOURCE OF SPECIMEN\(S\)\b[:]?", re.IGNORECASE),
            re.compile(r"^SOURCE OF SPECIMENS?\b[:]?", re.IGNORECASE),
        ],
    ),
    (
        "Gross Description",
        [
            re.compile(r"^GROSS DESCRIPTION\b[:]?", re.IGNORECASE),
            re.compile(r"^MACROSCOPIC DESCRIPTION\b[:]?", re.IGNORECASE),
            re.compile(r"^[A-Z<]{0,5}OSS DESCRIPTION\b[:]?", re.IGNORECASE),
        ],
    ),
    ("Microscopic Description", [re.compile(r"^MICROSCOPIC DESCRIPTION\b[:]?", re.IGNORECASE)]),
    (
        "Diagnosis",
        [
            re.compile(r"^DIAGNOSIS\b[:]?", re.IGNORECASE),
            re.compile(r"^AMENDED DIAGNOSIS\b[:]?", re.IGNORECASE),
            re.compile(r"^PATHOLOGIC DIAGNOSIS\b[:]?", re.IGNORECASE),
            re.compile(r"^PATHOLOGICAL DIAGNOSIS\b[:]?", re.IGNORECASE),
            re.compile(r"^MICROSCOPIC DIAGNOSIS\b[:]?", re.IGNORECASE),
            re.compile(r"^HISTOPATHOLOGICAL DIAGNOSIS\b[:]?", re.IGNORECASE),
            re.compile(r"^DIAGNOSTIC INTERPRETATION\b[:]?", re.IGNORECASE),
        ],
    ),
    (
        "Final Diagnosis",
        [
            re.compile(r"^FINAL DIAGNOSIS\b[:]?", re.IGNORECASE),
            re.compile(r"^PATH FINAL DIAGNOSIS\b[:]?", re.IGNORECASE),
            re.compile(r"^POST[ -]?OP DIAGNOSIS\b[:]?", re.IGNORECASE),
            re.compile(r"^(?:PRE|PRO)[ -]?OP DIAGNOSIS\b[:]?", re.IGNORECASE),
        ],
    ),
    ("Comment", [re.compile(r"^COMMENT(?:\(S\))?\b[:]?", re.IGNORECASE)]),
    (
        "Synoptic Report",
        [
            re.compile(r"^SYNOPTIC(?: REPORT)?\b.*$", re.IGNORECASE),
            re.compile(r"^SURGICAL PATHOLOGY CANCER CASE SUMMARY\b.*$", re.IGNORECASE),
            re.compile(r"^CASE SUMMARY\b.*$", re.IGNORECASE),
            re.compile(r"^CHECKLIST\b.*$", re.IGNORECASE),
        ],
    ),
    ("Ancillary Studies", [re.compile(r"^ANCILLARY STUDIES\b[:]?", re.IGNORECASE)]),
    ("Intraoperative Consultation", [re.compile(r"^INTRAOPERATIVE CONSULTATION\b[:]?", re.IGNORECASE)]),
    ("Summary of Sections", [re.compile(r"^SUMMARY OF SECTIONS\b[:]?", re.IGNORECASE)]),
]

BRCA_SECTION_RULES: list[tuple[str, Sequence[re.Pattern[str]]]] = [
    ("Patient History", [re.compile(r"^PATIENT HISTORY\b[:]?", re.IGNORECASE)]),
    ("Clinical Information", [re.compile(r"^CLINICAL INFORMATION\b[:]?", re.IGNORECASE)]),
    ("Procedure", [re.compile(r"^PROCEDURE\b[:]?", re.IGNORECASE)]),
    (
        "Final Diagnosis",
        [
            re.compile(r"^FINAL DIAGNOSIS\b[:]?", re.IGNORECASE),
            re.compile(r"^HISTOPATHOLOGICAL DIAGNOSIS\b[:]?", re.IGNORECASE),
            re.compile(r"^MICROSCOPIC DIAGNOSIS\b[:]?", re.IGNORECASE),
            re.compile(r"^DIAGNOSIS\b[:]?", re.IGNORECASE),
        ],
    ),
    (
        "Gross Description",
        [
            re.compile(r"^GROSS DESCRIPTION\b[:]?", re.IGNORECASE),
            re.compile(r"^MACROSCOPIC DESCRIPTION\b[:]?", re.IGNORECASE),
            re.compile(r"^[A-Z<]{0,5}OSS DESCRIPTION\b[:]?", re.IGNORECASE),
        ],
    ),
    ("Microscopic Description", [re.compile(r"^MICROSCOPIC DESCRIPTION\b[:]?", re.IGNORECASE)]),
    ("Comment", [re.compile(r"^COMMENT(?:\(S\))?\b[:]?", re.IGNORECASE)]),
    (
        "Synoptic Report",
        [
            re.compile(r"^SYNOPTIC\b.*$", re.IGNORECASE),
            re.compile(r"^SURGICAL PATHOLOGY CANCER CASE SUMMARY\b.*$", re.IGNORECASE),
            re.compile(r"^CASE SUMMARY\b.*$", re.IGNORECASE),
            re.compile(r"^CHECKLIST\b.*$", re.IGNORECASE),
        ],
    ),
    (
        "Ancillary Studies",
        [
            re.compile(r"^ANCILLARY STUDIES\b[:]?", re.IGNORECASE),
            re.compile(r"^RESULTS OF IMMUNOHISTOCHEMICAL EXAMINATION\b.*$", re.IGNORECASE),
        ],
    ),
]

KIRC_SECTION_RULES: list[tuple[str, Sequence[re.Pattern[str]]]] = [
    (
        "Clinical Information",
        [
            re.compile(r"^CLINICAL DIAGNOSIS\s*&\s*HISTORY\b[:]?", re.IGNORECASE),
            re.compile(r"^CLINICAL HISTORY/DIAGNOSIS\b[:]?", re.IGNORECASE),
            re.compile(r"^CLINICAL HISTORY\b[:]?", re.IGNORECASE),
        ],
    ),
    (
        "Specimen Submitted",
        [
            re.compile(r"^SPECIMENS SUBMITTED\b[:]?", re.IGNORECASE),
            re.compile(r"^SOURCE OF SPECIMEN\(S\)\b[:]?", re.IGNORECASE),
            re.compile(r"^SOURCE OF SPECIMENS?\b[:]?", re.IGNORECASE),
        ],
    ),
    (
        "Diagnosis",
        [
            re.compile(r"^AMENDED DIAGNOSIS\b[:]?", re.IGNORECASE),
            re.compile(r"^FINAL DIAGNOSIS\b[:]?", re.IGNORECASE),
            re.compile(r"^DIAGNOSIS\b[:]?", re.IGNORECASE),
        ],
    ),
    (
        "Gross Description",
        [
            re.compile(r"^GROSS DESCRIPTION\b[:]?", re.IGNORECASE),
            re.compile(r"^[A-Z<]{0,5}OSS DESCRIPTION\b[:]?", re.IGNORECASE),
        ],
    ),
    (
        "Intraoperative Consultation",
        [
            re.compile(r"^INTRAOPERATIVE CONSULTATION\b[:]?", re.IGNORECASE),
            re.compile(r"^INTRAOPERATIVE CONS\w+TATION\b[:]?", re.IGNORECASE),
        ],
    ),
    ("Summary of Sections", [re.compile(r"^SUMMARY OF SECTIONS\b[:]?", re.IGNORECASE)]),
    ("Comment", [re.compile(r"^COMMENT\b[:]?", re.IGNORECASE)]),
]

BRCA_EXTRA_NOISE = [
    r"^RUN DATE.*$",
    r"^RUN TIME.*$",
    r"^RUN USER.*$",
    r"^COMPLIANCE VALIDATED BY.*$",
    r"^TCGA-[A-Z0-9-]+(?:-[A-Z0-9]+)?$",
    r"^REDACTED$",
]

KIRC_EXTRA_NOISE = [
    r"^\*{5,}$",
    r"^\*.*REPORT ELECTRONICALLY SIGNED OUT.*$",
    r"^I.?ATTEST.*$",
    r"^REVIEWED AND APPROVED THIS REPORT\.?$",
    r"^SURGICAL PATHOLOGY REPORT$",
]

ALL_SECTION_RULES = COMMON_SECTION_RULES + [
    rule for rule in BRCA_SECTION_RULES + KIRC_SECTION_RULES if rule[0] not in {title for title, _ in COMMON_SECTION_RULES}
]
KNOWN_TITLE_SET = {title for title, _ in ALL_SECTION_RULES}

GENERAL_NOISE_PATTERNS = [
    r"^PAGE\s+\d+\s*(?:/|OF)\s*\d+$",
    r"^PAGE\s+\d+$",
    r"^END OF REPORT$",
    r"^SURGICAL PATHOLOGY REPORT$",
    r"^REPORT ELECTRONICALLY SIGNED OUT.*$",
    r"^REVIEWED AND APPROVED THIS REPORT\.?$",
    r"^RUN DATE.*$",
    r"^RUN TIME.*$",
    r"^RUN USER.*$",
    r"^TCGA-[A-Z0-9-]+(?:-[A-Z0-9]+)?$",
    r"^REDACTED$",
]

DIAGNOSIS_TITLE_PATTERNS = [
    re.compile(r"\bFINAL DIAGNOSIS\b", re.IGNORECASE),
    re.compile(r"\bDIAGNOSIS\b", re.IGNORECASE),
    re.compile(r"\bDIAGNOSTIC\b", re.IGNORECASE),
]

NO_DIAGNOSIS_SECTION_DROP_PATTERNS = {
    "COMMON": [
        re.compile(r"\b(final diagnosis|diagnosis|pathologic(?:al)? diagnosis|diagnostic interpretation|amended diagnosis)\b", re.IGNORECASE),
        re.compile(r"\b(synoptic(?: report)?|surgical pathology cancer case summary|case summary|checklist|cap cancer protocol)\b", re.IGNORECASE),
        re.compile(r"\b(ajcc|tnm|staging summary|stage summary|pathologic stage|pathological stage|impression)\b", re.IGNORECASE),
        re.compile(r"\b(intraoperative consultation|frozen section diagnosis|permanent diagnosis|summary of sections)\b", re.IGNORECASE),
    ],
    "BRCA": [re.compile(r"\b(breast tissue checklist)\b", re.IGNORECASE)],
    "KIRC": [],
}

MASKING_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("ajcc_stage", re.compile(r"\bAJCC\s*stage\s*[:\-]?\s*[IVX0-9A-D]+\b", re.IGNORECASE), "[MASK_STAGE]"),
    ("pathologic_stage_assessment", re.compile(r"\bpatholog(?:y|ic(?:al)?)\s+stage\s+assessment\b", re.IGNORECASE), "[MASK_STAGE]"),
    ("pathologic_tnm_stage_heading", re.compile(r"\bpatholog(?:y|ic(?:al)?)\s+(?:tnm\s+)?stage(?:\s*\([^)]*\))?\s*:?", re.IGNORECASE), "[MASK_STAGE]"),
    ("pathologic_stage_heading", re.compile(r"\bpatholog(?:y|ic(?:al)?)\s+stage(?:\s*\([^)]*\))?\s*:?", re.IGNORECASE), "[MASK_STAGE]"),
    ("pathologic_tumor_stage", re.compile(r"\bpathologic(?:al)?\s+tumou?r\s+stage\b", re.IGNORECASE), "[MASK_STAGE]"),
    ("pathologic_tumor_staging", re.compile(r"\binvasive\s+pathologic\s+tumou?r\s+staging(?:\s*\([^)]*\))?", re.IGNORECASE), "[MASK_STAGE]"),
    ("pathologic_staging", re.compile(r"\bpathologic(?:al)?\s+staging\b", re.IGNORECASE), "[MASK_STAGE]"),
    ("pathologic_stage", re.compile(r"\bpathologic(?:al)?\s+stage\s*[:\-]?\s*[IVX0-9A-D]+\b", re.IGNORECASE), "[MASK_STAGE]"),
    ("stage", re.compile(r"\bstage\s*[:\-]?\s*[IVX0-9A-D]+\b", re.IGNORECASE), "[MASK_STAGE]"),
    ("ptnm", re.compile(r"\bpTNM\b", re.IGNORECASE), "[MASK_TNM]"),
    ("tnm", re.compile(r"\b(?:AJCC\s*)?TNM\b", re.IGNORECASE), "[MASK_TNM]"),
    ("ajcc_ref", re.compile(r"\bAJCC\b", re.IGNORECASE), "[MASK_AJCC]"),
    ("primary_t_field", re.compile(r"\bprimary tumor\s*\([cpy]?t\)\b", re.IGNORECASE), "[MASK_T_FIELD]"),
    ("regional_n_field", re.compile(r"\bregional lymph nodes?\s*\([cpy]?n\)\b", re.IGNORECASE), "[MASK_N_FIELD]"),
    ("distant_m_field", re.compile(r"\bdistant metast(?:asis|ases)\s*\([cpy]?m\)\b", re.IGNORECASE), "[MASK_M_FIELD]"),
    (
        "tnm_compound",
        re.compile(
            r"\b[cpy]?T(?:is|[x0o]|[1-4il])(?:[a-d]|mi|lc|1c)?\s*"
            r"[,/;-]?\s*[cpy]?N(?:[x0o]|[1-3il])(?:[a-c]|mi|ia|ib|ic|i+)?(?:\s*\([^)]*\))?\s*"
            r"[,/;-]?\s*[cpy]?M(?:[x0o]|1)\b",
            re.IGNORECASE,
        ),
        "[MASK_TNM_SEQUENCE]",
    ),
    ("t_category", re.compile(r"\b[cpy]?T(?:is|x|0|[1-4])(?:[a-d]|mi)?\b", re.IGNORECASE), "[MASK_T]"),
    ("n_category", re.compile(r"\b[cpy]?N(?:x|0|[1-3])(?:[a-c]|mi)?\b", re.IGNORECASE), "[MASK_N]"),
    ("m_category", re.compile(r"\b[cpy]?M(?:x|0|1)\b", re.IGNORECASE), "[MASK_M]"),
    (
        "t_category_loose",
        re.compile(r"(?<![A-Za-z0-9])[cpy]T(?:is|[x0o]|[1-4il])(?:[a-d]|mi|lc|1c)?(?![A-Za-z])(?:\s*\([^)]*\))?", re.IGNORECASE),
        "[MASK_T]",
    ),
    (
        "n_category_loose",
        re.compile(r"(?<![A-Za-z0-9])[cpy]N(?:[x0o]|[1-3il])(?:[a-c]|mi|ia|ib|ic|i\+)?(?![A-Za-z])(?:\s*\([^)]*\))?", re.IGNORECASE),
        "[MASK_N]",
    ),
    (
        "m_category_loose",
        re.compile(r"(?<![A-Za-z0-9])[cpy]M(?:[x0o]|1)(?![A-Za-z])(?:\s*\([^)]*\))?", re.IGNORECASE),
        "[MASK_M]",
    ),
]

STAGE_CONTEXT_MASKING_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    (
        "stage_context_t",
        re.compile(r"(?<![A-Za-z0-9])T(?:is|X|0|O|[1-4Il])(?:[A-Da-d]|mi|lc|1c)?(?![A-Za-z])(?:\s*\([^)]*\))?"),
        "[MASK_T]",
    ),
    (
        "stage_context_n",
        re.compile(r"(?<![A-Za-z0-9])N(?:X|0|O|[1-3Il])(?:[A-Ca-c]|mi|ia|ib|ic|IA|IB|IC|I\+)?(?![A-Za-z])(?:\s*\([^)]*\))?"),
        "[MASK_N]",
    ),
    (
        "stage_context_m",
        re.compile(r"(?<![A-Za-z0-9])M(?:X|0|O|1)(?![A-Za-z])(?:\s*\([^)]*\))?"),
        "[MASK_M]",
    ),
]

STAGE_SIGNAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bAJCC\s*stage\b", re.IGNORECASE),
    re.compile(r"\bpTNM\s*stage\b", re.IGNORECASE),
    re.compile(r"\bTNM\s*stage\b", re.IGNORECASE),
    re.compile(r"\bpatholog(?:y|ic(?:al)?)\s+stage\b", re.IGNORECASE),
    re.compile(r"\bclinical\s+stage\b", re.IGNORECASE),
    re.compile(r"\b[Ss]tage\s*[IVX0-9A-D]+\b"),
    re.compile(r"\b[cp]?T[0-4][a-d]?\b", re.IGNORECASE),
    re.compile(r"\b[cp]?N[0-3][a-c]?\b", re.IGNORECASE),
    re.compile(r"\b[cp]?M[0-1xX]\b", re.IGNORECASE),
    re.compile(r"\bT[0-4][a-d]?N[0-3][a-c]?M[0-1xX]\b", re.IGNORECASE),
]

KEEP_HINT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bhistologic(?:al)? type\b", re.IGNORECASE),
    re.compile(r"\btumou?r size\b", re.IGNORECASE),
    re.compile(r"\btumou?r site\b", re.IGNORECASE),
    re.compile(r"\bmacroscopic extent\b", re.IGNORECASE),
    re.compile(r"\bextent of tumou?r\b", re.IGNORECASE),
    re.compile(r"\bnuclear grade\b", re.IGNORECASE),
    re.compile(r"\bfuhrman grade\b", re.IGNORECASE),
    re.compile(r"\bnottingham grade\b", re.IGNORECASE),
    re.compile(r"\blymphovascular\b", re.IGNORECASE),
    re.compile(r"\btranscapsular\b", re.IGNORECASE),
    re.compile(r"\brenal vein\b", re.IGNORECASE),
    re.compile(r"\bvena?\s*caval\b", re.IGNORECASE),
    re.compile(r"\bmargin(?:s)?\b", re.IGNORECASE),
    re.compile(r"\blymph node(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bgross description\b", re.IGNORECASE),
    re.compile(r"\bmicroscopic description\b", re.IGNORECASE),
    re.compile(r"\bspecimen\b", re.IGNORECASE),
    re.compile(r"\binvasion\b", re.IGNORECASE),
    re.compile(r"\bkidney\b", re.IGNORECASE),
    re.compile(r"\bpelvis\b", re.IGNORECASE),
    re.compile(r"\bvessel(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bsoft tissue\b", re.IGNORECASE),
]


PROTECTED_ABBREVIATIONS = [
    "Dr.",
    "No.",
    "Fig.",
    "vs.",
    "cm.",
    "mm.",
    "mg.",
    "ml.",
    "dept.",
    "approx.",
]


def setup_logging(output_dir: Path) -> None:
    ensure_dir(output_dir)
    log_path = output_dir / "preprocess.log"
    handlers = [logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def iter_pdf_files_recursive(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf")


def detect_dataset(pdf_path: Path) -> str | None:
    for part in pdf_path.parts:
        upper = part.upper()
        if upper == "BRCA":
            return "BRCA"
        if upper == "KIRC":
            return "KIRC"
        if upper == "LUSC":
            return "LUSC"
    return None


def choose_dataset_rules(dataset: str | None) -> tuple[list[tuple[str, Sequence[re.Pattern[str]]]], list[str]]:
    if dataset == "BRCA":
        return BRCA_SECTION_RULES, BRCA_EXTRA_NOISE
    if dataset == "KIRC":
        return KIRC_SECTION_RULES, KIRC_EXTRA_NOISE
    return COMMON_SECTION_RULES, []


def choose_page_text(native_text: str, ocr_text: str) -> tuple[str, str]:
    native_text = (native_text or "").strip()
    ocr_text = (ocr_text or "").strip()

    if native_text and not ocr_text:
        return native_text, "native_text"
    if ocr_text and not native_text:
        return ocr_text, "ocr_only"
    if not native_text and not ocr_text:
        return "", "empty"
    if len(ocr_text) >= len(native_text) * 1.15:
        return ocr_text, "ocr_fallback"
    if native_text == ocr_text:
        return native_text, "native_text"
    return f"{native_text}\n{ocr_text}".strip(), "native_plus_ocr"


def extract_pages(pdf_path: Path, native_char_threshold: int, ocr_zoom: float) -> list[dict]:
    pages: list[dict] = []
    with fitz.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            native_text = extract_native_text_from_page(page)
            final_text = native_text.strip()
            strategy = "native_text"

            if len(final_text) < native_char_threshold:
                try:
                    ocr_payload = ocr_page(page, zoom=ocr_zoom, deskew=True)
                    final_text, strategy = choose_page_text(native_text, ocr_payload.get("text", ""))
                except Exception as exc:
                    LOGGER.warning("OCR failed for %s page %s: %s", pdf_path.name, page_number, exc)
                    final_text = native_text.strip()
                    strategy = "native_text_ocr_failed"

            pages.append({"page_number": page_number, "strategy": strategy, "text": final_text})
    return pages


def margin_candidate(line: str) -> bool:
    line = normalize_line(line)
    if not line:
        return False
    if len(line) < 4 or len(line) > 100:
        return False
    if re.fullmatch(r"[\W_]+", line):
        return False
    return True


def build_repeated_margin_patterns(page_texts: Sequence[str], edge_line_count: int = 3) -> list[str]:
    if len(page_texts) < 3:
        return []

    counts: Counter[str] = Counter()
    for text in page_texts:
        lines = [normalize_line(line) for line in text.splitlines() if normalize_line(line)]
        edge_lines = lines[:edge_line_count] + lines[-edge_line_count:]
        counts.update({line for line in edge_lines if margin_candidate(line)})

    threshold = max(2, math.ceil(len(page_texts) * 0.6))
    repeated_lines = [line for line, count in counts.items() if count >= threshold]
    return [rf"^{re.escape(line)}$" for line in repeated_lines]


def normalize_heading_title(title: str) -> str:
    stripped = normalize_line(title).strip(":- ")
    if not stripped:
        return DEFAULT_SECTION_TITLE

    for canonical_title, patterns in ALL_SECTION_RULES:
        if canonical_title.lower() == stripped.lower():
            return canonical_title
        for pattern in patterns:
            if pattern.match(stripped):
                return canonical_title

    words = [word.capitalize() if not word.isupper() else word for word in stripped.split()]
    normalized = " ".join(words)
    return normalized or DEFAULT_SECTION_TITLE


def split_sections_by_rules(lines: Sequence[str], heading_rules: Sequence[tuple[str, Sequence[re.Pattern[str]]]]) -> list[dict[str, str]]:
    sections: list[dict[str, str]] = []
    current_title = DEFAULT_SECTION_TITLE
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        text = "\n".join(line for line in buffer if line.strip()).strip()
        if text:
            sections.append({"section_title": current_title, "text": text})
        buffer = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if buffer and buffer[-1] != "":
                buffer.append("")
            continue

        matched_title = None
        remainder = ""
        for canonical_title, patterns in heading_rules:
            for pattern in patterns:
                match = pattern.match(line)
                if match:
                    matched_title = canonical_title
                    remainder = line[match.end():].strip(" :-")
                    break
            if matched_title:
                break

        if matched_title:
            flush()
            current_title = matched_title
            if remainder:
                buffer.append(remainder)
            continue

        buffer.append(line)

    flush()
    return sections


def looks_like_generic_heading(line: str) -> bool:
    stripped = normalize_line(line)
    if not stripped or len(stripped) > 90:
        return False
    if stripped.endswith("."):
        return False

    if ":" in stripped:
        prefix, _ = stripped.split(":", 1)
        prefix = prefix.strip()
        if not prefix or len(prefix) > 45:
            return False
        if any(char.isdigit() for char in prefix):
            return False
        word_count = len(prefix.split())
        return 1 <= word_count <= 6 and (prefix.isupper() or prefix.istitle())

    if stripped == stripped.upper() and re.search(r"[A-Z]", stripped):
        word_count = len(stripped.split())
        return 1 <= word_count <= 8
    return False


def match_generic_heading(line: str, current_title: str) -> tuple[str, str] | None:
    if current_title == "Synoptic Report":
        return None
    if not looks_like_generic_heading(line):
        return None

    if ":" in line:
        raw_title, remainder = line.split(":", 1)
        title = normalize_heading_title(raw_title)
        if title in KNOWN_TITLE_SET or raw_title.isupper():
            return title, remainder.strip()
        return None

    if line.strip().upper() in {"PART 1", "PART 2", "PART 3"}:
        return None

    return normalize_heading_title(line), ""


def split_generic_sections(lines: Sequence[str]) -> list[dict[str, str]]:
    sections: list[dict[str, str]] = []
    current_title = DEFAULT_SECTION_TITLE
    buffer: list[str] = []
    previous_blank = True

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        text = "\n".join(buffer).strip()
        if text:
            sections.append({"section_title": current_title, "text": text})
        buffer = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if buffer and buffer[-1] != "":
                buffer.append("")
            previous_blank = True
            continue

        heading_match = match_generic_heading(line, current_title) if previous_blank else None
        if heading_match is not None:
            title, remainder = heading_match
            flush()
            current_title = title
            if remainder:
                buffer.append(remainder)
            previous_blank = False
            continue

        buffer.append(line)
        previous_blank = False

    flush()
    return sections


def score_sections(sections: Sequence[dict[str, str]]) -> tuple[int, int, int]:
    non_default = sum(1 for section in sections if section["section_title"] != DEFAULT_SECTION_TITLE)
    non_empty = sum(1 for section in sections if section["text"].strip())
    total_chars = sum(len(section["text"]) for section in sections)
    return non_default, non_empty, total_chars


def choose_best_sections(rule_sections: list[dict[str, str]], generic_sections: list[dict[str, str]]) -> tuple[list[dict[str, str]], str]:
    if score_sections(rule_sections) >= score_sections(generic_sections):
        return rule_sections, "rule_based"
    return generic_sections, "generic_heading_fallback"


def is_diagnosis_section(section_title: str) -> bool:
    title = normalize_heading_title(section_title)
    return any(pattern.search(title) for pattern in DIAGNOSIS_TITLE_PATTERNS)


def empty_filter_metadata() -> dict:
    return {
        "removed_section_titles": [],
        "removed_sentence_count": 0,
        "removed_sentence_reasons": Counter(),
        "emptied_section_titles": [],
        "masked_sentence_count": 0,
        "masked_sentence_reasons": Counter(),
    }


def should_drop_section_no_diagnosis(dataset: str | None, section_title: str) -> bool:
    title = normalize_heading_title(section_title)
    if is_diagnosis_section(title):
        return True
    for pattern in NO_DIAGNOSIS_SECTION_DROP_PATTERNS["COMMON"]:
        if pattern.search(title):
            return True
    if dataset in NO_DIAGNOSIS_SECTION_DROP_PATTERNS:
        for pattern in NO_DIAGNOSIS_SECTION_DROP_PATTERNS[dataset]:
            if pattern.search(title):
                return True
    return False


def normalize_paragraphs(text: str) -> list[str]:
    text = normalize_text(text)
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    normalized: list[str] = []
    for paragraph in paragraphs:
        merged = re.sub(r"(?<!\n)\n(?!\n)", " ", paragraph).strip()
        if merged:
            normalized.append(merged)
    return normalized



def split_sentences_robust(text: str) -> list[str]:
    paragraphs = normalize_paragraphs(text)
    results: list[str] = []
    for paragraph in paragraphs:
        protected = paragraph
        for token in PROTECTED_ABBREVIATIONS:
            protected = protected.replace(token, token.replace(".", "<prd>"))
        protected = re.sub(r"(?<=\d)\.(?=\d)", "<prd>", protected)
        pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])", protected)
        for piece in pieces:
            sentence = piece.replace("<prd>", ".").strip()
            if sentence:
                results.append(sentence)
    return results or split_sentences(text)


def apply_sentence_masks(sentence: str) -> tuple[str, Counter]:
    masked = normalize_line(sentence)
    reasons: Counter = Counter()
    for reason, pattern, replacement in MASKING_PATTERNS:
        masked, count = pattern.subn(replacement, masked)
        if count:
            reasons[reason] += count
    if "[MASK_STAGE]" in masked or "[MASK_TNM]" in masked or "[MASK_TNM_SEQUENCE]" in masked:
        for reason, pattern, replacement in STAGE_CONTEXT_MASKING_PATTERNS:
            masked, count = pattern.subn(replacement, masked)
            if count:
                reasons[reason] += count
    masked = re.sub(r"\s{2,}", " ", masked).strip()
    masked = re.sub(r"\s+([,.;:])", r"\1", masked)
    return masked, reasons


def sentence_has_stage_signal(sentence: str) -> bool:
    normalized = normalize_line(sentence)
    return any(pattern.search(normalized) for pattern in STAGE_SIGNAL_PATTERNS)


def sentence_has_keep_hint(sentence: str) -> bool:
    normalized = normalize_line(sentence)
    return any(pattern.search(normalized) for pattern in KEEP_HINT_PATTERNS)


def filter_sentence_masked(sentence: str) -> tuple[str | None, Counter, str | None]:
    normalized = normalize_line(sentence)
    masked_sentence, reasons = apply_sentence_masks(normalized)
    has_stage_signal = sentence_has_stage_signal(normalized)
    has_keep_hint = sentence_has_keep_hint(normalized)

    if has_stage_signal and not has_keep_hint:
        return None, reasons, "stage_only"

    return masked_sentence, reasons, None


def finalize_section(section_title: str, sentences: Sequence[str], metadata: dict) -> tuple[dict[str, object] | None, dict]:
    clean_sentences = [normalize_line(sentence) for sentence in sentences if normalize_line(sentence)]
    if not clean_sentences:
        metadata["emptied_section_titles"].append(section_title)
        return None, metadata
    return {"section_title": section_title, "sentences": clean_sentences}, metadata


def filter_section_full(section: dict[str, str]) -> tuple[dict[str, object] | None, dict]:
    metadata = empty_filter_metadata()
    sentences = split_sentences_robust(section["text"])
    return finalize_section(section["section_title"], sentences, metadata)


def filter_section_no_diagnosis(dataset: str | None, section: dict[str, str]) -> tuple[dict[str, object] | None, dict]:
    metadata = empty_filter_metadata()
    if should_drop_section_no_diagnosis(dataset, section["section_title"]):
        metadata["removed_section_titles"].append(section["section_title"])
        return None, metadata

    sentences = split_sentences_robust(section["text"])
    return finalize_section(section["section_title"], sentences, metadata)


def filter_section_masked(section: dict[str, str]) -> tuple[dict[str, object] | None, dict]:
    metadata = empty_filter_metadata()
    masked_sentences: list[str] = []
    for sentence in split_sentences_robust(section["text"]):
        masked_sentence, reasons, drop_reason = filter_sentence_masked(sentence)
        if reasons:
            metadata["masked_sentence_count"] += 1
            metadata["masked_sentence_reasons"].update(reasons)
        if drop_reason:
            metadata["removed_sentence_count"] += 1
            metadata["removed_sentence_reasons"].update([drop_reason])
            continue
        if masked_sentence:
            masked_sentences.append(masked_sentence)
    return finalize_section(section["section_title"], masked_sentences, metadata)


def filter_section_no_diagnosis_masked(dataset: str | None, section: dict[str, str]) -> tuple[dict[str, object] | None, dict]:
    metadata = empty_filter_metadata()
    if should_drop_section_no_diagnosis(dataset, section["section_title"]):
        metadata["removed_section_titles"].append(section["section_title"])
        return None, metadata

    masked_sentences: list[str] = []
    for sentence in split_sentences_robust(section["text"]):
        masked_sentence, reasons = apply_sentence_masks(sentence)
        if reasons:
            metadata["masked_sentence_count"] += 1
            metadata["masked_sentence_reasons"].update(reasons)
        if masked_sentence:
            masked_sentences.append(masked_sentence)
    return finalize_section(section["section_title"], masked_sentences, metadata)


def merge_filter_metadata(items: Sequence[dict]) -> dict:
    merged = {
        "removed_section_titles": [],
        "removed_sentence_count": 0,
        "removed_sentence_reasons": Counter(),
        "emptied_section_titles": [],
        "masked_sentence_count": 0,
        "masked_sentence_reasons": Counter(),
    }
    for item in items:
        merged["removed_section_titles"].extend(item["removed_section_titles"])
        merged["removed_sentence_count"] += item["removed_sentence_count"]
        merged["removed_sentence_reasons"].update(item["removed_sentence_reasons"])
        merged["emptied_section_titles"].extend(item["emptied_section_titles"])
        merged["masked_sentence_count"] += item.get("masked_sentence_count", 0)
        merged["masked_sentence_reasons"].update(item.get("masked_sentence_reasons", {}))
    return merged


def filter_sections(sections_raw: Sequence[dict[str, str]], dataset: str | None, filter_mode: str) -> tuple[list[dict[str, object]], dict]:
    outputs: list[dict[str, object]] = []
    all_metadata: list[dict] = []

    for section in sections_raw:
        if filter_mode == "full":
            filtered, metadata = filter_section_full(section)
        elif filter_mode == "no_diagnosis":
            filtered, metadata = filter_section_no_diagnosis(dataset, section)
        elif filter_mode == "masked":
            filtered, metadata = filter_section_masked(section)
        elif filter_mode == "no_diagnosis_masked":
            filtered, metadata = filter_section_no_diagnosis_masked(dataset, section)
        else:
            raise ValueError(f"Unsupported filter mode: {filter_mode}")
        all_metadata.append(metadata)
        if filtered is not None:
            outputs.append(filtered)

    merged = merge_filter_metadata(all_metadata)
    merged["removed_sentence_reasons"] = dict(merged["removed_sentence_reasons"])
    merged["masked_sentence_reasons"] = dict(merged["masked_sentence_reasons"])
    return outputs, merged


def build_sections(pdf_path: Path, native_char_threshold: int, ocr_zoom: float) -> tuple[list[dict[str, str]], int, str]:
    dataset = detect_dataset(pdf_path)
    heading_rules, dataset_noise = choose_dataset_rules(dataset)
    pages = extract_pages(pdf_path, native_char_threshold=native_char_threshold, ocr_zoom=ocr_zoom)
    page_texts = [page["text"] for page in pages]
    repeated_margin_patterns = build_repeated_margin_patterns(page_texts)
    heading_patterns = [pattern for _, patterns in heading_rules for pattern in patterns]

    cleaned_lines = clean_lines(
        "\n\n".join(page_texts).splitlines(),
        extra_noise_patterns=[*GENERAL_NOISE_PATTERNS, *dataset_noise, *repeated_margin_patterns],
        heading_patterns=heading_patterns,
    )

    rule_sections = split_sections_by_rules(cleaned_lines, heading_rules)
    generic_sections = split_generic_sections(cleaned_lines)
    sections, parser_mode = choose_best_sections(rule_sections, generic_sections)
    return sections, len(pages), parser_mode


def preprocess_document(
    pdf_path: Path,
    input_root: Path,
    output_root: Path,
    native_char_threshold: int,
    ocr_zoom: float,
    filter_mode: str,
) -> dict:
    dataset = detect_dataset(pdf_path)
    sections_raw, page_count, parser_mode = build_sections(
        pdf_path,
        native_char_threshold=native_char_threshold,
        ocr_zoom=ocr_zoom,
    )
    output_sections, filter_metadata = filter_sections(sections_raw, dataset=dataset, filter_mode=filter_mode)

    relative_path = pdf_path.relative_to(input_root)
    json_path = output_root / relative_path.with_suffix(".json")
    ensure_dir(json_path.parent)

    payload = {
        "document_id": extract_case_id(pdf_path),
        "file_name": pdf_path.name,
        "source_path": str(pdf_path),
        "dataset": dataset,
        "page_count": page_count,
        "filter_mode": filter_mode,
        "sections": output_sections,
        "filter_audit": {
            "removed_section_titles": filter_metadata["removed_section_titles"],
            "removed_sentence_count": filter_metadata["removed_sentence_count"],
            "removed_sentence_reasons": filter_metadata["removed_sentence_reasons"],
            "emptied_section_titles": filter_metadata["emptied_section_titles"],
            "masked_sentence_count": filter_metadata["masked_sentence_count"],
            "masked_sentence_reasons": filter_metadata["masked_sentence_reasons"],
        },
    }
    write_json(json_path, payload)

    sentence_count = sum(len(section["sentences"]) for section in output_sections)
    return {
        "document_id": payload["document_id"],
        "file_name": pdf_path.name,
        "source_path": str(pdf_path),
        "output_path": str(json_path),
        "dataset": dataset,
        "page_count": page_count,
        "section_count": len(output_sections),
        "sentence_count": sentence_count,
        "parser_mode": parser_mode,
        "filter_mode": filter_mode,
        "removed_section_titles": filter_metadata["removed_section_titles"],
        "removed_sentence_count": filter_metadata["removed_sentence_count"],
        "removed_sentence_reasons": filter_metadata["removed_sentence_reasons"],
        "emptied_section_titles": filter_metadata["emptied_section_titles"],
        "masked_sentence_count": filter_metadata["masked_sentence_count"],
        "masked_sentence_reasons": filter_metadata["masked_sentence_reasons"],
        "status": "success",
    }


def process_all_pdfs(
    input_dir: Path,
    output_dir: Path,
    native_char_threshold: int,
    ocr_zoom: float,
    limit: int | None,
    filter_mode: str,
) -> dict:
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    ensure_dir(output_root)

    log_path = output_root / "preprocess.log"
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

    pdf_paths = sorted(input_root.rglob("*.pdf"))
    if limit is not None:
        pdf_paths = pdf_paths[:limit]

    summaries: list[dict] = []
    total = len(pdf_paths)
    LOGGER.info("Discovered %s PDF files under %s", total, input_root)

    for index, pdf_path in enumerate(pdf_paths, start=1):
        LOGGER.info("[%s/%s] Processing %s", index, total, pdf_path)
        try:
            summary = preprocess_document(
                pdf_path=pdf_path,
                input_root=input_root,
                output_root=output_root,
                native_char_threshold=native_char_threshold,
                ocr_zoom=ocr_zoom,
                filter_mode=filter_mode,
            )
            LOGGER.info(
                "Finished %s | dataset=%s | sections=%s | sentences=%s | removed_sentences=%s | masked_sentences=%s",
                pdf_path.name,
                summary["dataset"],
                summary["section_count"],
                summary["sentence_count"],
                summary["removed_sentence_count"],
                summary.get("masked_sentence_count", 0),
            )
        except Exception as exc:
            LOGGER.exception("Failed to process %s", pdf_path)
            summary = {
                "document_id": extract_case_id(pdf_path),
                "file_name": pdf_path.name,
                "source_path": str(pdf_path),
                "output_path": None,
                "dataset": detect_dataset(pdf_path),
                "page_count": 0,
                "section_count": 0,
                "sentence_count": 0,
                "parser_mode": None,
                "filter_mode": filter_mode,
                "removed_section_titles": [],
                "removed_sentence_count": 0,
                "removed_sentence_reasons": {},
                "emptied_section_titles": [],
                "masked_sentence_count": 0,
                "masked_sentence_reasons": {},
                "status": "failed",
                "error": str(exc),
            }
        summaries.append(summary)

    successes = [item for item in summaries if item["status"] == "success"]
    failures = [item for item in summaries if item["status"] == "failed"]
    empty_output_pdfs = sum(1 for item in successes if item["sentence_count"] == 0)

    dataset_counter = Counter(item.get("dataset") or "UNKNOWN" for item in summaries)
    aggregate_removed_reasons = Counter()
    aggregate_masked_reasons = Counter()
    for item in successes:
        aggregate_removed_reasons.update(item["removed_sentence_reasons"])
        aggregate_masked_reasons.update(item.get("masked_sentence_reasons", {}))

    run_summary = {
        "input_dir": str(input_root),
        "output_dir": str(output_root),
        "filter_mode": filter_mode,
        "total_pdfs": total,
        "success_count": len(successes),
        "failure_count": len(failures),
        "empty_output_pdfs": empty_output_pdfs,
        "datasets": dict(dataset_counter),
        "removed_sentence_reasons": dict(aggregate_removed_reasons),
        "masked_sentence_reasons": dict(aggregate_masked_reasons),
        "files": summaries,
    }
    write_json(output_root / "run_summary.json", run_summary)
    LOGGER.info(
        "Completed run | total=%s | success=%s | failed=%s | empty_output=%s | filter_mode=%s",
        total,
        len(successes),
        len(failures),
        empty_output_pdfs,
        filter_mode,
    )
    return run_summary


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=None, help="Optional YAML config preset.")
    pre_args, _ = pre_parser.parse_known_args()
    raw_config, config_path = load_yaml_config(pre_args.config)
    config = get_stage_config(raw_config, "preprocess")

    parser = argparse.ArgumentParser(description="Preprocess pathology report PDFs into Document -> Section -> Sentence JSON.")
    parser.add_argument("--config", type=Path, default=pre_args.config, help="Optional YAML config preset.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=get_path(config, "input_dir", DEFAULT_INPUT_DIR, config_path),
        help="Root directory containing PDF files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory for preprocessed JSON outputs.",
    )
    parser.add_argument(
        "--native_char_threshold",
        type=int,
        default=int(get_value(config, "native_char_threshold", 40)),
        help="If a page yields fewer than this many characters natively, OCR is used as fallback.",
    )
    parser.add_argument(
        "--ocr_zoom",
        type=float,
        default=float(get_value(config, "ocr_zoom", 2.0)),
        help="Zoom factor for OCR rasterization.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None if get_value(config, "limit", None) is None else int(get_value(config, "limit", None)),
        help="Optional cap on number of PDFs to process.",
    )
    parser.add_argument(
        "--filter_mode",
        choices=["full", "no_diagnosis", "masked", "no_diagnosis_masked"],
        default=str(get_value(config, "filter_mode", DEFAULT_FILTER_MODE)),
        help="Filtering strategy. masked is the recommended default for stage classification; no_diagnosis matches the paper-style workflow; full keeps all sections; no_diagnosis_masked removes diagnosis-oriented sections and then masks remaining stage/TNM tokens.",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        explicit_output_dir = config.get("output_dir")
        if explicit_output_dir not in (None, ""):
            args.output_dir = get_path(config, "output_dir", DEFAULT_OUTPUT_DIR, config_path)
        else:
            output_root = get_path(config, "output_root", DEFAULT_OUTPUT_ROOT, config_path)
            output_subdirs = dict(PREPROCESS_OUTPUT_SUBDIRS)
            output_subdirs.update(get_value(config, "output_subdirs", {}) or {})
            output_subdir = output_subdirs.get(args.filter_mode, PREPROCESS_OUTPUT_SUBDIRS[DEFAULT_FILTER_MODE])
            args.output_dir = output_root / output_subdir

    return args


def main() -> None:
    args = parse_args()
    summary = process_all_pdfs(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        native_char_threshold=args.native_char_threshold,
        ocr_zoom=args.ocr_zoom,
        limit=args.limit,
        filter_mode=args.filter_mode,
    )
    print(
        f"Processed {summary['total_pdfs']} PDFs | success={summary['success_count']} | "
        f"failed={summary['failure_count']} | empty_output={summary['empty_output_pdfs']} | "
        f"filter_mode={summary['filter_mode']}"
    )


if __name__ == "__main__":
    main()









