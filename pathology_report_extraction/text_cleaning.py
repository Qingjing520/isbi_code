from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Sequence


DEFAULT_NOISE_PATTERNS = [
    r'^page\s+\d+\s*(?:/|of)\s*\d+$',
    r'^Page\s+\d+\s*(?:/|of)\s*\d+$',
    r'^END OF REPORT$',
    r'^CONTACT YOUR DOCTOR WITH THIS REPORT!?$',
    r'^UUID[:\s-].*$',
    r'^TCGA-[A-Z0-9-]+.*REDACTED$',
    r'^Criteria$',
    r'^Diagnosis Discrepancy.*$',
    r'^Primary Tumor Site Discrepancy.*$',
    r'^HI.?PAA Discrepancy.*$',
    r'^Prior Malignancy History.*$',
    r'^Dual/Synchronous Primary.*$',
    r'^Case is.*$',
    r'^Review.*$',
]

ITEM_START_RE = re.compile(r'^(?:PART\s+\d+|[A-Z]\.|\d+[\).:])', re.IGNORECASE)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize('NFKC', text or '')
    text = text.replace('\xa0', ' ')
    text = text.replace('\u200b', '')
    text = text.replace('\r', '\n')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def normalize_line(line: str) -> str:
    line = normalize_text(line)
    line = re.sub(r'\s+', ' ', line)
    return line.strip()


def is_barcode_like(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 16:
        return False
    return bool(re.fullmatch(r'[I1l| ]+', stripped))


def looks_like_heading(line: str, heading_patterns: Sequence[re.Pattern[str]] | None = None) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if heading_patterns and any(pattern.match(stripped) for pattern in heading_patterns):
        return True
    if len(stripped) <= 70 and stripped == stripped.upper() and re.search(r'[A-Z]', stripped):
        return True
    return False


def drop_noise_lines(lines: Iterable[str], extra_noise_patterns: Sequence[str] | None = None) -> list[str]:
    patterns = [re.compile(pattern, re.IGNORECASE) for pattern in DEFAULT_NOISE_PATTERNS]
    patterns.extend(re.compile(pattern, re.IGNORECASE) for pattern in (extra_noise_patterns or []))

    cleaned: list[str] = []
    for raw_line in lines:
        line = normalize_line(raw_line)
        if not line:
            cleaned.append('')
            continue
        if is_barcode_like(line):
            continue
        if any(pattern.match(line) for pattern in patterns):
            continue
        if len(line) > 40 and sum(ch in 'I1l|' for ch in line) / len(line) > 0.65:
            continue
        cleaned.append(line)
    return cleaned


def should_join_lines(prev_line: str, current_line: str, heading_patterns: Sequence[re.Pattern[str]] | None = None) -> bool:
    if not prev_line or not current_line:
        return False
    if looks_like_heading(current_line, heading_patterns):
        return False
    if ITEM_START_RE.match(current_line):
        return False
    if prev_line.endswith(':'):
        return False
    if prev_line.endswith(('-', '/')):
        return True
    if re.search(r'[.!?;)]$', prev_line):
        return False
    if current_line[0].islower() or re.match(r'^[\d(<]', current_line):
        return True
    if len(prev_line) > 60 and current_line[:1].islower():
        return True
    return False


def merge_wrapped_lines(lines: Sequence[str], heading_patterns: Sequence[re.Pattern[str]] | None = None) -> list[str]:
    merged: list[str] = []
    buffer = ''
    for line in lines:
        if not line:
            if buffer:
                merged.append(buffer)
                buffer = ''
            merged.append('')
            continue
        if not buffer:
            buffer = line
            continue
        if should_join_lines(buffer, line, heading_patterns):
            joiner = '' if buffer.endswith('-') else ' '
            buffer = f"{buffer.rstrip('-')}{joiner}{line}"
        else:
            merged.append(buffer)
            buffer = line
    if buffer:
        merged.append(buffer)
    return merged


def clean_lines(
    lines: Iterable[str],
    extra_noise_patterns: Sequence[str] | None = None,
    heading_patterns: Sequence[re.Pattern[str]] | None = None,
) -> list[str]:
    dropped = drop_noise_lines(lines, extra_noise_patterns=extra_noise_patterns)
    merged = merge_wrapped_lines(dropped, heading_patterns=heading_patterns)
    compacted: list[str] = []
    blank_streak = 0
    for line in merged:
        if line:
            compacted.append(line)
            blank_streak = 0
        else:
            blank_streak += 1
            if blank_streak <= 1:
                compacted.append('')
    while compacted and compacted[0] == '':
        compacted.pop(0)
    while compacted and compacted[-1] == '':
        compacted.pop()
    return compacted


def clean_text(
    text: str,
    extra_noise_patterns: Sequence[str] | None = None,
    heading_patterns: Sequence[re.Pattern[str]] | None = None,
) -> str:
    lines = clean_lines(text.splitlines(), extra_noise_patterns=extra_noise_patterns, heading_patterns=heading_patterns)
    return '\n'.join(lines).strip()


ABBREVIATIONS = ['Dr.', 'No.', 'Fig.', 'vs.', 'cm.', 'mm.']


def split_sentences(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    protected = text
    for token in ABBREVIATIONS:
        protected = protected.replace(token, token.replace('.', '<prd>'))
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z(\[])', protected)
    sentences = [part.replace('<prd>', '.').strip() for part in parts if part.strip()]
    if not sentences:
        return [text]
    return sentences


def sanitize_key(label: str) -> str:
    label = normalize_line(label).lower()
    label = re.sub(r'[^a-z0-9]+', '_', label)
    return label.strip('_')
