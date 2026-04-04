from __future__ import annotations

"""Expand case-level labels to slide-level labels using a split CSV.

Input case label CSV format:
case_id,slide_id,label

Only `case_id` and `label` are used. The output CSV matches split slide_ids:
case_id,slide_id,label
"""

import argparse
import csv
import re
from pathlib import Path

from pdf_utils import ensure_dir


CASE_ID_RE = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", re.IGNORECASE)


def extract_case_id(text: str) -> str:
    value = str(text or "").strip()
    match = CASE_ID_RE.search(value)
    if match:
        return match.group(1).upper()
    return value[:12].upper()


def load_case_labels(case_label_csv: Path) -> dict[str, int]:
    with case_label_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    case_to_label: dict[str, int] = {}
    for row in rows:
        case_id = extract_case_id(row.get("case_id", ""))
        label_raw = row.get("label")
        if not case_id or label_raw in (None, ""):
            continue
        case_to_label[case_id] = int(label_raw)
    return case_to_label


def collect_split_slide_ids(split_csv: Path) -> list[str]:
    with split_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    slide_ids: list[str] = []
    for row in rows:
        for column in ("train", "val", "test"):
            slide_id = str(row.get(column, "") or "").strip()
            if slide_id:
                slide_ids.append(slide_id)
    return slide_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand case-level labels to split-matched slide-level labels.")
    parser.add_argument("--case_label_csv", type=Path, required=True, help="Case-level label CSV.")
    parser.add_argument("--split_csv", type=Path, required=True, help="Split CSV containing WSI slide_ids.")
    parser.add_argument("--output_csv", type=Path, required=True, help="Output slide-level label CSV.")
    args = parser.parse_args()

    case_to_label = load_case_labels(args.case_label_csv)
    slide_ids = collect_split_slide_ids(args.split_csv)

    rows: list[dict[str, str | int]] = []
    unmatched = 0
    seen = set()
    for slide_id in slide_ids:
        if slide_id in seen:
            continue
        seen.add(slide_id)
        case_id = extract_case_id(slide_id)
        label = case_to_label.get(case_id)
        if label is None:
            unmatched += 1
            continue
        rows.append({"case_id": case_id, "slide_id": slide_id, "label": label})

    ensure_dir(args.output_csv.parent)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "slide_id", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Output CSV written to: {args.output_csv}")
    print(f"Matched slide rows: {len(rows)}")
    print(f"Unmatched slide rows: {unmatched}")


if __name__ == "__main__":
    main()
