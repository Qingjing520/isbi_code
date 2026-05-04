from __future__ import annotations

"""Prepare pathologic-stage labels for report/slide training.

Subcommands:
- from_clinical: extract binary stage labels from TCGA clinical XML files and
  align them to pathology report PDF names.
- expand_to_split: expand case-level labels to split slide IDs.
"""

import argparse
import csv
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from pathology_report_extraction.common.pdf_utils import ensure_dir, extract_case_id as extract_case_id_from_path, iter_pdf_files
from pathology_report_extraction.common.pipeline_defaults import TASKS_ROOT


DEFAULT_CLINICAL_ROOT = TASKS_ROOT / "Pathologic_Stage_Label" / "BRCA"
DEFAULT_REPORT_DIR = TASKS_ROOT / "Pathology Report" / "BRCA"
DEFAULT_OUTPUT_CSV = TASKS_ROOT / "Pathologic_Stage_Label" / "BRCA_pathologic_stage.csv"

CASE_ID_RE = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", re.IGNORECASE)
STAGE_TOKEN_RE = re.compile(r"\bstage\s+([ivx0-9]+[a-d]?)\b", re.IGNORECASE)


def extract_case_id(text: str) -> str:
    value = str(text or "").strip()
    match = CASE_ID_RE.search(value)
    if match:
        return match.group(1).upper()
    return value[:12].upper()


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def get_first_text(root: ET.Element, target_local_name: str) -> str:
    for elem in root.iter():
        if local_name(elem.tag) == target_local_name:
            value = (elem.text or "").strip()
            if value:
                return value
    return ""


def normalize_stage(value: str) -> str:
    text = " ".join((value or "").strip().split())
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.replace("STAGE", "Stage").replace("stage", "Stage"))


def map_stage_to_binary_label(stage_value: str) -> int | None:
    stage = normalize_stage(stage_value)
    if not stage:
        return None

    match = STAGE_TOKEN_RE.search(stage)
    if not match:
        return None

    token = match.group(1).upper()
    if token.startswith("I") and not token.startswith("III") and not token.startswith("IV"):
        return 0
    if token.startswith("II") and not token.startswith("III"):
        return 0
    if token.startswith("III") or token.startswith("IV"):
        return 1
    return None


def load_case_label_map(clinical_root: Path) -> tuple[dict[str, int], Counter[str], int]:
    xml_paths = sorted(clinical_root.rglob("nationwidechildrens.org_clinical*.xml"))
    case_to_label: dict[str, int] = {}
    stage_counter: Counter[str] = Counter()
    failed_xmls = 0

    for xml_path in xml_paths:
        try:
            root = ET.parse(xml_path).getroot()
            case_id = get_first_text(root, "bcr_patient_barcode")
            if not case_id:
                case_id = xml_path.stem.split(".")[-1].upper()

            stage = normalize_stage(get_first_text(root, "pathologic_stage"))
            stage_counter[stage or "[MISSING]"] += 1
            label = map_stage_to_binary_label(stage)
            if label is not None:
                case_to_label[case_id.upper()] = label
        except Exception:
            failed_xmls += 1

    return case_to_label, stage_counter, failed_xmls


def build_report_label_rows(report_dir: Path, case_to_label: dict[str, int]) -> tuple[list[dict[str, int | str]], int]:
    rows: list[dict[str, int | str]] = []
    skipped_reports = 0

    for pdf_path in iter_pdf_files(report_dir):
        case_id = extract_case_id_from_path(pdf_path)
        label = case_to_label.get(case_id)
        if label is None:
            skipped_reports += 1
            continue

        rows.append({"case_id": case_id, "slide_id": pdf_path.stem, "label": label})

    return rows, skipped_reports


def write_label_csv(rows: list[dict[str, int | str]], output_csv: Path) -> None:
    ensure_dir(output_csv.parent)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "slide_id", "label"])
        writer.writeheader()
        writer.writerows(rows)


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


def run_from_clinical(args: argparse.Namespace) -> None:
    clinical_root = Path(args.clinical_root)
    report_dir = Path(args.report_dir)
    output_csv = Path(args.output_csv)

    if not clinical_root.exists():
        raise FileNotFoundError(f"Clinical root not found: {clinical_root}")
    if not report_dir.exists():
        raise FileNotFoundError(f"Report directory not found: {report_dir}")

    case_to_label, stage_counter, failed_xmls = load_case_label_map(clinical_root)
    rows, skipped_reports = build_report_label_rows(report_dir, case_to_label)
    write_label_csv(rows, output_csv)

    label_counter = Counter(str(row["label"]) for row in rows)
    print(f"Output CSV written to: {output_csv}")
    print(f"Clinical XMLs with usable labels: {len(case_to_label)}")
    print(f"Failed XML files: {failed_xmls}")
    print(f"Matched report rows: {len(rows)}")
    print(f"Skipped report PDFs: {skipped_reports}")
    print(f"Label distribution: {dict(sorted(label_counter.items()))}")
    print(f"Top stage values: {dict(stage_counter.most_common(10))}")


def run_expand_to_split(args: argparse.Namespace) -> None:
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

    write_label_csv(rows, args.output_csv)
    print(f"Output CSV written to: {args.output_csv}")
    print(f"Matched slide rows: {len(rows)}")
    print(f"Unmatched slide rows: {unmatched}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare pathologic-stage label CSV files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    from_clinical = subparsers.add_parser("from_clinical", help="Extract labels from TCGA clinical XML files.")
    from_clinical.add_argument("--clinical_root", type=Path, default=DEFAULT_CLINICAL_ROOT)
    from_clinical.add_argument("--report_dir", type=Path, default=DEFAULT_REPORT_DIR)
    from_clinical.add_argument("--output_csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    from_clinical.set_defaults(func=run_from_clinical)

    expand = subparsers.add_parser("expand_to_split", help="Expand case labels to split slide IDs.")
    expand.add_argument("--case_label_csv", type=Path, required=True, help="Case-level label CSV.")
    expand.add_argument("--split_csv", type=Path, required=True, help="Split CSV containing slide IDs.")
    expand.add_argument("--output_csv", type=Path, required=True, help="Output slide-level label CSV.")
    expand.set_defaults(func=run_expand_to_split)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
