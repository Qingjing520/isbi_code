from __future__ import annotations

"""从 TCGA clinical XML 提取二分类 pathologic stage 标签。

输出 CSV 形式与 `LUSC_pathologic_stage.csv` 保持一致：
case_id,slide_id,label

这里的 `slide_id` 会直接使用病理报告 PDF 的 basename，
这样可以和当前 pathology report / text graph 流程直接对齐。
"""

import argparse
import csv
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from pdf_utils import ensure_dir, extract_case_id, iter_pdf_files


DEFAULT_CLINICAL_ROOT = Path(r"D:\Tasks\Pathologic_Stage_Label\BRCA")
DEFAULT_REPORT_DIR = Path(r"D:\Tasks\Pathology Report\BRCA")
DEFAULT_OUTPUT_CSV = Path(r"D:\Tasks\Pathologic_Stage_Label\BRCA_pathologic_stage.csv")

STAGE_TOKEN_RE = re.compile(r"\bstage\s+([ivx0-9]+[a-d]?)\b", re.IGNORECASE)


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
    if token.startswith("I") and not token.startswith("IV") and not token.startswith("III"):
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


def build_rows(report_dir: Path, case_to_label: dict[str, int]) -> tuple[list[dict[str, int | str]], int]:
    rows: list[dict[str, int | str]] = []
    skipped_reports = 0

    for pdf_path in iter_pdf_files(report_dir):
        case_id = extract_case_id(pdf_path)
        label = case_to_label.get(case_id)
        if label is None:
            skipped_reports += 1
            continue

        rows.append(
            {
                "case_id": case_id,
                "slide_id": pdf_path.stem,
                "label": label,
            }
        )

    return rows, skipped_reports


def write_csv(rows: list[dict[str, int | str]], output_csv: Path) -> None:
    ensure_dir(output_csv.parent)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "slide_id", "label"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="提取 TCGA pathologic stage 标签，并生成 case_id,slide_id,label 三列表。",
    )
    parser.add_argument(
        "--clinical_root",
        type=Path,
        default=DEFAULT_CLINICAL_ROOT,
        help="clinical XML 根目录，例如 D:\\Tasks\\Pathologic_Stage_Label\\BRCA",
    )
    parser.add_argument(
        "--report_dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="病理报告 PDF 目录，例如 D:\\Tasks\\Pathology Report\\BRCA",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="输出 CSV 路径。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clinical_root = Path(args.clinical_root)
    report_dir = Path(args.report_dir)
    output_csv = Path(args.output_csv)

    if not clinical_root.exists():
        raise FileNotFoundError(f"Clinical root not found: {clinical_root}")
    if not report_dir.exists():
        raise FileNotFoundError(f"Report directory not found: {report_dir}")

    case_to_label, stage_counter, failed_xmls = load_case_label_map(clinical_root)
    rows, skipped_reports = build_rows(report_dir, case_to_label)
    write_csv(rows, output_csv)

    label_counter = Counter(str(row["label"]) for row in rows)
    print(f"Output CSV written to: {output_csv}")
    print(f"Clinical XMLs with usable labels: {len(case_to_label)}")
    print(f"Failed XML files: {failed_xmls}")
    print(f"Matched report rows: {len(rows)}")
    print(f"Skipped report PDFs: {skipped_reports}")
    print(f"Label distribution: {dict(sorted(label_counter.items()))}")
    print(f"Top stage values: {dict(stage_counter.most_common(10))}")


if __name__ == "__main__":
    main()
