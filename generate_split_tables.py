from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


DATASETS = ("BRCA", "KIRC", "LUSC")
NUM_SPLITS = 150
TRAIN_RATIO = 26
TEST_RATIO = 10

WSI_ROOT = Path(r"D:\Tasks\WSI_extract_features")
LABEL_ROOT = Path(r"D:\Tasks\Pathologic_Stage_Label")
OUTPUT_ROOT = Path(r"D:\Tasks\Split_Table")


def discover_slide_ids(dataset: str) -> List[str]:
    dataset_dir = WSI_ROOT / f"{dataset}_WSI_extract_features"
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    slide_ids = sorted({path.stem for path in dataset_dir.rglob("*.pt")})
    if not slide_ids:
        raise RuntimeError(f"No .pt feature files found under: {dataset_dir}")
    return slide_ids


def load_label_map(dataset: str) -> Dict[str, int]:
    label_csv = LABEL_ROOT / f"{dataset}_pathologic_stage.csv"
    if not label_csv.exists():
        raise FileNotFoundError(f"Label file not found: {label_csv}")

    label_map: Dict[str, int] = {}
    with label_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            slide_id = row["slide_id"].strip()
            label = int(row["label"])
            label_map[slide_id] = label
    if not label_map:
        raise RuntimeError(f"No labels loaded from: {label_csv}")
    return label_map


def allocate_test_counts(groups: Dict[int, Sequence[str]], total_test_count: int) -> Dict[int, int]:
    raw_targets: Dict[int, float] = {}
    floors: Dict[int, int] = {}
    remainders: List[Tuple[float, int]] = []

    total_items = sum(len(items) for items in groups.values())
    for label, items in groups.items():
        target = len(items) * total_test_count / total_items
        raw_targets[label] = target
        floors[label] = math.floor(target)
        remainders.append((target - floors[label], label))

    assigned = sum(floors.values())
    remaining = total_test_count - assigned

    for _, label in sorted(remainders, key=lambda x: (-x[0], x[1])):
        if remaining <= 0:
            break
        if floors[label] < len(groups[label]):
            floors[label] += 1
            remaining -= 1

    return floors


def make_stratified_split(slide_ids: Sequence[str], label_map: Dict[str, int], rng: random.Random) -> Tuple[List[str], List[str]]:
    grouped: Dict[int, List[str]] = defaultdict(list)
    for slide_id in slide_ids:
        grouped[label_map[slide_id]].append(slide_id)

    for items in grouped.values():
        rng.shuffle(items)

    total_count = len(slide_ids)
    test_count = round(total_count * TEST_RATIO / (TRAIN_RATIO + TEST_RATIO))
    test_count = max(1, min(test_count, total_count - 1))

    per_label_test = allocate_test_counts(grouped, test_count)

    test_ids: List[str] = []
    train_ids: List[str] = []
    for label in sorted(grouped):
        items = grouped[label]
        split_at = per_label_test[label]
        test_ids.extend(items[:split_at])
        train_ids.extend(items[split_at:])

    rng.shuffle(train_ids)
    rng.shuffle(test_ids)
    return train_ids, test_ids


def write_split_csv(output_csv: Path, train_ids: Sequence[str], test_ids: Sequence[str]) -> None:
    row_count = max(len(train_ids), len(test_ids))
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "train", "test"])
        for idx in range(row_count):
            train_value = train_ids[idx] if idx < len(train_ids) else ""
            test_value = test_ids[idx] if idx < len(test_ids) else ""
            writer.writerow([idx, train_value, test_value])


def generate_dataset_splits(dataset: str) -> Dict[str, object]:
    all_slide_ids = discover_slide_ids(dataset)
    label_map = load_label_map(dataset)

    usable_slide_ids = [slide_id for slide_id in all_slide_ids if slide_id in label_map]
    if len(usable_slide_ids) < 2:
        raise RuntimeError(f"Not enough labeled slides for dataset {dataset}")

    output_dir = OUTPUT_ROOT / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    train_counts: List[int] = []
    test_counts: List[int] = []

    for split_idx in range(NUM_SPLITS):
        rng = random.Random(20260404 + split_idx)
        train_ids, test_ids = make_stratified_split(usable_slide_ids, label_map, rng)
        write_split_csv(output_dir / f"split_{split_idx}.csv", train_ids, test_ids)
        train_counts.append(len(train_ids))
        test_counts.append(len(test_ids))

    label_counter = Counter(label_map[slide_id] for slide_id in usable_slide_ids)
    summary = {
        "dataset": dataset,
        "feature_slide_count": len(all_slide_ids),
        "labeled_slide_count": len(usable_slide_ids),
        "unlabeled_slide_count": len(all_slide_ids) - len(usable_slide_ids),
        "label_distribution": dict(sorted(label_counter.items())),
        "num_splits": NUM_SPLITS,
        "train_count_per_split": sorted(set(train_counts)),
        "test_count_per_split": sorted(set(test_counts)),
        "output_dir": str(output_dir),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for dataset in DATASETS:
        summaries.append(generate_dataset_splits(dataset))

    with (OUTPUT_ROOT / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    print("Generated split tables:")
    for item in summaries:
        print(
            f"- {item['dataset']}: labeled={item['labeled_slide_count']}, "
            f"train={item['train_count_per_split']}, test={item['test_count_per_split']}"
        )


if __name__ == "__main__":
    main()
