from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_DATASETS = ("BRCA", "KIRC", "LUSC")
DEFAULT_WSI_ROOT = Path(r"F:\Tasks\WSI_extract_features")
DEFAULT_LABEL_ROOT = Path(r"F:\Tasks\Pathologic_Stage_Label")
DEFAULT_OUTPUT_ROOT = Path(r"F:\Tasks\Split_Table")


def discover_slide_ids(pt_dir: Path) -> list[str]:
    if not pt_dir.exists():
        raise FileNotFoundError(f"Feature directory not found: {pt_dir}")
    slide_ids = sorted({path.stem for path in pt_dir.rglob("*.pt") if path.is_file()})
    if not slide_ids:
        raise RuntimeError(f"No .pt feature files found under: {pt_dir}")
    return slide_ids


def load_label_map(label_csv: Path | None) -> dict[str, int]:
    if label_csv is None:
        return {}
    if not label_csv.exists():
        raise FileNotFoundError(f"Label file not found: {label_csv}")

    label_map: dict[str, int] = {}
    with label_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            slide_id = str(row.get("slide_id", "")).strip()
            label_raw = str(row.get("label", "")).strip()
            if slide_id and label_raw:
                label_map[slide_id] = int(label_raw)
    if not label_map:
        raise RuntimeError(f"No slide labels loaded from: {label_csv}")
    return label_map


def split_counts(total_count: int, train_ratio: int, test_ratio: int) -> tuple[int, int]:
    if total_count < 2:
        raise ValueError("At least two slides are required to make a train/test split.")
    if train_ratio <= 0 or test_ratio <= 0:
        raise ValueError("train_ratio and test_ratio must be positive.")
    train_count = int(round(total_count * train_ratio / (train_ratio + test_ratio)))
    train_count = max(1, min(train_count, total_count - 1))
    return train_count, total_count - train_count


def allocate_group_test_counts(groups: dict[int, list[str]], total_test_count: int) -> dict[int, int]:
    total_items = sum(len(items) for items in groups.values())
    floors: dict[int, int] = {}
    remainders: list[tuple[float, int]] = []
    for label, items in groups.items():
        target = len(items) * total_test_count / total_items
        floors[label] = math.floor(target)
        remainders.append((target - floors[label], label))

    remaining = total_test_count - sum(floors.values())
    for _, label in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if remaining <= 0:
            break
        if floors[label] < len(groups[label]):
            floors[label] += 1
            remaining -= 1
    return floors


def make_random_split(slide_ids: Sequence[str], rng: random.Random, train_ratio: int, test_ratio: int) -> tuple[list[str], list[str]]:
    train_count, _ = split_counts(len(slide_ids), train_ratio, test_ratio)
    shuffled = list(slide_ids)
    rng.shuffle(shuffled)
    return shuffled[:train_count], shuffled[train_count:]


def make_stratified_split(slide_ids: Sequence[str], label_map: dict[str, int], rng: random.Random, train_ratio: int, test_ratio: int) -> tuple[list[str], list[str]]:
    _, test_count = split_counts(len(slide_ids), train_ratio, test_ratio)
    groups: dict[int, list[str]] = defaultdict(list)
    for slide_id in slide_ids:
        groups[label_map[slide_id]].append(slide_id)
    for items in groups.values():
        rng.shuffle(items)

    per_label_test = allocate_group_test_counts(groups, test_count)
    train_ids: list[str] = []
    test_ids: list[str] = []
    for label in sorted(groups):
        items = groups[label]
        cutoff = per_label_test[label]
        test_ids.extend(items[:cutoff])
        train_ids.extend(items[cutoff:])
    rng.shuffle(train_ids)
    rng.shuffle(test_ids)
    return train_ids, test_ids


def write_split_csv(output_csv: Path, train_ids: Sequence[str], test_ids: Sequence[str]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    row_count = max(len(train_ids), len(test_ids))
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["idx", "train", "test"])
        for idx in range(row_count):
            train_value = train_ids[idx] if idx < len(train_ids) else ""
            test_value = test_ids[idx] if idx < len(test_ids) else ""
            writer.writerow([idx, train_value, test_value])


def split_signature(train_ids: Iterable[str]) -> str:
    return ",".join(sorted(train_ids))


def generate_splits(
    *,
    slide_ids: Sequence[str],
    label_map: dict[str, int],
    output_dir: Path,
    n_splits: int,
    seed: int,
    train_ratio: int,
    test_ratio: int,
) -> dict[str, int | str]:
    usable_slide_ids = list(slide_ids)
    if label_map:
        usable_slide_ids = [slide_id for slide_id in usable_slide_ids if slide_id in label_map]
    if len(usable_slide_ids) < 2:
        raise RuntimeError("Not enough usable slides to generate splits.")

    rng = random.Random(seed)
    seen: set[str] = set()
    created = 0
    attempts = 0
    max_attempts = max(10_000, n_splits * 100)

    while created < n_splits:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Only created {created}/{n_splits} unique splits after {attempts} attempts."
            )
        if label_map:
            train_ids, test_ids = make_stratified_split(usable_slide_ids, label_map, rng, train_ratio, test_ratio)
        else:
            train_ids, test_ids = make_random_split(usable_slide_ids, rng, train_ratio, test_ratio)
        signature = split_signature(train_ids)
        if signature in seen:
            continue
        seen.add(signature)
        write_split_csv(output_dir / f"split_{created}.csv", train_ids, test_ids)
        created += 1

    return {
        "slides": len(slide_ids),
        "usable_slides": len(usable_slide_ids),
        "splits": created,
        "output_dir": str(output_dir),
        "stratified": int(bool(label_map)),
    }


def dataset_paths(dataset: str, wsi_root: Path, label_root: Path, output_root: Path) -> tuple[Path, Path, Path]:
    dataset = dataset.upper()
    return (
        wsi_root / f"{dataset}_WSI_extract_features",
        label_root / f"{dataset}_pathologic_stage.csv",
        output_root / dataset,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate train/test split CSVs from WSI feature files, optionally stratified by labels."
    )
    parser.add_argument("--datasets", nargs="*", choices=DEFAULT_DATASETS, help="Dataset names to generate from standard roots.")
    parser.add_argument("--wsi_root", type=Path, default=DEFAULT_WSI_ROOT)
    parser.add_argument("--label_root", type=Path, default=DEFAULT_LABEL_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--pt_dir", type=Path, default=None, help="Single feature directory mode.")
    parser.add_argument("--label_csv", type=Path, default=None, help="Optional label CSV for single directory mode.")
    parser.add_argument("--out_dir", type=Path, default=None, help="Output directory for single directory mode.")
    parser.add_argument("--n_splits", type=int, default=150)
    parser.add_argument("--train_ratio", type=int, default=26)
    parser.add_argument("--test_ratio", type=int, default=10)
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    jobs: list[tuple[str, Path, Path | None, Path]] = []

    if args.pt_dir is not None:
        out_dir = args.out_dir or (args.pt_dir.parent / "Splits")
        jobs.append(("single", args.pt_dir, args.label_csv, out_dir))

    for dataset in args.datasets or ():
        pt_dir, label_csv, out_dir = dataset_paths(dataset, args.wsi_root, args.label_root, args.output_root)
        jobs.append((dataset, pt_dir, label_csv, out_dir))

    if not jobs:
        for dataset in DEFAULT_DATASETS:
            pt_dir, label_csv, out_dir = dataset_paths(dataset, args.wsi_root, args.label_root, args.output_root)
            jobs.append((dataset, pt_dir, label_csv, out_dir))

    for name, pt_dir, label_csv, out_dir in jobs:
        slide_ids = discover_slide_ids(pt_dir)
        labels = load_label_map(label_csv) if label_csv is not None and label_csv.exists() else {}
        summary = generate_splits(
            slide_ids=slide_ids,
            label_map=labels,
            output_dir=out_dir,
            n_splits=args.n_splits,
            seed=args.seed,
            train_ratio=args.train_ratio,
            test_ratio=args.test_ratio,
        )
        print(
            f"[{name}] generated {summary['splits']} splits | "
            f"slides={summary['slides']} usable={summary['usable_slides']} "
            f"stratified={bool(summary['stratified'])} | output={summary['output_dir']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
