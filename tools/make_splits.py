from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


def list_slide_ids(pt_dir: Path) -> list[str]:
    if not pt_dir.exists():
        raise FileNotFoundError(f"pt_dir not found: {pt_dir}")
    if not pt_dir.is_dir():
        raise NotADirectoryError(f"pt_dir is not a directory: {pt_dir}")

    slide_ids: list[str] = []
    for p in pt_dir.iterdir():
        if p.is_file() and p.suffix.lower() == ".pt":
            slide_ids.append(p.stem)
    slide_ids.sort()  # 排序只是为了结果可复现、便于检查
    return slide_ids


def write_split_csv(out_path: Path, train_ids: list[str], test_ids: list[str]) -> None:
    # Match current code expectation: columns named "train" and "test"
    n_rows = max(len(train_ids), len(test_ids))
    train_col = train_ids + [None] * (n_rows - len(train_ids))
    test_col = test_ids + [None] * (n_rows - len(test_ids))
    df = pd.DataFrame({"train": train_col, "test": test_col})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def split_signature(train_ids: list[str]) -> str:
    # 用排好序的 train 集 id 作为签名（test 集由补集唯一确定）
    return ",".join(sorted(train_ids))


def main():
    ap = argparse.ArgumentParser(
        description=(
            "根据给定目录下 .pt 文件总数，按 train:test = 26:10（可约分）比例，"
            "随机划分为若干组 train/test split，且每组划分都覆盖目录中所有文件，"
            "并保证不同组之间划分不完全相同。"
        )
    )
    ap.add_argument(
        "--pt_dir",
        type=str,
        default=r"D:\CLAM_output\features\pt_files",
        help="Directory containing .pt files. Each filename stem is treated as a slide_id.",
    )
    ap.add_argument(
        "--train_ratio_numer",
        type=int,
        default=26,
        help="Train 集比例分子（例如 26，对应 26:10）。",
    )
    ap.add_argument(
        "--test_ratio_numer",
        type=int,
        default=10,
        help="Test 集比例分子（例如 10，对应 26:10）。",
    )
    ap.add_argument(
        "--n_splits",
        type=int,
        default=150,
        help="要生成多少组不同的划分（默认 150 组）。",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for split CSV files. "
        "If not set, will use '<pt_dir>/../Splits'.",
    )
    ap.add_argument("--seed", type=int, default=23, help="Random seed for reproducibility.")
    args = ap.parse_args()

    pt_dir = Path(args.pt_dir)
    if args.out_dir is None:
        # 默认：特征目录的上一级目录下的 Splits 文件夹
        out_dir = pt_dir.parent / "Splits"
    else:
        out_dir = Path(args.out_dir)

    slide_ids = list_slide_ids(pt_dir)
    n_total = len(slide_ids)
    if n_total == 0:
        raise ValueError(f"No .pt files found in {pt_dir}")

    # 根据 26:10（可约分 13:5）这样的比例，计算 train/test 数量
    r_train = float(args.train_ratio_numer)
    r_test = float(args.test_ratio_numer)
    if r_train <= 0 or r_test <= 0:
        raise ValueError("train_ratio_numer 和 test_ratio_numer 必须为正数。")

    train_frac = r_train / (r_train + r_test)
    # 四舍五入到最近的整数，确保二者加起来等于总数
    train_n = int(round(n_total * train_frac))
    # 保证两边都至少 1 个（在样本数足够的情况下）
    if n_total >= 2:
        train_n = max(1, min(train_n, n_total - 1))
    test_n = n_total - train_n

    rng = random.Random(args.seed)

    seen_sigs: set[str] = set()
    created = 0
    attempts = 0
    max_attempts = max(10_000, args.n_splits * 50)

    while created < args.n_splits:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Tried {attempts} times but only created {created} unique splits. "
                f"Try changing the seed or check that there are enough files."
            )

        ids = slide_ids.copy()
        rng.shuffle(ids)
        train_ids = ids[:train_n]
        test_ids = ids[train_n:]

        sig = split_signature(train_ids)
        if sig in seen_sigs:
            continue
        seen_sigs.add(sig)

        out_path = out_dir / f"split_{created}.csv"
        write_split_csv(out_path, train_ids, test_ids)
        created += 1

    print(
        f"Done. Found {n_total} files in {pt_dir}.\n"
        f"Train/Test counts per split: {train_n}/{test_n} "
        f"(target ratio {r_train}:{r_test}, actual ~{train_frac:.4f}/{1-train_frac:.4f}).\n"
        f"Generated {created} unique splits in: {out_dir}"
    )


if __name__ == "__main__":
    main()

