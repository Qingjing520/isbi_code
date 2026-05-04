from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import yaml


def find_max_trained_split(exp_dir: Path) -> int:
    """
    在 experiments 目录下查找已训练好的最大 split 编号。
    支持目录名形如: split_0, split_1, split0, split1 等。
    若不存在任何 split 目录，则返回 -1。
    """
    if not exp_dir.exists():
        return -1

    max_idx = -1
    pattern = re.compile(r"^split_?(\d+)$")

    for p in exp_dir.iterdir():
        if not p.is_dir():
            continue
        m = pattern.match(p.name)
        if not m:
            continue
        idx = int(m.group(1))
        if idx > max_idx:
            max_idx = idx
    return max_idx


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def train_next_split(project_root: Path, *, dry: bool = False) -> bool:
    """
    训练“下一组”划分。
    返回值:
      - True: 成功启动了一次训练
      - False: 没有可训练的下一组（已经全部训练完）
    """
    config_path = project_root / "configs" / "config.yaml"
    experiments_dir = project_root / "experiments"

    # 1. 找到当前 experiments 目录下已经训练到的最大 split_x
    max_trained_idx = find_max_trained_split(experiments_dir)
    next_idx = max_trained_idx + 1

    print(f"[info] Detected max trained split index: {max_trained_idx}")
    print(f"[info] Next split index to train on: {next_idx}")

    # 2. 读取并更新 config.yaml
    cfg = load_yaml(config_path)

    if "data" not in cfg:
        raise KeyError("config.yaml 缺少顶层键 'data'")

    # 当前 split_file 用来推断 Splits 目录
    split_file_str = cfg["data"].get("split_file", None)
    if not split_file_str:
        raise KeyError("config.yaml 的 data.split_file 为空，请先手动设置一份有效的 split 文件路径。")

    split_file_path = Path(split_file_str)
    splits_dir = split_file_path.parent

    # 在 Splits 目录中查找现有 split_x.csv，防止超出范围
    if not splits_dir.exists():
        raise FileNotFoundError(f"Splits 目录不存在: {splits_dir}")

    available_indices = []
    pattern_csv = re.compile(r"^split_(\d+)\.csv$")
    for p in splits_dir.iterdir():
        if not p.is_file():
            continue
        m = pattern_csv.match(p.name)
        if not m:
            continue
        available_indices.append(int(m.group(1)))

    if not available_indices:
        raise RuntimeError(f"在 {splits_dir} 中未找到任何 split_*.csv 文件。")

    max_split_idx = max(available_indices)
    if next_idx > max_split_idx:
        print(
            f"[info] All splits have been trained. "
            f"最大可用 split index = {max_split_idx}, 下一个 = {next_idx}，不再继续训练。"
        )
        return False

    new_split_file = splits_dir / f"split_{next_idx}.csv"

    # 更新 data.split_file
    cfg["data"]["split_file"] = str(new_split_file)

    # 更新 output.exp_dir -> experiments/split_{next_idx}
    if "output" not in cfg:
        cfg["output"] = {}
    cfg["output"]["exp_dir"] = str(experiments_dir / f"split_{next_idx}")

    save_yaml(config_path, cfg)

    print(f"[info] Updated config.yaml:")
    print(f"       data.split_file -> {new_split_file}")
    print(f"       output.exp_dir  -> {cfg['output']['exp_dir']}")

    if dry:
        print("[info] Dry run enabled, skip launching training.")
        return False

    # 3. 调用 main.py 开始训练
    cmd = ["python", "main.py", "--config", "configs/config.yaml", "--mode", "train"]
    print(f"[info] Running training command: {' '.join(cmd)}")

    subprocess.run(cmd, cwd=str(project_root), check=True)
    return True


def main():
    parser = argparse.ArgumentParser(
        description=(
            "自动根据已完成的 experiments/split_x，选择下一组 split_k.csv 训练。\n"
            "可以通过 --num_runs 一次性连续训练多组。"
        )
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="本次脚本最多连续训练多少组划分（默认 1 组）。",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent

    runs = max(1, int(args.num_runs))
    for i in range(runs):
        print(f"\n[info] ===== Run {i + 1}/{runs} =====")
        ok = train_next_split(project_root)
        if not ok:
            print("[info] 没有更多可训练的划分，本次脚本提前结束。")
            break


if __name__ == "__main__":
    main()

