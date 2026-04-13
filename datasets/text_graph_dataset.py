from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


VALID_SPLITS = {"train", "val", "test"}


def _read_manifest_rows(manifest_csv: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(manifest_csv)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    normalized: list[dict[str, Any]] = []
    for row in rows:
        graph_pt = (row.get("graph_pt") or "").strip()
        label_raw = row.get("label")
        if not graph_pt or label_raw in (None, ""):
            continue
        normalized.append(
            {
                "case_id": (row.get("case_id") or "").strip(),
                "slide_id": (row.get("slide_id") or "").strip(),
                "dataset": (row.get("dataset") or "").strip(),
                "split": (row.get("split") or "").strip(),
                "label": int(label_raw),
                "graph_pt": graph_pt,
                "graph_json": (row.get("graph_json") or "").strip(),
                "report_id": (row.get("report_id") or "").strip(),
                "file_name": (row.get("file_name") or "").strip(),
                "filter_mode": (row.get("filter_mode") or "").strip(),
                "source_concept_json": (row.get("source_concept_json") or "").strip(),
                "concept_count": int(row.get("concept_count") or 0),
                "has_concept_level": (row.get("has_concept_level") or "").strip().lower() in {"1", "true", "yes"},
            }
        )
    return normalized


class TextGraphDataset(Dataset):
    """Dataset for hierarchy or concept-enhanced text graph tensors."""

    def __init__(
        self,
        manifest_csv: str | Path,
        mode: str | None = None,
        datasets: list[str] | tuple[str, ...] | None = None,
    ):
        if mode is not None and mode not in VALID_SPLITS:
            raise ValueError(f"mode must be one of {sorted(VALID_SPLITS)} or None. Got: {mode}")

        rows = _read_manifest_rows(manifest_csv)
        if mode is not None:
            rows = [row for row in rows if row.get("split") == mode]

        if datasets:
            dataset_set = {name.upper() for name in datasets}
            rows = [row for row in rows if row.get("dataset", "").upper() in dataset_set]

        self.samples = rows

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        graph = torch.load(sample["graph_pt"], map_location="cpu")
        label = torch.tensor(sample["label"], dtype=torch.long)
        meta = {
            "case_id": sample["case_id"],
            "slide_id": sample["slide_id"],
            "dataset": sample["dataset"],
            "split": sample["split"],
            "graph_pt": sample["graph_pt"],
            "graph_json": sample["graph_json"],
            "report_id": sample["report_id"],
            "file_name": sample["file_name"],
            "filter_mode": sample["filter_mode"],
            "source_concept_json": sample["source_concept_json"],
            "concept_count": sample["concept_count"],
            "has_concept_level": sample["has_concept_level"],
        }
        return graph, label, meta


def collate_text_graphs(batch):
    graphs, labels, metas = zip(*batch)
    return list(graphs), torch.stack(labels, dim=0), list(metas)


def get_text_graph_dataloader(
    manifest_csv: str | Path,
    mode: str | None,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool | None = None,
    datasets: list[str] | tuple[str, ...] | None = None,
) -> DataLoader:
    dataset = TextGraphDataset(
        manifest_csv=manifest_csv,
        mode=mode,
        datasets=datasets,
    )
    if shuffle is None:
        shuffle = mode == "train"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_text_graphs,
    )
