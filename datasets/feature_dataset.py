from __future__ import annotations
import os
import re
from typing import List, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

CASE_ID_RE = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", re.IGNORECASE)


def _extract_case_id(identifier: str) -> str:
    text = str(identifier or "").strip()
    match = CASE_ID_RE.search(text)
    if match:
        return match.group(1).upper()
    return text[:12].upper()


def _build_text_index(text_dir: str) -> tuple[dict[str, str], dict[str, list[str]]]:
    exact_index: dict[str, str] = {}
    case_index: dict[str, list[str]] = {}

    for root, _dirs, files in os.walk(text_dir):
        for name in files:
            if not name.lower().endswith(".pt"):
                continue
            stem = os.path.splitext(name)[0]
            path = os.path.join(root, name)
            exact_index[stem] = path
            case_id = _extract_case_id(stem)
            case_index.setdefault(case_id, []).append(path)

    for case_id, paths in case_index.items():
        paths.sort()
    return exact_index, case_index


class MultiModalFeatureDataset(Dataset):

    def __init__(
        self,
        split_csv: str,
        label_csv: str,
        image_dir: str,
        text_dir: str,
        text_mode: str = "sentence_pt",
        text_graph_feature: str = "node_features",
        mode: str = "train",
    ):
        assert mode in ["train", "val", "test"]
        self.image_dir = image_dir
        self.text_dir = text_dir
        self.text_mode = text_mode
        self.text_graph_feature = text_graph_feature

        valid_text_modes = {"sentence_pt", "hierarchy_graph"}
        if self.text_mode not in valid_text_modes:
            raise ValueError(f"text_mode must be one of {sorted(valid_text_modes)}. Got: {self.text_mode}")

        self._text_exact_index, self._text_case_index = _build_text_index(text_dir)

        split_df = pd.read_csv(split_csv)
        if mode not in split_df.columns:
            raise ValueError(f"Split CSV must contain column '{mode}'. Got: {list(split_df.columns)}")

        slide_ids = split_df[mode].dropna().astype(str).tolist()

        label_df = pd.read_csv(label_csv)
        required = {"case_id", "slide_id", "label"}
        if not required.issubset(set(label_df.columns)):
            raise ValueError(f"Label CSV must contain columns {required}. Got: {set(label_df.columns)}")

        label_dict = {(r["case_id"], r["slide_id"]): int(r["label"]) for _, r in label_df.iterrows()}

        self.samples: List[Tuple[str, str, int, str]] = []
        for slide_id in slide_ids:
            case_id = _extract_case_id(slide_id)
            image_path = os.path.join(image_dir, f"{slide_id}.pt")
            text_path = self._resolve_text_path(slide_id, case_id)
            if not (os.path.exists(image_path) and os.path.exists(text_path)):
                continue
            label = label_dict.get((case_id, slide_id), None)
            if label is None:
                continue
            self.samples.append((image_path, text_path, label, slide_id))

    def _resolve_text_path(self, slide_id: str, case_id: str) -> str:
        if self.text_mode == "sentence_pt":
            return os.path.join(self.text_dir, f"{case_id}.pt")

        exact_path = self._text_exact_index.get(slide_id)
        if exact_path:
            return exact_path

        case_exact = self._text_exact_index.get(case_id)
        if case_exact:
            return case_exact

        case_matches = self._text_case_index.get(case_id, [])
        if case_matches:
            return case_matches[0]

        return os.path.join(self.text_dir, f"{slide_id}.pt")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, text_path, label, slide_id = self.samples[idx]
        image_feat = torch.load(image_path)  # [n, 512]
        text_payload = torch.load(text_path)
        if self.text_mode == "sentence_pt":
            text_feat = text_payload    # [N, 512]
        else:
            if not isinstance(text_payload, dict):
                raise TypeError(f"Expected a dict graph payload for hierarchy_graph mode: {text_path}")
            if self.text_graph_feature not in text_payload:
                raise KeyError(
                    f"text_graph_feature='{self.text_graph_feature}' not found in {text_path}. "
                    f"Available keys: {sorted(text_payload.keys())}"
                )
            text_feat = text_payload[self.text_graph_feature]
            if isinstance(text_feat, torch.Tensor) and text_feat.dim() == 1:
                text_feat = text_feat.unsqueeze(0)
        return image_feat, text_feat, torch.tensor(label, dtype=torch.long), slide_id


def collate_multimodal(batch):

    imgs, txts, labels, ids = zip(*batch)
    return list(imgs), list(txts), torch.stack(labels, dim=0), list(ids)


def get_dataloaders(cfg):
    train_set = MultiModalFeatureDataset(
        cfg.data.split_file,
        cfg.data.label_file,
        cfg.data.image_dir,
        cfg.data.text_dir,
        text_mode=getattr(cfg.data, "text_mode", "sentence_pt"),
        text_graph_feature=getattr(cfg.data, "text_graph_feature", "node_features"),
        mode="train",
    )
    # val_set = MultiModalFeatureDataset(
    #     cfg.data.split_file, cfg.data.label_file, cfg.data.image_dir, cfg.data.text_dir, mode="val"
    # )
    test_set = MultiModalFeatureDataset(
        cfg.data.split_file,
        cfg.data.label_file,
        cfg.data.image_dir,
        cfg.data.text_dir,
        text_mode=getattr(cfg.data, "text_mode", "sentence_pt"),
        text_graph_feature=getattr(cfg.data, "text_graph_feature", "node_features"),
        mode="test",
    )

    train_loader = DataLoader(
        train_set, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=cfg.train.num_workers, pin_memory=True,
        collate_fn=collate_multimodal
    )
    # val_loader = DataLoader(
    #     val_set, batch_size=cfg.train.batch_size, shuffle=False,
    #     num_workers=cfg.train.num_workers, pin_memory=True,
    #     collate_fn=collate_multimodal
    # )
    test_loader = DataLoader(
        test_set, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=cfg.train.num_workers, pin_memory=True,
        collate_fn=collate_multimodal
    )
    return train_loader, test_loader
    # return train_loader, val_loader, test_loader
