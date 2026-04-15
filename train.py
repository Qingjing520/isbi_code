from __future__ import annotations
import os
import json
import math
import random
import re
from datetime import datetime
from dataclasses import asdict
from typing import List, Tuple
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from datasets.feature_dataset import get_dataloaders
from losses.mmd_loss import mmd_rbf
from models.mapper import SharedMapper
from models.pooling import AttentionPooling
from models.fusion import MoEFusion
from models.classifier import StageClassifier
from models.hier_text import DualTextFusion, HierTextBranch
from models.text_graph import TextHierarchyGraphEncoder
from utils.graph_utils import build_graph_features
from utils.metrics import safe_auc


_SECTION_TITLE_CACHE: dict[str, list[str]] = {}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EarlyStopper:
    def __init__(self, patience: int = 3, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.count = 0

    def step(self, score: float) -> bool:
        if self.best is None or score > self.best + self.min_delta:
            self.best = score
            self.count = 0
            return False
        self.count += 1
        return self.count >= self.patience


def pad_and_mask(seq_list: List[torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    seq_list: list of [L_i, D] -> padded [B, Lmax, D], mask [B, Lmax] (1 valid, 0 pad)
    """
    if len(seq_list) == 0:
        raise ValueError("pad_and_mask: empty seq_list")
    B = len(seq_list)
    D = seq_list[0].shape[1]
    Lmax = max(x.shape[0] for x in seq_list)
    padded = torch.zeros((B, Lmax, D), device=device, dtype=seq_list[0].dtype)
    mask = torch.zeros((B, Lmax), device=device, dtype=torch.long)
    for i, x in enumerate(seq_list):
        L = x.shape[0]
        padded[i, :L] = x
        mask[i, :L] = 1
    return padded, mask


def move_text_batch_to_device(
    txt_list: List[torch.Tensor | dict],
    device: torch.device,
    text_mode: str,
    text_use_graph_structure: bool,
) -> List[torch.Tensor | dict]:
    if text_mode == "dual_text":
        moved: List[dict] = []
        for payload in txt_list:
            if not isinstance(payload, dict):
                raise TypeError("Expected dict payloads for dual_text mode.")
            graph_payload = payload["hierarchy_graph"]
            if not isinstance(graph_payload, dict):
                raise TypeError("dual_text hierarchy_graph payload must be a dict.")
            graph_item = {}
            for key, value in graph_payload.items():
                if isinstance(value, torch.Tensor):
                    graph_item[key] = value.to(device)
                else:
                    graph_item[key] = value
            moved.append(
                {
                    "sentence_pt": payload["sentence_pt"].to(device).float(),
                    "hierarchy_graph": graph_item,
                    "sentence_path": payload.get("sentence_path", ""),
                    "graph_path": payload.get("graph_path", ""),
                    "graph_json_path": payload.get("graph_json_path", ""),
                }
            )
        return moved

    if text_mode not in {"hierarchy_graph", "concept_graph"} or not text_use_graph_structure:
        return [x.to(device).float() for x in txt_list]

    moved: List[dict] = []
    for payload in txt_list:
        if not isinstance(payload, dict):
            raise TypeError(f"Expected dict payloads for graph-structured {text_mode} mode.")
        item = {}
        for key, value in payload.items():
            if isinstance(value, torch.Tensor):
                item[key] = value.to(device)
            else:
                item[key] = value
        moved.append(item)
    return moved


def encode_text_batch(
    txt_list: List[torch.Tensor | dict],
    mapper: SharedMapper,
    pool_txt: AttentionPooling | None,
    text_graph_encoder: TextHierarchyGraphEncoder | None,
    hier_text_branch: HierTextBranch | None,
    dual_text_fusion: DualTextFusion | None,
    device: torch.device,
    text_mode: str,
    text_use_graph_structure: bool,
    text_graph_use_next_edges: bool = True,
    use_section_title_embedding: bool = False,
    section_title_to_id: dict[str, int] | None = None,
) -> Tuple[List[torch.Tensor], torch.Tensor, dict]:
    if text_mode == "dual_text":
        if pool_txt is None or hier_text_branch is None or dual_text_fusion is None:
            raise RuntimeError("dual_text mode requires pool_txt, hier_text_branch, and dual_text_fusion.")

        sentence_nodes = [mapper(payload["sentence_pt"]) for payload in txt_list]  # type: ignore[index]
        txt_pad, txt_mask = pad_and_mask(sentence_nodes, device)
        sentence_vec, _ = pool_txt(txt_pad, txt_mask)

        graph_sentence_nodes: List[torch.Tensor] = []
        graph_section_vecs: List[torch.Tensor] = []
        graph_doc_vecs: List[torch.Tensor] = []
        graph_doc_section_attn: List[torch.Tensor] = []
        graph_section_sentence_attn: List[list[torch.Tensor]] = []
        graph_json_paths: List[str] = []
        graph_section_attention_mix: List[torch.Tensor] = []
        graph_document_attention_mix: List[torch.Tensor] = []
        for payload in txt_list:
            graph_payload = payload["hierarchy_graph"]  # type: ignore[index]
            sentence_features = graph_payload["sentence_features"].float()
            mapped_sentence_features = mapper(sentence_features)
            graph_json_path = str(payload.get("graph_json_path", ""))
            section_title_ids = None
            if use_section_title_embedding and section_title_to_id:
                section_title_ids = _section_title_ids_from_graph_json(
                    graph_json_path=graph_json_path,
                    num_sections=int(graph_payload["section_spans"].shape[0]),
                    title_to_id=section_title_to_id,
                    device=device,
                )
            sentence_out, section_out, document_out, branch_analysis = hier_text_branch(
                sentence_features=mapped_sentence_features,
                section_spans=graph_payload["section_spans"].long(),
                section_title_ids=section_title_ids,
            )
            graph_sentence_nodes.append(sentence_out)
            graph_section_vecs.append(section_out)
            graph_doc_vecs.append(document_out)
            graph_doc_section_attn.append(branch_analysis["document_section_attn"])
            graph_section_sentence_attn.append(branch_analysis["section_sentence_attn"])
            graph_json_paths.append(graph_json_path)
            graph_section_attention_mix.append(branch_analysis["section_attention_mix"])
            graph_document_attention_mix.append(branch_analysis["document_attention_mix"])

        graph_vec = torch.stack(graph_doc_vecs, dim=0)
        fused_vec, gates = dual_text_fusion(sentence_vec, graph_vec)
        fused_nodes = [fused_vec[i].unsqueeze(0) for i in range(fused_vec.shape[0])]
        extras = {
            "sentence_vec": sentence_vec,
            "graph_vec": graph_vec,
            "graph_sentence_nodes": graph_sentence_nodes,
            "graph_section_vecs": graph_section_vecs,
            "fusion_gate": gates,
            "graph_doc_section_attn": graph_doc_section_attn,
            "graph_section_sentence_attn": graph_section_sentence_attn,
            "graph_json_paths": graph_json_paths,
            "graph_section_attention_mix": graph_section_attention_mix,
            "graph_document_attention_mix": graph_document_attention_mix,
        }
        return fused_nodes, fused_vec, extras

    if text_mode in {"hierarchy_graph", "concept_graph"} and text_use_graph_structure:
        if text_graph_encoder is None:
            raise RuntimeError("text_graph_encoder is required when text_use_graph_structure=True")

        node_lists: List[torch.Tensor] = []
        pooled_vecs: List[torch.Tensor] = []
        graph_topology_descs: List[torch.Tensor] = []
        for payload in txt_list:
            if not isinstance(payload, dict):
                raise TypeError("Expected dict payload for graph-aware text encoding.")
            if "node_features" not in payload:
                raise KeyError("Graph payload missing 'node_features'.")

            x = mapper(payload["node_features"].float())
            allowed_forward_relation_ids = _allowed_forward_relation_ids_from_payload(
                payload=payload,
                text_graph_use_next_edges=text_graph_use_next_edges,
            )
            updated_nodes, pooled_vec = text_graph_encoder(
                node_features=x,
                edge_index=payload["edge_index"].long(),
                edge_type=payload["edge_type"].long(),
                node_type=payload["node_type"].long(),
                allowed_forward_relation_ids=allowed_forward_relation_ids,
            )
            node_lists.append(updated_nodes)
            pooled_vecs.append(pooled_vec)
            graph_topology_descs.append(_graph_structure_descriptor(payload))
        return node_lists, torch.stack(pooled_vecs, dim=0), {"graph_topology_descs": graph_topology_descs}

    if pool_txt is None:
        raise RuntimeError("pool_txt must be available for non-graph text encoding.")

    mapped = [mapper(x) for x in txt_list]  # type: ignore[arg-type]
    txt_pad, txt_mask = pad_and_mask(mapped, device)
    txt_vec, _ = pool_txt(txt_pad, txt_mask)
    return mapped, txt_vec, {}


def _allowed_forward_relation_ids_from_payload(
    payload: dict,
    text_graph_use_next_edges: bool,
) -> list[int] | None:
    edge_type_mapping = payload.get("edge_type_mapping", {})
    if not isinstance(edge_type_mapping, dict) or text_graph_use_next_edges:
        return None

    allowed: list[int] = []
    for edge_name, rel_id in edge_type_mapping.items():
        try:
            rel_index = int(rel_id)
        except (TypeError, ValueError):
            continue
        if str(edge_name).lower() == "next":
            continue
        allowed.append(rel_index)
    return sorted(set(allowed)) or None


def _graph_mode_uses_structure(text_mode: str) -> bool:
    return text_mode in {"hierarchy_graph", "concept_graph"}


def _auto_text_graph_num_node_types(text_mode: str) -> int:
    if text_mode == "concept_graph":
        return 4
    return 3


def _auto_text_graph_num_base_relations(text_mode: str) -> int:
    if text_mode == "concept_graph":
        return 6
    return 2


def _text_graph_num_node_types(cfg) -> int:
    configured = int(_get_optional(cfg, "model.text_graph_num_node_types", 0))
    if configured > 0:
        return configured
    return _auto_text_graph_num_node_types(getattr(cfg.data, "text_mode", "sentence_pt"))


def _text_graph_num_base_relations(cfg) -> int:
    configured = int(_get_optional(cfg, "model.text_graph_num_base_relations", 0))
    if configured > 0:
        return configured
    return _auto_text_graph_num_base_relations(getattr(cfg.data, "text_mode", "sentence_pt"))


def _node_type_id_from_payload(payload: dict, node_type_name: str) -> int | None:
    mapping = payload.get("node_type_mapping", {})
    if not isinstance(mapping, dict):
        return None
    raw = mapping.get(node_type_name)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _collect_node_type_features(
    node_lists: List[torch.Tensor],
    payloads: List[torch.Tensor | dict],
    node_type_name: str,
) -> torch.Tensor | None:
    collected: list[torch.Tensor] = []
    for nodes, payload in zip(node_lists, payloads):
        if not isinstance(payload, dict):
            continue
        node_type_id = _node_type_id_from_payload(payload, node_type_name)
        if node_type_id is None:
            continue
        node_type_tensor = payload.get("node_type")
        if not isinstance(node_type_tensor, torch.Tensor):
            continue
        mask = node_type_tensor.long() == int(node_type_id)
        if bool(mask.any()):
            collected.append(nodes[mask])
    if not collected:
        return None
    return torch.cat(collected, dim=0)


def _graph_structure_descriptor(payload: dict) -> torch.Tensor:
    if not isinstance(payload, dict):
        raise TypeError("Expected dict payload for graph structure descriptor.")

    node_type_tensor = payload.get("node_type")
    edge_type_tensor = payload.get("edge_type")
    if not isinstance(node_type_tensor, torch.Tensor) or not isinstance(edge_type_tensor, torch.Tensor):
        raise KeyError("Graph payload missing node_type or edge_type tensors.")

    node_type_mapping = payload.get("node_type_mapping", {})
    edge_type_mapping = payload.get("edge_type_mapping", {})
    if not isinstance(node_type_mapping, dict):
        node_type_mapping = {}
    if not isinstance(edge_type_mapping, dict):
        edge_type_mapping = {}

    node_dim = max((int(v) for v in node_type_mapping.values()), default=int(node_type_tensor.max().item()) if node_type_tensor.numel() else -1) + 1
    edge_dim = max((int(v) for v in edge_type_mapping.values()), default=int(edge_type_tensor.max().item()) if edge_type_tensor.numel() else -1) + 1

    node_counts = torch.zeros(node_dim, device=node_type_tensor.device, dtype=torch.float32)
    edge_counts = torch.zeros(edge_dim, device=edge_type_tensor.device, dtype=torch.float32)
    if node_type_tensor.numel() > 0:
        node_counts.scatter_add_(
            0,
            node_type_tensor.long(),
            torch.ones_like(node_type_tensor, dtype=torch.float32),
        )
    if edge_type_tensor.numel() > 0:
        edge_counts.scatter_add_(
            0,
            edge_type_tensor.long(),
            torch.ones_like(edge_type_tensor, dtype=torch.float32),
        )

    node_counts = node_counts / max(int(node_type_tensor.numel()), 1)
    edge_counts = edge_counts / max(int(edge_type_tensor.numel()), 1)

    concept_ic = payload.get("concept_ic")
    concept_depth = payload.get("concept_depth")
    concept_direct_mentions = payload.get("concept_direct_mentions")
    concept_stats = torch.zeros(4, device=node_type_tensor.device, dtype=torch.float32)
    if isinstance(concept_ic, torch.Tensor) and concept_ic.numel() > 0:
        concept_stats[0] = concept_ic.float().mean()
        concept_stats[1] = concept_ic.float().max()
    if isinstance(concept_depth, torch.Tensor) and concept_depth.numel() > 0:
        concept_stats[2] = concept_depth.float().mean()
    if isinstance(concept_direct_mentions, torch.Tensor) and concept_direct_mentions.numel() > 0:
        concept_stats[3] = concept_direct_mentions.float().mean()

    size_stats = torch.tensor(
        [
            float(node_type_tensor.numel()),
            float(edge_type_tensor.numel()),
        ],
        device=node_type_tensor.device,
        dtype=torch.float32,
    )
    size_stats = torch.log1p(size_stats)
    return torch.cat([node_counts, edge_counts, concept_stats, size_stats], dim=0)


def _get_optional(cfg, path: str, default):

    cur = cfg
    for p in path.split("."):
        if not hasattr(cur, p):
            return default
        cur = getattr(cur, p)
    return cur


def _make_class_weighted_ce(train_loader, device: torch.device):

    if not hasattr(train_loader.dataset, "samples"):
        return None

    ys = [int(s[2]) for s in train_loader.dataset.samples]
    if len(ys) == 0:
        return None

    n0 = sum(y == 0 for y in ys)
    n1 = sum(y == 1 for y in ys)
    n = n0 + n1

    # If only one class appears in source train, fall back to unweighted CE
    if n0 == 0 or n1 == 0:
        return None

    w0 = n / (2.0 * n0)
    w1 = n / (2.0 * n1)
    weight = torch.tensor([w0, w1], device=device, dtype=torch.float32)
    return torch.nn.CrossEntropyLoss(weight=weight)


def _linear_decay_factor(epoch: int, total_epochs: int, to: float) -> float:
    """
    Linearly decay from 1.0 at epoch=0 to 'to' at epoch=total_epochs-1.
    """
    if total_epochs <= 1:
        return float(to)
    t = epoch / float(total_epochs - 1)
    return float((1.0 - t) * 1.0 + t * to)


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _normalize_section_title(title: str) -> str:
    text = re.sub(r"\s+", " ", str(title or "").strip()).upper()
    return text


def _section_title_to_id_from_cfg(cfg) -> dict[str, int]:
    vocab = list(getattr(cfg.model, "section_title_vocab", []) or [])
    # 0 is reserved for unknown / missing titles.
    return {_normalize_section_title(title): idx + 1 for idx, title in enumerate(vocab)}


def _num_section_title_types_from_cfg(cfg) -> int:
    return len(list(getattr(cfg.model, "section_title_vocab", []) or [])) + 1


def _load_section_titles(graph_json_path: str, cache: dict[str, list[str]]) -> list[str]:
    graph_json_path = str(graph_json_path or "").strip()
    if not graph_json_path:
        return []
    cached = cache.get(graph_json_path)
    if cached is not None:
        return cached
    if not os.path.exists(graph_json_path):
        cache[graph_json_path] = []
        return []
    with open(graph_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    sections = []
    for node in payload.get("nodes", []):
        if node.get("node_type") != "section":
            continue
        sections.append((int(node.get("section_index", len(sections))), str(node.get("section_title", "Section"))))
    sections.sort(key=lambda x: x[0])
    titles = [title for _idx, title in sections]
    cache[graph_json_path] = titles
    return titles


def _section_title_ids_from_graph_json(
    graph_json_path: str,
    num_sections: int,
    title_to_id: dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    titles = _load_section_titles(graph_json_path, _SECTION_TITLE_CACHE)
    ids = [int(title_to_id.get(_normalize_section_title(title), 0)) for title in titles[:num_sections]]
    if len(ids) < num_sections:
        ids.extend([0] * (num_sections - len(ids)))
    return torch.tensor(ids[:num_sections], dtype=torch.long, device=device)


def _attention_entropy(attn: torch.Tensor) -> float | None:
    attn = attn.detach().cpu().view(-1)
    if attn.numel() == 0:
        return None
    entropy = -torch.sum(attn * torch.log(attn.clamp_min(1e-12))).item()
    if attn.numel() > 1:
        entropy = entropy / math.log(attn.numel())
    return float(entropy)


@torch.no_grad()
def evaluate(
    cfg,
    loader,
    mapper,
    pool_img,
    pool_txt,
    text_graph_encoder,
    hier_text_branch,
    dual_text_fusion,
    moe,
    classifier,
    print_dist: bool = False,
) -> dict:
    mapper.eval()
    pool_img.eval()
    if pool_txt is not None:
        pool_txt.eval()
    if text_graph_encoder is not None:
        text_graph_encoder.eval()
    if hier_text_branch is not None:
        hier_text_branch.eval()
    if dual_text_fusion is not None:
        dual_text_fusion.eval()
    moe.eval()
    classifier.eval()

    device = next(mapper.parameters()).device
    all_labels, all_probs, all_preds = [], [], []
    analysis: dict[str, object] = {}
    fusion_gates: list[float] = []
    doc_attn_entropies: list[float] = []
    doc_attn_maxima: list[float] = []
    section_attention_mixes: list[float] = []
    document_attention_mixes: list[float] = []
    top_section_counter: Counter[str] = Counter()
    section_title_cache: dict[str, list[str]] = {}
    sample_details: list[dict[str, object]] = []
    text_mode = getattr(cfg.data, "text_mode", "sentence_pt")
    text_use_graph_structure = bool(getattr(cfg.data, "text_use_graph_structure", False))
    use_section_title_embedding = bool(getattr(cfg.model, "use_section_title_embedding", False))
    section_title_to_id = _section_title_to_id_from_cfg(cfg) if use_section_title_embedding else None

    for img_list, txt_list, labels, _ids in loader:
        img_list = [x.to(device).float() for x in img_list]
        txt_list = move_text_batch_to_device(
            txt_list,
            device=device,
            text_mode=text_mode,
            text_use_graph_structure=text_use_graph_structure,
        )
        labels = labels.to(device)

        # Mapper
        img_list = [mapper(x) for x in img_list]
        _txt_nodes, txt_vec, _txt_extra = encode_text_batch(
            txt_list=txt_list,
            mapper=mapper,
            pool_txt=pool_txt,
            text_graph_encoder=text_graph_encoder,
            hier_text_branch=hier_text_branch,
            dual_text_fusion=dual_text_fusion,
            device=device,
            text_mode=text_mode,
            text_use_graph_structure=text_use_graph_structure,
            text_graph_use_next_edges=bool(getattr(cfg.model, "text_graph_use_next_edges", True)),
            use_section_title_embedding=use_section_title_embedding,
            section_title_to_id=section_title_to_id,
        )

        # Attention pooling
        img_pad, img_mask = pad_and_mask(img_list, device)
        img_vec, _ = pool_img(img_pad, img_mask)  # [B, D]

        # MoE + classifier (eval hard=True)
        fused, _g = moe(img_vec, txt_vec, hard=True)
        out = classifier(fused)

        probs = out["probs"][:, 1]
        preds = out["preds"]

        all_labels.extend(labels.detach().cpu().tolist())
        all_probs.extend(probs.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

        batch_size = len(labels.detach().cpu().view(-1).tolist())
        batch_ids = [str(x) for x in _ids]
        batch_labels = [int(x) for x in labels.detach().cpu().view(-1).tolist()]
        batch_probs = [float(x) for x in probs.detach().cpu().view(-1).tolist()]
        batch_preds = [int(x) for x in preds.detach().cpu().view(-1).tolist()]

        if text_mode == "dual_text":
            gate_values: list[float | None] = [None] * batch_size
            gate_tensor = _txt_extra.get("fusion_gate")
            if isinstance(gate_tensor, torch.Tensor):
                gate_values = [float(x) for x in gate_tensor.detach().cpu().view(-1).tolist()]
                fusion_gates.extend(gate_values)

            section_mix_values: list[float | None] = [None] * batch_size
            section_mix_list = _txt_extra.get("graph_section_attention_mix", [])
            if isinstance(section_mix_list, list):
                section_mix_values = [
                    float(x.detach().cpu().view(-1)[0].item()) if isinstance(x, torch.Tensor) and x.numel() else None
                    for x in section_mix_list[:batch_size]
                ]
                section_attention_mixes.extend([x for x in section_mix_values if x is not None])

            document_mix_values: list[float | None] = [None] * batch_size
            document_mix_list = _txt_extra.get("graph_document_attention_mix", [])
            if isinstance(document_mix_list, list):
                document_mix_values = [
                    float(x.detach().cpu().view(-1)[0].item()) if isinstance(x, torch.Tensor) and x.numel() else None
                    for x in document_mix_list[:batch_size]
                ]
                document_attention_mixes.extend([x for x in document_mix_values if x is not None])

            doc_attn_list = _txt_extra.get("graph_doc_section_attn", [])
            graph_json_paths = _txt_extra.get("graph_json_paths", [])
            if not isinstance(doc_attn_list, list):
                doc_attn_list = []
            if not isinstance(graph_json_paths, list):
                graph_json_paths = []

            for sample_idx in range(batch_size):
                graph_json_path = str(graph_json_paths[sample_idx]) if sample_idx < len(graph_json_paths) else ""
                attn = doc_attn_list[sample_idx] if sample_idx < len(doc_attn_list) else None
                top_idx: int | None = None
                top_section = None
                top_weight = None
                entropy = None
                section_titles = _load_section_titles(graph_json_path, section_title_cache)
                if isinstance(attn, torch.Tensor):
                    attn_cpu = attn.detach().cpu().view(-1)
                    if attn_cpu.numel() > 0:
                        top_idx = int(attn_cpu.argmax().item())
                        top_weight = float(attn_cpu.max().item())
                        entropy = _attention_entropy(attn_cpu)
                        doc_attn_maxima.append(top_weight)
                        if entropy is not None:
                            doc_attn_entropies.append(entropy)
                        if top_idx < len(section_titles):
                            top_section = section_titles[top_idx]
                        else:
                            top_section = f"section_{top_idx}"
                        top_section_counter[str(top_section)] += 1

                sample_details.append(
                    {
                        "slide_id": batch_ids[sample_idx] if sample_idx < len(batch_ids) else "",
                        "label": batch_labels[sample_idx],
                        "pred": batch_preds[sample_idx],
                        "prob_late": batch_probs[sample_idx],
                        "correct": bool(batch_labels[sample_idx] == batch_preds[sample_idx]),
                        "fusion_gate": gate_values[sample_idx] if sample_idx < len(gate_values) else None,
                        "sentence_branch_weight": gate_values[sample_idx] if sample_idx < len(gate_values) else None,
                        "graph_branch_weight": (
                            1.0 - gate_values[sample_idx]
                            if sample_idx < len(gate_values) and gate_values[sample_idx] is not None
                            else None
                        ),
                        "graph_json_path": graph_json_path,
                        "top_section_index": top_idx,
                        "top_section_title": top_section,
                        "top_section_attention": top_weight,
                        "doc_attention_entropy": entropy,
                        "section_attention_mix": section_mix_values[sample_idx]
                        if sample_idx < len(section_mix_values)
                        else None,
                        "document_attention_mix": document_mix_values[sample_idx]
                        if sample_idx < len(document_mix_values)
                        else None,
                    }
                )
        else:
            for sample_idx in range(batch_size):
                sample_details.append(
                    {
                        "slide_id": batch_ids[sample_idx] if sample_idx < len(batch_ids) else "",
                        "label": batch_labels[sample_idx],
                        "pred": batch_preds[sample_idx],
                        "prob_late": batch_probs[sample_idx],
                        "correct": bool(batch_labels[sample_idx] == batch_preds[sample_idx]),
                    }
                )

    if len(all_labels) == 0:
        return {"acc": 0.0, "auc": float("nan")}

    y_true = np.array(all_labels, dtype=int)
    y_pred = np.array(all_preds, dtype=int)
    acc = float(np.mean((y_pred == y_true).astype(np.float32)))
    auc = safe_auc(all_labels, all_probs)

    if text_mode == "dual_text":
        graph_branch_weights = [1.0 - x for x in fusion_gates]
        analysis = {
            "fusion_gate_is_sentence_branch_weight": True,
            "fusion_gate_mean": float(np.mean(fusion_gates)) if fusion_gates else None,
            "fusion_gate_std": float(np.std(fusion_gates)) if fusion_gates else None,
            "fusion_gate_min": float(np.min(fusion_gates)) if fusion_gates else None,
            "fusion_gate_max": float(np.max(fusion_gates)) if fusion_gates else None,
            "sentence_branch_weight_mean": float(np.mean(fusion_gates)) if fusion_gates else None,
            "graph_branch_weight_mean": float(np.mean(graph_branch_weights)) if graph_branch_weights else None,
            "doc_attention_entropy_mean": float(np.mean(doc_attn_entropies)) if doc_attn_entropies else None,
            "doc_attention_max_mean": float(np.mean(doc_attn_maxima)) if doc_attn_maxima else None,
            "section_attention_mix_mean": float(np.mean(section_attention_mixes)) if section_attention_mixes else None,
            "document_attention_mix_mean": float(np.mean(document_attention_mixes)) if document_attention_mixes else None,
            "top_section_counts": dict(top_section_counter.most_common()),
            "sample_details": sample_details,
        }
    else:
        analysis = {"sample_details": sample_details}

    # if print_dist:
    #     print("tgt label dist:", np.bincount(y_true, minlength=2).tolist())
    #     print("tgt pred  dist:", np.bincount(y_pred, minlength=2).tolist())

    return {"acc": acc, "auc": auc, "analysis": analysis}


def train_one_split(cfg):

    os.makedirs(cfg.output.exp_dir, exist_ok=True)
    with open(os.path.join(cfg.output.exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, target_loader = get_dataloaders(cfg)

    mapper = SharedMapper(
        dim=cfg.model.feat_dim,
        hidden=cfg.model.mapper_hidden,
        depth=cfg.model.mapper_depth,
        dropout=cfg.model.mapper_dropout,
    ).to(device)

    pool_img = AttentionPooling(
        dim=cfg.model.feat_dim,
        hidden=cfg.model.attn_pool_hidden,
        dropout=cfg.model.attn_pool_dropout,
    ).to(device)

    text_mode = getattr(cfg.data, "text_mode", "sentence_pt")
    text_use_graph_structure = bool(getattr(cfg.data, "text_use_graph_structure", False))
    if text_use_graph_structure and not _graph_mode_uses_structure(text_mode):
        raise ValueError("text_use_graph_structure=True requires text_mode='hierarchy_graph' or 'concept_graph'.")
    use_section_title_embedding = bool(getattr(cfg.model, "use_section_title_embedding", False))
    section_title_to_id = _section_title_to_id_from_cfg(cfg) if use_section_title_embedding else None

    pool_txt = None
    text_graph_encoder = None
    hier_text_branch = None
    dual_text_fusion = None
    if _graph_mode_uses_structure(text_mode) and text_use_graph_structure:
        text_graph_encoder = TextHierarchyGraphEncoder(
            dim=cfg.model.feat_dim,
            num_node_types=_text_graph_num_node_types(cfg),
            num_base_relations=_text_graph_num_base_relations(cfg),
            num_layers=cfg.model.text_graph_layers,
            dropout=cfg.model.text_graph_dropout,
            use_next_edges=cfg.model.text_graph_use_next_edges,
        ).to(device)
    else:
        pool_txt = AttentionPooling(
            dim=cfg.model.feat_dim,
            hidden=cfg.model.attn_pool_hidden,
            dropout=cfg.model.attn_pool_dropout,
        ).to(device)
        if text_mode == "dual_text":
            hier_text_branch = HierTextBranch(
                dim=cfg.model.feat_dim,
                hidden=cfg.model.hier_readout_hidden,
                dropout=cfg.model.hier_readout_dropout,
                sentence_local_layers=cfg.model.sentence_local_layers,
                sentence_local_heads=cfg.model.sentence_local_heads,
                readout_attention_init=cfg.model.hier_readout_attention_init,
                use_section_title_embedding=use_section_title_embedding,
                num_section_title_types=_num_section_title_types_from_cfg(cfg),
            ).to(device)
            dual_text_fusion = DualTextFusion(
                dim=cfg.model.feat_dim,
                dropout=cfg.model.text_dual_fusion_dropout,
            ).to(device)

    moe = MoEFusion(
        dim=cfg.model.feat_dim,
        n_heads=cfg.model.moe_heads,
        dropout=cfg.model.moe_dropout,
        tau_start=cfg.model.moe_tau_start,
        tau_min=cfg.model.moe_tau_min,
        decay=cfg.model.moe_tau_decay,
        hard_start_epoch=cfg.model.moe_hard_start_epoch,
    ).to(device)

    classifier = StageClassifier(
        dim=cfg.model.feat_dim,
        hidden=cfg.model.classifier_hidden,
        dropout=cfg.model.classifier_dropout,
        n_classes=2,
    ).to(device)

    params = (
        list(mapper.parameters())
        + list(pool_img.parameters())
        + list(moe.parameters())
        + list(classifier.parameters())
    )
    if pool_txt is not None:
        params += list(pool_txt.parameters())
    if text_graph_encoder is not None:
        params += list(text_graph_encoder.parameters())
    if hier_text_branch is not None:
        params += list(hier_text_branch.parameters())
    if dual_text_fusion is not None:
        params += list(dual_text_fusion.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    warmup_epochs = int(_get_optional(cfg, "loss.warmup_epochs", 3))
    align_decay_to = float(_get_optional(cfg, "loss.align_decay_to", 0.3))

    es_patience = int(_get_optional(cfg, "train.early_stop_patience", 3))
    es_min_delta = float(_get_optional(cfg, "train.early_stop_min_delta", 1e-4))
    stopper = EarlyStopper(patience=es_patience, min_delta=es_min_delta)

    ce_weighted = _make_class_weighted_ce(train_loader, device)
    if ce_weighted is None:
        # fallback
        ce_weighted = torch.nn.CrossEntropyLoss()
        # print("[CE] use unweighted CE (source train single-class or missing samples)")
    # else:
        # print("[CE] use class-weighted CE on source train")

    best_metric = -1e9
    best_path = os.path.join(cfg.output.exp_dir, "best_model.pt")

    target_iter = iter(target_loader)

    printed_dist = False

    for epoch in range(cfg.train.num_epochs):
        mapper.train()
        pool_img.train()
        if pool_txt is not None:
            pool_txt.train()
        if text_graph_encoder is not None:
            text_graph_encoder.train()
        if hier_text_branch is not None:
            hier_text_branch.train()
        if dual_text_fusion is not None:
            dual_text_fusion.train()
        moe.train()
        classifier.train()
        moe.set_epoch(epoch)

        # alignment weight factor (decay)
        decay_factor = _linear_decay_factor(epoch, cfg.train.num_epochs, to=align_decay_to)

        running = {
            "loss_total": 0.0,
            "loss_cls": 0.0,
            "loss_txt": 0.0,
            "loss_concept": 0.0,
            "loss_node": 0.0,
            "loss_topo": 0.0,
            "loss_text_topology": 0.0,
            "loss_gate": 0.0,
            "graph_weight": 0.0,
            "steps": 0,
        }

        for img_s, txt_s, y_s, _ids_s in train_loader:
            try:
                img_t, txt_t, _y_t, _ids_t = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                img_t, txt_t, _y_t, _ids_t = next(target_iter)

            img_s = [x.to(device).float() for x in img_s]
            txt_s = move_text_batch_to_device(
                txt_s,
                device=device,
                text_mode=text_mode,
                text_use_graph_structure=text_use_graph_structure,
            )
            y_s = y_s.to(device)

            img_t = [x.to(device).float() for x in img_t]
            txt_t = move_text_batch_to_device(
                txt_t,
                device=device,
                text_mode=text_mode,
                text_use_graph_structure=text_use_graph_structure,
            )

            # ===== Mapper =====
            img_s_m = [mapper(x) for x in img_s]
            img_t_m = [mapper(x) for x in img_t]
            txt_s_nodes, txt_vec, txt_s_extra = encode_text_batch(
                txt_list=txt_s,
                mapper=mapper,
                pool_txt=pool_txt,
                text_graph_encoder=text_graph_encoder,
                hier_text_branch=hier_text_branch,
                dual_text_fusion=dual_text_fusion,
                device=device,
                text_mode=text_mode,
                text_use_graph_structure=text_use_graph_structure,
                text_graph_use_next_edges=cfg.model.text_graph_use_next_edges,
                use_section_title_embedding=use_section_title_embedding,
                section_title_to_id=section_title_to_id,
            )
            txt_t_nodes, _txt_t_vec, txt_t_extra = encode_text_batch(
                txt_list=txt_t,
                mapper=mapper,
                pool_txt=pool_txt,
                text_graph_encoder=text_graph_encoder,
                hier_text_branch=hier_text_branch,
                dual_text_fusion=dual_text_fusion,
                device=device,
                text_mode=text_mode,
                text_use_graph_structure=text_use_graph_structure,
                text_graph_use_next_edges=cfg.model.text_graph_use_next_edges,
                use_section_title_embedding=use_section_title_embedding,
                section_title_to_id=section_title_to_id,
            )

            # ===== Classification on SOURCE only =====
            img_pad, img_mask = pad_and_mask(img_s_m, device)
            img_vec, _ = pool_img(img_pad, img_mask)  # [B, D]

            fused, _g = moe(img_vec, txt_vec)         # [B, D]
            out = classifier(fused)
            loss_cls = ce_weighted(out["logits"], y_s)

            # ===== Alignment losses (after warmup) =====
            loss_txt = torch.zeros((), device=device)
            loss_concept = torch.zeros((), device=device)
            loss_node = torch.zeros((), device=device)
            loss_topo = torch.zeros((), device=device)
            loss_text_topology = torch.zeros((), device=device)
            loss_gate = torch.zeros((), device=device)
            graph_weight_mean = torch.zeros((), device=device)

            gate_reg_weight = float(_get_optional(cfg, "loss.dual_text_gate_reg_weight", 0.0))
            graph_weight_target = float(_get_optional(cfg, "loss.dual_text_graph_weight_target", 0.2))
            if text_mode == "dual_text" and gate_reg_weight > 0:
                gate_tensors = []
                for extra in (txt_s_extra, txt_t_extra):
                    gate = extra.get("fusion_gate")
                    if isinstance(gate, torch.Tensor):
                        gate_tensors.append(gate.view(-1))
                if gate_tensors:
                    sentence_weights = torch.cat(gate_tensors, dim=0)
                    graph_weights = 1.0 - sentence_weights
                    graph_weight_mean = graph_weights.detach().mean()
                    loss_gate = (graph_weights - graph_weight_target).pow(2).mean()

            if epoch >= warmup_epochs:
                nodes_s_list, topo_s_list = [], []
                nodes_t_list, topo_t_list = [], []

                for x in img_s_m:
                    nodes, topo = build_graph_features(
                        x,
                        m=cfg.graph.num_nodes_m,
                        k=cfg.graph.topk_k,
                        kmeans_iters=cfg.graph.kmeans_iters,
                        max_patches_for_kmeans=cfg.graph.max_patches_for_kmeans,
                    )
                    nodes_s_list.append(nodes)
                    topo_s_list.append(topo)

                for x in img_t_m:
                    nodes, topo = build_graph_features(
                        x,
                        m=cfg.graph.num_nodes_m,
                        k=cfg.graph.topk_k,
                        kmeans_iters=cfg.graph.kmeans_iters,
                        max_patches_for_kmeans=cfg.graph.max_patches_for_kmeans,
                    )
                    nodes_t_list.append(nodes)
                    topo_t_list.append(topo)

                nodes_s_all = torch.cat(nodes_s_list, dim=0)  # [B*m, D]
                nodes_t_all = torch.cat(nodes_t_list, dim=0)  # [B*m, D]
                topo_s_all = torch.cat(topo_s_list, dim=0)    # [B*m, 3]
                topo_t_all = torch.cat(topo_t_list, dim=0)    # [B*m, 3]

                loss_node = mmd_rbf(
                    nodes_s_all, nodes_t_all,
                    sigma_multipliers=cfg.loss.mmd_sigma_multipliers,
                    unbiased=cfg.loss.mmd_unbiased,
                    clamp_nonneg=cfg.loss.mmd_clamp_nonneg,
                )
                loss_topo = mmd_rbf(
                    topo_s_all, topo_t_all,
                    sigma_multipliers=cfg.loss.mmd_sigma_multipliers,
                    unbiased=cfg.loss.mmd_unbiased,
                    clamp_nonneg=cfg.loss.mmd_clamp_nonneg,
                )

                if text_mode == "dual_text":
                    txt_s_mean = txt_vec
                    txt_t_mean = _txt_t_vec
                else:
                    txt_s_mean = torch.stack([x.mean(dim=0) for x in txt_s_nodes], dim=0)  # [B, D]
                    txt_t_mean = torch.stack([x.mean(dim=0) for x in txt_t_nodes], dim=0)  # [B, D]
                loss_txt = mmd_rbf(
                    txt_s_mean, txt_t_mean,
                    sigma_multipliers=cfg.loss.mmd_sigma_multipliers,
                    unbiased=cfg.loss.mmd_unbiased,
                    clamp_nonneg=cfg.loss.mmd_clamp_nonneg,
                )

                if text_mode == "concept_graph" and text_use_graph_structure:
                    concept_nodes_s = _collect_node_type_features(txt_s_nodes, txt_s, "concept")
                    concept_nodes_t = _collect_node_type_features(txt_t_nodes, txt_t, "concept")
                    if concept_nodes_s is not None and concept_nodes_t is not None:
                        loss_concept = mmd_rbf(
                            concept_nodes_s,
                            concept_nodes_t,
                            sigma_multipliers=cfg.loss.mmd_sigma_multipliers,
                            unbiased=cfg.loss.mmd_unbiased,
                            clamp_nonneg=cfg.loss.mmd_clamp_nonneg,
                        )

                    topo_descs_s = txt_s_extra.get("graph_topology_descs", [])
                    topo_descs_t = txt_t_extra.get("graph_topology_descs", [])
                    if isinstance(topo_descs_s, list) and isinstance(topo_descs_t, list) and topo_descs_s and topo_descs_t:
                        loss_text_topology = mmd_rbf(
                            torch.stack(topo_descs_s, dim=0),
                            torch.stack(topo_descs_t, dim=0),
                            sigma_multipliers=cfg.loss.mmd_sigma_multipliers,
                            unbiased=cfg.loss.mmd_unbiased,
                            clamp_nonneg=cfg.loss.mmd_clamp_nonneg,
                        )

            loss = (
                loss_cls
                + decay_factor
                * (
                    cfg.loss.alpha_txt * loss_txt
                    + cfg.loss.alpha_concept * loss_concept
                    + cfg.loss.beta_node * loss_node
                    + cfg.loss.gamma_topo * loss_topo
                    + cfg.loss.gamma_text_topology * loss_text_topology
                )
                + gate_reg_weight * loss_gate
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if getattr(cfg.train, "grad_clip", None) is not None and cfg.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
            optimizer.step()

            running["loss_total"] += float(loss.item())
            running["loss_cls"] += float(loss_cls.item())
            running["loss_txt"] += float(loss_txt.item())
            running["loss_concept"] += float(loss_concept.item())
            running["loss_node"] += float(loss_node.item())
            running["loss_topo"] += float(loss_topo.item())
            running["loss_text_topology"] += float(loss_text_topology.item())
            running["loss_gate"] += float(loss_gate.item())
            running["graph_weight"] += float(graph_weight_mean.item())
            running["steps"] += 1

        steps = max(running["steps"], 1)
        for k in [
            "loss_total",
            "loss_cls",
            "loss_txt",
            "loss_concept",
            "loss_node",
            "loss_topo",
            "loss_text_topology",
            "loss_gate",
            "graph_weight",
        ]:
            running[k] /= steps

        # ===== Evaluate on target(test) =====
        tgt_metrics = evaluate(
            cfg,
            target_loader,
            mapper,
            pool_img,
            pool_txt,
            text_graph_encoder,
            hier_text_branch,
            dual_text_fusion,
            moe,
            classifier,
            print_dist=(not printed_dist)
        )
        printed_dist = True

        score = tgt_metrics["auc"] if not np.isnan(tgt_metrics["auc"]) else tgt_metrics["acc"]

        if score > best_metric:
            best_metric = score
            torch.save(
                {
                    "mapper": mapper.state_dict(),
                    "pool_img": pool_img.state_dict(),
                    "pool_txt": pool_txt.state_dict() if pool_txt is not None else None,
                    "text_graph_encoder": text_graph_encoder.state_dict() if text_graph_encoder is not None else None,
                    "hier_text_branch": hier_text_branch.state_dict() if hier_text_branch is not None else None,
                    "dual_text_fusion": dual_text_fusion.state_dict() if dual_text_fusion is not None else None,
                    "moe": moe.state_dict(),
                    "classifier": classifier.state_dict(),
                    "epoch": epoch,
                    "best_metric": best_metric,
                },
                best_path,
            )
            print(f"[{_ts()}] [best] save best @ epoch={epoch:03d}, score={best_metric:.4f}")

        stop = stopper.step(score)

        summary = {
            "epoch": epoch,
            "train": running,
            "target": tgt_metrics,
            "score_for_earlystop": float(score),
            "best_metric_so_far": float(best_metric),
            "early_stop": bool(stop),
            "warmup_epochs": warmup_epochs,
            "align_decay_to": align_decay_to,
            "align_decay_factor": float(decay_factor),
        }
        with open(os.path.join(cfg.output.exp_dir, "log.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")

        # log line
        if epoch < warmup_epochs:
            mode_str = f"warmup(cls only)"
        else:
            mode_str = f"align(decay={decay_factor:.3f})"

        print(
            f"[{_ts()}] [Epoch {epoch:03d}] {mode_str} "
            f"train(loss={running['loss_total']:.4f}, cls={running['loss_cls']:.4f}, "
            f"txt={running['loss_txt']:.4f}, concept={running['loss_concept']:.4f}, "
            f"node={running['loss_node']:.4f}, topo={running['loss_topo']:.4f}, "
            f"text_topo={running['loss_text_topology']:.4f}, "
            f"gate={running['loss_gate']:.4f}, graph_w={running['graph_weight']:.3f}) | "
            f"tgt(acc={tgt_metrics['acc']:.4f}, auc={tgt_metrics['auc']:.4f}) | "
            f"best={best_metric:.4f}"
        )
        print("")

        if stop:
            print(f"[{_ts()}] [EarlyStop] stop at epoch={epoch:03d} (best={best_metric:.4f})")
            break

    return best_path


def load_and_eval(cfg, ckpt_path: str):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # _, _val_loader, target_loader = get_dataloaders(cfg)
    _, target_loader = get_dataloaders(cfg)

    mapper = SharedMapper(
        dim=cfg.model.feat_dim,
        hidden=cfg.model.mapper_hidden,
        depth=cfg.model.mapper_depth,
        dropout=cfg.model.mapper_dropout,
    ).to(device)

    pool_img = AttentionPooling(
        dim=cfg.model.feat_dim,
        hidden=cfg.model.attn_pool_hidden,
        dropout=cfg.model.attn_pool_dropout,
    ).to(device)

    text_mode = getattr(cfg.data, "text_mode", "sentence_pt")
    text_use_graph_structure = bool(getattr(cfg.data, "text_use_graph_structure", False))
    if text_use_graph_structure and not _graph_mode_uses_structure(text_mode):
        raise ValueError("text_use_graph_structure=True requires text_mode='hierarchy_graph' or 'concept_graph'.")
    use_section_title_embedding = bool(getattr(cfg.model, "use_section_title_embedding", False))

    pool_txt = None
    text_graph_encoder = None
    hier_text_branch = None
    dual_text_fusion = None
    if _graph_mode_uses_structure(text_mode) and text_use_graph_structure:
        text_graph_encoder = TextHierarchyGraphEncoder(
            dim=cfg.model.feat_dim,
            num_node_types=_text_graph_num_node_types(cfg),
            num_base_relations=_text_graph_num_base_relations(cfg),
            num_layers=cfg.model.text_graph_layers,
            dropout=cfg.model.text_graph_dropout,
            use_next_edges=cfg.model.text_graph_use_next_edges,
        ).to(device)
    else:
        pool_txt = AttentionPooling(
            dim=cfg.model.feat_dim,
            hidden=cfg.model.attn_pool_hidden,
            dropout=cfg.model.attn_pool_dropout,
        ).to(device)
        if text_mode == "dual_text":
            hier_text_branch = HierTextBranch(
                dim=cfg.model.feat_dim,
                hidden=cfg.model.hier_readout_hidden,
                dropout=cfg.model.hier_readout_dropout,
                sentence_local_layers=cfg.model.sentence_local_layers,
                sentence_local_heads=cfg.model.sentence_local_heads,
                readout_attention_init=cfg.model.hier_readout_attention_init,
                use_section_title_embedding=use_section_title_embedding,
                num_section_title_types=_num_section_title_types_from_cfg(cfg),
            ).to(device)
            dual_text_fusion = DualTextFusion(
                dim=cfg.model.feat_dim,
                dropout=cfg.model.text_dual_fusion_dropout,
            ).to(device)

    moe = MoEFusion(
        dim=cfg.model.feat_dim,
        n_heads=cfg.model.moe_heads,
        dropout=cfg.model.moe_dropout,
        tau_start=cfg.model.moe_tau_start,
        tau_min=cfg.model.moe_tau_min,
        decay=cfg.model.moe_tau_decay,
        hard_start_epoch=cfg.model.moe_hard_start_epoch,
    ).to(device)

    classifier = StageClassifier(
        dim=cfg.model.feat_dim,
        hidden=cfg.model.classifier_hidden,
        dropout=cfg.model.classifier_dropout,
        n_classes=2,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    mapper.load_state_dict(ckpt["mapper"])
    pool_img.load_state_dict(ckpt["pool_img"])
    if pool_txt is not None and ckpt.get("pool_txt") is not None:
        pool_txt.load_state_dict(ckpt["pool_txt"])
    if text_graph_encoder is not None and ckpt.get("text_graph_encoder") is not None:
        text_graph_encoder.load_state_dict(ckpt["text_graph_encoder"])
    if hier_text_branch is not None and ckpt.get("hier_text_branch") is not None:
        hier_text_branch.load_state_dict(ckpt["hier_text_branch"], strict=False)
    if dual_text_fusion is not None and ckpt.get("dual_text_fusion") is not None:
        dual_text_fusion.load_state_dict(ckpt["dual_text_fusion"])
    moe.load_state_dict(ckpt["moe"])
    classifier.load_state_dict(ckpt["classifier"])

    tgt_metrics = evaluate(
        cfg,
        target_loader,
        mapper,
        pool_img,
        pool_txt,
        text_graph_encoder,
        hier_text_branch,
        dual_text_fusion,
        moe,
        classifier,
        print_dist=True,
    )
    return {
        "target": tgt_metrics,
        "analysis": tgt_metrics.get("analysis", {}),
        "epoch": int(ckpt.get("epoch", -1)),
        "best_metric": float(ckpt.get("best_metric", float("nan"))),
    }
