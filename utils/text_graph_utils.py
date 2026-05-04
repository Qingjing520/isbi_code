from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


DEFAULT_GENERIC_CONCEPT_DENYLIST = {
    "disease",
    "neoplasm",
    "patient",
    "procedure",
    "finding",
    "clinical finding",
    "body structure",
    "anatomical structure",
    "anatomic structure",
}

DEFAULT_LEAKAGE_CONCEPT_DENYLIST = {
    "stage",
    "staging",
    "pathologic stage",
    "pathological stage",
    "tnm",
    "ajcc",
    "grade",
}

DEFAULT_EVIDENCE_KEYWORDS = {
    "tumor size",
    "size",
    "invasion",
    "invasive",
    "lymph node",
    "lymph nodes",
    "lymphatic",
    "nodal",
    "margin",
    "margins",
    "metastasis",
    "metastatic",
    "anatomic extent",
    "extension",
    "extranodal",
    "capsule",
    "vascular invasion",
    "lymphovascular invasion",
    "perineural invasion",
}

DEFAULT_EVIDENCE_DENY_KEYWORDS = {
    "estrogen receptor",
    "progesterone receptor",
    "hormone receptor",
    " er ",
    " pr ",
    "her2",
    "immunohistochemistry",
    "procedure",
    "mastectomy",
    "biopsy",
    "carcinoma",
    "adenocarcinoma",
    "neoplasm",
}

LEAKAGE_CODE_RE = re.compile(r"\b(?:pt?[1-4][a-z]?|n[0-3][a-z]?|m[01][a-z]?|t[1-4][a-z]?)\b", re.IGNORECASE)


@dataclass
class GraphFilterStats:
    concept_total: int = 0
    concept_kept: int = 0
    concept_filtered: int = 0
    hierarchy_edges: int = 0
    ontology_edges: int = 0
    edges_before: int = 0
    edges_after: int = 0

    def as_dict(self) -> dict[str, float]:
        return {
            "concept_total": float(self.concept_total),
            "concept_kept": float(self.concept_kept),
            "concept_filtered": float(self.concept_filtered),
            "hierarchy_edges": float(self.hierarchy_edges),
            "ontology_edges": float(self.ontology_edges),
            "edges_before": float(self.edges_before),
            "edges_after": float(self.edges_after),
        }


def _cfg_get(obj: Any, name: str, default: Any) -> Any:
    return getattr(obj, name, default) if obj is not None else default


def _edge_name_to_id(mapping: dict[str, Any], name: str) -> int | None:
    raw = mapping.get(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _node_name_to_id(mapping: dict[str, Any], name: str) -> int | None:
    raw = mapping.get(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _load_concept_meta(graph_json_path: str) -> list[dict[str, Any]]:
    if not graph_json_path or not os.path.exists(graph_json_path):
        return []
    try:
        with open(graph_json_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return []
    concepts = [node for node in payload.get("nodes", []) if node.get("node_type") == "concept"]
    concepts.sort(key=lambda node: int(node.get("node_index", 10**9)))
    return concepts


def _concept_name(meta: dict[str, Any], fallback_id: str = "") -> str:
    for key in ("concept_name", "name", "label", "title", "node_id"):
        value = str(meta.get(key, "") or "").strip()
        if value:
            return value
    return str(fallback_id or "")


def _is_denied_concept(name: str, ontology_cfg: Any) -> bool:
    text = re.sub(r"\s+", " ", str(name or "").strip().lower())
    if not text:
        return False

    deny_generic = set(DEFAULT_GENERIC_CONCEPT_DENYLIST)
    deny_generic.update(str(x).strip().lower() for x in _cfg_get(ontology_cfg, "generic_concept_denylist", []) or [])
    deny_leakage = set(DEFAULT_LEAKAGE_CONCEPT_DENYLIST)
    deny_leakage.update(str(x).strip().lower() for x in _cfg_get(ontology_cfg, "leakage_concept_denylist", []) or [])

    if bool(_cfg_get(ontology_cfg, "remove_generic_concepts", True)):
        if text in deny_generic or any(text == item or text.endswith(" " + item) for item in deny_generic):
            return True
    if bool(_cfg_get(ontology_cfg, "remove_leakage_concepts", True)):
        if any(item in text for item in deny_leakage):
            return True
        if LEAKAGE_CODE_RE.search(text):
            return True
    return False


def _is_evidence_concept(name: str, ontology_cfg: Any) -> bool:
    text = " " + re.sub(r"\s+", " ", str(name or "").strip().lower()) + " "
    if not text.strip():
        return False

    deny = set(DEFAULT_EVIDENCE_DENY_KEYWORDS)
    deny.update(str(x).strip().lower() for x in _cfg_get(ontology_cfg, "evidence_deny_keywords", []) or [])
    for item in deny:
        item = str(item or "").strip().lower()
        if not item:
            continue
        needle = item if item.startswith(" ") or item.endswith(" ") else f" {item} "
        if needle in text:
            return False

    keywords = set(DEFAULT_EVIDENCE_KEYWORDS)
    keywords.update(str(x).strip().lower() for x in _cfg_get(ontology_cfg, "evidence_keywords", []) or [])
    for item in keywords:
        item = str(item or "").strip().lower()
        if item and item in text:
            return True
    return False


def _concept_scores(payload: dict[str, Any], concept_positions: torch.Tensor) -> torch.Tensor:
    device = concept_positions.device
    scores = torch.zeros(concept_positions.numel(), device=device, dtype=torch.float32)
    direct = payload.get("concept_direct_mentions")
    if isinstance(direct, torch.Tensor) and direct.numel() >= concept_positions.numel():
        scores += direct[: concept_positions.numel()].to(device=device, dtype=torch.float32)
    ic = payload.get("concept_ic")
    if isinstance(ic, torch.Tensor) and ic.numel() >= concept_positions.numel():
        scores += 0.1 * ic[: concept_positions.numel()].to(device=device, dtype=torch.float32)
    depth = payload.get("concept_depth")
    if isinstance(depth, torch.Tensor) and depth.numel() >= concept_positions.numel():
        scores += 0.05 * depth[: concept_positions.numel()].to(device=device, dtype=torch.float32)
    return scores


def _build_keep_nodes(
    payload: dict[str, Any],
    ontology_cfg: Any,
    hierarchy_cfg: Any,
    graph_json_path: str,
) -> tuple[torch.Tensor, GraphFilterStats]:
    node_type = payload["node_type"].long()
    keep = torch.ones(node_type.numel(), device=node_type.device, dtype=torch.bool)
    stats = GraphFilterStats()

    max_sentences = int(_cfg_get(hierarchy_cfg, "max_sentences_per_report", 0) or 0)
    sentence_type = _node_name_to_id(payload.get("node_type_mapping", {}), "sentence")
    if max_sentences > 0 and sentence_type is not None:
        sentence_positions = torch.nonzero(node_type == sentence_type, as_tuple=False).view(-1)
        if sentence_positions.numel() > max_sentences:
            keep[sentence_positions[max_sentences:]] = False

    ontology_enabled = bool(_cfg_get(ontology_cfg, "enabled", False))
    concept_type = _node_name_to_id(payload.get("node_type_mapping", {}), "concept")
    if concept_type is None:
        return keep, stats

    concept_positions = torch.nonzero(node_type == concept_type, as_tuple=False).view(-1)
    stats.concept_total = int(concept_positions.numel())
    if concept_positions.numel() == 0:
        return keep, stats
    if not ontology_enabled:
        keep[concept_positions] = False
        stats.concept_filtered = stats.concept_total
        return keep, stats

    keep_concepts = torch.ones(concept_positions.numel(), device=node_type.device, dtype=torch.bool)
    meta = _load_concept_meta(graph_json_path)
    concept_ids = payload.get("concept_ids", [])

    confidence = payload.get("concept_confidence")
    min_conf = float(_cfg_get(ontology_cfg, "min_concept_confidence", 0.7))
    if isinstance(confidence, torch.Tensor) and confidence.numel() >= concept_positions.numel():
        keep_concepts &= confidence[: concept_positions.numel()].to(node_type.device).float() >= min_conf

    for idx in range(concept_positions.numel()):
        fallback = str(concept_ids[idx]) if isinstance(concept_ids, list) and idx < len(concept_ids) else ""
        name = _concept_name(meta[idx], fallback_id=fallback) if idx < len(meta) else fallback
        if _is_denied_concept(name, ontology_cfg):
            keep_concepts[idx] = False
            continue
        if bool(_cfg_get(ontology_cfg, "evidence_only", False)) and not _is_evidence_concept(name, ontology_cfg):
            keep_concepts[idx] = False

    max_concepts = int(_cfg_get(ontology_cfg, "max_concepts_per_report", 30))
    if max_concepts > 0 and int(keep_concepts.sum().item()) > max_concepts:
        scores = _concept_scores(payload, concept_positions)
        scores = scores.masked_fill(~keep_concepts, -float("inf"))
        top_indices = torch.topk(scores, k=max_concepts).indices
        limited = torch.zeros_like(keep_concepts)
        limited[top_indices] = True
        keep_concepts &= limited

    keep[concept_positions] = keep_concepts
    stats.concept_kept = int(keep_concepts.sum().item())
    stats.concept_filtered = stats.concept_total - stats.concept_kept
    return keep, stats


def _filter_nodes(payload: dict[str, Any], keep: torch.Tensor) -> dict[str, Any]:
    if bool(keep.all()):
        return dict(payload)
    old_to_new = torch.full((keep.numel(),), -1, device=keep.device, dtype=torch.long)
    old_to_new[keep] = torch.arange(int(keep.sum().item()), device=keep.device)
    filtered: dict[str, Any] = dict(payload)
    for key in ("node_features", "node_type"):
        value = payload.get(key)
        if isinstance(value, torch.Tensor) and value.shape[0] == keep.numel():
            filtered[key] = value[keep]

    edge_index = payload["edge_index"].long()
    edge_keep = keep[edge_index[0]] & keep[edge_index[1]]
    filtered["edge_index"] = old_to_new[edge_index[:, edge_keep]]
    for key in ("edge_type", "edge_weight"):
        value = payload.get(key)
        if isinstance(value, torch.Tensor) and value.shape[0] == edge_keep.numel():
            filtered[key] = value[edge_keep]
    return filtered


def _prune_sentence_similarity_edges(
    payload: dict[str, Any],
    hierarchy_cfg: Any,
    stats: GraphFilterStats,
) -> dict[str, Any]:
    if not bool(_cfg_get(hierarchy_cfg, "use_sentence_similarity_edges", True)):
        return _drop_edge_names(payload, {"next", "same_sentence"}, stats)

    edge_mapping = payload.get("edge_type_mapping", {})
    same_sentence_id = _edge_name_to_id(edge_mapping, "same_sentence")
    if same_sentence_id is None:
        return payload

    edge_index = payload["edge_index"].long()
    edge_type = payload["edge_type"].long()
    same_mask = edge_type == same_sentence_id
    if not bool(same_mask.any()):
        return payload

    node_features = payload["node_features"].float()
    src = edge_index[0, same_mask]
    dst = edge_index[1, same_mask]
    sim = F.cosine_similarity(node_features[src], node_features[dst], dim=1)
    min_sim = float(_cfg_get(hierarchy_cfg, "min_sentence_sim", 0.3))
    topk = int(_cfg_get(hierarchy_cfg, "sentence_topk", 5))

    local_keep = sim >= min_sim
    if topk > 0:
        per_src: dict[int, list[int]] = {}
        kept_positions = torch.nonzero(local_keep, as_tuple=False).view(-1).tolist()
        for pos in kept_positions:
            per_src.setdefault(int(src[pos].item()), []).append(pos)
        limited = torch.zeros_like(local_keep)
        for positions in per_src.values():
            positions = sorted(positions, key=lambda i: float(sim[i].item()), reverse=True)[:topk]
            limited[positions] = True
        local_keep &= limited

    full_keep = torch.ones(edge_type.numel(), device=edge_type.device, dtype=torch.bool)
    same_positions = torch.nonzero(same_mask, as_tuple=False).view(-1)
    full_keep[same_positions] = local_keep
    return _filter_edges(payload, full_keep)


def _drop_edge_names(payload: dict[str, Any], names: set[str], stats: GraphFilterStats) -> dict[str, Any]:
    edge_mapping = payload.get("edge_type_mapping", {})
    drop_ids = {
        edge_id
        for name in names
        for edge_id in [_edge_name_to_id(edge_mapping, name)]
        if edge_id is not None
    }
    if not drop_ids:
        return payload
    edge_type = payload["edge_type"].long()
    keep = torch.ones(edge_type.numel(), device=edge_type.device, dtype=torch.bool)
    for edge_id in drop_ids:
        keep &= edge_type != edge_id
    return _filter_edges(payload, keep)


def _filter_edges(payload: dict[str, Any], keep: torch.Tensor) -> dict[str, Any]:
    if bool(keep.all()):
        return payload
    filtered = dict(payload)
    filtered["edge_index"] = payload["edge_index"][:, keep]
    for key in ("edge_type", "edge_weight"):
        value = payload.get(key)
        if isinstance(value, torch.Tensor) and value.shape[0] == keep.numel():
            filtered[key] = value[keep]
    return filtered


def _limit_ontology_edges(payload: dict[str, Any], ontology_cfg: Any) -> dict[str, Any]:
    hop_limit = int(_cfg_get(ontology_cfg, "edge_hop_limit", 1))
    if hop_limit >= 1:
        return payload
    return _drop_edge_names(payload, {"ontology"}, GraphFilterStats())


def _apply_edge_dropout(payload: dict[str, Any], hierarchy_cfg: Any, ontology_cfg: Any) -> dict[str, Any]:
    edge_type = payload["edge_type"].long()
    if edge_type.numel() == 0:
        return payload
    edge_mapping = payload.get("edge_type_mapping", {})
    keep = torch.ones(edge_type.numel(), device=edge_type.device, dtype=torch.bool)

    parent_id = _edge_name_to_id(edge_mapping, "parent")
    mention_id = _edge_name_to_id(edge_mapping, "mention")
    ontology_id = _edge_name_to_id(edge_mapping, "ontology")

    h_drop = float(_cfg_get(hierarchy_cfg, "edge_dropout", 0.0) or 0.0)
    o_drop = float(_cfg_get(ontology_cfg, "edge_dropout", 0.0) or 0.0)
    if h_drop > 0:
        base_keep = torch.ones_like(keep)
        if parent_id is not None:
            base_keep &= edge_type != parent_id
        if mention_id is not None:
            base_keep &= edge_type != mention_id
        if ontology_id is not None:
            base_keep &= edge_type != ontology_id
        rand = torch.rand(edge_type.numel(), device=edge_type.device)
        keep &= (~base_keep) | (rand >= h_drop)
    if o_drop > 0 and ontology_id is not None:
        ontology_mask = edge_type == ontology_id
        rand = torch.rand(edge_type.numel(), device=edge_type.device)
        keep &= (~ontology_mask) | (rand >= o_drop)
    return _filter_edges(payload, keep)


def _count_edges(payload: dict[str, Any], stats: GraphFilterStats) -> None:
    edge_type = payload.get("edge_type")
    if not isinstance(edge_type, torch.Tensor):
        return
    mapping = payload.get("edge_type_mapping", {})
    parent_id = _edge_name_to_id(mapping, "parent")
    ontology_id = _edge_name_to_id(mapping, "ontology")
    if parent_id is not None:
        stats.hierarchy_edges = int((edge_type.long() == parent_id).sum().item())
    if ontology_id is not None:
        stats.ontology_edges = int((edge_type.long() == ontology_id).sum().item())
    stats.edges_after = int(edge_type.numel())


def prepare_text_graph_payload(
    payload: dict[str, Any],
    *,
    hierarchy_cfg: Any = None,
    ontology_cfg: Any = None,
    graph_json_path: str = "",
    training: bool = False,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Return a noise-aware graph payload without mutating the dataset cache."""
    if not isinstance(payload, dict) or "edge_index" not in payload or "edge_type" not in payload:
        return payload, GraphFilterStats().as_dict()
    if not bool(_cfg_get(hierarchy_cfg, "enabled", False)) and not bool(_cfg_get(ontology_cfg, "enabled", False)):
        stats = GraphFilterStats(
            edges_before=int(payload["edge_type"].numel()),
            edges_after=int(payload["edge_type"].numel()),
        )
        _count_edges(payload, stats)
        return payload, stats.as_dict()

    out = dict(payload)
    stats = GraphFilterStats(edges_before=int(out["edge_type"].numel()))

    keep_nodes, node_stats = _build_keep_nodes(out, ontology_cfg, hierarchy_cfg, graph_json_path)
    stats.concept_total = node_stats.concept_total
    stats.concept_kept = node_stats.concept_kept
    stats.concept_filtered = node_stats.concept_filtered
    out = _filter_nodes(out, keep_nodes)

    if not bool(_cfg_get(hierarchy_cfg, "use_section_edges", True)):
        out = _drop_edge_names(out, {"parent"}, stats)
    out = _prune_sentence_similarity_edges(out, hierarchy_cfg, stats)

    if not bool(_cfg_get(ontology_cfg, "enabled", False)):
        out = _drop_edge_names(out, {"mention", "ontology"}, stats)
    else:
        out = _limit_ontology_edges(out, ontology_cfg)

    if training:
        out = _apply_edge_dropout(out, hierarchy_cfg, ontology_cfg)

    if out["edge_type"].numel() == 0:
        # Keep the graph encoder safe: add a self parent-like edge on node 0.
        out["edge_index"] = torch.zeros((2, 1), device=out["node_type"].device, dtype=torch.long)
        out["edge_type"] = torch.zeros((1,), device=out["node_type"].device, dtype=torch.long)
        out["edge_weight"] = torch.ones((1,), device=out["node_type"].device, dtype=torch.float32)

    _count_edges(out, stats)
    out["graph_filter_stats"] = stats.as_dict()
    return out, stats.as_dict()
