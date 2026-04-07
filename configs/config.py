from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import os

try:
    import yaml
except Exception as e:
    yaml = None

@dataclass
class DataConfig:
    split_file: str
    label_file: str
    image_dir: str
    text_dir: str
    text_mode: str = "sentence_pt"
    text_graph_feature: str = "node_features"
    text_use_graph_structure: bool = False
    sentence_text_dir: str = ""
    graph_text_dir: str = ""


@dataclass
class GraphConfig:
    num_nodes_m: int = 64
    topk_k: int = 32
    kmeans_iters: int = 10
    max_patches_for_kmeans: Optional[int] = 2048


@dataclass
class ModelConfig:
    feat_dim: int = 512

    mapper_hidden: int = 1024
    mapper_depth: int = 2
    mapper_dropout: float = 0.10

    attn_pool_hidden: int = 256
    attn_pool_dropout: float = 0.10

    moe_heads: int = 8
    moe_dropout: float = 0.10
    moe_tau_start: float = 1.0
    moe_tau_min: float = 0.1
    moe_tau_decay: float = 0.95
    moe_hard_start_epoch: int = 10

    classifier_hidden: int = 256
    classifier_dropout: float = 0.20

    text_graph_layers: int = 1
    text_graph_dropout: float = 0.05
    text_graph_use_next_edges: bool = True
    sentence_local_layers: int = 1
    sentence_local_heads: int = 8
    hier_readout_hidden: int = 256
    hier_readout_dropout: float = 0.10
    hier_readout_attention_init: float = -0.5
    text_dual_fusion_dropout: float = 0.10


@dataclass
class LossConfig:
    alpha_txt: float = 0.5
    beta_node: float = 0.1
    gamma_topo: float = 0.1
    dual_text_gate_reg_weight: float = 0.0
    dual_text_graph_weight_target: float = 0.2

    mmd_num_kernels: int = 3
    mmd_sigma_multipliers: List[float] = None
    mmd_unbiased: bool = True
    mmd_clamp_nonneg: bool = True


@dataclass
class TrainConfig:
    num_epochs: int = 20
    batch_size: int = 2
    num_workers: int = 4

    lr: float = 1e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0

    early_stop_patience: int = 20
    early_stop_min_delta: float = 1e-4


@dataclass
class OutputConfig:
    exp_dir: str = "experiments/default"
    save_best_by: str = "val_auc"  # "val_auc" or "val_avg"


@dataclass
class Config:
    seed: int
    data: DataConfig
    graph: GraphConfig
    model: ModelConfig
    loss: LossConfig
    train: TrainConfig
    output: OutputConfig


def _require_yaml():
    if yaml is None:
        raise ImportError(
            "PyYAML is required for configs/config.py. Install with: pip install pyyaml"
        )


def _expand(p: str) -> str:
    if p is None or str(p).strip() == "":
        return ""
    return os.path.abspath(os.path.expanduser(p))


def get_config(path: str) -> Config:
    _require_yaml()
    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    data = raw["data"]
    graph = raw.get("graph", {})
    model = raw.get("model", {})
    loss = raw.get("loss", {})
    train = raw.get("train", {})
    output = raw.get("output", {})

    cfg = Config(
        seed=int(raw.get("seed", 42)),
        data=DataConfig(
            split_file=_expand(data["split_file"]),
            label_file=_expand(data["label_file"]),
            image_dir=_expand(data["image_dir"]),
            text_dir=_expand(data["text_dir"]),
            text_mode=str(data.get("text_mode", "sentence_pt")),
            text_graph_feature=str(data.get("text_graph_feature", "node_features")),
            text_use_graph_structure=bool(data.get("text_use_graph_structure", False)),
            sentence_text_dir=_expand(data.get("sentence_text_dir", data.get("text_dir", ""))),
            graph_text_dir=_expand(data.get("graph_text_dir", data.get("text_dir", ""))),
        ),
        graph=GraphConfig(
            num_nodes_m=int(graph.get("num_nodes_m", 64)),
            topk_k=int(graph.get("topk_k", 32)),
            kmeans_iters=int(graph.get("kmeans_iters", 10)),
            max_patches_for_kmeans=graph.get("max_patches_for_kmeans", 2048),
        ),
        model=ModelConfig(
            feat_dim=int(model.get("feat_dim", 512)),
            mapper_hidden=int(model.get("mapper_hidden", 1024)),
            mapper_depth=int(model.get("mapper_depth", 2)),
            mapper_dropout=float(model.get("mapper_dropout", 0.10)),
            attn_pool_hidden=int(model.get("attn_pool_hidden", 256)),
            attn_pool_dropout=float(model.get("attn_pool_dropout", 0.10)),
            moe_heads=int(model.get("moe_heads", 8)),
            moe_dropout=float(model.get("moe_dropout", 0.10)),
            moe_tau_start=float(model.get("moe_tau_start", 1.0)),
            moe_tau_min=float(model.get("moe_tau_min", 0.1)),
            moe_tau_decay=float(model.get("moe_tau_decay", 0.95)),
            moe_hard_start_epoch=int(model.get("moe_hard_start_epoch", 10)),
            classifier_hidden=int(model.get("classifier_hidden", 256)),
            classifier_dropout=float(model.get("classifier_dropout", 0.20)),
            text_graph_layers=int(model.get("text_graph_layers", 1)),
            text_graph_dropout=float(model.get("text_graph_dropout", 0.05)),
            text_graph_use_next_edges=bool(model.get("text_graph_use_next_edges", True)),
            sentence_local_layers=int(model.get("sentence_local_layers", 1)),
            sentence_local_heads=int(model.get("sentence_local_heads", 8)),
            hier_readout_hidden=int(model.get("hier_readout_hidden", 256)),
            hier_readout_dropout=float(model.get("hier_readout_dropout", 0.10)),
            hier_readout_attention_init=float(model.get("hier_readout_attention_init", -0.5)),
            text_dual_fusion_dropout=float(model.get("text_dual_fusion_dropout", 0.10)),
        ),
        loss=LossConfig(
            alpha_txt=float(loss.get("alpha_txt", 0.5)),
            beta_node=float(loss.get("beta_node", 0.1)),
            gamma_topo=float(loss.get("gamma_topo", 0.1)),
            dual_text_gate_reg_weight=float(loss.get("dual_text_gate_reg_weight", 0.0)),
            dual_text_graph_weight_target=float(loss.get("dual_text_graph_weight_target", 0.2)),
            mmd_num_kernels=int(loss.get("mmd_num_kernels", 3)),
            mmd_sigma_multipliers=loss.get("mmd_sigma_multipliers", [0.5, 1.0, 2.0]),
            mmd_unbiased=bool(loss.get("mmd_unbiased", True)),
            mmd_clamp_nonneg=bool(loss.get("mmd_clamp_nonneg", True)),
        ),
        train=TrainConfig(
            num_epochs=int(train.get("num_epochs", 20)),
            batch_size=int(train.get("batch_size", 2)),
            num_workers=int(train.get("num_workers", 4)),
            lr=float(train.get("lr", 1e-4)),
            weight_decay=float(train.get("weight_decay", 1e-2)),
            grad_clip=float(train.get("grad_clip", 1.0)),
            early_stop_patience=int(train.get("early_stop_patience", 20)),
            early_stop_min_delta=float(train.get("early_stop_min_delta", 1e-4)),
        ),
        output=OutputConfig(
            exp_dir=_expand(output.get("exp_dir", "experiments/default")),
            save_best_by=str(output.get("save_best_by", "val_auc")),
        ),
    )
    return cfg
