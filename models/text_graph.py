from __future__ import annotations

import torch
import torch.nn as nn


class RelationalGraphLayer(nn.Module):
    """A lightweight relation-aware message passing block for text hierarchy graphs."""

    def __init__(self, dim: int = 512, num_relations: int = 4, dropout: float = 0.1):
        super().__init__()
        self.self_proj = nn.Linear(dim, dim)
        self.rel_projs = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_relations)])
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        src, dst = edge_index

        agg = torch.zeros_like(x)
        deg = torch.zeros(n, device=x.device, dtype=x.dtype)

        for rel_id, proj in enumerate(self.rel_projs):
            rel_mask = edge_type == rel_id
            if not torch.any(rel_mask):
                continue
            rel_src = src[rel_mask]
            rel_dst = dst[rel_mask]
            msg = proj(x[rel_src])
            agg.index_add_(0, rel_dst, msg)
            deg.index_add_(0, rel_dst, torch.ones(rel_dst.shape[0], device=x.device, dtype=x.dtype))

        agg = agg / deg.clamp_min(1.0).unsqueeze(1)
        h = self.self_proj(x) + agg
        x = self.norm1(x + self.dropout(h))
        x = self.norm2(x + self.ffn(x))
        return x


class TextHierarchyGraphEncoder(nn.Module):
    """
    Graph-aware encoder for Document -> Section -> Sentence hierarchy graphs.

    It explicitly uses:
    - node_type via learned node-type embeddings
    - edge_index / edge_type via relation-aware message passing
    """

    def __init__(
        self,
        dim: int = 512,
        num_node_types: int = 3,
        num_base_relations: int = 2,
        num_layers: int = 1,
        dropout: float = 0.05,
        use_next_edges: bool = True,
    ):
        super().__init__()
        self.use_next_edges = bool(use_next_edges)
        self.default_allowed_forward_relations = tuple(range(num_base_relations)) if self.use_next_edges else (0,)
        self.num_base_relations = int(num_base_relations)
        self.num_total_relations = self.num_base_relations * 2  # forward + reverse
        self.node_type_embed = nn.Embedding(num_node_types, dim)
        self.input_norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(
            [RelationalGraphLayer(dim=dim, num_relations=self.num_total_relations, dropout=dropout) for _ in range(num_layers)]
        )
        self.output_norm = nn.LayerNorm(dim)
        # Start from a conservative regime: graph updates act as a small residual
        # correction on top of the original hierarchy node features.
        self.graph_mix_logit = nn.Parameter(torch.tensor(-2.0))
        # Favor the document node while still letting other nodes contribute.
        self.doc_mix_logit = nn.Parameter(torch.tensor(1.0))

    def _augment_edges(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        allowed_forward_relation_ids: list[int] | tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        relation_ids = (
            tuple(int(x) for x in allowed_forward_relation_ids)
            if allowed_forward_relation_ids is not None
            else self.default_allowed_forward_relations
        )
        allowed = torch.zeros_like(edge_type, dtype=torch.bool)
        for rel_id in relation_ids:
            allowed |= edge_type == rel_id
        edge_index = edge_index[:, allowed]
        edge_type = edge_type[allowed]
        if edge_type.numel() == 0:
            raise RuntimeError("No graph edges remain after relation filtering.")

        src, dst = edge_index
        reverse_edge_index = torch.stack([dst, src], dim=0)
        reverse_edge_type = edge_type + self.num_base_relations
        edge_index_full = torch.cat([edge_index, reverse_edge_index], dim=1)
        edge_type_full = torch.cat([edge_type, reverse_edge_type], dim=0)
        return edge_index_full, edge_type_full

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        node_type: torch.Tensor,
        allowed_forward_relation_ids: list[int] | tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base = self.input_norm(node_features + self.node_type_embed(node_type))
        x = base
        edge_index_full, edge_type_full = self._augment_edges(
            edge_index,
            edge_type,
            allowed_forward_relation_ids=allowed_forward_relation_ids,
        )
        for layer in self.layers:
            x = layer(x, edge_index_full, edge_type_full)
        x = self.output_norm(x)
        graph_mix = torch.sigmoid(self.graph_mix_logit)
        x = base + graph_mix * (x - base)
        doc_mix = torch.sigmoid(self.doc_mix_logit)
        pooled = doc_mix * x[0] + (1.0 - doc_mix) * x.mean(dim=0)
        return x, pooled
