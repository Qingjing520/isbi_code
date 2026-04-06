from __future__ import annotations

import torch
import torch.nn as nn


class ScalarAttentionPool(nn.Module):
    """Attention pooling that works on [L, D] or [B, L, D] inputs."""

    def __init__(self, dim: int = 512, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.Tanh()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        h = self.drop(self.act(self.fc1(x)))
        scores = self.fc2(h).squeeze(-1)
        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=1)
        pooled = torch.bmm(attn.unsqueeze(1), x).squeeze(1)
        if squeeze:
            return pooled.squeeze(0), attn.squeeze(0)
        return pooled, attn


class LocalSentenceEncoderBlock(nn.Module):
    """A lightweight local self-attention block for sentence sequences."""

    def __init__(self, dim: int = 512, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + out)
        x = self.norm2(x + self.ffn(x))
        return x


class SentenceLocalEncoder(nn.Module):
    def __init__(self, dim: int = 512, n_heads: int = 8, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [LocalSentenceEncoderBlock(dim=dim, n_heads=n_heads, dropout=dropout) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"Expected [N, D] sentence features, got shape {tuple(x.shape)}")
        x = x.unsqueeze(0)
        for layer in self.layers:
            x = layer(x)
        return x.squeeze(0)


class HierarchicalReadout(nn.Module):
    """
    Sentence -> Section -> Document hierarchical attention pooling.
    """

    def __init__(self, dim: int = 512, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.section_pool = ScalarAttentionPool(dim=dim, hidden=hidden, dropout=dropout)
        self.document_pool = ScalarAttentionPool(dim=dim, hidden=hidden, dropout=dropout)

    def forward(
        self,
        sentence_nodes: torch.Tensor,
        section_spans: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sentence_nodes.dim() != 2:
            raise ValueError(f"Expected [N, D] sentence nodes, got shape {tuple(sentence_nodes.shape)}")

        section_vecs: list[torch.Tensor] = []

        if section_spans.numel() == 0:
            pooled_doc, _ = self.document_pool(sentence_nodes)
            return sentence_nodes.new_zeros((0, sentence_nodes.shape[1])), pooled_doc

        for start_end in section_spans.tolist():
            start, end = int(start_end[0]), int(start_end[1])
            chunk = sentence_nodes[start:end]
            if chunk.numel() == 0:
                continue
            pooled_sec, _ = self.section_pool(chunk)
            section_vecs.append(pooled_sec)

        if not section_vecs:
            pooled_doc, _ = self.document_pool(sentence_nodes)
            return sentence_nodes.new_zeros((0, sentence_nodes.shape[1])), pooled_doc

        section_tensor = torch.stack(section_vecs, dim=0)
        pooled_doc, _ = self.document_pool(section_tensor)
        return section_tensor, pooled_doc


class DualTextFusion(nn.Module):
    """Fuse sentence baseline text vector and hierarchy-graph text vector with a gate."""

    def __init__(self, dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, sentence_vec: torch.Tensor, graph_vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate = torch.sigmoid(self.gate(torch.cat([sentence_vec, graph_vec], dim=-1)))
        fused = gate * sentence_vec + (1.0 - gate) * graph_vec
        return self.out_norm(fused), gate


class HierTextBranch(nn.Module):
    """
    Graph-derived text branch:
    sentence features -> local sentence encoder -> hierarchical pooling.
    """

    def __init__(
        self,
        dim: int = 512,
        hidden: int = 256,
        dropout: float = 0.1,
        sentence_local_layers: int = 1,
        sentence_local_heads: int = 8,
    ):
        super().__init__()
        self.local_encoder = SentenceLocalEncoder(
            dim=dim,
            n_heads=sentence_local_heads,
            num_layers=sentence_local_layers,
            dropout=dropout,
        )
        self.readout = HierarchicalReadout(dim=dim, hidden=hidden, dropout=dropout)

    def forward(
        self,
        sentence_features: torch.Tensor,
        section_spans: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sentence_nodes = self.local_encoder(sentence_features)
        section_vecs, document_vec = self.readout(sentence_nodes=sentence_nodes, section_spans=section_spans)
        return sentence_nodes, section_vecs, document_vec
