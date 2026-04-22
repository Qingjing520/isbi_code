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

    def __init__(
        self,
        dim: int = 512,
        hidden: int = 256,
        dropout: float = 0.1,
        attention_init: float = -0.5,
        use_section_title_embedding: bool = False,
        num_section_title_types: int = 0,
    ):
        super().__init__()
        self.section_pool = ScalarAttentionPool(dim=dim, hidden=hidden, dropout=dropout)
        self.document_pool = ScalarAttentionPool(dim=dim, hidden=hidden, dropout=dropout)
        self.use_section_title_embedding = bool(use_section_title_embedding)
        self.section_title_embedding = (
            nn.Embedding(num_section_title_types, dim)
            if self.use_section_title_embedding and num_section_title_types > 0
            else None
        )
        # Attention can be sharp on noisy pathology sections, so mix it with a mean-pooling fallback.
        self.section_attention_mix_logit = nn.Parameter(torch.tensor(float(attention_init)))
        self.document_attention_mix_logit = nn.Parameter(torch.tensor(float(attention_init)))

    def _mix_attention_with_mean(
        self,
        attention_vec: torch.Tensor,
        mean_vec: torch.Tensor,
        mix_logit: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mix = torch.sigmoid(mix_logit).to(dtype=attention_vec.dtype, device=attention_vec.device)
        return mix * attention_vec + (1.0 - mix) * mean_vec, mix

    def forward(
        self,
        sentence_nodes: torch.Tensor,
        section_spans: torch.Tensor,
        section_title_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        if sentence_nodes.dim() != 2:
            raise ValueError(f"Expected [N, D] sentence nodes, got shape {tuple(sentence_nodes.shape)}")

        section_vecs: list[torch.Tensor] = []
        section_sentence_attn: list[torch.Tensor] = []

        if section_spans.numel() == 0:
            pooled_doc, doc_attn = self.document_pool(sentence_nodes)
            pooled_doc, doc_mix = self._mix_attention_with_mean(
                attention_vec=pooled_doc,
                mean_vec=sentence_nodes.mean(dim=0),
                mix_logit=self.document_attention_mix_logit,
            )
            return sentence_nodes.new_zeros((0, sentence_nodes.shape[1])), pooled_doc, {
                "section_sentence_attn": [],
                "document_section_attn": doc_attn,
                "section_attention_mix": torch.sigmoid(self.section_attention_mix_logit).detach(),
                "document_attention_mix": doc_mix.detach(),
                "section_title_embedding_used": False,
            }

        for start_end in section_spans.tolist():
            start, end = int(start_end[0]), int(start_end[1])
            chunk = sentence_nodes[start:end]
            if chunk.numel() == 0:
                continue
            pooled_sec, sent_attn = self.section_pool(chunk)
            pooled_sec, _section_mix = self._mix_attention_with_mean(
                attention_vec=pooled_sec,
                mean_vec=chunk.mean(dim=0),
                mix_logit=self.section_attention_mix_logit,
            )
            section_vecs.append(pooled_sec)
            section_sentence_attn.append(sent_attn)

        if not section_vecs:
            pooled_doc, doc_attn = self.document_pool(sentence_nodes)
            pooled_doc, doc_mix = self._mix_attention_with_mean(
                attention_vec=pooled_doc,
                mean_vec=sentence_nodes.mean(dim=0),
                mix_logit=self.document_attention_mix_logit,
            )
            return sentence_nodes.new_zeros((0, sentence_nodes.shape[1])), pooled_doc, {
                "section_sentence_attn": [],
                "document_section_attn": doc_attn,
                "section_attention_mix": torch.sigmoid(self.section_attention_mix_logit).detach(),
                "document_attention_mix": doc_mix.detach(),
                "section_title_embedding_used": False,
            }

        section_tensor = torch.stack(section_vecs, dim=0)
        section_title_used = False
        if self.section_title_embedding is not None and section_title_ids is not None and section_title_ids.numel() > 0:
            title_ids = section_title_ids.to(device=section_tensor.device, dtype=torch.long).view(-1)
            if title_ids.numel() < section_tensor.shape[0]:
                title_ids = torch.cat(
                    [
                        title_ids,
                        title_ids.new_zeros(section_tensor.shape[0] - title_ids.numel()),
                    ],
                    dim=0,
                )
            title_ids = title_ids[: section_tensor.shape[0]].clamp(
                min=0,
                max=self.section_title_embedding.num_embeddings - 1,
            )
            section_tensor = section_tensor + self.section_title_embedding(title_ids)
            section_title_used = True
        pooled_doc, doc_attn = self.document_pool(section_tensor)
        pooled_doc, doc_mix = self._mix_attention_with_mean(
            attention_vec=pooled_doc,
            mean_vec=section_tensor.mean(dim=0),
            mix_logit=self.document_attention_mix_logit,
        )
        return section_tensor, pooled_doc, {
            "section_sentence_attn": section_sentence_attn,
            "document_section_attn": doc_attn,
            "section_attention_mix": torch.sigmoid(self.section_attention_mix_logit).detach(),
            "document_attention_mix": doc_mix.detach(),
            "section_title_embedding_used": section_title_used,
        }


class DualTextFusion(nn.Module):
    """Fuse sentence baseline text vector and hierarchy-graph text vector with a gate."""

    def __init__(self, dim: int = 512, dropout: float = 0.1, graph_weight_max: float = 1.0):
        super().__init__()
        if not 0.0 <= graph_weight_max <= 1.0:
            raise ValueError("graph_weight_max must be in [0, 1].")
        self.graph_weight_max = float(graph_weight_max)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, sentence_vec: torch.Tensor, graph_vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sentence_gate = torch.sigmoid(self.gate(torch.cat([sentence_vec, graph_vec], dim=-1)))
        graph_weight = 1.0 - sentence_gate
        if self.graph_weight_max < 1.0:
            graph_weight = graph_weight * self.graph_weight_max
            sentence_gate = 1.0 - graph_weight
        fused = sentence_gate * sentence_vec + graph_weight * graph_vec
        return self.out_norm(fused), sentence_gate


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
        readout_attention_init: float = -0.5,
        use_section_title_embedding: bool = False,
        num_section_title_types: int = 0,
    ):
        super().__init__()
        self.local_encoder = SentenceLocalEncoder(
            dim=dim,
            n_heads=sentence_local_heads,
            num_layers=sentence_local_layers,
            dropout=dropout,
        )
        self.readout = HierarchicalReadout(
            dim=dim,
            hidden=hidden,
            dropout=dropout,
            attention_init=readout_attention_init,
            use_section_title_embedding=use_section_title_embedding,
            num_section_title_types=num_section_title_types,
        )

    def forward(
        self,
        sentence_features: torch.Tensor,
        section_spans: torch.Tensor,
        section_title_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        sentence_nodes = self.local_encoder(sentence_features)
        section_vecs, document_vec, analysis = self.readout(
            sentence_nodes=sentence_nodes,
            section_spans=section_spans,
            section_title_ids=section_title_ids,
        )
        return sentence_nodes, section_vecs, document_vec, analysis
