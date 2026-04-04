# models/fusion.py
from __future__ import annotations
import torch
import torch.nn as nn


def diff_softmax(logits: torch.Tensor, tau: float = 1.0, hard: bool = False, dim: int = -1) -> torch.Tensor:

    y_soft = (logits / tau).softmax(dim=dim)
    if not hard:
        return y_soft
    index = y_soft.max(dim, keepdim=True).indices
    y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
    return y_hard - y_soft.detach() + y_soft


class CatSelfAttentionExpert(nn.Module):
    """
    EX1 = SA(concat(F1, F2))
    """
    def __init__(self, dim: int = 512, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, img_vec: torch.Tensor, txt_vec: torch.Tensor) -> torch.Tensor:
        x = torch.stack([img_vec, txt_vec], dim=1)  # [B, 2, D]
        out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + out)
        x = self.norm2(x + self.ffn(x))
        return x.mean(dim=1)  # [B, D]


class AddSelfAttentionExpert(nn.Module):
    """
    EX2 = SA(F1 + F2)
    """
    def __init__(self, dim: int = 512, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, img_vec: torch.Tensor, txt_vec: torch.Tensor) -> torch.Tensor:
        x = (img_vec + txt_vec).unsqueeze(1)  # [B, 1, D]
        out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + out)
        x = self.norm2(x + self.ffn(x))
        return x.squeeze(1)  # [B, D]


class CrossAttentionExpert(nn.Module):
    """
    EX3 = CA(F1, F2)
    """
    def __init__(self, dim: int = 512, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.img_to_txt = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.txt_to_img = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.proj = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, img_vec: torch.Tensor, txt_vec: torch.Tensor) -> torch.Tensor:
        q_img = img_vec.unsqueeze(1)  # [B, 1, D]
        q_txt = txt_vec.unsqueeze(1)  # [B, 1, D]
        out_img, _ = self.img_to_txt(q_img, q_txt, q_txt, need_weights=False)
        out_txt, _ = self.txt_to_img(q_txt, q_img, q_img, need_weights=False)
        fused = torch.cat([out_img.squeeze(1), out_txt.squeeze(1)], dim=-1)  # [B, 2D]
        return self.norm(self.proj(fused))  # [B, D]


class RoutingNetwork(nn.Module):

    def __init__(self, dim: int = 512, n_experts: int = 3, dropout: float = 0.1):
        super().__init__()
        self.w_img = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(dim))
        self.w_txt = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(dim))
        self.out = nn.Linear(dim, n_experts)

    def forward(self, img_vec: torch.Tensor, txt_vec: torch.Tensor) -> torch.Tensor:
        h = self.w_img(img_vec) + self.w_txt(txt_vec)
        return self.out(h)  # [B, E]


class MoEFusion(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        n_heads: int = 8,
        dropout: float = 0.1,
        tau_start: float = 1.0,
        tau_min: float = 0.1,
        decay: float = 0.95,
        hard_start_epoch: int = 10,
    ):
        super().__init__()
        self.experts = nn.ModuleList([
            CatSelfAttentionExpert(dim, n_heads, dropout),
            AddSelfAttentionExpert(dim, n_heads, dropout),
            CrossAttentionExpert(dim, n_heads, dropout),
        ])
        self.router = RoutingNetwork(dim, n_experts=len(self.experts), dropout=dropout)
        self.tau_start = tau_start
        self.tau_min = tau_min
        self.decay = decay
        self.hard_start_epoch = hard_start_epoch
        self._epoch = 0

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def _tau(self) -> float:
        return max(self.tau_start * (self.decay ** self._epoch), self.tau_min)

    def forward(self, img_vec: torch.Tensor, txt_vec: torch.Tensor, hard: bool | None = None):

        logits = self.router(img_vec, txt_vec)  # [B, E]
        if hard is None:
            hard = self.training and (self._epoch >= self.hard_start_epoch)
        gates = diff_softmax(logits, tau=self._tau(), hard=hard, dim=-1)  # [B, E]

        outs = [ex(img_vec, txt_vec) for ex in self.experts]  # list of [B, D]
        outs = torch.stack(outs, dim=1)  # [B, E, D]
        fused = torch.sum(gates.unsqueeze(-1) * outs, dim=1)  # [B, D]
        return fused, gates
