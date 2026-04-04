from __future__ import annotations
import torch
import torch.nn as nn


class AttentionPooling(nn.Module):

    def __init__(self, dim: int = 512, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.Tanh()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        h = self.drop(self.act(self.fc1(x)))          # [B, L, hidden]
        scores = self.fc2(h).squeeze(-1)              # [B, L]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=1)           # [B, L]
        pooled = torch.bmm(attn.unsqueeze(1), x).squeeze(1)  # [B, D]
        return pooled, attn
