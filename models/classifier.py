from __future__ import annotations
import torch
import torch.nn as nn


class StageClassifier(nn.Module):
    def __init__(self, dim: int = 512, hidden: int = 256, dropout: float = 0.2, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor):
        logits = self.net(x)                # [B, 2]
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        return {"logits": logits, "probs": probs, "preds": preds}
