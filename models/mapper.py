from __future__ import annotations
import torch
import torch.nn as nn

class MLPBlock(nn.Module):
    def __init__(self, dim: int = 512, hidden: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class SharedMapper(nn.Module):

    def __init__(self, dim: int = 512, hidden: int = 1024, depth: int = 2, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([MLPBlock(dim, hidden, dropout) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x
