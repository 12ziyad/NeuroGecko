from __future__ import annotations

import torch
from torch import nn


class AnatomyGraphEncoder(nn.Module):
    """Compact body-state encoder standing in for the later anatomy graph."""

    def __init__(self, input_dim: int, out_dim: int = 96):
        super().__init__()
        if input_dim < 1:
            raise ValueError("AnatomyGraphEncoder input_dim must be positive")
        self.input_dim = int(input_dim)
        self.out_dim = int(out_dim)
        hidden = max(64, min(192, self.input_dim))
        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, body: torch.Tensor) -> torch.Tensor:
        x = body.float()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self.net(x)
