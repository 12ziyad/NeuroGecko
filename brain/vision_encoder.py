from __future__ import annotations

import torch
from torch import nn


class VisionEncoder(nn.Module):
    """Small CNN for the 64x64 head camera stream."""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.out_dim = int(out_dim)
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, self.out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = image
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.ndim != 4:
            raise ValueError(f"VisionEncoder expected 4D image tensor, got shape {tuple(x.shape)}")
        if x.shape[1] not in (1, 3) and x.shape[-1] in (1, 3):
            x = x.permute(0, 3, 1, 2)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if x.shape[1] != 3:
            raise ValueError(f"VisionEncoder expected 3 channels, got shape {tuple(x.shape)}")
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        else:
            x = x.float()
        return self.net(x)
