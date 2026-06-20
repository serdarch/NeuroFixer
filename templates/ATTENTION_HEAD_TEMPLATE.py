"""Template for a NeuroFixer attention head or fusion module.

Copy this file when proposing a new attention module.

Expected style:
- PyTorch only unless optional extras are clearly separated.
- Clear input/output tensor shapes.
- CPU smoke-test compatible.
- Citation or attribution when based on a paper/preprint.
"""

from __future__ import annotations

import torch
from torch import nn


class ExampleAttentionHead(nn.Module):
    """Example attention head template.

    Parameters
    ----------
    channels:
        Number of input and output channels.

    Input shape
    -----------
    x: torch.Tensor
        Shape [B, C, H, W].

    Output shape
    ------------
    y: torch.Tensor
        Shape [B, C, H, W].
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x with shape [B, C, H, W], got {tuple(x.shape)}")
        return x + self.proj(x) * self.gate(x)
