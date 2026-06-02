"""EncodingGate: per-location gated attention modulation for CNN/ViT features."""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .controller import ControllerConfig, NeuromodulationController


class EncodingGate(nn.Module):
    """Backbone-agnostic feature gate with residual dense prediction support.

    Input and output are ``[B, C, H, W]``. ViT tokens can be converted with
    :func:`neurofixer.utils.token_ops.tokens_to_grid` before calling this module.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        num_classes: Optional[int] = None,
        heads: int = 4,
        budget_gb: float = 8.0,
    ) -> None:
        super().__init__()
        hidden = max(16, channels // reduction)
        self.controller = NeuromodulationController(
            ControllerConfig(channels=channels, heads=heads, budget_gb=budget_gb)
        )
        self.local_gate = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=7, padding=3, groups=1),
            nn.Sigmoid(),
        )
        self.norm = nn.BatchNorm2d(channels)
        self.pred_head = nn.Conv2d(channels, num_classes, kernel_size=1) if num_classes else None

    def forward(self, x: torch.Tensor, return_prediction: bool = False):
        if x.ndim != 4:
            raise ValueError(f"EncodingGate expects [B,C,H,W], got {tuple(x.shape)}")
        controls = self.controller(x)
        channel_gate = controls["channel_gate"]
        local_gate = torch.sigmoid(self.local_gate(x))
        spatial_gate = self.spatial_gate(x)
        gated = x * (0.5 + channel_gate) * (0.5 + local_gate) * (0.5 + spatial_gate)
        out = self.norm(x + gated)
        if return_prediction:
            pred = self.pred_head(out) if self.pred_head is not None else None
            return out, pred, controls
        return out
