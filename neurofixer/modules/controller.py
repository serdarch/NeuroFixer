"""Resource-aware neuromodulation utilities for NeuroFixer.

The controller is intentionally lightweight and dependency-free beyond PyTorch.
It emits differentiable gates and normalized dilation/branch weights that can be
shared by CNN and ViT-style backbones.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ControllerConfig:
    """Configuration for the neuromodulation controller.

    Args:
        channels: Feature dimension used to summarize the current stage.
        heads: Number of attention heads requested by the caller.
        dilations: Candidate dilation radii for grouped attention modulation.
        budget_gb: Soft memory budget metadata. It is not used to allocate memory;
            it conditions gate temperature so experiments can keep a stable budget.
        min_temperature: Lower bound for sigmoid/softmax temperature.
    """

    channels: int
    heads: int = 4
    dilations: Tuple[int, ...] = (1, 2, 3)
    budget_gb: float = 8.0
    min_temperature: float = 0.35


class NeuromodulationController(nn.Module):
    """Differentiable controller for gates, branch weights, and dilation mixtures.

    The module turns a feature tensor ``[B, C, H, W]`` into small control vectors.
    It avoids custom kernels, FlashAttention, MMCV, Detectron2, and external repos,
    so the same component can be installed as a simple PyPI package later.
    """

    def __init__(self, config: ControllerConfig):
        super().__init__()
        self.config = config
        hidden = max(16, config.channels // 4)
        self.summary = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.LayerNorm(config.channels),
            nn.Linear(config.channels, hidden),
            nn.GELU(),
        )
        self.gate_head = nn.Linear(hidden, config.channels)
        self.dilation_head = nn.Linear(hidden, len(config.dilations))
        self.head_head = nn.Linear(hidden, config.heads)
        self.temperature_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Controller expects [B,C,H,W], got {tuple(x.shape)}")
        z = self.summary(x)
        temperature = F.softplus(self.temperature_head(z)) + self.config.min_temperature
        # Budget-aware temperature scaling: larger budgets can keep gates slightly sharper.
        budget_scale = torch.as_tensor(self.config.budget_gb, dtype=x.dtype, device=x.device).clamp_min(1.0)
        temperature = temperature / budget_scale.sqrt().clamp_min(1.0)
        channel_gate = torch.sigmoid(self.gate_head(z) / temperature).view(x.shape[0], x.shape[1], 1, 1)
        dilation_mix = torch.softmax(self.dilation_head(z) / temperature, dim=-1)
        head_mix = torch.softmax(self.head_head(z) / temperature, dim=-1)
        return {
            "channel_gate": channel_gate,
            "dilation_mix": dilation_mix,
            "head_mix": head_mix,
            "temperature": temperature,
        }


def normalize_branches(scores: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Stable branch-wise normalization used by FusionBridge.

    Args:
        scores: Branch scores, normally ``[B, K, 1, H, W]``.
        dim: Branch dimension.
    """
    return torch.softmax(scores - scores.amax(dim=dim, keepdim=True), dim=dim)
