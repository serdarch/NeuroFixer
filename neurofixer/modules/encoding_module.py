"""EncodingModule: latent-grid alignment with grouped dilation modulation."""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .controller import ControllerConfig, NeuromodulationController


class EncodingModule(nn.Module):
    """Aligns a stage feature map to a latent grid and optional dense logits.

    The module combines pointwise projection, depthwise dilation mixtures, and a
    controller-emitted channel gate. It is designed to be safe for public demos:
    no custom CUDA, no external segmentation framework, and no pretrained weights.
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim: int = 256,
        out_size: Optional[Tuple[int, int]] = None,
        num_classes: Optional[int] = None,
        dilations: Sequence[int] = (1, 2, 3),
        heads: int = 4,
        budget_gb: float = 8.0,
    ) -> None:
        super().__init__()
        self.out_size = out_size
        self.dilations = tuple(int(d) for d in dilations)
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, latent_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(latent_dim),
            nn.GELU(),
        )
        self.controller = NeuromodulationController(
            ControllerConfig(channels=latent_dim, heads=heads, dilations=self.dilations, budget_gb=budget_gb)
        )
        self.dilated = nn.ModuleList([
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=d, dilation=d, groups=latent_dim, bias=False)
            for d in self.dilations
        ])
        self.mix = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(latent_dim),
            nn.GELU(),
        )
        self.pred_head = nn.Conv2d(latent_dim, num_classes, kernel_size=1) if num_classes else None

    def forward(self, x: torch.Tensor, out_size: Optional[Tuple[int, int]] = None, return_prediction: bool = False):
        if x.ndim != 4:
            raise ValueError(f"EncodingModule expects [B,C,H,W], got {tuple(x.shape)}")
        z = self.project(x)
        target_size = out_size or self.out_size
        if target_size is not None and tuple(z.shape[-2:]) != tuple(target_size):
            z = F.interpolate(z, size=target_size, mode="bilinear", align_corners=False)
        controls = self.controller(z)
        branches = torch.stack([conv(z) for conv in self.dilated], dim=1)  # [B,K,C,H,W]
        mix = controls["dilation_mix"].view(z.shape[0], len(self.dilations), 1, 1, 1)
        z_dil = (branches * mix).sum(dim=1)
        out = self.mix(z + z_dil * controls["channel_gate"])
        if return_prediction:
            pred = self.pred_head(out) if self.pred_head is not None else None
            return out, pred, controls
        return out
