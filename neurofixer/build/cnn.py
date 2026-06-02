"""Small CNN demo backbone and NeuroFuser-style decoder."""
from __future__ import annotations

from typing import List, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from neurofixer.modules import EncodingGate, EncodingModule, FusionBridge


class SimpleCNNEncoder(nn.Module):
    """Dependency-free CNN encoder for public demos and smoke tests."""

    def __init__(self, in_channels: int = 3, channels: Sequence[int] = (32, 64, 128, 256)) -> None:
        super().__init__()
        layers = []
        prev = in_channels
        for i, c in enumerate(channels):
            stride = 1 if i == 0 else 2
            layers.append(nn.Sequential(
                nn.Conv2d(prev, c, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(c),
                nn.GELU(),
                nn.Conv2d(c, c, 3, padding=1, bias=False),
                nn.BatchNorm2d(c),
                nn.GELU(),
            ))
            prev = c
        self.stages = nn.ModuleList(layers)
        self.channels = tuple(channels)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats


class CNNNeuroFuserDemo(nn.Module):
    """Buildable CNN NeuroFixer demo using EG, EM, and FBr."""

    def __init__(self, num_classes: int = 19, in_channels: int = 3, latent_dim: int = 128) -> None:
        super().__init__()
        self.encoder = SimpleCNNEncoder(in_channels=in_channels)
        ch = self.encoder.channels
        self.gates = nn.ModuleList([EncodingGate(c, num_classes=None) for c in ch])
        self.align = nn.ModuleList([EncodingModule(c, latent_dim=latent_dim) for c in ch])
        self.fuse43 = FusionBridge([latent_dim, latent_dim], latent_dim)
        self.fuse32 = FusionBridge([latent_dim, latent_dim], latent_dim)
        self.fuse21 = FusionBridge([latent_dim, latent_dim], latent_dim)
        self.head = nn.Conv2d(latent_dim, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        image_size = x.shape[-2:]
        feats = [gate(f) for gate, f in zip(self.gates, self.encoder(x))]
        latent = [em(f) for em, f in zip(self.align, feats)]
        y = latent[-1]
        y = self.fuse43([F.interpolate(y, size=latent[-2].shape[-2:], mode="bilinear", align_corners=False), latent[-2]])
        y = self.fuse32([F.interpolate(y, size=latent[-3].shape[-2:], mode="bilinear", align_corners=False), latent[-3]])
        y = self.fuse21([F.interpolate(y, size=latent[-4].shape[-2:], mode="bilinear", align_corners=False), latent[-4]])
        logits = self.head(y)
        return F.interpolate(logits, size=image_size, mode="bilinear", align_corners=False)


def build_cnn_neurofixer(num_classes: int = 19, in_channels: int = 3, latent_dim: int = 128) -> CNNNeuroFuserDemo:
    return CNNNeuroFuserDemo(num_classes=num_classes, in_channels=in_channels, latent_dim=latent_dim)
