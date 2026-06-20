"""Small ViT-style demo backbone and NeuroFuser-style decoder."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from neurofixer.modules import EncodingGate, EncodingModule, FusionBridge


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int = 3, embed_dim: int = 128, patch_size: int = 8) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class SimpleViTGridEncoder(nn.Module):
    """Grid-preserving ViT-like encoder using PyTorch Transformer blocks.

    It keeps the demo installable without timm or HuggingFace dependencies.
    """

    def __init__(self, in_channels: int = 3, embed_dim: int = 128, patch_size: int = 8, depth: int = 4, heads: int = 4) -> None:
        super().__init__()
        self.patch = PatchEmbed(in_channels, embed_dim, patch_size)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=heads,
                dim_feedforward=embed_dim * 4,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            for _ in range(depth)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(depth)])
        self.channels = tuple([embed_dim] * depth)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        grid = self.patch(x)
        b, c, h, w = grid.shape
        tokens = grid.flatten(2).transpose(1, 2)
        feats = []
        for block, norm in zip(self.blocks, self.norms):
            tokens = block(tokens)
            z = norm(tokens).transpose(1, 2).reshape(b, c, h, w).contiguous()
            feats.append(z)
        return feats


class NeuroFuserBasicViT(nn.Module):
    """Buildable ViT NeuroFixer demo using EG, EM, and FBr."""

    def __init__(self, num_classes: int = 19, in_channels: int = 3, latent_dim: int = 128, patch_size: int = 8) -> None:
        super().__init__()
        self.encoder = SimpleViTGridEncoder(in_channels=in_channels, embed_dim=latent_dim, patch_size=patch_size)
        ch = self.encoder.channels
        self.gates = nn.ModuleList([EncodingGate(c, num_classes=None) for c in ch])
        self.align = nn.ModuleList([EncodingModule(c, latent_dim=latent_dim) for c in ch])
        self.fuse = nn.ModuleList([FusionBridge([latent_dim, latent_dim], latent_dim) for _ in range(len(ch) - 1)])
        self.head = nn.Conv2d(latent_dim, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        image_size = x.shape[-2:]
        feats = [gate(f) for gate, f in zip(self.gates, self.encoder(x))]
        latent = [em(f) for em, f in zip(self.align, feats)]
        y = latent[-1]
        for skip, fuse in zip(reversed(latent[:-1]), self.fuse):
            y = fuse([y, skip])
        logits = self.head(y)
        return F.interpolate(logits, size=image_size, mode="bilinear", align_corners=False)


def build_neurofuser_basic_vit(num_classes: int = 19, in_channels: int = 3, latent_dim: int = 128, patch_size: int = 8) -> NeuroFuserBasicViT:
    return NeuroFuserBasicViT(num_classes=num_classes, in_channels=in_channels, latent_dim=latent_dim, patch_size=patch_size)
