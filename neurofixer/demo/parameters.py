"""Default demo parameters."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DemoParameters:
    backbone: str = "cnn"  # "cnn" or "vit"
    num_classes: int = 19
    in_channels: int = 3
    latent_dim: int = 128
    image_size: int = 128
    patch_size: int = 8
