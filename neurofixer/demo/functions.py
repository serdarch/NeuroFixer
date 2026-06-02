"""Demo helpers."""
from __future__ import annotations

import torch

from neurofixer.build import build_cnn_neurofixer, build_vit_neurofixer


def build_demo_model(backbone: str = "cnn", num_classes: int = 19, latent_dim: int = 128, patch_size: int = 8):
    if backbone == "cnn":
        return build_cnn_neurofixer(num_classes=num_classes, latent_dim=latent_dim)
    if backbone == "vit":
        return build_vit_neurofixer(num_classes=num_classes, latent_dim=latent_dim, patch_size=patch_size)
    raise ValueError("backbone must be 'cnn' or 'vit'")


@torch.no_grad()
def run_smoke_demo(backbone: str = "cnn", num_classes: int = 19, latent_dim: int = 128, image_size: int = 128, patch_size: int = 8, device: str = "cpu"):
    model = build_demo_model(backbone, num_classes=num_classes, latent_dim=latent_dim, patch_size=patch_size).to(device)
    model.eval()
    x = torch.randn(1, 3, image_size, image_size, device=device)
    y = model(x)
    return {"backbone": backbone, "input_shape": tuple(x.shape), "output_shape": tuple(y.shape)}
