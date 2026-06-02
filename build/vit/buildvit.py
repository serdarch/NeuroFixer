"""Compatibility builder for ViT NeuroFixer demo."""
from neurofixer.build import build_vit_neurofixer


def build_model(num_classes: int = 19, latent_dim: int = 128, patch_size: int = 8):
    return build_vit_neurofixer(num_classes=num_classes, latent_dim=latent_dim, patch_size=patch_size)


if __name__ == "__main__":
    import torch
    model = build_model()
    y = model(torch.randn(1, 3, 128, 128))
    print(tuple(y.shape))
