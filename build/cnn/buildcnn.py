"""Compatibility builder for CNN NeuroFixer demo."""
from neurofixer.build import build_cnn_neurofixer


def build_model(num_classes: int = 19, latent_dim: int = 128):
    return build_cnn_neurofixer(num_classes=num_classes, latent_dim=latent_dim)


if __name__ == "__main__":
    import torch
    model = build_model()
    y = model(torch.randn(1, 3, 128, 128))
    print(tuple(y.shape))
