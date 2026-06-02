import torch

from neurofixer import EncodingGate, EncodingModule, FusionBridge, build_cnn_neurofixer, build_vit_neurofixer


def test_modules_forward():
    x = torch.randn(1, 32, 32, 32)
    eg = EncodingGate(32, num_classes=3)
    y, pred, _ = eg(x, return_prediction=True)
    assert y.shape == x.shape
    assert pred.shape == (1, 3, 32, 32)

    em = EncodingModule(32, latent_dim=64, num_classes=3, out_size=(16, 16))
    y, pred, _ = em(x, return_prediction=True)
    assert y.shape == (1, 64, 16, 16)
    assert pred.shape == (1, 3, 16, 16)

    fb = FusionBridge([64, 32], out_channels=64, num_classes=3)
    y, pred, weights = fb([y, x], return_prediction=True)
    assert y.shape == (1, 64, 16, 16)
    assert pred.shape == (1, 3, 16, 16)
    assert weights.shape[1] == 2


def test_demo_builders_forward():
    x = torch.randn(1, 3, 64, 64)
    cnn = build_cnn_neurofixer(num_classes=5, latent_dim=32)
    assert cnn(x).shape == (1, 5, 64, 64)
    vit = build_vit_neurofixer(num_classes=5, latent_dim=32, patch_size=8)
    assert vit(x).shape == (1, 5, 64, 64)
