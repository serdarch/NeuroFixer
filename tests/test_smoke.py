import torch

from neurofixer import (
    EncodingGate,
    EncodingModule,
    FusionBridge,
    build_cnn_neurofixer,
    build_vit_neurofixer,
    build_neurofuser_basic_cnn,
    build_neurofuser_basic_vit,
)
from neurofixer.models.registry import build_model, list_models
from neurofixer.models.neurofuser_basic.blocks import (
    EncodingGate as BasicEG,
    EncodingModule as BasicEM,
    FusionBridge as BasicFBr,
)


def test_neurofuser_basic_blocks_are_exposed():
    assert BasicEG is EncodingGate
    assert BasicEM is EncodingModule
    assert BasicFBr is FusionBridge


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


def test_neurofuser_basic_builders_forward():
    x = torch.randn(1, 3, 64, 64)

    cnn = build_neurofuser_basic_cnn(num_classes=5, latent_dim=32)
    assert cnn(x).shape == (1, 5, 64, 64)

    vit = build_neurofuser_basic_vit(num_classes=5, latent_dim=32, patch_size=8)
    assert vit(x).shape == (1, 5, 64, 64)


def test_backward_compatible_builders_forward():
    x = torch.randn(1, 3, 64, 64)

    cnn = build_cnn_neurofixer(num_classes=5, latent_dim=32)
    assert cnn(x).shape == (1, 5, 64, 64)

    vit = build_vit_neurofixer(num_classes=5, latent_dim=32, patch_size=8)
    assert vit(x).shape == (1, 5, 64, 64)


def test_registry():
    names = list_models()
    assert "neurofuser_basic_cnn" in names
    assert "neurofuser_basic_vit" in names

    x = torch.randn(1, 3, 64, 64)
    model = build_model("neurofuser_basic_cnn", num_classes=5, latent_dim=32)
    assert model(x).shape == (1, 5, 64, 64)
