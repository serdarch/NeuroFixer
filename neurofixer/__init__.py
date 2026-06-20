"""NeuroFixer: lightweight attention-fusion modules and model-family builders."""

from neurofixer.modules import (
    EncodingGate,
    EncodingModule,
    FusionBridge,
    NeuromodulationController,
)

from neurofixer.models.neurofuser_basic import (
    NeuroFuserBasicCNN,
    NeuroFuserBasicViT,
    build_neurofuser_basic_cnn,
    build_neurofuser_basic_vit,
)

# Backward-compatible v0.1.0 aliases.
build_cnn_neurofixer = build_neurofuser_basic_cnn
build_vit_neurofixer = build_neurofuser_basic_vit

__version__ = "0.1.1"

__all__ = [
    "EncodingGate",
    "EncodingModule",
    "FusionBridge",
    "NeuromodulationController",
    "NeuroFuserBasicCNN",
    "NeuroFuserBasicViT",
    "build_neurofuser_basic_cnn",
    "build_neurofuser_basic_vit",
    "build_cnn_neurofixer",
    "build_vit_neurofixer",
]
