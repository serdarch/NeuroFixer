"""NeuroFixer: lightweight attention-fusion modules and model-family demos."""

from neurofixer.modules import (
    EncodingGate,
    EncodingModule,
    FusionBridge,
    NeuromodulationController,
)

from neurofixer.models.neurofuser_demo import (
    build_cnn_neurofixer,
    build_vit_neurofixer,
)

__version__ = "0.1.0"

__all__ = [
    "EncodingGate",
    "EncodingModule",
    "FusionBridge",
    "NeuromodulationController",
    "build_cnn_neurofixer",
    "build_vit_neurofixer",
]
