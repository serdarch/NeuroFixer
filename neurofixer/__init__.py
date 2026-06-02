"""NeuroFixer: build and support tools for NeuroFuser-style attention fusion."""

from neurofixer.modules import EncodingGate, EncodingModule, FusionBridge, NeuromodulationController
from neurofixer.build import build_cnn_neurofixer, build_vit_neurofixer

__version__ = "0.1.0"

__all__ = [
    "EncodingGate",
    "EncodingModule",
    "FusionBridge",
    "NeuromodulationController",
    "build_cnn_neurofixer",
    "build_vit_neurofixer",
]
