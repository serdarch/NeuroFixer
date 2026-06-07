"""NeuroFuser-style public demo models for NeuroFixer."""

from neurofixer.models.neurofuser_demo.cnn import (
    CNNNeuroFuserDemo,
    build_cnn_neurofixer,
)
from neurofixer.models.neurofuser_demo.vit import (
    ViTNeuroFuserDemo,
    build_vit_neurofixer,
)

__all__ = [
    "CNNNeuroFuserDemo",
    "ViTNeuroFuserDemo",
    "build_cnn_neurofixer",
    "build_vit_neurofixer",
]
