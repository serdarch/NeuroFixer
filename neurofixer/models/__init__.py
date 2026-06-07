"""Model families available in NeuroFixer."""

from neurofixer.models.neurofuser_demo import (
    build_cnn_neurofixer,
    build_vit_neurofixer,
)

__all__ = [
    "build_cnn_neurofixer",
    "build_vit_neurofixer",
]
