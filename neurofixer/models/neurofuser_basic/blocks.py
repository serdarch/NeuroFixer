"""NeuroFuser Basic blocks.

This module provides the first official NeuroFuser-style public block family
inside NeuroFixer. The blocks are intentionally dependency-light and are built
from the reusable core modules in ``neurofixer.modules``.

Public blocks
-------------
EncodingGate
    Per-location/channel feature calibration with residual flow and optional
    dense prediction support.

EncodingModule
    Latent-grid feature alignment with grouped dilation modulation and optional
    dense prediction support.

FusionBridge
    Branch-normalized multi-scale fusion for decoder, skip, and auxiliary
    pathways.

NeuromodulationController
    Lightweight controller for resource-aware gates, dilation mixtures, branch
    weights, and fusion controls.

These blocks are aligned with the public design principles of the NeuroFuser
Pattern Recognition paper, while remaining lightweight enough for public demos,
PyPI installation, and future extension.
"""

from neurofixer.modules import (
    EncodingGate,
    EncodingModule,
    FusionBridge,
    NeuromodulationController,
)

__all__ = [
    "EncodingGate",
    "EncodingModule",
    "FusionBridge",
    "NeuromodulationController",
]
