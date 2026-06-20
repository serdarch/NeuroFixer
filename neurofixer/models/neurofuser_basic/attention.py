"""Official NeuroFuser Basic attention functions.

This file exposes the public NeuroFuser Basic EG, EM, FBr, and controller
interfaces. The implementations are dependency-light PyTorch modules designed
for public use, PyPI installation, and future extension.

The core implementations live in ``neurofixer.modules`` so they can also be
reused by future attention model families.
"""

from __future__ import annotations

from neurofixer.modules import (
    EncodingGate,
    EncodingModule,
    FusionBridge,
    NeuromodulationController,
)


class NeuroFuserBasicEG(EncodingGate):
    """NeuroFuser Basic EncodingGate.

    Performs per-location and per-channel feature calibration with residual
    flow and optional dense prediction support.
    """


class NeuroFuserBasicEM(EncodingModule):
    """NeuroFuser Basic EncodingModule.

    Aligns stage features to a latent grid and applies grouped dilation
    modulation with optional dense prediction support.
    """


class NeuroFuserBasicFBr(FusionBridge):
    """NeuroFuser Basic FusionBridge.

    Performs branch-normalized multi-scale fusion for decoder, skip, and
    auxiliary feature pathways.
    """


class NeuroFuserBasicController(NeuromodulationController):
    """NeuroFuser Basic neuromodulation controller.

    Provides lightweight resource-aware controls for gates, dilation mixtures,
    branch weights, and fusion modulation.
    """


__all__ = [
    "NeuroFuserBasicEG",
    "NeuroFuserBasicEM",
    "NeuroFuserBasicFBr",
    "NeuroFuserBasicController",
]
