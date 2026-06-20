"""NeuroFuser Basic model family.

NeuroFuser Basic is the first official model family in NeuroFixer. It exposes
the public EG, EM, FBr, and neuromodulation-controller blocks and provides
lightweight CNN/ViT builders for testing attention-fusion pipelines.
"""

from neurofixer.models.neurofuser_basic.blocks import (
    EncodingGate,
    EncodingModule,
    FusionBridge,
    NeuromodulationController,
)

try:
    from neurofixer.models.neurofuser_basic.cnn import (
        NeuroFuserBasicCNN,
        build_neurofuser_basic_cnn,
    )
except Exception:  # pragma: no cover
    NeuroFuserBasicCNN = None
    build_neurofuser_basic_cnn = None

try:
    from neurofixer.models.neurofuser_basic.vit import (
        NeuroFuserBasicViT,
        build_neurofuser_basic_vit,
    )
except Exception:  # pragma: no cover
    NeuroFuserBasicViT = None
    build_neurofuser_basic_vit = None

__all__ = [
    "EncodingGate",
    "EncodingModule",
    "FusionBridge",
    "NeuromodulationController",
    "NeuroFuserBasicCNN",
    "NeuroFuserBasicViT",
    "build_neurofuser_basic_cnn",
    "build_neurofuser_basic_vit",
]
