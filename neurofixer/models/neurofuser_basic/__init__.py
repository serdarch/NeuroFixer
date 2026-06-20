"""NeuroFuser Basic model family.

NeuroFuser Basic is the first official model family in NeuroFixer. It exposes
public EG, EM, FBr, and neuromodulation-controller functions together with
lightweight CNN/ViT builders.
"""

from neurofixer.models.neurofuser_basic.attention import (
    NeuroFuserBasicEG,
    NeuroFuserBasicEM,
    NeuroFuserBasicFBr,
    NeuroFuserBasicController,
)

from neurofixer.models.neurofuser_basic.blocks import (
    EncodingGate,
    EncodingModule,
    FusionBridge,
    NeuromodulationController,
)

from neurofixer.models.neurofuser_basic.cnn import (
    NeuroFuserBasicCNN,
    build_neurofuser_basic_cnn,
)

from neurofixer.models.neurofuser_basic.vit import (
    NeuroFuserBasicViT,
    build_neurofuser_basic_vit,
)

__all__ = [
    "EncodingGate",
    "EncodingModule",
    "FusionBridge",
    "NeuromodulationController",
    "NeuroFuserBasicEG",
    "NeuroFuserBasicEM",
    "NeuroFuserBasicFBr",
    "NeuroFuserBasicController",
    "NeuroFuserBasicCNN",
    "NeuroFuserBasicViT",
    "build_neurofuser_basic_cnn",
    "build_neurofuser_basic_vit",
]
