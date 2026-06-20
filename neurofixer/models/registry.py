"""Model registry for NeuroFixer model families."""

from __future__ import annotations

from typing import Callable, Dict

from neurofixer.models.neurofuser_basic import (
    build_neurofuser_basic_cnn,
    build_neurofuser_basic_vit,
)

MODEL_REGISTRY: Dict[str, Callable] = {
    "neurofuser_basic_cnn": build_neurofuser_basic_cnn,
    "neurofuser_basic_vit": build_neurofuser_basic_vit,

    # Backward-compatible names from the first public demo release.
    "neurofuser_demo_cnn": build_neurofuser_basic_cnn,
    "neurofuser_demo_vit": build_neurofuser_basic_vit,
}


def list_models() -> list[str]:
    """Return available model identifiers."""
    return sorted(MODEL_REGISTRY.keys())


def build_model(name: str, **kwargs):
    """Build a registered NeuroFixer model by name."""
    if name not in MODEL_REGISTRY:
        available = ", ".join(list_models())
        raise KeyError(f"Unknown model '{name}'. Available models: {available}")
    return MODEL_REGISTRY[name](**kwargs)
