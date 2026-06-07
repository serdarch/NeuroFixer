"""Small model registry for NeuroFixer model-family demos."""

from __future__ import annotations

from typing import Callable, Dict

from neurofixer.models.neurofuser_demo import (
    build_cnn_neurofixer,
    build_vit_neurofixer,
)

MODEL_REGISTRY: Dict[str, Callable] = {
    "neurofuser_demo_cnn": build_cnn_neurofixer,
    "neurofuser_demo_vit": build_vit_neurofixer,
}


def list_models() -> list[str]:
    """Return available public demo model identifiers."""
    return sorted(MODEL_REGISTRY.keys())


def build_model(name: str, **kwargs):
    """Build a registered NeuroFixer demo model by name."""
    if name not in MODEL_REGISTRY:
        available = ", ".join(list_models())
        raise KeyError(f"Unknown model '{name}'. Available models: {available}")
    return MODEL_REGISTRY[name](**kwargs)
