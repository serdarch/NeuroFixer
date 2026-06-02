# NeuroFixer

**NeuroFixer** is a public-demo oriented build and support repository for NeuroFuser-style attention-fusion models. It is designed as a lightweight, backbone-agnostic library that grows toward:

```bash
pip install neurofixer
```

The current version provides dependency-light PyTorch implementations of the core NeuroFuser-inspired modules:

- `EncodingGate` — per-location/channel gated feature calibration with residual flow.
- `EncodingModule` — latent-grid alignment with grouped dilation modulation.
- `FusionBridge` — branch-normalized multi-scale fusion for decoder/skip pathways.
- `NeuromodulationController` — differentiable resource-aware controls for gates, heads, dilations, and fusion weights.
- Buildable CNN and ViT demo networks without custom dependencies.

## Design principles

1. **Backbone-agnostic:** CNN feature maps and ViT token grids can both use the same EG/EM/FBr interfaces.
2. **Minimal dependencies:** PyTorch is the only required runtime dependency.
3. **Public-demo safe:** no pretrained weights, no private backbones, and no journal-specific experimental assets are included.
4. **Research extensible:** the modules are intentionally modular so user feedback can guide future additions.

## Quick start from source

```bash
pip install -e .
python -m neurofixer.demo.run_demo --backbone cnn
python -m neurofixer.demo.run_demo --backbone vit
```

Expected output:

```python
{'backbone': 'cnn', 'input_shape': (1, 3, 128, 128), 'output_shape': (1, 19, 128, 128)}
```

## Python usage

```python
import torch
from neurofixer import build_cnn_neurofixer, build_vit_neurofixer

x = torch.randn(1, 3, 128, 128)
model = build_cnn_neurofixer(num_classes=19)
logits = model(x)
print(logits.shape)

vit_model = build_vit_neurofixer(num_classes=19, patch_size=8)
vit_logits = vit_model(x)
print(vit_logits.shape)
```

## Repository layout

```text
neurofixer/
  modules/       # EG, EM, FBr, neuromodulation controller
  build/         # CNN and ViT demo builders
  demo/          # CLI demo and smoke-test helpers
  utils/         # token/grid helpers for CNN-ViT interoperability
functions/       # compatibility wrappers for the early folder plan
build/           # compatibility build entry points
neurofuser_demo/ # compatibility demo wrappers
tests/           # smoke tests
```

## Development roadmap

- Add config dataclasses for larger CNN/ViT variants.
- Add optional FlashAttention/xFormers adapters only as optional extras, never as required dependencies.
- Add palette-alignment utilities for training-free domain transfer.
- Add public examples for Cityscapes-like and ADE20K-like class counts without releasing private weights.
- Add GitHub Actions smoke tests for CPU-only import and demo execution.

## Rights

Copyright © Serdar Erişen, 2026. All rights reserved.

A formal public license has not yet been selected. Until then, this repository should be treated as a controlled public demo / research preview rather than a permissively licensed release.

NeuroFixer is released under a custom **Research Preview License**.

You may view, clone, run, and evaluate the repository for personal, academic,
and non-commercial research purposes. Commercial use, redistribution as a
competing library, relicensing, sublicensing, or use of the project names to
imply endorsement require prior written permission from the copyright holder.

See [LICENSE](LICENSE) for the full terms.
