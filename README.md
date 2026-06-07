# NeuroFixer

**NeuroFixer** is a lightweight, installable PyTorch library for building and testing backbone-agnostic attention-fusion modules and model-family demos.

The first public model family is **`neurofuser_demo`**, a companion implementation inspired by the Pattern Recognition article:

**NeuroFuser: Resource-Aware Neuromodulation of Multi-scale Fusion Attention for Domain Adaptive Segmentation**
Serdar Erişen and André Borrmann
*Pattern Recognition*, 2026
DOI: **10.1016/j.patcog.2026.114167**

NeuroFixer is designed as a global attention-based model builder. It provides reusable modules such as `EncodingGate`, `EncodingModule`, `FusionBridge`, and `NeuromodulationController`, together with buildable CNN and ViT demo models. Future attention projects can be added as separate model families under `neurofixer.models`.

The current version provides dependency-light PyTorch implementations of the core NeuroFuser-inspired modules:

* `EncodingGate` — per-location/channel gated feature calibration with residual flow.
* `EncodingModule` — latent-grid alignment with grouped dilation modulation.
* `FusionBridge` — branch-normalized multi-scale fusion for decoder/skip pathways.
* `NeuromodulationController` — differentiable resource-aware controls for gates, heads, dilations, and fusion weights.
* Buildable CNN and ViT demo networks without custom dependencies.

## Relation to the paper

The published NeuroFuser paper introduces resource-aware neuromodulation of multi-scale fusion attention for domain-adaptive semantic segmentation. NeuroFixer is a public-facing, lightweight implementation and experimentation repository inspired by the paper’s main design principles: backbone-agnostic fusion, controllable attention modulation, CNN/ViT compatibility, and dependency-light modularity.

This repository is intentionally released as a public demo and development base. It does not include private training assets, pretrained weights, full experimental pipelines, journal submission files, or confidential reviewer-related material.

## Design principles

1. **Backbone-agnostic:** CNN feature maps and ViT token grids can both use the same EG/EM/FBr interfaces.
2. **Minimal dependencies:** PyTorch is the only required runtime dependency.
3. **Public-demo safe:** no pretrained weights, no private backbones, and no journal-specific experimental assets are included.
4. **Research extensible:** the modules are intentionally modular so user feedback can guide future additions.
5. **Model-family oriented:** new attention projects can be added under `neurofixer.models` without changing the core module API.

## Installation

Install the current public release from PyPI:

```bash
pip install neurofixer
```

For development from source:

```bash
git clone https://github.com/serdarch/NeuroFixer.git
cd NeuroFixer
pip install -e .
```

## Quick start

```bash
python -m neurofixer.demo.run_demo --backbone cnn
python -m neurofixer.demo.run_demo --backbone vit
```

Expected output:

```python
{'backbone': 'cnn', 'input_shape': (1, 3, 128, 128), 'output_shape': (1, 19, 128, 128)}
{'backbone': 'vit', 'input_shape': (1, 3, 128, 128), 'output_shape': (1, 19, 128, 128)}
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

## Global model-family structure

NeuroFixer is organized as a lightweight library for reusable attention-fusion modules and model-family demos. The first model family is:

```text
neurofixer.models.neurofuser_demo
```

This folder contains lightweight CNN and ViT demonstration networks that use the NeuroFuser-inspired `EncodingGate`, `EncodingModule`, `FusionBridge`, and `NeuromodulationController` components.

Example imports:

```python
from neurofixer.models.neurofuser_demo import build_cnn_neurofixer
from neurofixer.models.neurofuser_demo import build_vit_neurofixer
from neurofixer.models.registry import list_models, build_model
```

The long-term design allows future model families to be added under:

```text
neurofixer/models/
  neurofuser_demo/
  token_gate/
  crossfusion/
  cf3a/
  cf4a/
  adapters/
```

without changing the public API of the core attention modules.

## Repository layout

```text
neurofixer/
  modules/       # EG, EM, FBr, neuromodulation controller
  models/        # model-family demos, including neurofuser_demo
  build/         # CNN and ViT demo builders
  demo/          # CLI demo and smoke-test helpers
  utils/         # token/grid helpers for CNN-ViT interoperability

functions/       # compatibility wrappers for the early folder plan
build/           # compatibility build entry points
neurofuser_demo/ # compatibility demo wrappers
tests/           # smoke tests
```

## Development roadmap

* Add config dataclasses for larger CNN/ViT variants.
* Add optional FlashAttention/xFormers adapters only as optional extras, never as required dependencies.
* Add palette-alignment utilities for training-free domain transfer.
* Add public examples for Cityscapes-like and ADE20K-like class counts without releasing private weights.
* Add GitHub Actions smoke tests for CPU-only import and demo execution.
* Expand `neurofixer.models` with additional attention-fusion model families.

## Citation

If you use NeuroFixer, NeuroFuser-inspired modules, or the design ideas in this repository, please cite the associated Pattern Recognition article:

```bibtex
@article{erisen2026neurofuser,
  title   = {NeuroFuser: Resource-Aware Neuromodulation of Multi-scale Fusion Attention for Domain Adaptive Segmentation},
  author  = {Eri{\c{s}}en, Serdar and Borrmann, Andr{\'e}},
  journal = {Pattern Recognition},
  year    = {2026},
  doi     = {10.1016/j.patcog.2026.114167}
}
```

## License and rights

Copyright © Serdar Erişen, 2026. All rights reserved.

NeuroFixer is released under a custom **Research Preview License**.

You may view, clone, run, and evaluate the repository for personal, academic, and non-commercial research purposes. Commercial use, redistribution as a competing library, relicensing, sublicensing, or use of the project names to imply endorsement require prior written permission from the copyright holder.

See [LICENSE](LICENSE) for the full terms.

## Contact

For academic collaboration, licensing, or repository-related questions, please contact the repository owner through GitHub:

**GitHub:** `@serdarch`
**Repository:** `serdarch/NeuroFixer`
