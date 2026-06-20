# NeuroFixer

**NeuroFixer** is a lightweight, installable PyTorch library for backbone-agnostic attention-fusion modules and model-family builders.

The first official model family is **NeuroFuser Basic**, available under:

```python
neurofixer.models.neurofuser_basic
```

NeuroFuser Basic is a public, dependency-light implementation path inspired by the Pattern Recognition article:

**NeuroFuser: Resource-Aware Neuromodulation of Multi-scale Fusion Attention for Domain Adaptive Segmentation**  
Serdar Erişen and André Borrmann  
*Pattern Recognition*, 2026  
DOI: **10.1016/j.patcog.2026.114167**

NeuroFixer is designed as a global attention-based model builder. It provides reusable attention-fusion functions and buildable CNN/ViT model families. Future attention projects can be added under `neurofixer.models` without changing the core attention API.

## Core attention functions

The reusable attention functions live under:

```python
neurofixer.modules
```

and are exposed in the NeuroFuser Basic family as:

```python
from neurofixer.models.neurofuser_basic import (
    NeuroFuserBasicEG,
    NeuroFuserBasicEM,
    NeuroFuserBasicFBr,
    NeuroFuserBasicController,
)
```

The current public functions are:

- `NeuroFuserBasicEG` / `EncodingGate` — per-location and per-channel feature calibration with residual flow and optional dense prediction support.
- `NeuroFuserBasicEM` / `EncodingModule` — latent-grid feature alignment with grouped dilation modulation and optional dense prediction support.
- `NeuroFuserBasicFBr` / `FusionBridge` — branch-normalized multi-scale fusion for decoder, skip, and auxiliary pathways.
- `NeuroFuserBasicController` / `NeuromodulationController` — lightweight resource-aware control for gates, dilation mixtures, branch weights, and fusion modulation.

## Relation to the paper

The NeuroFuser paper introduces resource-aware neuromodulation of multi-scale fusion attention for domain-adaptive semantic segmentation. NeuroFixer provides a public-facing, lightweight implementation base aligned with the paper’s main design principles: backbone-agnostic fusion, controllable attention modulation, CNN/ViT compatibility, and dependency-light modularity.

This repository is intentionally released as a public research-preview implementation. It does not include private training assets, pretrained weights, full experimental pipelines, journal submission files, or confidential reviewer-related material.

## Installation

Install from PyPI:

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

from neurofixer.models.neurofuser_basic import (
    build_neurofuser_basic_cnn,
    build_neurofuser_basic_vit,
)

x = torch.randn(1, 3, 128, 128)

cnn_model = build_neurofuser_basic_cnn(num_classes=19)
cnn_logits = cnn_model(x)
print(cnn_logits.shape)

vit_model = build_neurofuser_basic_vit(num_classes=19, patch_size=8)
vit_logits = vit_model(x)
print(vit_logits.shape)
```

## Attention-function usage

```python
import torch

from neurofixer.models.neurofuser_basic import (
    NeuroFuserBasicEG,
    NeuroFuserBasicEM,
    NeuroFuserBasicFBr,
)

x = torch.randn(1, 64, 64, 64)

eg = NeuroFuserBasicEG(channels=64, num_classes=19)
x_eg, logits_eg, controls = eg(x, return_prediction=True)

em = NeuroFuserBasicEM(in_channels=64, latent_dim=128, out_size=(32, 32), num_classes=19)
x_em, logits_em, controls = em(x_eg, return_prediction=True)

fbr = NeuroFuserBasicFBr(in_channels=[128, 64], out_channels=128, num_classes=19)
x_fused, logits_fused, branch_weights = fbr([x_em, x_eg], return_prediction=True)
```

## Model registry

```python
from neurofixer.models.registry import list_models, build_model

print(list_models())

model = build_model("neurofuser_basic_cnn", num_classes=19)
```

## Repository layout

```text
neurofixer/
  modules/                  # Core EG, EM, FBr, controller implementations
  models/
    neurofuser_basic/        # Official NeuroFuser Basic model family
  demo/                      # CLI smoke demos
  utils/                     # token/grid helpers for CNN-ViT interoperability
tests/                       # smoke tests
requirements/                # optional requirements files
```

## Design principles

1. **Backbone-agnostic:** CNN feature maps and ViT token grids can use the same EG/EM/FBr interfaces.
2. **Minimal dependencies:** PyTorch is the only required runtime dependency.
3. **Public research-preview safe:** no pretrained weights, private backbones, or confidential experimental assets are included.
4. **Model-family oriented:** future attention projects can be added under `neurofixer.models`.
5. **PyPI-ready:** the repository is structured as an installable package.

## Citation

If you use NeuroFixer, NeuroFuser Basic, or the design ideas in this repository, please cite the associated Pattern Recognition article:

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
