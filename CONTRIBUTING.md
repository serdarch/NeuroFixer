# Contributing to NeuroFixer

Thank you for your interest in contributing to NeuroFixer.

NeuroFixer is a lightweight PyTorch research library for backbone-agnostic attention-fusion modules and model-family builders. We welcome contributions of attention heads, fusion modules, gating mechanisms, token-mixing blocks, CNN/ViT adapters, and compact model-family demos.

## Contribution types

You may contribute:

1. New attention or fusion modules under `neurofixer/modules/`.
2. New model families under `neurofixer/models/`.
3. Tests, examples, or documentation.
4. Bug fixes and packaging improvements.

## Requirements

Contributions should:

- use PyTorch as the core runtime dependency,
- avoid heavy required dependencies unless they are optional extras,
- include clear tensor-shape documentation,
- include at least one CPU smoke test,
- avoid pretrained weights unless explicitly approved,
- avoid private datasets or confidential research assets,
- include citation information when the method is based on a paper, preprint, or public repository,
- preserve compatibility with `pip install neurofixer`.

## Suggested structure for a new attention module

```text
neurofixer/modules/attention_heads/my_attention_head.py
tests/test_my_attention_head.py
