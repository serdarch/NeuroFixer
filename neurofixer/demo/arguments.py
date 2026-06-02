"""Command-line arguments for NeuroFixer demos."""
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a NeuroFixer CNN/ViT smoke demo.")
    parser.add_argument("--backbone", choices=["cnn", "vit"], default="cnn")
    parser.add_argument("--num-classes", type=int, default=19)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    return parser
