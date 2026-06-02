"""Token/grid helpers for ViT and CNN interoperability."""
from __future__ import annotations

from typing import Optional, Tuple

import math
import torch


def tokens_to_grid(tokens: torch.Tensor, grid_size: Optional[Tuple[int, int]] = None, has_cls_token: bool = False) -> torch.Tensor:
    """Convert ViT tokens ``[B, N, C]`` to a feature grid ``[B, C, H, W]``."""
    if tokens.ndim != 3:
        raise ValueError(f"tokens_to_grid expects [B,N,C], got {tuple(tokens.shape)}")
    if has_cls_token:
        tokens = tokens[:, 1:, :]
    b, n, c = tokens.shape
    if grid_size is None:
        side = int(math.sqrt(n))
        if side * side != n:
            raise ValueError("Provide grid_size when token count is not a square.")
        grid_size = (side, side)
    h, w = grid_size
    if h * w != n:
        raise ValueError(f"grid_size {grid_size} is incompatible with {n} tokens")
    return tokens.transpose(1, 2).reshape(b, c, h, w).contiguous()


def grid_to_tokens(grid: torch.Tensor) -> torch.Tensor:
    """Convert feature grid ``[B, C, H, W]`` to tokens ``[B, H*W, C]``."""
    if grid.ndim != 4:
        raise ValueError(f"grid_to_tokens expects [B,C,H,W], got {tuple(grid.shape)}")
    return grid.flatten(2).transpose(1, 2).contiguous()
