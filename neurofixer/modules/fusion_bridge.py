"""FusionBridge: branch-normalized multi-scale feature fusion."""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .controller import normalize_branches


class FusionBridge(nn.Module):
    """Fuse decoder, skip, and optional auxiliary branches by learned branch weights.

    Args:
        in_channels: Channels of each input branch. Provide one entry per branch.
        out_channels: Output channels after alignment.
        num_classes: Optional dense prediction classes.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int = 256,
        num_classes: Optional[int] = None,
        use_depthwise_refine: bool = True,
    ) -> None:
        super().__init__()
        if len(in_channels) < 2:
            raise ValueError("FusionBridge needs at least two branches, e.g. decoder and skip.")
        self.align = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            )
            for c in in_channels
        ])
        self.score = nn.ModuleList([nn.Conv2d(out_channels, 1, kernel_size=1) for _ in in_channels])
        if use_depthwise_refine:
            self.refine = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            )
        else:
            self.refine = nn.Identity()
        self.pred_head = nn.Conv2d(out_channels, num_classes, kernel_size=1) if num_classes else None

    def forward(self, branches: Sequence[torch.Tensor], return_prediction: bool = False):
        if len(branches) != len(self.align):
            raise ValueError(f"Expected {len(self.align)} branches, got {len(branches)}")
        target_size = branches[0].shape[-2:]
        aligned: List[torch.Tensor] = []
        scores: List[torch.Tensor] = []
        for x, align, score in zip(branches, self.align, self.score):
            if x.ndim != 4:
                raise ValueError(f"FusionBridge expects [B,C,H,W], got {tuple(x.shape)}")
            z = align(x)
            if z.shape[-2:] != target_size:
                z = F.interpolate(z, size=target_size, mode="bilinear", align_corners=False)
            aligned.append(z)
            scores.append(score(z))
        score_tensor = torch.stack(scores, dim=1)  # [B,K,1,H,W]
        weights = normalize_branches(score_tensor, dim=1)
        stacked = torch.stack(aligned, dim=1)
        fused = (stacked * weights).sum(dim=1)
        out = self.refine(fused)
        if return_prediction:
            pred = self.pred_head(out) if self.pred_head is not None else None
            return out, pred, weights.squeeze(2)
        return out
