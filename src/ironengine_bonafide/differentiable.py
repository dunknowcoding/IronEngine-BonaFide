"""Convenience helpers for the autograd path.

For now this re-exports `render_differentiable` and provides a small
`make_optimizable` that flips `requires_grad` on the right tensors of an
asset. Full custom `torch.autograd.Function` wrappers (for non-torch CUDA
kernels) land alongside the warp / completion modules.
"""
from __future__ import annotations

from typing import Any

import torch

from ironengine_bonafide.api import render_differentiable
from ironengine_bonafide.core.mesh import Mesh
from ironengine_bonafide.core.pointcloud import PointCloud

__all__ = ["render_differentiable", "make_optimizable"]


def make_optimizable(asset: Any, *, fields: tuple[str, ...] = ("colors", "positions")) -> list[torch.Tensor]:
    """Flip `requires_grad=True` on the named tensor fields of `asset`.

    Returns the list of optimizable parameters — pass directly to a torch
    optimizer (`torch.optim.Adam(params, lr=...)`).
    """
    out: list[torch.Tensor] = []
    if isinstance(asset, (PointCloud, Mesh)):
        for f in fields:
            t = getattr(asset, f, None)
            if isinstance(t, torch.Tensor):
                t.requires_grad_(True)
                out.append(t)
    return out
