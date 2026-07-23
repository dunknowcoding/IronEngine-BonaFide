"""gsplat wrapper.

Exposes :func:`render_gsplat_full` (3DGS rasterization) and
:func:`intrinsics_from_fov` (camera intrinsics builder).

Both raise a clear ``RuntimeError`` when ``gsplat`` isn't importable —
callers are expected to have probed ``backend.supports("gsplat")`` first.
"""
from __future__ import annotations

import math

import torch


def render_gsplat_full(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    view_matrix: torch.Tensor,           # (4, 4) world → camera
    K: torch.Tensor,                     # (3, 3) pinhole intrinsics
    width: int,
    height: int,
    *,
    background: tuple[float, float, float] = (0.0, 0.0, 0.0),
    near: float = 0.05,
    far: float = 200.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from gsplat import rasterization  # type: ignore[import-not-found]
    bg = torch.tensor(background, device=means.device, dtype=torch.float32)
    # gsplat 1.x signature
    out = rasterization(
        means=means, quats=quats, scales=scales,
        opacities=opacities, colors=colors,
        viewmats=view_matrix.unsqueeze(0),
        Ks=K.unsqueeze(0),
        width=width, height=height,
        backgrounds=bg.unsqueeze(0),
        near_plane=near, far_plane=far,
    )
    # gsplat returns (render_colors, render_alphas, info)
    rgb = out[0][0]                      # (H, W, 3)
    alpha = out[1][0]                    # (H, W, 1) or (H, W)
    if alpha.ndim == 3:
        alpha = alpha.squeeze(-1)
    info = out[2] if len(out) > 2 else {}
    depth = info.get("depth", torch.zeros_like(alpha))
    return rgb, depth, alpha


def intrinsics_from_fov(fov_deg: float, width: int, height: int) -> torch.Tensor:
    """Build a 3x3 pinhole K matrix from horizontal FOV + image dims."""
    fy = (height * 0.5) / math.tan(math.radians(fov_deg) * 0.5)
    fx = fy                              # square pixels
    cx, cy = width * 0.5, height * 0.5
    return torch.tensor(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
