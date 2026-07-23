"""Fit a PointCloud's per-point colors against a target image via the
differentiable splat path.

Minimal reference loop — for production gsplat training (per-Gaussian
positions/scales/rotations) use `nerfstudio` or `gsplat`'s native
`SplatfactoModel`. This helper exists so users can experiment without
those heavier deps.
"""
from __future__ import annotations

import torch
from torch import optim

from ironengine_bonafide.api import (
    Engine,
    PerspectiveCamera,
    RenderConfig,
    Scene,
    render_differentiable,
)
from ironengine_bonafide.core.pointcloud import PointCloud
from ironengine_bonafide.logging import logger, progress
from ironengine_bonafide.training.losses import l2


def optimize_gsplat(
    cloud: PointCloud,
    *,
    target: torch.Tensor,                          # (H, W, 3) linear HDR
    camera: PerspectiveCamera,
    engine: Engine | None = None,
    iterations: int = 200,
    lr: float = 5e-3,
    width: int | None = None,
    height: int | None = None,
) -> PointCloud:
    """Differentiably optimize `cloud.colors` to match `target` from the
    given camera. Returns the same PointCloud instance with `colors`
    updated in place."""
    engine = engine or Engine.auto()
    h, w, _ = target.shape
    width = int(width or w)
    height = int(height or h)
    target = target.to(engine.backend.device)

    cloud.colors = (cloud.colors if cloud.colors is not None else
                    torch.full(cloud.positions.shape, 0.5, device=engine.backend.device))
    cloud.colors = cloud.colors.detach().clone().to(engine.backend.device).requires_grad_(True)
    cloud.use_gsplat = True

    scene = Scene().add(cloud)
    cfg = RenderConfig(width=width, height=height,
                       gsplat=__import__("ironengine_bonafide.core.config",
                                          fromlist=["GsplatConfig"]).GsplatConfig(enabled=True),
                       output_color_space="linear", samples=1)
    opt = optim.Adam([cloud.colors], lr=lr)

    with progress("optimize_gsplat", total=iterations) as bar:
        for it in range(iterations):
            out = render_differentiable(engine, scene, camera, cfg)
            loss = l2(out.rgb, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            bar.update(1)
            if it % 20 == 0:
                logger.info(f"optimize_gsplat iter={it} loss={float(loss):.4f}")
    cloud.colors = cloud.colors.detach()
    return cloud
