"""Sky / background pass — first in the default graph.

Paints ``targets.rgb`` wherever depth is still empty (+inf) so geometry
passes composite over it. Three modes via :class:`Background`:

  * ``solid``    — flat color
  * ``gradient`` — horizon → zenith blend by ray elevation (default)
  * ``envmap``   — equirect sample of ``scene.ibl`` (gradient fallback)

Ray directions are derived from the camera's view matrix and fov; for
orthographic cameras every pixel shares the camera forward direction.
"""
from __future__ import annotations

import math

import torch

from ironengine_bonafide.core.camera import (
    OrthographicCamera,
    PerspectiveCamera,
    SensorCamera,
)
from ironengine_bonafide.core.envmap import equirect_sample
from ironengine_bonafide.passes.base import PassContext, RenderPass

# Elevation range (radians) over which the gradient blends horizon→zenith
# and horizon→ground.
_ZENITH_SPAN = 0.5
_GROUND_SPAN = 0.25


class SkyPass(RenderPass):
    name = "sky"

    def is_active(self, ctx: PassContext) -> bool:
        return ctx.scene.background is not None

    def run(self, ctx: PassContext) -> None:
        bg = ctx.scene.background
        assert bg is not None  # guaranteed by is_active
        depth = ctx.targets.depth
        empty = ~torch.isfinite(depth)
        if not torch.any(empty):
            return
        device = ctx.targets.rgb.device
        h, w = depth.shape

        if bg.mode == "solid":
            sky = torch.tensor(bg.color, dtype=torch.float32, device=device)
            ctx.targets.rgb[empty] = sky * bg.intensity
            return

        dirs = ray_directions(ctx.camera, ctx.aspect, w, h, device)

        if bg.mode == "envmap" and ctx.scene.ibl is not None:
            try:
                from ironengine_bonafide.core.light import IBL
                ibl: IBL = ctx.scene.ibl
                env = torch.as_tensor(ibl.load(), dtype=torch.float32, device=device)
                if env.ndim == 3 and env.shape[-1] >= 3:
                    sky = equirect_sample(env[..., :3].contiguous(), dirs)
                    ctx.targets.rgb[empty] = (sky * ibl.intensity * bg.intensity)[empty]
                    return
            except Exception:                                   # noqa: BLE001
                ctx.skipped.append("sky:envmap_load_failed→gradient")

        # Gradient (default + fallback).
        elev = dirs[..., 1]                                     # sin(elevation)
        t_up = (elev / math.sin(_ZENITH_SPAN)).clamp(0.0, 1.0).unsqueeze(-1)
        t_dn = (-elev / math.sin(_GROUND_SPAN)).clamp(0.0, 1.0).unsqueeze(-1)
        zenith = torch.tensor(bg.zenith_color, dtype=torch.float32, device=device)
        horizon = torch.tensor(bg.horizon_color, dtype=torch.float32, device=device)
        ground = torch.tensor(bg.ground_color, dtype=torch.float32, device=device)
        above = horizon * (1.0 - t_up) + zenith * t_up
        below = horizon * (1.0 - t_dn) + ground * t_dn
        sky = torch.where((elev >= 0.0).unsqueeze(-1), above, below) * bg.intensity
        ctx.targets.rgb[empty] = sky[empty]


def ray_directions(
    camera: PerspectiveCamera | OrthographicCamera | SensorCamera,
    aspect: float,
    width: int,
    height: int,
    device: str | torch.device,
) -> torch.Tensor:
    """Unit world-space ray direction per pixel, (H, W, 3)."""
    import numpy as np

    view = camera.view_matrix()                                 # (4, 4) world→eye
    rot = torch.from_numpy(np.linalg.inv(view)[:3, :3]).to(
        device=device, dtype=torch.float32,
    )
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing="ij",
    )
    if isinstance(camera, OrthographicCamera):
        # All rays parallel to the camera forward (-Z in eye space).
        fwd = rot @ torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=device)
        fwd = fwd / torch.linalg.norm(fwd).clamp(min=1e-9)
        return fwd.expand(height, width, 3).contiguous()

    fov = math.radians(getattr(camera, "fov_deg", 45.0))
    tan_half = math.tan(fov * 0.5)
    ndc_x = ((xx + 0.5) / width) * 2.0 - 1.0
    ndc_y = 1.0 - ((yy + 0.5) / height) * 2.0
    dir_cam = torch.stack([
        ndc_x * tan_half * aspect,
        ndc_y * tan_half,
        -torch.ones_like(ndc_x),
    ], dim=-1)                                                  # (H, W, 3)
    dirs = dir_cam @ rot.T
    return dirs / torch.linalg.norm(dirs, dim=-1, keepdim=True).clamp(min=1e-9)
