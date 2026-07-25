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

        sky: torch.Tensor | None = None
        if bg.mode == "envmap" and ctx.scene.ibl is not None:
            try:
                from ironengine_bonafide.core.light import IBL
                ibl: IBL = ctx.scene.ibl
                env = torch.as_tensor(ibl.load(), dtype=torch.float32, device=device)
                if env.ndim == 3 and env.shape[-1] >= 3:
                    sky = equirect_sample(env[..., :3].contiguous(), dirs)
                    sky = sky * ibl.intensity * bg.intensity
            except Exception:                                   # noqa: BLE001
                ctx.skipped.append("sky:envmap_load_failed→gradient")
                sky = None

        if sky is None:
            # Gradient (default + fallback).
            elev = dirs[..., 1]                                 # sin(elevation)
            t_up = (elev / math.sin(_ZENITH_SPAN)).clamp(0.0, 1.0).unsqueeze(-1)
            t_dn = (-elev / math.sin(_GROUND_SPAN)).clamp(0.0, 1.0).unsqueeze(-1)
            zenith = torch.tensor(bg.zenith_color, dtype=torch.float32, device=device)
            horizon = torch.tensor(bg.horizon_color, dtype=torch.float32, device=device)
            ground = torch.tensor(bg.ground_color, dtype=torch.float32, device=device)
            above = horizon * (1.0 - t_up) + zenith * t_up
            below = horizon * (1.0 - t_dn) + ground * t_dn
            sky = torch.where((elev >= 0.0).unsqueeze(-1), above, below) * bg.intensity

        sky = _add_celestial_discs(ctx, bg, sky, dirs)
        ctx.targets.rgb[empty] = sky[empty]


def _sun_direction(ctx: PassContext) -> torch.Tensor | None:
    """Unit vector toward the sun = negated direction of the first
    DirectionalLight, or None when the scene has none."""
    from ironengine_bonafide.core.light import DirectionalLight
    for lt in ctx.scene.lights:
        if isinstance(lt, DirectionalLight):
            d = torch.tensor(lt.direction, dtype=torch.float32,
                             device=ctx.targets.rgb.device)
            n = torch.linalg.norm(d)
            if float(n) < 1e-9:
                return None
            return -d / n
    return None


def _angular_disc(dirs: torch.Tensor, toward: torch.Tensor,
                  radius_deg: float) -> torch.Tensor:
    """Smooth-edged angular disc mask ∈ [0, 1], (H, W).

    1.0 inside ~80% of the angular radius, smoothstepping to 0 at the edge.
    """
    cos_r = math.cos(math.radians(radius_deg))
    cos_inner = math.cos(math.radians(radius_deg * 0.8))
    cosang = (dirs * toward).sum(dim=-1)
    t = ((cosang - cos_r) / max(1e-9, cos_inner - cos_r)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _add_celestial_discs(ctx: PassContext, bg, sky: torch.Tensor,  # type: ignore[no-untyped-def]
                         dirs: torch.Tensor) -> torch.Tensor:
    """Sun/moon discs (+ cheap sun-side horizon glow), opt-in via Background."""
    device = sky.device
    if getattr(bg, "sun_disc", False):
        sun_dir = _sun_direction(ctx)
        if sun_dir is None:
            ctx.skipped.append("sky:sun_disc_no_directional_light")
        else:
            sun_col = torch.tensor((1.0, 0.97, 0.90), dtype=torch.float32, device=device)
            disc = _angular_disc(dirs, sun_dir, float(bg.sun_disc_radius_deg))
            sky = sky + sun_col * (float(bg.sun_disc_intensity) * disc).unsqueeze(-1)
            glow = float(getattr(bg, "sun_horizon_glow", 0.0))
            if glow > 0.0:
                # Warm forward-scatter band hugging the horizon on the sun
                # side: strong looking toward the sun, tight around elev ≈ 0.
                elev = dirs[..., 1]
                horiz = (1.0 - (elev / math.sin(0.35)).abs()).clamp(0.0, 1.0)
                forward = ((dirs * sun_dir).sum(dim=-1).clamp(min=0.0)) ** 4
                warm = torch.tensor((1.0, 0.62, 0.38), dtype=torch.float32, device=device)
                sky = sky + warm * (glow * horiz * forward).unsqueeze(-1)
    if getattr(bg, "moon_disc", False):
        md = torch.tensor(bg.moon_direction, dtype=torch.float32, device=device)
        md = md / torch.linalg.norm(md).clamp(min=1e-9)
        disc = _angular_disc(dirs, md, float(bg.moon_disc_radius_deg))
        mc = torch.tensor(bg.moon_color, dtype=torch.float32, device=device)
        sky = sky + mc * (float(bg.moon_disc_intensity) * disc).unsqueeze(-1)
    return sky


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
