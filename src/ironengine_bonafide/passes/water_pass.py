"""Water pass — Gerstner-wave surface with fresnel reflection/refraction.

For every :class:`~ironengine_bonafide.core.water.WaterSurface` in the
scene the pass raymarches nothing: it intersects each pixel ray with the
water plane analytically, evaluates three directional Gerstner components
at the hit point for the surface normal, then shades:

* **reflection** — mirror direction through the wave normal, sampled from
  the scene IBL (equirect) or the background gradient/solid, plus a tight
  sun glint from the first directional light;
* **refraction** — the already-rendered framebuffer behind the surface,
  absorbed with path length toward the deep body colour (Beer-style);
* **fresnel** — Schlick F0 = 0.02 blends the two; ``reflectivity`` scales it.

Hits write ``rgb`` / ``depth`` / ``normals`` / ``albedo`` (wave normal and
water body colour), so downstream passes (SSGI, denoise, fog) treat the
water as real scene geometry. Scene depth clips the surface; the surface
also clips the volumetric raymarch. ``WaterSurface.time`` advances by
``RenderConfig.simulation_dt`` per frame for deterministic animation.
"""
from __future__ import annotations

import math

import torch

from ironengine_bonafide.core.envmap import equirect_sample
from ironengine_bonafide.core.light import DirectionalLight
from ironengine_bonafide.passes.base import PassContext, RenderPass
from ironengine_bonafide.passes.sky_pass import ray_directions
from ironengine_bonafide.passes.volumetric_pass import (
    _camera_origin,
    _fragment_world_pos,
)

_F0 = 0.02
_N_WAVES = 3


class WaterPass(RenderPass):
    name = "water"

    def is_active(self, ctx: PassContext) -> bool:
        return bool(getattr(ctx.scene, "waters", None))

    def run(self, ctx: PassContext) -> None:
        dt = float(getattr(ctx.config, "simulation_dt", 1.0 / 60.0))
        for surf in ctx.scene.waters:
            self._render_one(ctx, surf)
            surf.time += dt

    # ------------------------------------------------------------ per surface
    def _render_one(self, ctx: PassContext, surf: object) -> None:
        device = ctx.targets.rgb.device
        dtype = ctx.targets.rgb.dtype
        h, w = ctx.targets.depth.shape
        rays = ray_directions(ctx.camera, ctx.aspect, w, h, device)   # (H, W, 3)
        origin = _camera_origin(ctx, device, dtype)

        n = torch.tensor(surf.normal, device=device, dtype=dtype)     # type: ignore[attr-defined]
        n = n / torch.linalg.norm(n).clamp(min=1e-9)
        center = torch.tensor(surf.center, device=device, dtype=dtype)  # type: ignore[attr-defined]

        denom = (rays * n).sum(dim=-1)
        t = ((center - origin) * n).sum() / denom.clamp(
            min=1e-9).where(denom >= 0, denom.clamp(max=-1e-9))
        hit = (denom.abs() > 1e-6) & (t > 0.0)
        if not bool(hit.any()):
            return

        # Scene-depth clip: the surface is only visible before geometry.
        world = _fragment_world_pos(ctx, ctx.targets.depth)
        finite = torch.isfinite(ctx.targets.depth)
        t_scene = torch.linalg.norm(world - origin, dim=-1)
        hit &= t < torch.where(finite, t_scene, torch.full_like(t_scene, float("inf")))
        # Rectangle clip in the plane basis.
        anchor = torch.tensor(
            (0.0, 0.0, 1.0) if abs(float(n[1])) > 0.9 else (0.0, 1.0, 0.0),
            device=device, dtype=dtype)
        u_ax = torch.cross(n, anchor, dim=-1)
        u_ax = u_ax / torch.linalg.norm(u_ax).clamp(min=1e-9)
        v_ax = torch.cross(n, u_ax, dim=-1)
        p = origin + rays * t.unsqueeze(-1)
        rel = p - center
        hu, hv = float(surf.half_size[0]), float(surf.half_size[1])  # type: ignore[attr-defined]
        hit &= ((rel * u_ax).sum(dim=-1).abs() <= hu) & ((rel * v_ax).sum(dim=-1).abs() <= hv)
        if not bool(hit.any()):
            return

        # ---- Gerstner wave normal at p ---------------------------------
        pu = (rel * u_ax).sum(dim=-1)
        pv = (rel * v_ax).sum(dim=-1)
        amp = float(surf.wave_amplitude)                            # type: ignore[attr-defined]
        wl = float(surf.wave_length)                                # type: ignore[attr-defined]
        speed = float(surf.wave_speed)                              # type: ignore[attr-defined]
        steep = float(surf.steepness)                               # type: ignore[attr-defined]
        time = float(surf.time)                                     # type: ignore[attr-defined]
        n_wave = torch.zeros_like(p)
        n_wave += n
        height = torch.zeros_like(pu)
        for i in range(_N_WAVES):
            ang = math.radians(i * 137.5)                           # golden-angle spread
            du = math.cos(ang)
            dv = math.sin(ang)
            k = 2.0 * math.pi / (wl * (1.0 + 0.35 * i))
            omega = speed * math.sqrt(9.81 * k)
            a = amp / (1.0 + 0.5 * i)
            phase = k * (du * pu + dv * pv) - omega * time
            c = torch.cos(phase)
            # Steepness sharpens crests (larger dH/dx at the crest).
            slope = a * k * c * (1.0 + steep * torch.sin(phase).clamp(min=0.0))
            grad = (du * u_ax.unsqueeze(0).unsqueeze(0)
                    + dv * v_ax.unsqueeze(0).unsqueeze(0)) * slope.unsqueeze(-1)
            n_wave = n_wave - grad
            height = height + a * torch.sin(phase)
        n_wave = n_wave / torch.linalg.norm(n_wave, dim=-1, keepdim=True).clamp(min=1e-9)

        # ---- fresnel + reflection direction -----------------------------
        cos_v = (-rays * n_wave).sum(dim=-1).clamp(0.0, 1.0)
        fres = (_F0 + (1.0 - _F0) * (1.0 - cos_v).pow(5)) \
            * float(surf.reflectivity)                              # type: ignore[attr-defined]
        fres = fres.clamp(0.0, 1.0)
        refl = rays - 2.0 * (rays * n_wave).sum(dim=-1, keepdim=True) * n_wave
        refl = refl / torch.linalg.norm(refl, dim=-1, keepdim=True).clamp(min=1e-9)

        refl_col = _sky_sample(ctx, refl)

        # Sun glint: tight lobe toward the first directional light.
        for lt in ctx.scene.lights:
            if isinstance(lt, DirectionalLight):
                sun = -torch.tensor(lt.direction, device=device, dtype=dtype)
                sun = sun / torch.linalg.norm(sun).clamp(min=1e-9)
                glint = (refl * sun).sum(dim=-1).clamp(min=0.0).pow(700.0)
                sun_col = torch.tensor(lt.color, device=device, dtype=dtype)
                refl_col = refl_col + sun_col * (glint * lt.intensity).unsqueeze(-1)
                break

        # ---- refraction + absorption ------------------------------------
        backdrop = ctx.targets.rgb
        absorb = (1.0 - torch.exp(-0.15 * t)).clamp(0.0, 0.9).unsqueeze(-1)
        body = torch.tensor(surf.color, device=device, dtype=dtype)  # type: ignore[attr-defined]
        scatter = torch.tensor(surf.scatter_color, device=device, dtype=dtype)  # type: ignore[attr-defined]
        crest = (height / max(1e-6, _N_WAVES * amp) * 0.5 + 0.5).clamp(0.0, 1.0)
        body_col = body + scatter * crest.unsqueeze(-1) * 0.5
        transmitted = backdrop * (1.0 - absorb) + body_col * absorb

        out = fres.unsqueeze(-1) * refl_col + (1.0 - fres).unsqueeze(-1) * transmitted

        # ---- write targets ----------------------------------------------
        ctx.targets.rgb[hit] = out[hit]
        vp = ctx.camera.view_proj_torch(ctx.aspect, device=device)
        homog = torch.cat([p, torch.ones((h, w, 1), device=device, dtype=dtype)], dim=-1)
        clip = homog @ vp.T
        z_ndc = clip[..., 2] / clip[..., 3].clamp(min=1e-6)
        ctx.targets.depth[hit] = z_ndc[hit]
        ctx.targets.normals[hit] = n_wave[hit]
        ctx.targets.albedo[hit] = body_col[hit]


def _sky_sample(ctx: PassContext, dirs: torch.Tensor) -> torch.Tensor:
    """Sample the scene's backdrop for reflection rays: IBL envmap when the
    background is in envmap mode and an IBL is set, otherwise the same
    gradient/solid the sky pass paints."""
    bg = ctx.scene.background
    device = dirs.device
    if bg is not None and bg.mode == "envmap" and ctx.scene.ibl is not None:
        try:
            env = torch.as_tensor(ctx.scene.ibl.load(), dtype=torch.float32, device=device)
            if env.ndim == 3 and env.shape[-1] >= 3:
                return equirect_sample(env[..., :3].contiguous(), dirs) \
                    * ctx.scene.ibl.intensity * bg.intensity
        except Exception:                                           # noqa: BLE001
            pass
    if bg is None:
        return torch.zeros((*dirs.shape[:-1], 3), device=device, dtype=dirs.dtype)
    if bg.mode == "solid":
        c = torch.tensor(bg.color, device=device, dtype=dirs.dtype)
        return c.expand(*dirs.shape[:-1], 3) * bg.intensity
    zenith = torch.tensor(bg.zenith_color, device=device, dtype=dirs.dtype)
    horizon = torch.tensor(bg.horizon_color, device=device, dtype=dirs.dtype)
    ground = torch.tensor(bg.ground_color, device=device, dtype=dirs.dtype)
    elev = dirs[..., 1]
    t_up = (elev / math.sin(0.5)).clamp(0.0, 1.0).unsqueeze(-1)
    t_dn = (-elev / math.sin(0.25)).clamp(0.0, 1.0).unsqueeze(-1)
    above = horizon * (1.0 - t_up) + zenith * t_up
    below = horizon * (1.0 - t_dn) + ground * t_dn
    return torch.where((elev >= 0.0).unsqueeze(-1), above, below) * bg.intensity
