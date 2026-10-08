"""Particle pass — deterministic CPU particle simulation + splat render.

Each frame, for every :class:`ParticleSystem` in the scene:

  1. **step** — semi-implicit Euler: ``v += g·dt``, ``v *= 1/(1+drag·dt)``,
     ``p += v·dt``, ``age += dt``. Particles older than ``lifetime``
     respawn at the emitter (when ``respawn`` is on) with a spread jitter
     drawn from the system's seeded generator.
  2. **render** — each live particle splats as a depth-tested disk whose
     colour fades out over the last 25% of its life.

No Warp/XPBD required — the old warp-gated stub is gone.
"""
from __future__ import annotations

import torch

from ironengine_bonafide.backends import torch_raster
from ironengine_bonafide.core.particles import ParticleSystem
from ironengine_bonafide.passes.base import PassContext, RenderPass

_FADE = 0.25                       # last fraction of life used for fade-out


class ParticlePass(RenderPass):
    name = "particles"

    def is_active(self, ctx: PassContext) -> bool:
        return bool(getattr(ctx.scene, "particles", None))

    def run(self, ctx: PassContext) -> None:
        dt = float(getattr(ctx.config, "simulation_dt", 1.0 / 60.0))
        for ps in ctx.scene.particles:
            _step(ps, dt, seed=int(ctx.config.seed))
            self._render_one(ctx, ps)

    def _render_one(self, ctx: PassContext, ps: ParticleSystem) -> None:
        device = ctx.backend.device
        h, w = ctx.targets.rgb.shape[:2]
        vp = ctx.camera.view_proj_torch(ctx.aspect, device=device)
        ages = ps.ages if ps.ages is not None else torch.zeros(
            ps.num_particles, device=device)
        life = max(float(ps.lifetime), 1e-6)
        frac = (ages.to(device) / life).clamp(0.0, 1.0)
        fade = ((1.0 - frac) / _FADE).clamp(0.0, 1.0)               # 1 → 0 at end
        col = torch.tensor(ps.color, device=device, dtype=torch.float32)
        colors = col.unsqueeze(0) * fade.unsqueeze(-1)
        live = fade > 0.0
        if not bool(live.any()):
            return
        rgb, depth = torch_raster.raster_points(
            ps.positions.to(device)[live], colors[live], vp, w, h,
            point_size_px=float(ps.size_px),
        )
        better = depth < ctx.targets.depth
        if not bool(better.any()):
            return
        ctx.targets.rgb[better] = rgb[better]
        ctx.targets.depth[better] = depth[better]


def _step(ps: ParticleSystem, dt: float, *, seed: int) -> None:
    """Advance the system one frame (mutates positions/velocities/ages)."""
    n = ps.num_particles
    device = ps.positions.device
    if ps.velocities is None:
        ps.velocities = torch.zeros((n, 3), dtype=torch.float32, device=device)
    if ps.ages is None:
        ps.ages = torch.zeros(n, dtype=torch.float32, device=device)
    g = torch.tensor(ps.gravity, dtype=torch.float32, device=device)
    ps.velocities = ps.velocities + g * dt
    ps.velocities = ps.velocities / (1.0 + float(ps.drag) * dt)
    ps.positions = ps.positions + ps.velocities * dt
    ps.ages = ps.ages + dt

    dead = ps.ages > float(ps.lifetime)
    if not bool(dead.any()) or not ps.respawn:
        return
    if ps._rng is None:
        ps._rng = torch.Generator(device="cpu").manual_seed(seed)
    k = int(dead.sum())
    ep = torch.tensor(ps.emitter_position, dtype=torch.float32, device=device)
    ev = torch.tensor(ps.emitter_velocity, dtype=torch.float32, device=device)
    jitter = (torch.rand((k, 3), generator=ps._rng) - 0.5) \
        * (2.0 * float(ps.emitter_spread))
    ps.positions[dead] = ep + jitter.to(device)
    ps.velocities[dead] = ev
    ps.ages[dead] = 0.0
