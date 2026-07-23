"""Volumetric (fog / cloud) pass.

Single-scatter fog in screen space. Fog is uniform exponential density
with an optional altitude falloff; VDB-backed volumes sample the grid
trilinearly. Both write into `targets.rgb` after PBR but before
post-processing.

Depth handling: `targets.depth` stores NDC z in [-1, 1]. Fog density is
physical (per meter), so the pass first reconstructs **linear eye depth
in meters** from the camera's near/far planes — using raw NDC z as
"meters" made fog effectively resolution-independent but distance-blind.
"""
from __future__ import annotations

import torch

from ironengine_bonafide.core.camera import OrthographicCamera
from ironengine_bonafide.passes.base import PassContext, RenderPass


class VolumetricPass(RenderPass):
    name = "volumetric"

    def is_active(self, ctx: PassContext) -> bool:
        return bool(ctx.scene.volumes) or ctx.config.fog.enabled

    def run(self, ctx: PassContext) -> None:
        # Combine config-level fog with scene-level fog volumes.
        if ctx.config.fog.enabled:
            self._apply_uniform_fog(
                ctx,
                density=ctx.config.fog.density,
                color=ctx.config.fog.color,
                height_falloff=ctx.config.fog.height_falloff,
            )
        for v in ctx.scene.volumes:
            if v.kind == "fog":
                self._apply_uniform_fog(ctx, density=v.density, color=v.color,
                                        height_falloff=v.height_falloff)
            else:
                # Grid / VDB volume: skip for now (Warp/CuPy raymarch lands in 0.2)
                ctx.skipped.append(f"volumetric:{v.kind}_unimplemented")

    def _apply_uniform_fog(self, ctx: PassContext, *, density: float,
                           color: tuple[float, float, float], height_falloff: float) -> None:
        depth = ctx.targets.depth
        finite = torch.isfinite(depth)
        # Linear eye depth in meters; empty pixels fog at the far plane.
        dist = _linear_depth_meters(ctx, depth)
        if height_falloff > 0.0:
            # Height fog: density decays with world altitude of the
            # fragment (background pixels keep the uniform density).
            world_y = _fragment_world_y(ctx, depth)
            fall = torch.where(
                finite,
                torch.exp(-height_falloff * world_y.clamp(min=0.0)),
                torch.ones_like(world_y),
            )
            eff_density = density * fall
        else:
            eff_density = torch.full_like(dist, density)
        amount = (1.0 - torch.exp(-eff_density * dist)).clamp(0.0, 1.0).unsqueeze(-1)
        c = torch.tensor(color, device=ctx.targets.rgb.device, dtype=ctx.targets.rgb.dtype)
        ctx.targets.rgb = ctx.targets.rgb * (1.0 - amount) + c * amount


def _near_far(ctx: PassContext) -> tuple[float, float]:
    cam = ctx.camera
    near = float(getattr(cam, "near", 0.05))
    far = float(getattr(cam, "far", 200.0))
    return near, far


def _linear_depth_meters(ctx: PassContext, depth: torch.Tensor) -> torch.Tensor:
    """NDC z ∈ [-1, 1] → eye-space distance in meters.

    Perspective: d = 2·near·far / (z·(near−far) + far + near).
    Orthographic: d = near + (z + 1)/2 · (far − near).
    Empty (+inf) pixels resolve to the far plane.
    """
    near, far = _near_far(ctx)
    z = torch.where(torch.isfinite(depth), depth, torch.ones_like(depth))
    if isinstance(ctx.camera, OrthographicCamera):
        d = near + (z + 1.0) * 0.5 * (far - near)
    else:
        d = (2.0 * near * far) / (z * (near - far) + far + near)
    return torch.where(torch.isfinite(depth), d, torch.full_like(d, far))


def _fragment_world_y(ctx: PassContext, depth: torch.Tensor) -> torch.Tensor:
    """World-space Y per pixel (0 where depth is empty)."""
    h, w = depth.shape
    device = depth.device
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    ndc_x = (xx + 0.5) / w * 2.0 - 1.0
    ndc_y = 1.0 - ((yy + 0.5) / h * 2.0)
    z = torch.where(torch.isfinite(depth), depth, torch.zeros_like(depth))
    ndc = torch.stack([ndc_x, ndc_y, z, torch.ones_like(z)], dim=-1)
    view_proj = ctx.camera.view_proj_torch(ctx.aspect, device=device)
    inv = torch.linalg.inv(view_proj.double()).to(torch.float32)
    world_h = ndc.reshape(-1, 4) @ inv.T
    world_y = world_h[:, 1] / world_h[:, 3].clamp(min=1e-6)
    world_y = world_y.reshape(h, w)
    return torch.where(torch.isfinite(depth), world_y, torch.zeros_like(world_y))
