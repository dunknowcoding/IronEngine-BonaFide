"""Volumetric (fog / cloud) pass.

Single-scatter fog in screen space. Fog is uniform exponential density
with an optional altitude falloff; in-memory density grids
(:meth:`Volume.from_grid`, and VDB grids once loaded by the ``[formats]``
extra) are raymarched trilinearly with front-to-back emission/absorption,
depth-limited so geometry correctly occludes the volume.

Depth handling: `targets.depth` stores NDC z in [-1, 1]. Fog density is
physical (per meter), so the pass first reconstructs **linear eye depth
in meters** from the camera's near/far planes — using raw NDC z as
"meters" made fog effectively resolution-independent but distance-blind.
"""
from __future__ import annotations

import torch

from ironengine_bonafide.core.camera import OrthographicCamera
from ironengine_bonafide.passes.base import PassContext, RenderPass

_RAYMARCH_STEPS = 64


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
            elif v.grid is not None:
                self._raymarch_grid(ctx, v)
            else:
                ctx.skipped.append(f"volumetric:{v.kind}_no_grid")

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

    def _raymarch_grid(self, ctx: PassContext, v: object) -> None:
        """Emission/absorption raymarch of an in-memory density grid.

        The grid (D, H, W) maps to world axes (z, y, x): world =
        ``grid_origin + (x, y, z) * grid_voxel_size``. Rays are clipped to
        the grid's AABB and to the scene depth, so geometry occludes the
        volume and the volume occludes the sky. ``_RAYMARCH_STEPS``
        front-to-back steps per pixel; trilinear density sampling.
        """
        from ironengine_bonafide.passes.sky_pass import ray_directions

        device = ctx.targets.rgb.device
        dtype = ctx.targets.rgb.dtype
        h, w = ctx.targets.depth.shape
        grid = v.grid.to(device=device, dtype=torch.float32)           # type: ignore[attr-defined,union-attr]
        dims = torch.tensor([grid.shape[2], grid.shape[1], grid.shape[0]],
                            device=device, dtype=dtype)                # (x, y, z) dims
        vs = float(v.grid_voxel_size)                                  # type: ignore[attr-defined]
        box_min = torch.tensor(v.grid_origin, device=device, dtype=dtype)  # type: ignore[attr-defined]
        box_max = box_min + dims * vs
        density_scale = float(v.density)                               # type: ignore[attr-defined]
        color = torch.tensor(v.color, device=device, dtype=dtype)      # type: ignore[attr-defined]

        origin = _camera_origin(ctx, device, dtype)
        dirs = ray_directions(ctx.camera, ctx.aspect, w, h, device)    # (H, W, 3)

        # Slab AABB intersection per pixel.
        inv_d = 1.0 / torch.where(dirs.abs() < 1e-9,
                                  torch.full_like(dirs, 1e-9), dirs)
        t_a = (box_min - origin) * inv_d
        t_b = (box_max - origin) * inv_d
        t0 = torch.minimum(t_a, t_b).max(dim=-1).values.clamp(min=0.0)
        t1 = torch.maximum(t_a, t_b).min(dim=-1).values

        # Clip the far end at the first scene surface along the ray.
        world = _fragment_world_pos(ctx, ctx.targets.depth)            # (H, W, 3)
        finite = torch.isfinite(ctx.targets.depth)
        t_surf = torch.linalg.norm(world - origin, dim=-1)
        t1 = torch.minimum(t1, torch.where(finite, t_surf,
                                           torch.full_like(t1, float("inf"))))
        hit = t1 > t0
        if not bool(hit.any()):
            return

        steps = _RAYMARCH_STEPS
        step_len = ((t1 - t0) / steps).where(hit, torch.ones_like(t1))  # (H, W)
        T = torch.ones((h, w), device=device, dtype=dtype)             # transmittance
        accum = torch.zeros((h, w, 3), device=device, dtype=dtype)
        for s in range(steps):
            t = t0 + (s + 0.5) * step_len
            p = origin + dirs * t.unsqueeze(-1)                        # (H, W, 3)
            q = (p - box_min) / vs - 0.5                               # voxel coords
            dens = _trilinear(grid, q) * density_scale                 # (H, W)
            alpha = (1.0 - torch.exp(-dens * step_len)).clamp(0.0, 1.0)
            alpha = torch.where(hit, alpha, torch.zeros_like(alpha))
            accum = accum + (T * alpha).unsqueeze(-1) * color
            T = T * (1.0 - alpha)
        ctx.targets.rgb = ctx.targets.rgb * T.unsqueeze(-1) + accum


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
    world = _fragment_world_pos(ctx, depth)
    finite = torch.isfinite(depth)
    return torch.where(finite, world[..., 1], torch.zeros_like(world[..., 1]))


def _fragment_world_pos(ctx: PassContext, depth: torch.Tensor) -> torch.Tensor:
    """World-space position per pixel (H, W, 3); 0 where depth is empty.

    NDC (x, y, z) is unprojected through the inverse view-proj. Note the
    matrix includes any active TAA jitter, keeping volumes consistent with
    the geometry passes.
    """
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
    world = world_h[:, :3] / world_h[:, 3:4].clamp(min=1e-6)
    world = world.reshape(h, w, 3)
    return torch.where(torch.isfinite(depth).unsqueeze(-1), world,
                       torch.zeros_like(world))


def _camera_origin(ctx: PassContext, device: torch.device,
                   dtype: torch.dtype) -> torch.Tensor:
    """World-space eye position for Perspective/Orthographic/Sensor cameras."""
    cam = ctx.camera
    if hasattr(cam, "position"):
        return torch.tensor(cam.position, device=device, dtype=dtype)
    pose = getattr(cam, "pose", None)
    if pose is not None:
        return torch.as_tensor(pose[:3, 3]).to(device=device, dtype=dtype)
    return torch.zeros(3, device=device, dtype=dtype)


def _trilinear(grid: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Trilinear sample of a (D, H, W) grid at (H', W', 3) voxel coords.

    ``q[..., 0]`` indexes W (x), ``q[..., 1]`` indexes H (y), ``q[..., 2]``
    indexes D (z). Out-of-range samples return 0 (empty space).
    """
    d_dim, h_dim, w_dim = grid.shape
    x = q[..., 0].clamp(-0.5, w_dim - 0.5)
    y = q[..., 1].clamp(-0.5, h_dim - 0.5)
    z = q[..., 2].clamp(-0.5, d_dim - 0.5)
    x0 = torch.floor(x); y0 = torch.floor(y); z0 = torch.floor(z)
    x1 = x0 + 1.0; y1 = y0 + 1.0; z1 = z0 + 1.0
    fx = (x - x0).unsqueeze(-1); fy = (y - y0).unsqueeze(-1); fz = (z - z0).unsqueeze(-1)

    def _v(ix: torch.Tensor, iy: torch.Tensor, iz: torch.Tensor) -> torch.Tensor:
        inside = ((ix >= 0) & (ix <= w_dim - 1) & (iy >= 0) & (iy <= h_dim - 1)
                  & (iz >= 0) & (iz <= d_dim - 1))
        ci = ix.clamp(0, w_dim - 1).long()
        cj = iy.clamp(0, h_dim - 1).long()
        ck = iz.clamp(0, d_dim - 1).long()
        return torch.where(inside, grid[ck, cj, ci], torch.zeros_like(ix))

    c00 = _v(x0, y0, z0).unsqueeze(-1) * (1 - fx) + _v(x1, y0, z0).unsqueeze(-1) * fx
    c01 = _v(x0, y0, z1).unsqueeze(-1) * (1 - fx) + _v(x1, y0, z1).unsqueeze(-1) * fx
    c10 = _v(x0, y1, z0).unsqueeze(-1) * (1 - fx) + _v(x1, y1, z0).unsqueeze(-1) * fx
    c11 = _v(x0, y1, z1).unsqueeze(-1) * (1 - fx) + _v(x1, y1, z1).unsqueeze(-1) * fx
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    return (c0 * (1 - fz) + c1 * fz).squeeze(-1)
