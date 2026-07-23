"""Point-cloud rendering pass.

Capability flow:
  1. Backend supports `gsplat` AND user enabled gsplat → 3DGS path
  2. Else CPU disk splatting via `backend.raster_points`

Each PointCloud in the scene is rendered into the same `targets.rgb` /
`targets.depth`, with depth testing handled per-backend.
"""
from __future__ import annotations

import torch

from ironengine_bonafide.backends.cpu.backend import CpuBackend
from ironengine_bonafide.core.light import (
    AreaLight,
    DirectionalLight,
    PointLight,
    SpotLight,
)
from ironengine_bonafide.passes.base import PassContext, RenderPass


class SplatPass(RenderPass):
    name = "splat"

    def required_capabilities(self) -> tuple[str, ...]:
        return ("splat",)

    def is_active(self, ctx: PassContext) -> bool:
        return bool(ctx.scene.pointclouds)

    def run(self, ctx: PassContext) -> None:
        for cloud in ctx.scene.pointclouds:
            self._render_one(ctx, cloud)

    def _render_one(self, ctx: PassContext, cloud) -> None:  # type: ignore[no-untyped-def]
        # Path selection — fastest first:
        #   1. gsplat (gsplat library)              — best 3DGS quality
        #   2. native disk-splat (bonafide_native)  — best perf for plain disks
        #   3. CPU torch disk-splat                 — universal fallback
        use_gsplat = (
            ctx.config.gsplat.enabled
            and cloud.use_gsplat
            and ctx.backend.supports("gsplat")
        )
        use_native = (
            not use_gsplat
            and ctx.backend.supports("native_splat")
        )

        view_proj = ctx.camera.view_proj_torch(ctx.aspect, device=ctx.backend.device)
        positions = cloud.positions.to(ctx.backend.device)
        colors = (
            cloud.colors.to(ctx.backend.device)
            if cloud.colors is not None
            else _default_color(positions)
        )
        normals = cloud.normals
        # Apply the LOD subset the LodPass recorded for this frame (the
        # cloud itself is never mutated).
        from ironengine_bonafide.passes.lod_pass import lod_indices_for
        lod_idx = lod_indices_for(ctx, cloud)
        if lod_idx is not None:
            lod_idx = lod_idx.to(positions.device)
            positions = positions[lod_idx]
            colors = colors[lod_idx]
            normals = normals[lod_idx] if normals is not None else None
        # Pre-shade vertex colors (Lambert N·L + ambient) when the cloud
        # carries normals; without normals the raw colors pass through.
        colors = _shade_points(positions, colors, normals, ctx.scene.lights)

        if use_gsplat:
            self._render_gsplat(ctx, positions, colors, cloud)
            return

        h, w = ctx.targets.rgb.shape[:2]
        if use_native:
            self._render_native_splat(ctx, positions, colors, view_proj, w, h, cloud.point_size_px)
            return

        # CPU torch disk-splat fallback.
        be = ctx.backend if isinstance(ctx.backend, CpuBackend) else CpuBackend()
        rgb, depth = self._raster_points(
            be, positions.cpu(), colors.cpu(), view_proj.cpu(),
            w, h, cloud.point_size_px,
        )
        rgb = rgb.to(ctx.targets.rgb.device)
        depth = depth.to(ctx.targets.depth.device)
        better = depth < ctx.targets.depth
        if torch.any(better):
            ctx.targets.rgb[better] = rgb[better]
            ctx.targets.depth[better] = depth[better]

    def _render_native_splat(self, ctx: PassContext, positions: torch.Tensor,
                             colors: torch.Tensor, view_proj: torch.Tensor,
                             w: int, h: int, point_size: float) -> None:
        from ironengine_bonafide.backends.cuda.native_bridge import splat_render
        from ironengine_bonafide.backends.cuda.streams import with_stream
        with with_stream("splat"):
            rgb, depth = splat_render(
                positions, colors, view_proj, w, h, point_size_px=point_size,
            )
        better = depth < ctx.targets.depth
        if torch.any(better):
            ctx.targets.rgb[better] = rgb[better]
            ctx.targets.depth[better] = depth[better]

    def _raster_points(self, backend, positions, colors, view_proj, w, h, point_size):  # type: ignore[no-untyped-def]
        return backend.raster_points(
            positions=positions, colors=colors,
            view_proj=view_proj, width=w, height=h,
            point_size_px=point_size,
        )

    def _render_gsplat(self, ctx: PassContext, positions: torch.Tensor,
                       colors: torch.Tensor, cloud) -> None:                # type: ignore[no-untyped-def]
        from ironengine_bonafide.backends.cuda.splat import (
            intrinsics_from_fov,
            render_gsplat_full,
        )
        from ironengine_bonafide.backends.cuda.streams import (
            with_stream,
            workspace_cache,
        )

        device = ctx.backend.device
        cache = workspace_cache()
        ws = cache.ensure_gsplat(cloud, device)

        # Build a view matrix + intrinsics from the camera. Cache K against
        # the workspace so repeat renders at the same resolution skip the
        # tiny torch.tensor allocation.
        from ironengine_bonafide.core.camera import PerspectiveCamera, SensorCamera
        cam = ctx.camera
        if isinstance(cam, (PerspectiveCamera, SensorCamera)):
            fov = cam.fov_deg
        else:
            fov = 45.0
        h, w = ctx.targets.rgb.shape[:2]
        if (ws.K is None
                or ws.K.shape != (3, 3)
                or float(ws.K[0, 2]) != float(w * 0.5)
                or float(ws.K[1, 2]) != float(h * 0.5)):
            ws.K = intrinsics_from_fov(fov, w, h).to(device)
        view = torch.from_numpy(cam.view_matrix()).to(device=device, dtype=torch.float32)
        ws.last_view = view

        with with_stream("splat"):
            rgb, depth, alpha = render_gsplat_full(
                positions, ws.quats, ws.scales, ws.opacities, colors,
                view, ws.K, w, h,
            )
        # Alpha-composite over current targets
        a = alpha.unsqueeze(-1).clamp(0.0, 1.0)
        ctx.targets.rgb = ctx.targets.rgb * (1.0 - a) + rgb * a
        ctx.targets.depth = torch.where(alpha > 0.5, depth, ctx.targets.depth)


# Ambient floor for point-cloud Lambert shading.
_SPLAT_AMBIENT = 0.25


def _shade_points(positions: torch.Tensor, colors: torch.Tensor,
                  normals, lights) -> torch.Tensor:                      # type: ignore[no-untyped-def]
    """Per-point Lambert pre-shade: accumulate N·L per light against the
    cloud's world-space normals, plus a flat ambient term. Done at the
    vertex stage because disk splatting has no normal GBuffer. Clouds
    without normals keep their raw colors (previous behavior)."""
    if normals is None:
        return colors
    device, dtype = colors.device, colors.dtype
    nrm = normals.to(device=device, dtype=dtype)
    nrm = nrm / torch.linalg.norm(nrm, dim=-1, keepdim=True).clamp(min=1e-9)
    light_acc = torch.full_like(colors, _SPLAT_AMBIENT)
    for lt in lights:
        if isinstance(lt, DirectionalLight):
            ldir = -torch.tensor(lt.direction, device=device, dtype=dtype)
            ldir = ldir / (torch.linalg.norm(ldir) + 1e-9)
            ndotl = (nrm * ldir).sum(dim=-1).clamp(min=0.0).unsqueeze(-1)
            col = torch.tensor(lt.color, device=device, dtype=dtype)
            light_acc = light_acc + ndotl * col * lt.intensity

        elif isinstance(lt, (PointLight, AreaLight)):
            light_pos = torch.tensor(lt.position, device=device, dtype=dtype)
            to_light = light_pos - positions
            dist = torch.linalg.norm(to_light, dim=-1, keepdim=True).clamp(min=1e-3)
            ldir = to_light / dist
            ndotl = (nrm * ldir).sum(dim=-1).clamp(min=0.0).unsqueeze(-1)
            atten = ((1.0 - (dist / lt.range).clamp(0.0, 1.0)).pow(2)
                     if isinstance(lt, PointLight) else 1.0)
            col = torch.tensor(lt.color, device=device, dtype=dtype)
            light_acc = light_acc + ndotl * col * (lt.intensity * atten)

        elif isinstance(lt, SpotLight):
            light_pos = torch.tensor(lt.position, device=device, dtype=dtype)
            spot_dir = torch.tensor(lt.direction, device=device, dtype=dtype)
            spot_dir = spot_dir / (torch.linalg.norm(spot_dir) + 1e-9)
            to_light = light_pos - positions
            dist = torch.linalg.norm(to_light, dim=-1, keepdim=True).clamp(min=1e-3)
            ldir = to_light / dist
            cos_inner = float(torch.cos(torch.tensor(lt.inner_deg).deg2rad()))
            cos_outer = float(torch.cos(torch.tensor(lt.outer_deg).deg2rad()))
            cos_l = (-ldir * spot_dir).sum(dim=-1).clamp(min=0.0).unsqueeze(-1)
            spot_factor = ((cos_l - cos_outer) / max(1e-3, cos_inner - cos_outer)).clamp(0.0, 1.0)
            atten = (1.0 - (dist / lt.range).clamp(0.0, 1.0)).pow(2)
            ndotl = (nrm * ldir).sum(dim=-1).clamp(min=0.0).unsqueeze(-1)
            col = torch.tensor(lt.color, device=device, dtype=dtype)
            light_acc = light_acc + ndotl * col * (lt.intensity * spot_factor * atten)

    return colors * light_acc


def _default_color(positions: torch.Tensor) -> torch.Tensor:
    """Default cloud color: pale blue, on the same device as ``positions``."""
    base = torch.tensor(
        [0.85 * 0.85, 0.85 * 0.90, 0.85 * 1.0],
        dtype=torch.float32, device=positions.device,
    )
    return base.expand(positions.shape[0], 3).contiguous()
