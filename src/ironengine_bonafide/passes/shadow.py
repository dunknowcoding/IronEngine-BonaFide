"""Cascaded Shadow Map (CSM) depth pass.

For every directional light that casts shadows, build N cascade matrices
fitted to slices of the camera frustum, render scene-mesh depth from each
cascade's light view, and stash the resulting :class:`ShadowMap`s on
``targets.shadow_maps``. The PBR pass samples them with PCF.

Quality measures (see ``core.shadow`` for the math):

  * the light ortho frustum is tightened against the scene AABB and snapped
    to whole texels (stable texel grid, no shimmer);
  * a world-space depth bias (constant + worst-case slope, in texel units)
    is computed per cascade and baked into the depth map — a fixed NDC
    bias means a different world distance per cascade and breaks far
    cascades;
  * the bias can be overridden per light (duck-typed ``shadow_bias``
    attribute, world units) or globally via
    ``RenderConfig.shadow_bias_*``.

The depth raster is delegated to ``backend.raster_depth`` when it exists;
otherwise the pass cleanly skips and the PBR pass shades unshadowed.
"""
from __future__ import annotations

import numpy as np
import torch

from ironengine_bonafide.core.camera import PerspectiveCamera
from ironengine_bonafide.core.light import DirectionalLight
from ironengine_bonafide.core.shadow import (
    ShadowMap,
    bake_depth_bias,
    build_cascades_with_info,
    compute_receiver_bias_world,
    ground_slope_texels,
)
from ironengine_bonafide.passes.base import PassContext, RenderPass

_DEFAULT_RES = 512                       # per-cascade shadow map resolution
_N_CASCADES = 3


class CsmShadowPass(RenderPass):
    name = "shadow_csm"

    def required_capabilities(self) -> tuple[str, ...]:
        return ("shadow_csm",)

    def is_active(self, ctx: PassContext) -> bool:
        if ctx.config.shadows == "off":
            return False
        return any(isinstance(lt, DirectionalLight) and lt.cast_shadow
                   for lt in ctx.scene.lights)

    def run(self, ctx: PassContext) -> None:
        # Backend has to support depth-only raster. Skip cleanly if not.
        if not hasattr(ctx.backend, "raster_depth"):
            ctx.skipped.append("shadow_csm:no_raster_depth")
            return
        if not ctx.scene.meshes:
            ctx.targets.shadow_maps = []                # type: ignore[attr-defined]
            return

        # Use the first directional light that casts shadows.
        light = next(lt for lt in ctx.scene.lights
                     if isinstance(lt, DirectionalLight) and lt.cast_shadow)

        cfg = ctx.config
        res = int(getattr(cfg, "shadow_map_resolution", _DEFAULT_RES) or _DEFAULT_RES)

        cam = ctx.camera
        if isinstance(cam, PerspectiveCamera):
            fov = cam.fov_deg; near = cam.near; far = cam.far
        else:
            fov = 45.0; near = 0.1; far = 100.0
        view_inv = torch.from_numpy(
            _inv4(cam.view_matrix())                      # type: ignore[attr-defined]
        ).cpu().numpy()

        cascades = build_cascades_with_info(
            view_inv, fov, ctx.aspect, near, far,
            light.direction, n_cascades=_N_CASCADES,
            resolution=res,
            scene_bounds=_scene_aabb(ctx),
        )

        # Bias override: per-light attribute wins, then the config override,
        # else the constant+slope model (all in world units). The slope term
        # follows the light's elevation so flat receivers stay acne-free
        # from overhead sun down to grazing golden-hour light.
        override = getattr(light, "shadow_bias", None)
        if override is None:
            override = getattr(cfg, "shadow_bias_override", None)
        slope_texels = (float(getattr(cfg, "shadow_bias_slope", 1.0))
                        * ground_slope_texels(light.direction))

        shadow_maps: list[ShadowMap] = []
        for vp_np, z_n, z_f, frustum in cascades:
            vp = torch.from_numpy(vp_np).to(device=ctx.backend.device, dtype=torch.float32)
            depth = self._render_scene_depth(ctx, vp, res, res)
            bias_world = compute_receiver_bias_world(
                frustum.texel_size_x,
                constant_texels=float(getattr(cfg, "shadow_bias_constant", 0.2)),
                slope_texels=slope_texels,
                override_world=override,
            )
            depth = bake_depth_bias(depth, frustum, bias_world)
            shadow_maps.append(ShadowMap(
                light_view_proj=vp, depth=depth, z_split_near=z_n, z_split_far=z_f,
                texel_size_world=frustum.texel_size_x, bias_world=bias_world,
            ))
        ctx.targets.shadow_maps = shadow_maps           # type: ignore[attr-defined]

    def _render_scene_depth(self, ctx: PassContext, vp: torch.Tensor,
                            width: int, height: int) -> torch.Tensor:
        """Concatenate every mesh and run a single depth raster."""
        all_pos: list[torch.Tensor] = []
        all_idx: list[torch.Tensor] = []
        offset = 0
        for mesh in ctx.scene.meshes:
            pos = mesh.positions.to(ctx.backend.device)
            idx = mesh.indices.to(ctx.backend.device)
            all_pos.append(pos)
            all_idx.append(idx + offset)
            offset += pos.shape[0]
        positions = torch.cat(all_pos, dim=0) if all_pos else torch.zeros((0, 3), device=ctx.backend.device)
        indices = torch.cat(all_idx, dim=0) if all_idx else torch.zeros((0, 3), dtype=torch.int64, device=ctx.backend.device)
        # Every backend provides raster_depth (CPU reference / torch-on-device
        # for CUDA + WGPU), so depth stays on the backend's device.
        return ctx.backend.raster_depth(                # type: ignore[attr-defined]
            positions, indices, vp, width, height,
        ).to(ctx.backend.device)


def _scene_aabb(ctx: PassContext) -> tuple[np.ndarray, np.ndarray] | None:
    """World-space AABB over all scene meshes (None when empty)."""
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for mesh in ctx.scene.meshes:
        if mesh.positions.numel() == 0:
            continue
        p = mesh.positions.detach().cpu().numpy().astype(np.float64)
        lo = np.minimum(lo, p.min(axis=0))
        hi = np.maximum(hi, p.max(axis=0))
    if not np.isfinite(lo).all():
        return None
    return lo, hi


def _inv4(m):                                          # type: ignore[no-untyped-def]
    import numpy as _np
    return _np.linalg.inv(m)
