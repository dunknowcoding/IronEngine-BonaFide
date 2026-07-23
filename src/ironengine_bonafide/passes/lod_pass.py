"""Octree LOD pre-filter pass.

For every PointCloud with ``use_lod=True``:
  1. Lazily build an octree (cached on the cloud as ``_octree``).
  2. Walk the octree against the current camera + cloud-config SSE budget.
  3. Stash the visible-subset indices in ``ctx.frame_state["lod_indices"]``
     keyed by ``id(cloud)`` — the SplatPass applies the subset locally
     when rendering.

The cloud's ``positions`` / ``colors`` are NEVER mutated (the previous
implementation swapped them and never restored, which corrupted the
octree↔array indexing from frame 2 onward).

Native path (``backend.supports("native_octree")``) is ~50x faster than
the pure-Python walker; both produce the same index set so output is
identical.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from ironengine_bonafide.core.camera import PerspectiveCamera, SensorCamera
from ironengine_bonafide.passes.base import PassContext, RenderPass

# frame_state key under which the {id(cloud): indices} map is stored.
LOD_STATE_KEY = "lod_indices"


class LodPass(RenderPass):
    name = "lod"

    def is_active(self, ctx: PassContext) -> bool:
        return any(c.use_lod for c in ctx.scene.pointclouds) and ctx.config.lod.enabled

    def run(self, ctx: PassContext) -> None:
        cam = ctx.camera
        fov = math.radians(cam.fov_deg) if isinstance(cam, (PerspectiveCamera, SensorCamera)) \
            else math.radians(45.0)
        eye = _eye_position(cam)
        h = ctx.targets.rgb.shape[0]
        sse = float(ctx.config.lod.screen_space_error_px)

        state = ctx.frame_state.setdefault(LOD_STATE_KEY, {})
        for cloud in ctx.scene.pointclouds:
            if not cloud.use_lod:
                continue
            indices = self._select(ctx, cloud, eye, fov, h, sse)
            if indices is None or indices.numel() == cloud.num_points:
                state.pop(id(cloud), None)                       # full res
                continue
            state[id(cloud)] = indices

    # ------------------------------------------------------------ select
    def _select(self, ctx: PassContext, cloud: Any, eye: tuple[float, float, float],
                fov: float, h: int, sse: float) -> torch.Tensor | None:
        if ctx.backend.supports("native_octree"):
            from ironengine_bonafide.backends.cuda.native_bridge import (
                octree_build,
                octree_visible,
            )
            if cloud._octree is None:
                cloud._octree = octree_build(cloud.positions, leaf_capacity=4096)
            return octree_visible(
                cloud._octree, eye, fov, h, sse, cloud.num_points,
            ).long()

        from ironengine_bonafide.backends.cuda.lod import build_octree, select_visible
        if cloud._octree is None:
            cloud._octree = build_octree(cloud.positions)
        idx_np = select_visible(
            cloud._octree, np.asarray(eye, dtype=np.float64), fov, h, sse,
        )
        if idx_np.size == 0:
            return None
        return torch.from_numpy(idx_np).to(cloud.positions.device, dtype=torch.long)


def lod_indices_for(ctx: PassContext, cloud: Any) -> torch.Tensor | None:
    """Look up the LOD subset indices the LodPass recorded for ``cloud``."""
    state = ctx.frame_state.get(LOD_STATE_KEY)
    if not state:
        return None
    return state.get(id(cloud))


def _eye_position(cam: Any) -> tuple[float, float, float]:
    if isinstance(cam, PerspectiveCamera):
        return tuple(cam.position)                                          # type: ignore[return-value]
    if isinstance(cam, SensorCamera):
        return tuple(cam.pose[:3, 3].tolist())                              # type: ignore[return-value]
    return (0.0, 0.0, 3.0)
