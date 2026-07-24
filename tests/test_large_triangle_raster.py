"""Regression: very large triangles at grazing angles must rasterize cleanly.

A 2-triangle 4 km ground quad seen at a grazing angle used to render as
dotted vertical stripes — or nothing at all — because (a) near-plane clip
vertices landed back behind the clip plane after the float32 rounding of
the lerped position and were rejected, and (b) the float32 edge-function
products (~1e18) cancelled catastrophically for vertices projecting to
~1e9 px. The scan now runs in float64 and clip vertices are bisection-
verified against the rounded position.
"""
from __future__ import annotations

import numpy as np
import torch

from ironengine_bonafide.backends import torch_raster
from ironengine_bonafide.core.camera import PerspectiveCamera

_W = _H = 256
_S = 2000.0          # ground half-extent (4 km quad)


def _camera_vp() -> torch.Tensor:
    cam = PerspectiveCamera(
        position=(0.0, 2.0, 10.0), look_at=(0.0, 0.4, -50.0),
        fov_deg=60.0, near=0.1, far=8000.0,
    )
    return torch.from_numpy(cam.view_proj(1.0)).to(torch.float32)


def _quad() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pos = torch.tensor(
        [[-_S, 0.0, -_S], [_S, 0.0, -_S], [_S, 0.0, _S], [-_S, 0.0, _S]],
        dtype=torch.float32,
    )
    idx = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64)
    # Color ramp linear in world x only: perspective-correct interpolation
    # of a world-linear field is exact for ANY triangulation, so the
    # 2-triangle mesh and the subdivided reference must agree.
    ramp = (pos[:, 0:1] + _S) / (2 * _S)
    colors = torch.cat([ramp, torch.full((4, 2), 0.5)], dim=1)
    return pos, idx, colors


def _grid(n: int = 64) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs = np.linspace(-_S, _S, n + 1)
    zs = np.linspace(-_S, _S, n + 1)
    pos = torch.tensor([[x, 0.0, z] for z in zs for x in xs], dtype=torch.float32)
    idx = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            idx.append([a, a + 1, a + n + 2])
            idx.append([a, a + n + 2, a + n + 1])
    ramp = (pos[:, 0:1] + _S) / (2 * _S)
    colors = torch.cat([ramp, torch.full((pos.shape[0], 2), 0.5)], dim=1)
    return pos, torch.tensor(idx, dtype=torch.int64), colors


def test_large_grazing_triangle_covers_ground() -> None:
    pos, idx, colors = _quad()
    _, depth, _ = torch_raster.raster_mesh(pos, idx, colors, None, _camera_vp(), _W, _H)
    coverage = torch.isfinite(depth).float().mean()
    # Was 0.0 before the fix: the whole lower half of the frame is ground.
    assert float(coverage) > 0.45


def test_large_grazing_triangle_matches_subdivided_reference() -> None:
    vp = _camera_vp()
    pos, idx, colors = _quad()
    rgb, depth, _ = torch_raster.raster_mesh(pos, idx, colors, None, vp, _W, _H)
    gpos, gidx, gcol = _grid()
    rgb_ref, depth_ref, _ = torch_raster.raster_mesh(gpos, gidx, gcol, None, vp, _W, _H)

    hit = torch.isfinite(depth)
    hit_ref = torch.isfinite(depth_ref)
    both = hit & hit_ref

    # Coverage and recall against the fine-grid reference.
    recall = both.sum().float() / hit_ref.sum().clamp(min=1)
    assert float(recall) > 0.99
    assert abs(float(hit.float().mean()) - float(hit_ref.float().mean())) < 0.02

    # No banding: colors and depth agree tightly on shared pixels.
    diff_rgb = (rgb - rgb_ref).abs().max(dim=-1).values
    assert float(diff_rgb[both].max()) < 0.02
    diff_depth = (depth - depth_ref).abs()
    assert float(diff_depth[both].median()) < 1e-3


def test_large_grazing_triangle_has_no_stripes() -> None:
    pos, idx, colors = _quad()
    rgb, depth, _ = torch_raster.raster_mesh(pos, idx, colors, None, _camera_vp(), _W, _H)
    # Stripe signature: alternating covered/dropped columns in a band that
    # should be solid ground. The ramp channel must vary smoothly.
    band = slice(180, 250)
    covered = torch.isfinite(depth)[band]
    assert float(covered.float().mean()) > 0.98, "band must be solid, not dotted"
    col_std = rgb[band, :, 0][covered].float().std()
    assert float(col_std) < 0.02
