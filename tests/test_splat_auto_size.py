"""Regression: auto splat sizing for scene-scale point clouds.

The disk-splat size used to be ``point_size_px / eye_depth`` with no scene
scale awareness, so a kilometre-scale cloud rendered as 1 px dots. With
``PointCloud.auto_point_size`` the splat pass infers a world-space disk
radius from the cloud's density and converts it through the camera's
pixel focal length. The manual ``point_size_px`` override is unchanged.
"""
from __future__ import annotations

import numpy as np
import torch

from ironengine_bonafide.api import (
    Engine,
    PerspectiveCamera,
    PointCloud,
    RenderConfig,
    Scene,
    render,
)
from ironengine_bonafide.passes.splat_pass import auto_point_size_px


def _km_cloud(n: int = 16, extent: float = 4000.0) -> PointCloud:
    xs = np.linspace(-extent / 2, extent / 2, n)
    pos = np.array([[x, 0.0, z] for z in xs for x in xs], dtype=np.float32)
    colors = np.tile(np.array([[0.9, 0.5, 0.2]], dtype=np.float32), (pos.shape[0], 1))
    return PointCloud.from_arrays(pos, colors, name="terrain_km")


def test_auto_point_size_scales_with_scene() -> None:
    cam = PerspectiveCamera()
    km = _km_cloud()
    auto_km = auto_point_size_px(km, cam, 480)
    assert auto_km > 100.0, f"km-scale cloud needs a huge 1 m-equivalent size, got {auto_km}"

    # Same cloud shrunk to a 2 m tabletop: auto size must shrink with it.
    small = PointCloud.from_arrays(
        (km.positions * 0.001).numpy(), km.colors.numpy(),
    )
    auto_small = auto_point_size_px(small, cam, 480)
    assert auto_small < auto_km * 0.01
    assert auto_small > 0.0


def test_auto_point_size_planar_and_line_fallbacks() -> None:
    cam = PerspectiveCamera()
    planar = _km_cloud()                                   # d2 == 0 (flat sheet)
    r_planar = auto_point_size_px(planar, cam, 480)
    line = PointCloud.from_arrays(
        np.column_stack([np.linspace(0, 100, 50),
                         np.zeros(50), np.zeros(50)]).astype(np.float32),
    )
    r_line = auto_point_size_px(line, cam, 480)
    assert r_planar > 0.0 and r_line > 0.0


def _render(cloud: PointCloud | None) -> torch.Tensor:
    scene = Scene() if cloud is None else Scene().add(cloud)
    cam = PerspectiveCamera(
        position=(0.0, 500.0, 1500.0), look_at=(0.0, 0.0, 0.0),
        fov_deg=60.0, near=1.0, far=8000.0,
    )
    cfg = RenderConfig(width=96, height=72, samples=1)
    return render(Engine.cpu(), scene, cam, cfg).rgb


def _covered_px(rgb: torch.Tensor, bg: torch.Tensor) -> int:
    """Pixels that differ from the empty-scene (sky-only) render."""
    return int(((rgb - bg).abs().sum(dim=-1) > 0.02).sum())


def test_km_cloud_renders_visible_splats() -> None:
    bg = _render(None)
    manual = _render(_km_cloud())                          # default 2 px @ 1 m
    auto = _render(_km_cloud().with_auto_point_size())
    n_manual = _covered_px(manual, bg)
    n_auto = _covered_px(auto, bg)
    # Manual: ~256 tiny dots at ~1.6 km. Auto: spacing-wide disks.
    assert n_auto >= 3.5 * max(n_manual, 1), (n_auto, n_manual)
    assert n_auto > 2000


def test_manual_override_unchanged_when_auto_off() -> None:
    cloud = _km_cloud()
    assert cloud.auto_point_size is False
    assert cloud.point_size_px == 2.0
    # Explicit manual size still drives the render when auto is off.
    big = _render(PointCloud.from_arrays(
        cloud.positions.numpy(), cloud.colors.numpy(), name="terrain_km",
    ))
    bg = _render(None)
    assert _covered_px(big, bg) == _covered_px(_render(_km_cloud()), bg)
