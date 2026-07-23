"""Lambert lighting of point-cloud splats (pre-shaded vertex colors)."""
from __future__ import annotations

import numpy as np
import torch

from ironengine_bonafide.api import (
    DirectionalLight,
    Engine,
    PerspectiveCamera,
    PointCloud,
    RenderConfig,
    Scene,
    render,
)


def _plane_cloud(normals: np.ndarray | None) -> PointCloud:
    g = np.linspace(-0.8, 0.8, 17, dtype=np.float32)
    yy, xx = np.meshgrid(g, g, indexing="ij")
    positions = np.stack([xx.ravel(), yy.ravel(),
                          np.zeros(xx.size, dtype=np.float32)], axis=1)
    colors = np.ones_like(positions)
    return PointCloud.from_arrays(positions, colors, normals=normals, name="plane")


def _render(cloud: PointCloud) -> torch.Tensor:
    scene = Scene().add(cloud).add(
        DirectionalLight(direction=(0, 0, -1), intensity=2.0, cast_shadow=False))
    cam = PerspectiveCamera(position=(0, 0, 3.0), look_at=(0, 0, 0), fov_deg=45)
    cfg = RenderConfig(width=64, height=48)
    return render(Engine.cpu(), scene, cam, cfg).rgb


def test_normals_facing_light_are_brighter() -> None:
    toward = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (289, 1))
    away = np.tile(np.array([[0.0, 0.0, -1.0]], dtype=np.float32), (289, 1))
    lit = _render(_plane_cloud(toward))
    unlit = _render(_plane_cloud(away))
    assert float(lit.max()) > float(unlit.max()) + 0.5


def test_no_normals_keeps_raw_colors() -> None:
    rgb = _render(_plane_cloud(None))
    # Ambient-free raw white splats must still land on screen.
    assert float(rgb.max()) > 0.0
