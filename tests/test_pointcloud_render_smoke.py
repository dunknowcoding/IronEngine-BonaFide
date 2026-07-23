"""Smoke render of a synthetic point cloud on the CPU backend."""
from __future__ import annotations

import numpy as np
import torch

from ironengine_bonafide.api import (
    DirectionalLight, Engine, PerspectiveCamera, PointCloud, RenderConfig,
    Scene, render,
)


def test_pointcloud_smoke(cube_pointcloud: tuple[np.ndarray, np.ndarray]) -> None:
    positions, colors = cube_pointcloud
    cloud = PointCloud.from_arrays(positions, colors)
    scene = Scene().add(cloud).add(DirectionalLight(intensity=2.0))
    cam = PerspectiveCamera(position=(2.0, 1.5, 2.0), look_at=(0, 0, 0), fov_deg=45)
    cfg = RenderConfig(width=64, height=48, output_color_space="sRGB", samples=1)

    out = render(Engine.cpu(), scene, cam, cfg)

    assert isinstance(out.rgb, torch.Tensor)
    assert out.rgb.shape == (48, 64, 3)
    assert out.rgb.dtype == torch.float32
    assert float(out.rgb.max()) > 0.0           # we wrote at least one pixel


def test_render_outputs_helpers(cube_pointcloud: tuple[np.ndarray, np.ndarray]) -> None:
    positions, colors = cube_pointcloud
    scene = Scene().add(PointCloud.from_arrays(positions, colors))
    cam = PerspectiveCamera(position=(2, 1, 2), look_at=(0, 0, 0))
    cfg = RenderConfig(width=32, height=24)
    out = render(Engine.cpu(), scene, cam, cfg)

    u8 = out.rgb.to_uint8_srgb()
    assert u8.dtype == torch.uint8
    assert u8.shape == (24, 32, 3)

    aces = out.rgb.to_aces_srgb_uint8()
    assert aces.dtype == torch.uint8
