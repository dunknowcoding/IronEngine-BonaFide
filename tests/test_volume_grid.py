"""Grid-volume raymarching tests (Volume.from_grid → VolumetricPass)."""
from __future__ import annotations

import numpy as np
import torch

from ironengine_bonafide.api import (
    DirectionalLight,
    Engine,
    Mesh,
    PBRMaterial,
    PerspectiveCamera,
    RenderConfig,
    Scene,
    Volume,
    render,
)


def _cam() -> PerspectiveCamera:
    return PerspectiveCamera(position=(0, 0, 3), look_at=(0, 0, 0), fov_deg=45)


def _dense_box(color: tuple[float, float, float] = (1.0, 0.4, 0.1)) -> Volume:
    grid = np.zeros((16, 16, 16), dtype=np.float32)
    grid[4:12, 4:12, 4:12] = 1.0                       # dense cube core
    return Volume.from_grid(grid, origin=(-0.8, -0.8, -0.8), voxel_size=0.1,
                            color=color)


def test_dense_grid_renders_its_color() -> None:
    scene = Scene().add(_dense_box())
    out = render(Engine.cpu(), scene, _cam(),
                 RenderConfig(width=96, height=64, output_color_space="linear"))
    center = out.rgb[24:40, 40:56]
    # The volume's orange tint dominates the frame center.
    assert float(center[..., 0].mean()) > float(center[..., 2].mean()) + 0.05


def test_empty_grid_leaves_frame_unchanged() -> None:
    grid = np.zeros((8, 8, 8), dtype=np.float32)
    vol = Volume.from_grid(grid, origin=(-0.4, -0.4, -0.4), voxel_size=0.1)
    scene = Scene().add(vol)
    a = render(Engine.cpu(), scene, _cam(),
               RenderConfig(width=64, height=48, output_color_space="linear")).rgb
    b = render(Engine.cpu(), Scene(), _cam(),
               RenderConfig(width=64, height=48, output_color_space="linear")).rgb
    torch.testing.assert_close(a, b)


def test_geometry_occludes_volume() -> None:
    # Quad between camera and volume: the volume must not bleed through.
    pos = np.array([[-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0]],
                   dtype=np.float32)
    idx = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    quad = Mesh.from_arrays(pos, idx,
                            material=PBRMaterial(albedo=(0.1, 0.2, 0.9),
                                                 roughness=0.9))
    scene = Scene().add(quad).add(_dense_box()).add(
        DirectionalLight(direction=(0.0, 0.0, -1.0), intensity=3.0))
    out = render(Engine.cpu(), scene, _cam(),
                 RenderConfig(width=96, height=64, output_color_space="linear"))
    center = out.rgb[24:40, 40:56]
    # Blue quad wins over the orange volume behind it.
    assert float(center[..., 2].mean()) > float(center[..., 0].mean())


def test_volume_in_front_of_geometry_tints_it() -> None:
    pos = np.array([[-2, -2, -1.5], [2, -2, -1.5], [2, 2, -1.5], [-2, 2, -1.5]],
                   dtype=np.float32)
    idx = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    quad = Mesh.from_arrays(pos, idx,
                            material=PBRMaterial(albedo=(0.2, 0.2, 0.9),
                                                 roughness=0.9))
    scene = Scene().add(quad).add(_dense_box()).add(
        DirectionalLight(direction=(0.0, 0.0, -1.0), intensity=3.0))
    out = render(Engine.cpu(), scene, _cam(),
                 RenderConfig(width=96, height=64, output_color_space="linear"))
    center = out.rgb[24:40, 40:56]
    # Orange volume in front of the blue quad lifts the red channel.
    assert float(center[..., 0].mean()) > 0.05
