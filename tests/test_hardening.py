"""Hardening tests — edge cases, degenerate inputs, bundle fidelity, CLI.

Every test here targets a crash-or-corrupt class of bug: empty scenes,
zero-geometry meshes, 1x1 frames, near-plane straddling, ortho cameras,
extreme config combos, and bundle round-trips of texture-mapped assets.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ironengine_bonafide.api import (
    DirectionalLight,
    Engine,
    Mesh,
    OrthographicCamera,
    PBRMaterial,
    PerspectiveCamera,
    PointCloud,
    RenderConfig,
    Scene,
    SensorCamera,
    Volume,
    render,
)
from ironengine_bonafide.bundle import RenderBundle
from ironengine_bonafide.cli import main as cli_main


def _quad(material: PBRMaterial | None = None) -> Mesh:
    pos = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]],
                   dtype=np.float32)
    idx = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return Mesh.from_arrays(pos, idx,
                            uvs=np.array([[0, 1], [1, 1], [1, 0], [0, 0]],
                                         dtype=np.float32),
                            material=material or PBRMaterial(albedo=(0.7, 0.4, 0.2)))


def _cam() -> PerspectiveCamera:
    return PerspectiveCamera(position=(0, 0, 3), look_at=(0, 0, 0), fov_deg=45)


# ------------------------------------------------------------ edge cases
def test_empty_scene_renders_background() -> None:
    out = render(Engine.cpu(), Scene(), _cam(),
                 RenderConfig(width=32, height=24, output_color_space="linear"))
    assert out.rgb.shape == (24, 32, 3)
    assert torch.isfinite(out.rgb).all()
    assert float(out.rgb.sum()) > 0.0                       # gradient sky


def test_zero_geometry_mesh_is_skipped() -> None:
    mesh = Mesh.from_arrays(np.zeros((0, 3), np.float32),
                            np.zeros((0, 3), np.int64))
    scene = Scene().add(mesh)
    out = render(Engine.cpu(), scene, _cam(),
                 RenderConfig(width=16, height=16))
    assert torch.isfinite(out.rgb).all()


def test_render_1x1() -> None:
    out = render(Engine.cpu(), Scene().add(_quad()), _cam(),
                 RenderConfig(width=1, height=1))
    assert out.rgb.shape == (1, 1, 3)
    assert torch.isfinite(out.rgb).all()


def test_mesh_straddling_near_plane() -> None:
    # Camera sits inside the quad's plane: half the triangle clips.
    cam = PerspectiveCamera(position=(0, 0, 0.001), look_at=(0, 0, -1),
                            fov_deg=90, near=0.5)
    scene = Scene().add(_quad()).add(
        DirectionalLight(direction=(0, 0, -1), intensity=3.0))
    out = render(Engine.cpu(), scene, cam,
                 RenderConfig(width=32, height=32, output_color_space="linear"))
    assert torch.isfinite(out.rgb).all()


def test_mesh_fully_behind_camera() -> None:
    pos = np.array([[-1, -1, -10], [1, -1, -10], [1, 1, -10]], dtype=np.float32)
    idx = np.array([[0, 1, 2]], dtype=np.int64)
    scene = Scene().add(Mesh.from_arrays(pos, idx))
    out = render(Engine.cpu(), scene, _cam(), RenderConfig(width=16, height=16))
    assert torch.isfinite(out.rgb).all()


def test_orthographic_camera_with_taa() -> None:
    cam = OrthographicCamera(position=(0, 0, 3), look_at=(0, 0, 0),
                             half_width=1.5, half_height=1.5)
    scene = Scene().add(_quad())
    cfg = RenderConfig(width=32, height=32, aa="taa")
    for _ in range(3):
        out = render(Engine.cpu(), scene, cam, cfg)
    assert torch.isfinite(out.rgb).all()


def test_sensor_camera_renders() -> None:
    pose = np.eye(4)
    pose[:3, 3] = (0.0, 0.0, 3.0)
    cam = SensorCamera(pose=pose, fov_deg=50)
    out = render(Engine.cpu(), Scene().add(_quad()), cam,
                 RenderConfig(width=32, height=24, output_color_space="linear"))
    assert torch.isfinite(out.rgb).all()


def test_degenerate_zero_area_triangle() -> None:
    pos = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float32)
    idx = np.array([[0, 1, 2]], dtype=np.int64)
    scene = Scene().add(Mesh.from_arrays(pos, idx))
    out = render(Engine.cpu(), scene, _cam(), RenderConfig(width=16, height=16))
    assert torch.isfinite(out.rgb).all()


def test_single_point_cloud() -> None:
    cloud = PointCloud.from_arrays(np.array([[0, 0, 0]], dtype=np.float32),
                                   np.array([[1, 0, 0]], dtype=np.float32))
    out = render(Engine.cpu(), Scene().add(cloud), _cam(),
                 RenderConfig(width=32, height=32, output_color_space="linear"))
    center = out.rgb[16, 16]
    assert center[0] > center[2]                            # red splat visible


def test_volume_behind_camera_noop() -> None:
    grid = np.ones((4, 4, 4), dtype=np.float32)
    vol = Volume.from_grid(grid, origin=(0, 0, 10), voxel_size=1.0)
    base = render(Engine.cpu(), Scene(), _cam(), RenderConfig(width=16, height=16)).rgb
    out = render(Engine.cpu(), Scene().add(vol), _cam(), RenderConfig(width=16, height=16)).rgb
    torch.testing.assert_close(base, out)


def test_upscale_factor_one_is_noop() -> None:
    out = render(Engine.cpu(), Scene().add(_quad()), _cam(),
                 RenderConfig(width=64, height=48, neural_upscale="fsr",
                              upscale_factor=1.0, output_color_space="linear"))
    assert out.rgb.shape == (48, 64, 3)


def test_denoise_on_empty_frame_no_nan() -> None:
    out = render(Engine.cpu(), Scene(), _cam(),
                 RenderConfig(width=32, height=24, neural_denoise=True,
                              output_color_space="linear"))
    assert torch.isfinite(out.rgb).all()


def test_no_nan_anywhere_on_stress_scene() -> None:
    scene = (Scene()
             .add(_quad())
             .add(PointCloud.from_arrays(
                 np.random.default_rng(0).normal(size=(64, 3)).astype(np.float32)))
             .add(Volume.fog(density=0.05))
             .add(DirectionalLight(direction=(-0.4, -1, -0.3), intensity=3.0)))
    out = render(Engine.cpu(), scene, _cam(),
                 RenderConfig(width=64, height=48, neural_denoise=True,
                              neural_relight="ssgi", output_color_space="linear",
                              sensor_outputs=("rgb", "depth", "normals", "albedo")))
    assert torch.isfinite(out.rgb).all()
    assert out.normals is not None and torch.isfinite(out.normals).all()
    assert out.albedo is not None and torch.isfinite(out.albedo).all()


# ------------------------------------------------------------ bundle fidelity
def test_bundle_preserves_uvs_and_textured_render(tmp_path: Path) -> None:
    pytest.importorskip("pygltflib")
    from _glb_factory import build_full_glb

    from ironengine_bonafide.assets.loaders.gltf import load_primitives

    glb = tmp_path / "full.glb"
    build_full_glb(glb)
    prim = load_primitives(glb)[0]
    assert prim.mesh.uvs is not None

    scene = Scene().add(prim.mesh)
    cam = _cam()
    cfg = RenderConfig(width=64, height=48, output_color_space="linear")
    engine = Engine.cpu()
    before = render(engine, scene, cam, cfg).rgb.clone()

    bnf = tmp_path / "snap.bnf"
    RenderBundle.capture(scene, cam, cfg).save(bnf)
    loaded = RenderBundle.load(bnf)
    mesh = loaded.scene.meshes[0]
    assert mesh.uvs is not None                             # was lost pre-fix
    assert mesh.material.emissive_map == prim.mesh.material.emissive_map
    after = loaded.reproduce(engine).rgb.clone()
    torch.testing.assert_close(before, after)


def test_bundle_preserves_pointcloud_attributes(tmp_path: Path) -> None:
    cloud = PointCloud.from_arrays(
        np.array([[0, 0, 0], [0.1, 0, 0]], dtype=np.float32),
        np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
        np.array([[0, 0, 1], [0, 0, 1]], dtype=np.float32),
    )
    cloud.opacities = torch.tensor([1.0, 0.5])
    cloud = cloud.with_auto_point_size()
    scene = Scene().add(cloud)
    bnf = tmp_path / "pc.bnf"
    RenderBundle.capture(scene, _cam(), RenderConfig()).save(bnf)
    loaded = RenderBundle.load(bnf).scene.pointclouds[0]
    assert loaded.normals is not None and loaded.normals.shape == (2, 3)
    assert loaded.opacities is not None
    torch.testing.assert_close(loaded.opacities, torch.tensor([1.0, 0.5]))
    assert loaded.auto_point_size is True


# ------------------------------------------------------------ CLI smoke
def test_cli_render_smoke(tmp_path: Path) -> None:
    scene_json = tmp_path / "scene.json"
    scene_json.write_text(json.dumps({
        "name": "cli-smoke",
        "camera": {"position": [0, 0, 3], "look_at": [0, 0, 0], "fov_deg": 45},
    }), encoding="utf-8")
    out_png = tmp_path / "cli.png"
    rc = cli_main(["render", str(scene_json), "--out", str(out_png),
                   "--width", "32", "--height", "24", "--backend", "cpu"])
    assert rc == 0
    assert out_png.is_file() and out_png.stat().st_size > 0
    import imageio.v3 as iio
    img = np.asarray(iio.imread(out_png))
    assert img.shape[:2] == (24, 32)


def test_cli_info_smoke(capsys: pytest.CaptureFixture) -> None:
    rc = cli_main(["info"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "torch_version" in out
