"""Realism regression tests on the CPU backend.

Covers the 0.2 rendering fixes: sky background (W3), sRGB tonemap
contract (W4), perspective-scaled points (W5), shadows (W6),
near-plane clipping + perspective-correct raster (W7), deterministic
depth resolve (W8), LOD non-mutation (W9), fog in meters (W13),
real IBL (W2) and texture maps (W1).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image  # noqa: F401  (ensures pillow present for imageio png)

from ironengine_bonafide.api import (
    Background,
    DirectionalLight,
    Engine,
    IBL,
    Mesh,
    PBRMaterial,
    PerspectiveCamera,
    PointCloud,
    RenderConfig,
    Scene,
    render,
)
from ironengine_bonafide.core.color import aces_filmic, linear_to_srgb


def _quad(size: float = 1.0, y: float = 0.0, material: PBRMaterial | None = None,
          normal=(0.0, 1.0, 0.0), with_uv: bool = False) -> Mesh:
    s = size
    positions = np.array(
        [[-s, y, -s], [s, y, -s], [s, y, s], [-s, y, s]], dtype=np.float32,
    )
    normals = np.tile(np.array([normal], dtype=np.float32), (4, 1))
    indices = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int64)
    uvs = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32) if with_uv else None
    return Mesh.from_arrays(positions, indices, normals=normals, uvs=uvs,
                            material=material or PBRMaterial())


def _camera() -> PerspectiveCamera:
    return PerspectiveCamera(position=(0.0, 3.0, 4.0), look_at=(0.0, 0.0, 0.0), fov_deg=45)


# ---------------------------------------------------------------- sky (W3)
def test_background_is_not_black_by_default() -> None:
    scene = Scene()  # default Background gradient
    cam = PerspectiveCamera(position=(0, 0, 3), look_at=(0, 0, 0), fov_deg=45)
    cfg = RenderConfig(width=64, height=48)
    out = render(Engine.cpu(), scene, cam, cfg)
    assert float(out.rgb.abs().max()) > 0.05, "default background must not be void-black"
    # Gradient: top of frame differs from bottom of frame.
    top = out.rgb[2, 32]
    bottom = out.rgb[-3, 32]
    assert not torch.allclose(top, bottom, atol=1e-3)


def test_background_solid_mode() -> None:
    scene = Scene().add(Background(mode="solid", color=(0.5, 0.2, 0.1)))
    cam = PerspectiveCamera(position=(0, 0, 3), look_at=(0, 0, 0))
    cfg = RenderConfig(width=32, height=24)
    out = render(Engine.cpu(), scene, cam, cfg)
    assert torch.allclose(out.rgb[12, 16], torch.tensor([0.5, 0.2, 0.1]), atol=1e-5)


def test_background_can_be_disabled() -> None:
    scene = Scene(background=None)
    cam = PerspectiveCamera(position=(0, 0, 3), look_at=(0, 0, 0))
    out = render(Engine.cpu(), scene, cam, RenderConfig(width=32, height=24))
    assert float(out.rgb.abs().max()) == 0.0


def test_envmap_background() -> None:
    env = np.zeros((8, 16, 3), dtype=np.float32)
    env[:, :, 0] = 0.8                      # uniform red environment
    scene = Scene().add(IBL(pixels=env, intensity=1.0))
    scene.background = Background(mode="envmap")
    cam = PerspectiveCamera(position=(0, 0, 3), look_at=(0, 0, 0))
    out = render(Engine.cpu(), scene, cam, RenderConfig(width=32, height=24))
    assert torch.allclose(out.rgb[12, 16], torch.tensor([0.8, 0.0, 0.0]), atol=1e-4)


# ----------------------------------------------------------- tonemap (W4)
def test_srgb_output_is_display_ready() -> None:
    mesh = _quad(material=PBRMaterial(albedo=(0.5, 0.5, 0.5), roughness=0.9))
    scene = Scene().add(mesh).add(DirectionalLight(direction=(0, -1, 0), intensity=2.0))
    cam = _camera()
    lin = render(Engine.cpu(), scene, cam,
                 RenderConfig(width=64, height=48, output_color_space="linear"))
    srgb = render(Engine.cpu(), scene, cam,
                  RenderConfig(width=64, height=48, output_color_space="sRGB"))
    assert srgb.color_space == "sRGB"
    assert float(srgb.rgb.min()) >= 0.0 and float(srgb.rgb.max()) <= 1.0
    expected = linear_to_srgb(aces_filmic(lin.rgb * 1.0))
    assert torch.allclose(srgb.rgb, expected, atol=1e-5), (
        "sRGB output must be ACES + sRGB-encoded — final, display-ready"
    )
    u8 = srgb.rgb.to_uint8_display()
    assert u8.dtype == torch.uint8


# ---------------------------------------------------- points size (W5)
def test_point_size_scales_with_depth() -> None:
    from ironengine_bonafide.backends.cpu.backend import CpuBackend

    positions = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -4.0]], dtype=torch.float32)
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    cam = PerspectiveCamera(position=(0, 0, 0), look_at=(0, 0, -1), fov_deg=60)
    vp = torch.from_numpy(cam.view_proj(1.0)).float()
    rgb, _ = CpuBackend().raster_points(positions, colors, vp, 128, 128, point_size_px=8.0)
    near_px = int(((rgb[..., 0] > 0.5) & (rgb[..., 2] < 0.5)).sum())
    far_px = int(((rgb[..., 2] > 0.5) & (rgb[..., 0] < 0.5)).sum())
    assert near_px > far_px * 4, (
        f"near point must cover much more screen area (near={near_px}, far={far_px})"
    )


# -------------------------------------------------------------- shadows (W6)
def test_shadow_darkening_present() -> None:
    floor = _quad(size=3.0, y=0.0, material=PBRMaterial(albedo=(0.8, 0.8, 0.8), roughness=0.95))
    blocker = _quad(size=0.6, y=1.0, material=PBRMaterial(albedo=(0.8, 0.8, 0.8), roughness=0.95))
    cam = _camera()
    cfg = RenderConfig(width=96, height=72)

    def _render(cast: bool) -> torch.Tensor:
        scene = (Scene(background=None).add(floor).add(blocker)
                 .add(DirectionalLight(direction=(0.05, -1.0, 0.05), intensity=3.0,
                                       cast_shadow=cast)))
        return render(Engine.cpu(), scene, cam, cfg).rgb

    lit = _render(cast=False)
    shaded = _render(cast=True)
    diff = (lit - shaded).sum(dim=-1)
    assert float(diff.max()) > 0.15, "occluder must cast a visible shadow"
    dark_fraction = float((diff > 0.05).float().mean())
    assert dark_fraction < 0.5, "shadow must be localized, not a global darkening"


# ------------------------------------------- near plane / perspective (W7)
def test_triangle_crossing_near_plane_is_not_dropped() -> None:
    from ironengine_bonafide.backends.cpu.backend import CpuBackend

    # One vertex behind the camera, two in front — previously the whole
    # triangle was discarded.
    positions = torch.tensor(
        [[-1.0, -1.0, -2.0], [1.0, -1.0, -2.0], [0.0, 1.0, 0.5]], dtype=torch.float32,
    )
    indices = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    colors = torch.ones((3, 3), dtype=torch.float32)
    cam = PerspectiveCamera(position=(0, 0, 0), look_at=(0, 0, -1), fov_deg=60)
    vp = torch.from_numpy(cam.view_proj(1.0)).float()
    gb = CpuBackend().raster_mesh_gbuffer(positions, indices, colors, None, vp, 64, 64)
    assert float(gb.mask.sum()) > 0.0, "near-plane-crossing triangle must be split, not dropped"


def test_perspective_correct_uv_interpolation() -> None:
    """On a strongly foreshortened quad, perspective-correct interpolation
    shifts the texture midpoint toward the far end compared with affine."""
    from ironengine_bonafide.backends.cpu.backend import CpuBackend

    positions = torch.tensor(
        [[-1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -9.0], [-1.0, 0.0, -9.0]],
        dtype=torch.float32,
    )
    indices = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64)
    uvs = torch.tensor([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=torch.float32)
    colors = torch.ones((4, 3), dtype=torch.float32)
    normals = torch.tensor([[0, 1, 0]] * 4, dtype=torch.float32)
    cam = PerspectiveCamera(position=(0, 4.0, 0.0), look_at=(0, 0, -5.0), fov_deg=50)
    vp = torch.from_numpy(cam.view_proj(1.0)).float()
    gb = CpuBackend().raster_mesh_gbuffer(
        positions, indices, colors, normals, vp, 64, 64, uvs=uvs,
    )
    hit = gb.mask > 0.5
    assert hit.any()
    v = gb.uv[..., 1][hit]
    ys = torch.arange(64, dtype=torch.float32).unsqueeze(1).expand(64, 64)[hit]
    order = torch.argsort(ys)
    v_sorted = v[order]
    n = v_sorted.numel()
    # Perspective compression: the screen midpoint of the visible region
    # maps to v << 0.5 (affine interpolation would give ~0.5).
    mid_v = float(v_sorted[n // 2])
    assert 0.05 < mid_v < 0.45, f"affine interpolation would give ~0.5, got {mid_v}"


# ------------------------------------------------------- determinism (W8)
def test_render_is_deterministic() -> None:
    g = np.random.default_rng(7)
    pos = g.uniform(-0.5, 0.5, (3000, 3)).astype(np.float32)
    col = g.uniform(0.0, 1.0, (3000, 3)).astype(np.float32)
    cam = PerspectiveCamera(position=(1.5, 1.0, 1.5), look_at=(0, 0, 0))
    cfg = RenderConfig(width=64, height=48)

    def _render() -> torch.Tensor:
        scene = (Scene().add(PointCloud.from_arrays(pos, col))
                 .add(_quad(size=0.4, y=-0.2)))
        return render(Engine.cpu(), scene, cam, cfg).rgb

    a = _render()
    b = _render()
    assert torch.equal(a, b), "two identical renders must produce identical frames"


# ---------------------------------------------------------------- LOD (W9)
def test_lod_does_not_mutate_cloud() -> None:
    g = np.random.default_rng(3)
    pos = g.uniform(-1.0, 1.0, (20000, 3)).astype(np.float32)
    col = g.uniform(0.0, 1.0, (20000, 3)).astype(np.float32)
    cloud = PointCloud.from_arrays(pos, col).with_lod()
    n0 = cloud.num_points
    scene = Scene().add(cloud)
    # Far-away camera → aggressive LOD thinning.
    cam = PerspectiveCamera(position=(30.0, 30.0, 30.0), look_at=(0, 0, 0))
    cfg = RenderConfig(width=64, height=48)
    render(Engine.cpu(), scene, cam, cfg)
    assert cloud.num_points == n0, "LodPass must not mutate cloud arrays"
    # Second frame renders fine (previously: octree/subset index corruption).
    out = render(Engine.cpu(), scene, cam, cfg)
    assert torch.isfinite(out.rgb).all()


# ---------------------------------------------------------------- fog (W13)
def test_fog_uses_linear_meters() -> None:
    from types import SimpleNamespace

    from ironengine_bonafide.passes.volumetric_pass import _linear_depth_meters

    cam = PerspectiveCamera(position=(0, 0, 3), look_at=(0, 0, 0), near=0.5, far=100.0)
    ctx = SimpleNamespace(camera=cam, aspect=1.0)
    depth = torch.tensor([[-1.0], [1.0]])
    d = _linear_depth_meters(ctx, depth)
    assert abs(float(d[0, 0]) - 0.5) < 1e-4, "NDC -1 must map to the near plane"
    assert abs(float(d[1, 0]) - 100.0) < 1e-2, "NDC +1 must map to the far plane"


def test_fog_increases_with_distance() -> None:
    cam = PerspectiveCamera(position=(0, 2.0, 4.0), look_at=(0, 0, -5.0), fov_deg=45)
    fog_c = torch.tensor((0.7, 0.78, 0.86))

    def _render_at(z: float) -> torch.Tensor:
        q = _quad(size=0.8, y=0.0, material=PBRMaterial(albedo=(0.8, 0.8, 0.8)))
        q.positions = q.positions + torch.tensor([0.0, 0.0, z])
        scene = (Scene(background=None).add(q)
                 .add(DirectionalLight(direction=(0, -1, 0), intensity=3.0)))
        cfg = RenderConfig(width=96, height=72, sensor_outputs=("rgb", "ids"))
        cfg.fog.enabled = True
        cfg.fog.density = 0.08
        cfg.fog.color = (0.7, 0.78, 0.86)
        out = render(Engine.cpu(), scene, cam, cfg)
        d_to_fog = (out.rgb - fog_c).abs().sum(dim=-1)
        return d_to_fog[out.ids == 1]

    near_px = _render_at(-3.0)
    far_px = _render_at(-18.0)
    assert near_px.numel() > 0 and far_px.numel() > 0
    assert float(far_px.mean()) < float(near_px.mean()) * 0.5, (
        "far quad must sit much closer to the fog color than the near quad"
    )


# ---------------------------------------------------------------- IBL (W2)
def test_ibl_actually_lights_scene() -> None:
    env = np.zeros((16, 32, 3), dtype=np.float32)
    env[:8, :, :] = (0.9, 0.95, 1.0)                # bright sky hemisphere
    env[8:, :, :] = (0.2, 0.18, 0.15)               # dark ground
    mesh = _quad(material=PBRMaterial(albedo=(0.7, 0.7, 0.7), roughness=0.9))
    cam = _camera()

    def _render(intensity: float | None) -> torch.Tensor:
        scene = Scene(background=None).add(mesh)
        if intensity is not None:
            scene.add(IBL(pixels=env, intensity=intensity))
        return render(Engine.cpu(), scene, cam, RenderConfig(width=64, height=48)).rgb

    none = _render(None)
    one = _render(1.0)
    two = _render(2.0)
    hit = one.sum(dim=-1) > 0
    assert hit.any()
    assert float(one[hit].mean()) > float(none[hit].mean()) * 1.3, (
        "IBL must contribute real environment light (not a 0.05 placebo)"
    )
    assert float(two[hit].mean()) > float(one[hit].mean()) * 1.5


# --------------------------------------------------------- textures (W1)
def test_textured_quad_shows_texel_variation(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import imageio.v3 as iio

    # 4x4 checkerboard texture (sRGB black/white).
    tex = np.zeros((4, 4, 3), dtype=np.uint8)
    tex[::2, ::2] = 255
    tex[1::2, 1::2] = 255
    tex_path = tmp_path / "checker.png"
    iio.imwrite(tex_path, tex)

    mat = PBRMaterial(albedo=(1.0, 1.0, 1.0), roughness=0.9,
                      albedo_map=str(tex_path))
    mesh = _quad(size=1.0, material=mat, with_uv=True)
    scene = (Scene(background=None).add(mesh)
             .add(DirectionalLight(direction=(0, -1, 0), intensity=2.0)))
    cam = PerspectiveCamera(position=(0.0, 2.0, 0.5), look_at=(0, 0, 0), fov_deg=45)
    cfg = RenderConfig(width=64, height=64)
    out = render(Engine.cpu(), scene, cam, cfg)
    hit = out.rgb.sum(dim=-1) > 0
    assert hit.any()
    vals = out.rgb[hit].mean(dim=-1)
    assert float(vals.std()) > 0.05, (
        f"textured quad must show texel variation (std={float(vals.std()):.4f})"
    )


def test_no_nan_or_inf_in_lit_scene() -> None:
    env = np.random.default_rng(0).uniform(0.0, 1.0, (8, 16, 3)).astype(np.float32)
    mesh = _quad(material=PBRMaterial(albedo=(0.6, 0.5, 0.4), roughness=0.4, metallic=0.3),
                 with_uv=False)
    scene = (Scene().add(mesh).add(IBL(pixels=env, intensity=1.5))
             .add(DirectionalLight(direction=(-0.3, -1.0, -0.2), intensity=3.0)))
    cfg = RenderConfig(width=64, height=48, output_color_space="sRGB")
    out = render(Engine.cpu(), scene, _camera(), cfg)
    assert torch.isfinite(out.rgb).all(), "render must not contain NaN/Inf"
