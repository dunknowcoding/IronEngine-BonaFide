"""Visual-quality regression tests (CPU backend).

Covers the BF visual upgrade:
  * SSAA render-scale + area-average downsample (W-AA1)
  * FXAA-style luma-edge post pass (W-AA2)
  * threshold/separable-gaussian bloom with radius/intensity config (W-GLOW)
  * screen-space god rays from the sun (W-RAYS)
  * sun/moon sky discs (+ horizon glow) (W-SKY)
  * two-pass alpha-blended transparency for meshes and point clouds (W-A)

Staircase-energy metric: per-row sub-pixel edge-crossing positions of a
diagonal edge are fit to a line; the residual variance is the staircase
energy. Restricted to the central band of rows so triangle-corner
curvature cannot contaminate the measurement.
"""
from __future__ import annotations

import base64
import json
import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from ironengine_bonafide.api import (
    Background,
    DirectionalLight,
    Engine,
    Mesh,
    PBRMaterial,
    PerspectiveCamera,
    PointCloud,
    RenderConfig,
    Scene,
    render,
)
from ironengine_bonafide.errors import ConfigurationError

# ---------------------------------------------------------------- helpers
_CAM = dict(position=(0.0, 0.0, 0.0), look_at=(0.0, 0.0, -1.0), fov_deg=50)


def _cam() -> PerspectiveCamera:
    return PerspectiveCamera(**_CAM)


def _flat_quad(z: float, size: float, material: PBRMaterial) -> Mesh:
    s = size
    pos = np.array([[-s, -s, z], [s, -s, z], [s, s, z], [-s, s, z]], dtype=np.float32)
    nrm = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (4, 1))
    idx = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return Mesh.from_arrays(pos, idx, normals=nrm, material=material)


def _diagonal_edge_scene() -> Scene:
    """Emissive right triangle on void-black — one long diagonal edge."""
    positions = np.array([[-2.5, -2.0, -5.0], [2.0, -2.0, -5.0], [-2.5, 1.5, -5.0]],
                         dtype=np.float32)
    normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (3, 1))
    indices = np.array([[0, 1, 2]], dtype=np.int64)
    mat = PBRMaterial(albedo=(0.0, 0.0, 0.0), emissive=(1.0, 1.0, 1.0), roughness=1.0)
    return Scene(background=None).add(
        Mesh.from_arrays(positions, indices, normals=normals, material=mat)
    )


def _staircase_energy(rgb: torch.Tensor) -> float:
    luma = rgb.sum(-1).cpu().numpy()
    h, w = luma.shape
    pts = []
    for y in range(h):
        row = luma[y]
        above = row > 0.5
        trans = np.nonzero(above[:-1] != above[1:])[0]
        if len(trans) == 0:
            continue
        i = int(trans[0])
        v0, v1 = row[i], row[i + 1]
        t = (0.5 - v0) / (v1 - v0 + 1e-12)
        pts.append((y, i + t))
    pts = np.array(pts)
    n = len(pts)
    mid = pts[int(n * 0.25): int(n * 0.75)]             # corner-robust band
    fit = np.polyval(np.polyfit(mid[:, 0], mid[:, 1], 1), mid[:, 0])
    return float(np.mean((mid[:, 1] - fit) ** 2))


_BASE = dict(width=96, height=96, bloom=False, shadows="off")


# ------------------------------------------------------------------ SSAA
def test_ssaa_reduces_staircase_energy() -> None:
    scene = _diagonal_edge_scene()
    eng = Engine.cpu()
    off = render(eng, scene, _cam(), RenderConfig(aa="off", ssaa=1, **_BASE)).rgb
    ss4 = render(eng, scene, _cam(), RenderConfig(aa="off", ssaa=4, **_BASE)).rgb
    e_off, e_ss4 = _staircase_energy(off), _staircase_energy(ss4)
    assert e_ss4 < e_off * 0.7, (
        f"ssaa=4 must measurably reduce staircase energy ({e_ss4:.4f} vs {e_off:.4f})"
    )


def test_ssaa_output_resolution_and_area_average() -> None:
    scene = _diagonal_edge_scene()
    cfg = RenderConfig(aa="off", ssaa=2, **_BASE)
    out = render(Engine.cpu(), scene, _cam(), cfg)
    assert tuple(out.rgb.shape) == (96, 96, 3), "output must be config resolution"
    # Area averaging produces intermediate coverage values; nearest would not.
    uniq = torch.unique(out.rgb.sum(-1))
    assert uniq.numel() > 2, "ssaa downsample must area-average (coverage greys)"


def test_ssaa_downsample_helpers() -> None:
    from ironengine_bonafide.passes.aa_pass import (
        area_downsample,
        depth_downsample,
        ids_downsample,
    )
    x = torch.arange(16, dtype=torch.float32).reshape(4, 4, 1)
    d = area_downsample(x, 2)
    assert torch.allclose(d[:, :, 0],
                          torch.tensor([[2.5, 4.5], [10.5, 12.5]]))
    depth = torch.tensor([[0.1, 0.2], [0.3, float("inf")]])
    dd = depth_downsample(depth, 2)
    assert float(dd[0, 0]) == pytest.approx(0.1), "min-pool keeps the closest surface"
    all_inf = torch.full((2, 2), float("inf"))
    assert float(depth_downsample(all_inf, 2)[0, 0]) == float("inf")
    ids = torch.arange(16, dtype=torch.int64).reshape(4, 4)
    assert int(ids_downsample(ids, 2)[0, 0]) == int(ids[1, 1])


def test_ssaa_invalid_factor_rejected() -> None:
    with pytest.raises(ConfigurationError):
        RenderConfig(ssaa=3).validate()
    with pytest.raises(ConfigurationError):
        RenderConfig(bloom_radius=9).validate()
    with pytest.raises(ConfigurationError):
        RenderConfig(god_ray_decay=0.0).validate()


# ------------------------------------------------------------------ FXAA
def test_fxaa_reduces_staircase_energy() -> None:
    scene = _diagonal_edge_scene()
    eng = Engine.cpu()
    off = render(eng, scene, _cam(), RenderConfig(aa="off", ssaa=1, **_BASE)).rgb
    fx = render(eng, scene, _cam(), RenderConfig(aa="fxaa", ssaa=1, **_BASE)).rgb
    e_off, e_fx = _staircase_energy(off), _staircase_energy(fx)
    assert e_fx < e_off * 0.7, (
        f"fxaa must measurably reduce staircase energy ({e_fx:.4f} vs {e_off:.4f})"
    )


def test_fxaa_is_bit_exact_in_flat_regions() -> None:
    from ironengine_bonafide.passes.postprocess import _fxaa
    flat = torch.full((8, 8, 3), 0.37)
    assert torch.equal(_fxaa(flat), flat)


# ------------------------------------------------------------------ bloom
def _emissive_quad_scene(strength: float = 8.0) -> Scene:
    mat = PBRMaterial(albedo=(0.0, 0.0, 0.0), emissive=(strength,) * 3, roughness=1.0)
    return Scene(background=None).add(_flat_quad(-5.0, 0.5, mat))


def test_bloom_disc_produces_halo() -> None:
    scene = _emissive_quad_scene()
    eng = Engine.cpu()
    cfg_off = RenderConfig(bloom=False, aa="off", shadows="off",
                           width=64, height=64, sensor_outputs=("rgb", "ids"))
    cfg_on = RenderConfig(bloom=True, bloom_radius=2, bloom_intensity=1.0,
                          bloom_threshold=1.0, aa="off", shadows="off",
                          width=64, height=64, sensor_outputs=("rgb", "ids"))
    off = render(eng, scene, _cam(), cfg_off)
    on = render(eng, scene, _cam(), cfg_on)
    halo = (on.ids == 0) & ((on.rgb - off.rgb).sum(-1) > 1e-4)
    assert int(halo.sum()) > 8, "bright disc must bleed halo pixels beyond its silhouette"
    # Halo extends several pixels out with radius=2.
    ys, xs = torch.nonzero(halo, as_tuple=True)
    disc_ys, disc_xs = torch.nonzero(on.ids == 1, as_tuple=True)
    cy, cx = float(disc_ys.float().mean()), float(disc_xs.float().mean())
    far = ((ys.float() - cy) ** 2 + (xs.float() - cx) ** 2).sqrt().max()
    assert float(far) > 4.0, "radius=2 halo must reach well past the disc"


def test_bloom_default_is_legacy_bit_identical() -> None:
    """radius=1 + default threshold/intensity == legacy single-blur bloom."""
    from ironengine_bonafide.passes.postprocess import _blur5
    scene = _emissive_quad_scene()
    eng = Engine.cpu()
    cfg = RenderConfig(bloom=True, aa="off", shadows="off", width=64, height=64)
    out = render(eng, scene, _cam(), cfg).rgb
    cfg_nb = RenderConfig(bloom=False, aa="off", shadows="off", width=64, height=64)
    raw = render(eng, scene, _cam(), cfg_nb).rgb
    legacy = raw + 0.6 * _blur5((raw - 1.0).clamp(min=0.0))
    assert torch.allclose(out, legacy, atol=0.0), "default bloom changed behavior"


def test_bloom_scene_without_emissives_unchanged() -> None:
    mat = PBRMaterial(albedo=(0.05, 0.05, 0.05), roughness=0.95)
    scene = (Scene(background=None).add(_flat_quad(-5.0, 1.5, mat))
             .add(DirectionalLight(direction=(0.2, -0.6, -0.77), intensity=0.3,
                                   cast_shadow=False)))
    eng = Engine.cpu()
    cfg_off = RenderConfig(bloom=False, aa="off", shadows="off", width=64, height=64)
    cfg_on = RenderConfig(bloom=True, bloom_radius=3, bloom_intensity=2.0,
                          aa="off", shadows="off", width=64, height=64)
    off = render(eng, scene, _cam(), cfg_off).rgb
    on = render(eng, scene, _cam(), cfg_on).rgb
    assert float(off.max()) < 1.0, "test scene must stay below the bloom threshold"
    assert torch.equal(off, on), "nothing above threshold → bloom must be a no-op"


# -------------------------------------------------------------- god rays
def _sun_scene(**bg_kw) -> tuple[Scene, PerspectiveCamera]:
    sun_dir = np.array([0.35, 0.30, -0.89], dtype=np.float64)
    sun_dir /= np.linalg.norm(sun_dir)
    light = DirectionalLight(direction=tuple(-sun_dir), intensity=3.0, cast_shadow=False)
    scene = Scene(background=Background(**bg_kw)).add(light)
    cam = PerspectiveCamera(position=(0, 0, 0), look_at=tuple(sun_dir * 5), fov_deg=60)
    return scene, cam


def test_god_rays_brighten_radial_path() -> None:
    scene, cam = _sun_scene(sun_disc=True)
    eng = Engine.cpu()
    base = dict(width=96, height=96, aa="off", bloom=False, shadows="off")
    off = render(eng, scene, cam, RenderConfig(god_rays=False, **base)).rgb
    on = render(eng, scene, cam, RenderConfig(god_rays=True, **base)).rgb
    diff = (on - off).sum(-1)
    assert float(diff.max()) > 1.0, "god rays must add visible energy along the radial"
    center = diff[48, 48]
    ring = diff[48, 60:80].mean()
    assert float(ring) > 0.01 or float(center) > 1.0


def test_god_rays_skip_when_sun_behind() -> None:
    # Sun direction = -light.direction = (0, 0.2, +1) → behind the camera.
    light = DirectionalLight(direction=(0.0, -0.2, -1.0), intensity=3.0, cast_shadow=False)
    scene = Scene(background=Background(sun_disc=True)).add(light)
    cam = PerspectiveCamera(position=(0, 0, 0), look_at=(0, 0, -5), fov_deg=60)
    base = dict(width=64, height=64, aa="off", bloom=False, shadows="off")
    eng = Engine.cpu()
    off = render(eng, scene, cam, RenderConfig(god_rays=False, **base)).rgb
    on = render(eng, scene, cam, RenderConfig(god_rays=True, **base)).rgb
    assert torch.equal(off, on), "sun behind camera → pass must self-skip"


# ------------------------------------------------------------------ sky
def test_sun_disc_is_hdr_and_glows() -> None:
    scene, cam = _sun_scene(sun_disc=False)
    eng = Engine.cpu()
    base = dict(width=96, height=96, aa="off", shadows="off")
    plain = render(eng, scene, cam, RenderConfig(bloom=False, **base)).rgb

    scene_d, cam_d = _sun_scene(sun_disc=True)
    disc = render(eng, scene_d, cam_d, RenderConfig(bloom=False, **base)).rgb
    assert float(disc.max()) > 5.0, "sun disc must be HDR (feeds the bloom halo)"
    assert float(plain.max()) < 2.0, "no disc → plain gradient sky only"
    center = disc[48, 48]
    assert float(center.sum()) > 5.0, "disc must be centered on the sun direction"

    # With bloom on, the HDR disc grows a halo.
    glow = render(eng, _sun_scene(sun_disc=True)[0], _sun_scene(sun_disc=True)[1],
                  RenderConfig(bloom=True, **base)).rgb
    halo = (glow - disc).sum(-1)
    assert float(halo.max()) > 0.05, "bloom must halo the sun disc"


def test_sun_horizon_glow_warms_horizon() -> None:
    sun_dir = np.array([0.0, 0.06, -1.0])
    sun_dir /= np.linalg.norm(sun_dir)
    light = DirectionalLight(direction=tuple(-sun_dir), intensity=3.0, cast_shadow=False)
    cam = PerspectiveCamera(position=(0, 0, 0), look_at=tuple(sun_dir * 5), fov_deg=60)
    eng = Engine.cpu()
    base = dict(width=96, height=96, aa="off", bloom=False, shadows="off")
    off = render(eng, Scene(background=Background(sun_disc=False)).add(light), cam,
                 RenderConfig(**base)).rgb
    on = render(eng, Scene(background=Background(sun_disc=True, sun_horizon_glow=0.5))
                .add(light), cam, RenderConfig(**base)).rgb
    diff = (on - off)
    warm = diff[..., 0] > diff[..., 2]                # red channel leads blue
    assert int(warm.sum()) > 50, "horizon glow must add a warm (R>B) tint band"


def test_moon_disc_renders() -> None:
    moon_dir = np.array([0.0, 0.25, -0.97])
    moon_dir /= np.linalg.norm(moon_dir)
    scene = Scene(background=Background(
        moon_disc=True, moon_direction=tuple(moon_dir), moon_disc_intensity=8.0))
    cam = PerspectiveCamera(position=(0, 0, 0), look_at=tuple(moon_dir * 5), fov_deg=60)
    out = render(Engine.cpu(), scene, cam,
                 RenderConfig(width=64, height=64, aa="off", bloom=False, shadows="off"))
    assert float(out.rgb[32, 32].sum()) > 4.0, "moon disc must sit at moon_direction"


def test_default_background_has_no_disc() -> None:
    scene, cam = _sun_scene()
    out = render(Engine.cpu(), scene, cam,
                 RenderConfig(width=64, height=64, aa="off", bloom=False, shadows="off"))
    assert float(out.rgb.max()) < 2.0, "default background must stay disc-free"


# ----------------------------------------------------------- transparency
def _glass_scenes() -> tuple[Scene, Scene]:
    red = _flat_quad(-6.0, 2.0, PBRMaterial(albedo=(0.9, 0.1, 0.1), roughness=0.9))
    glass = _flat_quad(-3.0, 1.2, PBRMaterial(albedo=(0.9, 0.9, 0.95), roughness=0.2,
                                              alpha=0.5))
    lt = DirectionalLight(direction=(0.2, -0.6, -0.77), intensity=2.5, cast_shadow=False)
    both = Scene(background=None).add(red).add(glass).add(lt)
    red_only = Scene(background=None).add(red).add(lt)
    return both, red_only


def test_glass_pane_blends_over_object() -> None:
    both, red_only = _glass_scenes()
    eng = Engine.cpu()
    cfg_t = RenderConfig(transparency=True, aa="off", bloom=False, shadows="off",
                         width=64, height=64)
    cfg_o = RenderConfig(transparency=False, aa="off", bloom=False, shadows="off",
                         width=64, height=64)
    blend = render(eng, both, _cam(), cfg_t).rgb
    opaque = render(eng, both, _cam(), cfg_o).rgb       # glass rendered opaque (legacy)
    behind = render(eng, red_only, _cam(), cfg_o).rgb
    c = (32, 32)
    expected = 0.5 * opaque[c] + 0.5 * behind[c]
    assert torch.allclose(blend[c], expected, atol=1e-4), (
        f"glass must be a 50/50 blend (got {blend[c].tolist()}, "
        f"want {expected.tolist()})"
    )
    assert not torch.allclose(blend[c], opaque[c], atol=1e-3), (
        "blended glass must differ from the opaque render"
    )


def test_transparency_off_is_legacy_opaque() -> None:
    red = _flat_quad(-6.0, 2.0, PBRMaterial(albedo=(0.9, 0.1, 0.1), roughness=0.9))
    lt = DirectionalLight(direction=(0.2, -0.6, -0.77), intensity=2.5, cast_shadow=False)
    glass_half = _flat_quad(-3.0, 1.2, PBRMaterial(albedo=(0.9, 0.9, 0.95),
                                                   roughness=0.2, alpha=0.5))
    glass_full = _flat_quad(-3.0, 1.2, PBRMaterial(albedo=(0.9, 0.9, 0.95),
                                                   roughness=0.2, alpha=1.0))
    cfg = RenderConfig(transparency=False, aa="off", bloom=False, shadows="off",
                       width=64, height=64)
    eng = Engine.cpu()
    a = render(eng, Scene(background=None).add(red).add(glass_half).add(lt), _cam(), cfg)
    b = render(eng, Scene(background=None).add(red).add(glass_full).add(lt), _cam(), cfg)
    assert torch.equal(a.rgb, b.rgb), (
        "transparency off → alpha must be ignored (bit-identical legacy opaque)"
    )


def test_opaque_scene_bit_identical_with_transparency_on() -> None:
    mat = PBRMaterial(albedo=(0.6, 0.5, 0.4), roughness=0.6)
    lt = DirectionalLight(direction=(0.2, -0.6, -0.77), intensity=2.5, cast_shadow=False)
    scene = Scene(background=None).add(_flat_quad(-5.0, 1.5, mat)).add(lt)
    eng = Engine.cpu()
    base = dict(aa="off", bloom=False, shadows="off", width=64, height=64)
    off = render(eng, scene, _cam(), RenderConfig(transparency=False, **base)).rgb
    on = render(eng, scene, _cam(), RenderConfig(transparency=True, **base)).rgb
    assert torch.equal(off, on), "all-opaque scene must not change with transparency on"


def test_transparent_depth_not_written() -> None:
    """A nearer opaque object behind the glass plane must still draw through
    it (depth test vs opaque buffer, no depth write from the glass)."""
    red = _flat_quad(-6.0, 2.0, PBRMaterial(albedo=(0.9, 0.1, 0.1), roughness=0.9))
    glass = _flat_quad(-3.0, 1.2, PBRMaterial(albedo=(0.9, 0.9, 0.95), roughness=0.2,
                                              alpha=0.4))
    lt = DirectionalLight(direction=(0.2, -0.6, -0.77), intensity=2.5, cast_shadow=False)
    scene = Scene(background=None).add(red).add(glass).add(lt)
    cfg = RenderConfig(transparency=True, aa="off", bloom=False, shadows="off",
                       width=64, height=64, sensor_outputs=("rgb", "depth"))
    out = render(Engine.cpu(), scene, _cam(), cfg)
    c = (32, 32)
    # Depth at the glass pixel must be the RED quad's depth, not the glass's.
    red_only = Scene(background=None).add(red).add(lt)
    ref = render(Engine.cpu(), red_only, _cam(), cfg)
    assert float(out.depth[c]) == pytest.approx(float(ref.depth[c]), abs=1e-5), (
        "transparent surface must not write depth"
    )


def test_point_cloud_opacities_blend() -> None:
    g = np.random.default_rng(5)
    pos = g.uniform(-0.8, 0.8, (400, 3)).astype(np.float32)
    pos[:, 2] = -4.0
    col = np.tile(np.array([[0.1, 0.9, 0.2]], dtype=np.float32), (400, 1))
    cloud = PointCloud.from_arrays(pos, col)
    cloud.point_size_px = 6.0
    cloud.opacities = torch.full((400,), 0.5)
    bg = Background(mode="solid", color=(0.8, 0.1, 0.1))
    eng = Engine.cpu()
    base = dict(width=64, height=64, aa="off", bloom=False, shadows="off")
    opaque = render(eng, Scene(background=bg).add(cloud), _cam(),
                    RenderConfig(transparency=False, **base)).rgb
    blend = render(eng, Scene(background=bg).add(cloud), _cam(),
                   RenderConfig(transparency=True, **base)).rgb
    cov = opaque.sum(-1).argmax()
    cy, cx = np.unravel_index(int(cov), opaque.shape[:2])
    g_ch_o, g_ch_b = float(opaque[cy, cx, 1]), float(blend[cy, cx, 1])
    r_ch_o, r_ch_b = float(opaque[cy, cx, 0]), float(blend[cy, cx, 0])
    assert g_ch_b < g_ch_o and r_ch_b > r_ch_o, (
        "50% opacity splats must blend toward the red background"
    )


# ------------------------------------------------- GLB baseColorFactor alpha
def _build_alpha_glb(path: Path, alpha: float) -> None:
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    idx = np.array([0, 1, 2], dtype=np.uint32)
    blob = pos.tobytes() + idx.tobytes()
    doc = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(blob),
                     "uri": "data:application/octet-stream;base64,"
                            + base64.b64encode(blob).decode()}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 36},
            {"buffer": 0, "byteOffset": 36, "byteLength": 12},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5125, "count": 3, "type": "SCALAR"},
        ],
        "materials": [{
            "name": "glass",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.9, 0.95, 1.0, alpha],
                "roughnessFactor": 0.1,
            },
        }],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1,
                                    "material": 0}]}],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    js = json.dumps(doc).encode()
    pad = (4 - len(js) % 4) % 4
    js += b" " * pad
    glb = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js))
    glb += struct.pack("<II", len(js), 0x4E4F534A) + js
    path.write_bytes(glb)


def test_gltf_basecolor_alpha_lands_on_material(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("pygltflib", reason="pygltflib required for glTF tests")
    from ironengine_bonafide.assets.loaders.gltf import load_primitives
    p = tmp_path / "glass.glb"
    _build_alpha_glb(p, 0.5)
    prims = load_primitives(p)
    assert len(prims) == 1
    assert prims[0].alpha == pytest.approx(0.5)
    assert prims[0].mesh.material.alpha == pytest.approx(0.5), (
        "PBRMaterial.alpha must honor glTF baseColorFactor alpha"
    )


def test_pbrmaterial_alpha_round_trip() -> None:
    m = PBRMaterial(alpha=0.25)
    m2 = PBRMaterial.from_dict(m.to_dict())
    assert m2.alpha == pytest.approx(0.25)
    assert PBRMaterial.from_dict({}).alpha == 1.0, "alpha defaults to opaque"


# ------------------------------------------------------------ config misc
def test_new_fields_round_trip() -> None:
    cfg = RenderConfig(ssaa=2, bloom_threshold=0.8, bloom_intensity=1.2,
                       bloom_radius=3, god_rays=True, god_ray_samples=32,
                       god_ray_decay=0.9, god_ray_intensity=0.8, transparency=True)
    cfg2 = RenderConfig.from_dict(cfg.to_dict())
    assert cfg2.ssaa == 2 and cfg2.bloom_radius == 3 and cfg2.god_rays
    assert cfg2.god_ray_samples == 32 and cfg2.transparency
    json.dumps(cfg.to_dict())                           # must stay JSON-safe
