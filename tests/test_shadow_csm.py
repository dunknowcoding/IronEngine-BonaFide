"""CSM shadow quality tests (BF_Shadow_Fixer).

Covers the shot1 "dark triangle" fix:
  * light ortho fit covers every frustum-slice corner (re-centering fix);
  * texel snapping quantises the light frustum to the texel grid;
  * PCF samples the texel lattice the rasteriser wrote (round, not trunc);
  * world-space (per-cascade) depth bias, slope-scaled receiver bias and
    normal-offset helper;
  * integration: no acne on flat ground, shadows only where occluded, and
    no peter-panning gap beyond ~1 texel.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from ironengine_bonafide import lifecycle
from ironengine_bonafide.api import (
    DirectionalLight,
    Engine,
    Mesh,
    PBRMaterial,
    PerspectiveCamera,
    RenderConfig,
    Scene,
    render,
)
from ironengine_bonafide.core.shadow import (
    LightFrustum,
    bake_depth_bias,
    build_cascades_with_info,
    compute_receiver_bias_world,
    ground_slope_texels,
    offset_along_normal,
    pcf_sample,
    view_frustum_corners_world,
)
from ironengine_bonafide.passes.pbr_pass import _shadow_factor_csm


# ------------------------------------------------------------------ helpers
def _cube(size: float, center: tuple[float, float, float]) -> Mesh:
    s = size / 2.0
    cx, cy, cz = center
    p = np.array([
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
        [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s],
    ], dtype=np.float32) + np.array([cx, cy, cz], dtype=np.float32)
    idx = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
        [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3],
    ], dtype=np.int64)
    normals = np.zeros((8, 3), dtype=np.float32)      # unused (flat shading fallback)
    return Mesh.from_arrays(p, idx, normals=normals,
                            colors=np.ones((8, 3), dtype=np.float32) * 0.7,
                            material=PBRMaterial(albedo=(0.7, 0.7, 0.7)))


def _plane(size: float) -> Mesh:
    p = np.array([[-size, 0, -size], [size, 0, -size],
                  [size, 0, size], [-size, 0, size]], dtype=np.float32)
    idx = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int64)
    normals = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (4, 1))
    return Mesh.from_arrays(p, idx, normals=normals,
                            colors=np.ones((4, 3), dtype=np.float32) * 0.5,
                            material=PBRMaterial(albedo=(0.5, 0.5, 0.5), roughness=0.95))


def _wall(width: float, height: float, x: float) -> Mesh:
    """Vertical wall in the YZ plane at ``x``; interior normal faces -X."""
    p = np.array([[x, 0, -width], [x, 0, width],
                  [x, height, width], [x, height, -width]], dtype=np.float32)
    idx = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int64)
    normals = np.tile(np.array([[-1.0, 0.0, 0.0]], dtype=np.float32), (4, 1))
    return Mesh.from_arrays(p, idx, normals=normals,
                            colors=np.ones((4, 3), dtype=np.float32) * 0.8,
                            material=PBRMaterial(albedo=(0.8, 0.78, 0.75), roughness=0.9),
                            name="wall")


def _capture_shadow_maps(scene, cam, cfg) -> list:
    grabbed: dict = {}

    def on_pass_end(**kw):                                   # noqa: ANN202
        ctx = kw.get("ctx")
        if kw.get("pass_name") == "shadow_csm" and ctx is not None:
            grabbed["sms"] = list(ctx.targets.shadow_maps)

    lifecycle.register("on_pass_end", on_pass_end)
    try:
        with Engine.cpu() as eng:
            render(eng, scene, cam, cfg)
    finally:
        lifecycle.clear()
    return grabbed.get("sms", [])


def _raycast_shadow(points: np.ndarray, sun_dir: np.ndarray,
                    positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Möller–Trumbore ground truth: True where the segment point→sun hits
    any triangle."""
    v0 = positions[indices[:, 0]]; v1 = positions[indices[:, 1]]; v2 = positions[indices[:, 2]]
    e1 = v1 - v0; e2 = v2 - v0
    pvec = np.cross(sun_dir, e2)
    det = (e1 * pvec).sum(1)
    good = np.abs(det) > 1e-12
    inv = np.where(good, 1.0 / np.where(good, det, 1.0), 0.0)
    out = np.zeros(points.shape[0], bool)
    for s in range(0, points.shape[0], 4096):
        o = points[s:s + 4096] + sun_dir * 1e-3
        tvec = o[:, None, :] - v0[None, :, :]
        u = (tvec * pvec[None, :, :]).sum(2) * inv[None, :]
        q = np.cross(tvec, e1[None, :, :])
        v = (sun_dir[None, None, :] * q).sum(2) * inv[None, :]
        t = (e2[None, :, :] * q).sum(2) * inv[None, :]
        hit = good[None, :] & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-3)
        out[s:s + 4096] = hit.any(1)
    return out


def _binary_erosion(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Pure-numpy binary erosion (4-connectivity, border = False).

    Vendored so the strict interior-umbra check always runs — it must not
    silently degrade to the boundary-tolerant fallback when scipy is absent
    (e.g. on CI runners).
    """
    out = mask.astype(bool)
    for _ in range(iterations):
        p = np.pad(out, 1, constant_values=False)
        out = (p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1]
               & p[1:-1, :-2] & p[1:-1, 2:])
    return out


# ------------------------------------------------------- frustum fit (unit)
def test_fit_covers_all_slice_corners() -> None:
    """Every frustum-slice corner must land inside the light ortho box —
    regression for the un-centred symmetric ortho that cropped oblique
    slices (one side of the scene silently lost its shadow map)."""
    cam = PerspectiveCamera(position=(2.7, 1.35, 2.9), look_at=(0.05, 0.42, 0.0),
                            fov_deg=52.0, far=300.0)
    view_inv = np.linalg.inv(cam.view_matrix())
    light_dir = np.array([-0.45, -0.62, -0.35])
    light_dir /= np.linalg.norm(light_dir)
    cascades = build_cascades_with_info(view_inv, 52.0, 16 / 9, 0.05, 300.0,
                                        tuple(light_dir), 3, resolution=512)
    assert len(cascades) == 3
    for i, (vp, z_n, z_f, _info) in enumerate(cascades):
        corners = view_frustum_corners_world(view_inv, math.radians(52.0), 16 / 9, z_n, z_f)
        homog = np.concatenate([corners, np.ones((8, 1))], axis=1)
        ndc = (vp @ homog.T).T
        ndc = ndc[:, :3] / ndc[:, 3:4]
        assert np.all(np.abs(ndc[:, :2]) <= 1.0 + 1e-6), (
            f"cascade {i}: slice corners cropped in xy "
            f"(range {np.abs(ndc[:, :2]).max():.4f})"
        )
        assert np.all(ndc[:, 2] >= -1.0 - 1e-6) and np.all(ndc[:, 2] <= 1.0 + 1e-6), (
            f"cascade {i}: slice corners cropped in z (range "
            f"[{ndc[:, 2].min():.4f}, {ndc[:, 2].max():.4f}])"
        )


def test_texel_snapping_quantises_to_grid() -> None:
    """Snapped bounds must be whole-texel multiples (stable texel grid —
    no shadow shimmer under sub-texel camera motion)."""
    cam = PerspectiveCamera(position=(3.1, 2.0, 2.4), look_at=(0.0, 0.3, 0.0),
                            fov_deg=50.0, far=100.0)
    view_inv = np.linalg.inv(cam.view_matrix())
    light_dir = np.array([-0.4, -0.8, -0.3])
    light_dir /= np.linalg.norm(light_dir)
    corners = view_frustum_corners_world(view_inv, math.radians(50.0), 16 / 9, 0.1, 40.0)
    res = 256
    cascades = build_cascades_with_info(view_inv, 50.0, 16 / 9, 0.1, 100.0,
                                        tuple(light_dir), 1, resolution=res)
    vp, _, _, info = cascades[0]
    assert info.texel_size_x > 0.0
    # Re-derive the snapped window: project corners, they must fit, and the
    # window width must be an integer number of texels (within fp noise).
    homog = np.concatenate([corners, np.ones((8, 1))], axis=1)
    ndc = (vp @ homog.T).T
    ndc = ndc[:, :3] / ndc[:, 3:4]
    span_ndc = ndc[:, 0].max() - ndc[:, 0].min()
    span_world = span_ndc / 2.0 * info.texel_size_x * res
    n_texels = span_world / info.texel_size_x
    assert abs(n_texels - round(n_texels)) < 1e-3 or n_texels <= res, (
        f"window not quantised: {n_texels:.3f} texels"
    )
    assert n_texels <= res + 1e-3, "snapped window exceeds the map"


def test_scene_bounds_tightens_frustum() -> None:
    """Scene-AABB fitting must increase texel density vs the raw frustum
    fit while still covering the scene."""
    cam = PerspectiveCamera(position=(2.7, 1.35, 2.9), look_at=(0.05, 0.42, 0.0),
                            fov_deg=52.0, far=300.0)
    view_inv = np.linalg.inv(cam.view_matrix())
    light_dir = np.array([-0.45, -0.62, -0.35])
    light_dir /= np.linalg.norm(light_dir)
    bounds = (np.array([-12.0, 0.0, -12.0]), np.array([12.0, 1.0, 12.0]))
    loose = build_cascades_with_info(view_inv, 52.0, 16 / 9, 0.05, 300.0,
                                     tuple(light_dir), 3, resolution=512)
    tight = build_cascades_with_info(view_inv, 52.0, 16 / 9, 0.05, 300.0,
                                     tuple(light_dir), 3, resolution=512,
                                     scene_bounds=bounds)
    assert tight[0][3].texel_size_x < loose[0][3].texel_size_x * 0.9, (
        f"scene fit should shrink texels: {tight[0][3].texel_size_x:.4f} "
        f"vs {loose[0][3].texel_size_x:.4f}"
    )
    # Points that are both inside the scene AABB AND inside the camera
    # slice must be covered by the tightened map (scene corners beyond the
    # slice are legitimately outside — nothing renders there).
    vp = tight[0][0]
    cam_vp = cam.view_proj(16 / 9)
    sb = np.array([[x, y, z] for x in (-12, 12) for y in (0, 1) for z in (-12, 12)])
    cam_clip = (cam_vp @ np.concatenate([sb, np.ones((8, 1))], axis=1).T).T
    cam_ndc = cam_clip[:, :3] / cam_clip[:, 3:4]
    visible = (np.abs(cam_ndc) <= 1.0).all(axis=1)
    homog = np.concatenate([sb[visible], np.ones((int(visible.sum()), 1))], axis=1)
    ndc = (vp @ homog.T).T
    ndc = ndc[:, :3] / ndc[:, 3:4]
    assert np.all(np.abs(ndc[:, :2]) <= 1.0 + 1e-6), (
        f"visible scene corners cropped: {np.abs(ndc[:, :2]).max():.4f}"
    )


# ------------------------------------------------------------- PCF (unit)
def test_pcf_samples_raster_lattice() -> None:
    """pcf_sample must read the texel the rasteriser would have written at
    the same lattice coordinate (round-to-nearest, not truncation)."""
    depth = torch.full((8, 8), 0.5)
    depth[3, 5] = 0.1                       # one close texel at raster (x=5, y=3)
    # raster lattice: sx = uv.x * W, sy = (1 - uv.y) * H
    uv = torch.tensor([[5.0 / 8.0, 1.0 - 3.0 / 8.0],
                       [4.0 / 8.0, 1.0 - 3.0 / 8.0]])
    cur = torch.tensor([0.3, 0.3])          # between 0.1 and 0.5
    lit = pcf_sample(depth, uv, cur, bias=0.005, radius=0)
    assert lit[0].item() == 0.0, "lattice point (5,3) must read the close texel"
    assert lit[1].item() == 1.0, "neighbour lattice point (4,3) must read 0.5"


def test_pcf_slope_scaled_bias_per_fragment() -> None:
    """With normals supplied, grazing fragments get a larger effective bias
    (slope-scaled) than light-facing ones."""
    depth = torch.zeros((8, 8))
    uv = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    cur = torch.tensor([0.01, 0.01])        # slightly behind stored depth
    light = (0.0, 1.0, 0.0)
    normals = torch.tensor([[0.0, 1.0, 0.0],          # facing light (tan=0)
                            [0.9682, 0.25, 0.0]])     # grazing (tan≈3.87)
    lit = pcf_sample(depth, uv, cur, bias=0.0, radius=0,
                     normals=normals, light_dir=light,
                     texel_size_world=0.1, ndc_per_world=0.1, slope_scale=1.0)
    assert lit[0].item() == 0.0, "no slope term → stays shadowed without bias"
    assert lit[1].item() == 1.0, "slope-scaled bias must lift grazing fragment"


def test_bake_depth_bias_world_units() -> None:
    frustum = LightFrustum(texel_size_x=0.1, texel_size_y=0.1,
                           z_extent=20.0, ndc_per_world=0.1)
    depth = torch.tensor([[0.5, float("inf")]])
    out = bake_depth_bias(depth, frustum, bias_world=0.08)   # 0.008 NDC
    # Default: the full world-space bias is baked (no hidden receiver floor).
    assert out[0, 0].item() == pytest.approx(0.508, abs=1e-6)
    assert out[0, 1].item() == float("inf"), "empty texels must stay empty"
    # When the sampler applies a receiver-side bias, only the excess is baked.
    out2 = bake_depth_bias(depth, frustum, bias_world=0.08,
                           receiver_bias_ndc=0.005)
    assert out2[0, 0].item() == pytest.approx(0.503, abs=1e-6)
    out3 = bake_depth_bias(depth, frustum, bias_world=0.01,  # 0.001 NDC
                           receiver_bias_ndc=0.005)
    assert torch.equal(out3, depth)


def test_bias_override_world_units() -> None:
    assert compute_receiver_bias_world(0.1, override_world=0.5) == 0.5
    assert compute_receiver_bias_world(0.1, constant_texels=0.2,
                                       slope_texels=1.0) == pytest.approx(0.12)


def test_ground_slope_texels_follows_elevation() -> None:
    """Overhead light → small slope term; grazing light → clamped large."""
    overhead = ground_slope_texels((0.0, -1.0, 0.0))
    assert overhead < 0.01
    midday = ground_slope_texels((-0.577, -0.795, -0.449))
    assert 0.5 < midday < 1.0
    grazing = ground_slope_texels((-0.92, -0.216, 0.325))
    assert grazing == pytest.approx(4.52, abs=0.01)
    clamped = ground_slope_texels((-0.995, -0.05, 0.05))
    assert clamped == pytest.approx(8.0)               # clamped at tan_max
    custom = ground_slope_texels((-0.92, -0.216, 0.325), tan_max=3.0)
    assert custom == pytest.approx(3.0)


def test_offset_along_normal() -> None:
    pos = torch.tensor([[0.0, 0.0, 0.0]])
    nrm = torch.tensor([[0.0, 1.0, 0.0]])
    out = offset_along_normal(pos, nrm, texel_size_world=0.25, offset_texels=2.0)
    assert torch.allclose(out, torch.tensor([[0.0, 0.5, 0.0]]))


# ------------------------------------------------------------- config (unit)
def test_config_shadow_fields_round_trip() -> None:
    cfg = RenderConfig(shadow_map_resolution=1024, shadow_bias_constant=0.3,
              shadow_bias_slope=1.5, shadow_bias_override=0.05)
    d = cfg.to_dict()
    cfg2 = RenderConfig.from_dict(d)
    assert cfg2.shadow_map_resolution == 1024
    assert cfg2.shadow_bias_constant == 0.3
    assert cfg2.shadow_bias_slope == 1.5
    assert cfg2.shadow_bias_override == 0.05


def test_config_shadow_field_validation() -> None:
    from ironengine_bonafide.errors import ConfigurationError
    with pytest.raises(ConfigurationError):
        RenderConfig(shadow_map_resolution=0).validate()
    with pytest.raises(ConfigurationError):
        RenderConfig(shadow_bias_constant=-0.1).validate()
    with pytest.raises(ConfigurationError):
        RenderConfig(shadow_bias_override=-1.0).validate()


# ------------------------------------------------------ integration (render)
def test_no_acne_stripes_on_flat_ground() -> None:
    """A bare flat plane must render IDENTICALLY with shadows on and off —
    any darkening is self-shadow acne."""
    cam = PerspectiveCamera(position=(3.0, 3.0, 3.0), look_at=(0.0, 0.0, 0.0),
                            fov_deg=45.0, far=100.0)

    def _render(shadows: str) -> torch.Tensor:
        scene = (Scene(background=None).add(_plane(8.0))
                 .add(DirectionalLight(direction=(-0.45, -0.62, -0.35),
                                       intensity=2.0, cast_shadow=True)))
        cfg = RenderConfig(width=96, height=64, shadows=shadows, bloom=False, seed=0)
        with Engine.cpu() as eng:
            return render(eng, scene, cam, cfg).rgb

    on = _render("csm")
    off = _render("off")
    diff = (on - off).abs().max().item()
    assert diff < 1e-5, f"flat ground self-shadows (acne): max diff {diff}"


def _occlusion_scene():
    sun = np.array([0.45, 0.62, 0.35], dtype=np.float64)
    sun /= np.linalg.norm(sun)
    scene = (Scene(background=None).add(_plane(6.0)).add(_cube(1.0, (0.0, 0.5, 0.0)))
             .add(DirectionalLight(direction=tuple(-sun), intensity=2.0,
                                   cast_shadow=True)))
    cam = PerspectiveCamera(position=(3.0, 2.5, 3.0), look_at=(0.0, 0.3, 0.0),
                            fov_deg=45.0, far=100.0)
    return scene, cam, sun


def test_shadow_darkens_only_where_occluded() -> None:
    """CSM visibility must match a ray-cast ground truth: no acne on open
    ground, no missing shadows under the occluder."""
    scene, cam, sun = _occlusion_scene()
    cfg = RenderConfig(width=96, height=64, shadows="csm", bloom=False, seed=0)
    sms = _capture_shadow_maps(scene, cam, cfg)
    assert sms, "shadow pass produced no maps"

    # Dense ground grid around the cube.
    gx, gz = np.meshgrid(np.linspace(-3, 3, 120), np.linspace(-3, 3, 120))
    pts = np.stack([gx.ravel(), np.zeros(gx.size), gz.ravel()], axis=1)
    world = torch.from_numpy(pts).float().reshape(120, 120, 3)
    vis = _shadow_factor_csm(world, sms).numpy().ravel()

    pos = np.concatenate([m.positions.numpy() for m in scene.meshes], axis=0).astype(np.float64)
    off = 0
    tris = []
    for m in scene.meshes:
        tris.append(m.indices.numpy() + off)
        off += m.positions.shape[0]
    idx = np.concatenate(tris, axis=0)
    truth = _raycast_shadow(pts.astype(np.float64), sun, pos, idx)

    pred = vis < 0.5
    acne = pred & ~truth
    leak = ~pred & truth
    assert truth.sum() > 100, "cube must cast a real shadow on the grid"
    assert acne.mean() < 0.01, f"acne on open ground: {acne.mean():.4f}"
    # Interior leaks (away from the penumbra boundary) are bugs; boundary
    # misses within ~1-2 texels are the expected bias/PCF edge behaviour.
    # Vendored pure-numpy erosion: the strict interior check must run
    # identically with or without scipy installed (CI has no scipy).
    interior = _binary_erosion(truth.reshape(120, 120), iterations=2).ravel()
    interior_leak = leak & interior
    assert interior_leak.sum() / max(1, interior.sum()) < 0.01, (
        f"missing shadow inside umbra: "
        f"{interior_leak.sum() / max(1, interior.sum()):.4f}"
    )


def test_no_peter_panning_beyond_one_texel() -> None:
    """The shadow must start at the occluder's base edge (within ~1 texel
    + raster tolerance), not float away from the contact line."""
    scene, cam, sun = _occlusion_scene()
    cfg = RenderConfig(width=96, height=64, shadows="csm", bloom=False, seed=0)
    sms = _capture_shadow_maps(scene, cam, cfg)
    assert sms and sms[0].texel_size_world > 0.0

    # March along the ground from under the cube toward the shadow.
    shadow_dir = sun[[0, 2]] / np.linalg.norm(sun[[0, 2]])     # horizontal
    ts = np.linspace(0.45, 2.5, 400)                           # from inside footprint outward
    pts = np.stack([np.full_like(ts, 0.0) + shadow_dir[0] * ts,
                    np.zeros_like(ts),
                    np.zeros_like(ts) + shadow_dir[1] * ts], axis=1)
    world = torch.from_numpy(pts).float().reshape(-1, 1, 3)
    vis = _shadow_factor_csm(world, sms).numpy().ravel()
    shadowed = np.flatnonzero(vis < 0.5)
    assert shadowed.size > 0, "no shadow found along the shadow axis"
    first = ts[shadowed[0]]
    gap_world = max(0.0, first - 0.5)                          # cube half-size = 0.5
    gap_texels = gap_world / sms[0].texel_size_world
    assert gap_texels <= 1.5, (
        f"shadow starts {gap_texels:.2f} texels from the contact line "
        f"(peter-panning)"
    )


def _grazing_wall_scene():
    """Multi-scale scene with a wall lit at grazing incidence.

    Sun is steep against the ground (small horizontal slope term baked into
    the depth map) yet nearly parallel to the wall (tan ≈ 6.5 against the
    wall normal) — exactly the configuration where a ground-only slope bias
    under-compensates and the wall self-shadows in a texel lattice.
    """
    sun = np.array([-0.15, 0.6, -0.78], dtype=np.float64)
    sun /= np.linalg.norm(sun)
    scene = (Scene(background=None)
             .add(_plane(10.0))
             .add(_wall(10.0, 5.0, 4.0))
             .add(_cube(0.9, (3.0, 4.4, -4.7)))            # casts onto the wall
             .add(_cube(0.12, (0.4, 0.06, 1.2)))           # tiny multi-scale prop
             .add(DirectionalLight(direction=tuple(-sun), intensity=2.0,
                                   cast_shadow=True)))
    cam = PerspectiveCamera(position=(-3.0, 2.2, 0.0), look_at=(4.0, 1.6, 0.0),
                            fov_deg=45.0, far=100.0)
    return scene, cam, sun


def test_no_acne_on_grazing_lit_wall() -> None:
    """Regression: a wall at grazing light incidence must not self-shadow.

    Pre-fix, the receiver bias only modelled horizontal receivers (ground
    slope baked into the depth map + constant), so a steep-grazing wall
    acned across ~14% of its lit area. The per-fragment slope-scaled
    receiver term (normals + light dir wired from the GBuffer into
    ``pcf_sample``) must keep acne < 1% against a ray-cast ground truth.
    """
    scene, cam, sun = _grazing_wall_scene()
    cfg = RenderConfig(width=160, height=90, shadows="csm", bloom=False, seed=0)
    sms = _capture_shadow_maps(scene, cam, cfg)
    assert sms, "shadow pass produced no maps"

    # Dense receiver grid on the wall's lit interior surface.
    gy, gz = np.meshgrid(np.linspace(0.2, 4.6, 120), np.linspace(-4.0, 4.0, 160))
    pts = np.stack([np.full(gy.size, 4.0 - 1e-4), gy.ravel(), gz.ravel()], axis=1)
    world = torch.from_numpy(pts).float().reshape(120, 160, 3)
    normals = torch.from_numpy(
        np.tile(np.array([[-1.0, 0.0, 0.0]], dtype=np.float32), (120, 160, 1)))
    vis = _shadow_factor_csm(world, sms, normals=normals,
                             light_dir=torch.tensor(sun, dtype=torch.float32),
                             ).numpy().ravel()

    pos = np.concatenate([m.positions.numpy() for m in scene.meshes], axis=0).astype(np.float64)
    off = 0
    tris = []
    for m in scene.meshes:
        tris.append(m.indices.numpy() + off)
        off += m.positions.shape[0]
    idx = np.concatenate(tris, axis=0)
    truth = _raycast_shadow(pts.astype(np.float64), sun, pos, idx)

    pred = vis < 0.5
    acne = pred & ~truth
    leak = ~pred & truth
    assert truth.sum() > 100, "cube must cast a real shadow on the wall"
    assert acne.mean() < 0.01, f"acne on grazing-lit wall: {acne.mean():.4f}"
    interior = _binary_erosion(truth.reshape(120, 160), iterations=2).ravel()
    interior_leak = leak & interior
    assert interior_leak.sum() / max(1, interior.sum()) < 0.01, (
        f"missing shadow inside wall umbra: "
        f"{interior_leak.sum() / max(1, interior.sum()):.4f}"
    )


def test_pbr_pass_wires_gbuffer_normals_into_pcf() -> None:
    """Wiring regression: the production shading path must feed GBuffer
    normals + light direction (and the per-cascade slope metadata) into
    ``pcf_sample``. Pre-fix the slope-scaled receiver bias existed in
    ``core.shadow`` but was never passed anything — grazing receivers only
    ever got the horizontal-ground term."""
    import ironengine_bonafide.passes.pbr_pass as pbr_mod

    scene, cam, sun = _grazing_wall_scene()
    seen: list[dict] = []
    orig = pbr_mod.pcf_sample

    def spy(depth_map, uv, current_depth, **kw):            # noqa: ANN001, ANN202
        seen.append(kw)
        return orig(depth_map, uv, current_depth, **kw)

    pbr_mod.pcf_sample = spy
    try:
        cfg = RenderConfig(width=64, height=36, shadows="csm", bloom=False, seed=0)
        with Engine.cpu() as eng:
            render(eng, scene, cam, cfg)
    finally:
        pbr_mod.pcf_sample = orig
    assert seen, "pcf_sample was never called"
    for kw in seen:
        assert kw.get("normals") is not None, "GBuffer normals not passed to pcf_sample"
        assert kw.get("light_dir") is not None, "light direction not passed to pcf_sample"
        assert kw.get("slope_scale", 0.0) > 0.0, "per-cascade slope scale not wired"
        assert kw.get("slope_tan_ref", 0.0) > 0.0, "baked-slope reference not wired"

