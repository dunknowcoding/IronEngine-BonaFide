"""Cascaded Shadow Map plumbing.

Splits the camera frustum into N depth slices, fits an orthographic light
projection around each slice, and exposes the resulting matrices + a
container ``ShadowMap`` that the PBR pass samples with PCF.

References:
  * "Cascaded Shadow Maps", NVIDIA whitepaper, 2007
  * Zhao Pan-style PSSM cascade splits (linear + log mix)
  * MSDN "Common Techniques to Improve Shadow Depth Maps" (texel snapping,
    slope-scaled + normal-offset bias)

The actual depth raster is performed by the active backend's
``raster_depth`` helper. This module only does math.

Anti-aliasing / anti-acne measures implemented here:

  * The light ortho frustum is tightened against the scene AABB (visible
    receivers are scene geometry, so no receiver falls outside the map) and
    its XY extent is snapped to whole texels in light space — this removes
    shadow shimmering/swimming under sub-texel camera motion.
  * Depth bias is expressed in *world units* (texel-sized: constant +
    worst-case slope term) and converted to NDC per cascade — a fixed NDC
    bias is wrong because each cascade's ortho z-extent differs, so the
    same NDC epsilon means wildly different world distances per cascade.
  * :func:`pcf_sample` samples the texel lattice that the rasterizer
    actually wrote (round-to-nearest, not truncation), and optionally
    applies a receiver-side slope-scaled bias when normals are supplied.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from ironengine_bonafide.core.camera import look_at, orthographic


@dataclass(slots=True)
class LightFrustum:
    """Metadata of one fitted light ortho frustum (world units)."""
    texel_size_x: float        # world units per shadow-map texel, light X
    texel_size_y: float        # world units per shadow-map texel, light Y
    z_extent: float            # far - near of the ortho along the light axis
    ndc_per_world: float       # NDC depth units per world unit (= |m22|)


@dataclass(slots=True)
class ShadowMap:
    """One cascade's light-space matrix + depth texture."""
    light_view_proj: torch.Tensor                       # (4, 4) world -> light NDC
    depth: torch.Tensor                                 # (H, W) float32 NDC depth, +inf empty
    z_split_near: float                                 # camera-space near of this cascade
    z_split_far: float                                  # camera-space far of this cascade
    texel_size_world: float = 0.0                       # world units per texel (light X)
    bias_world: float = 0.0                             # world-space bias baked into ``depth``
    receiver_bias_ndc: float = 0.0                      # config-driven bias applied at sampling
    ndc_per_world: float = 0.0                          # NDC depth units per world unit


# --------------------------------------------------------------- splits
def cascade_splits(near: float, far: float, n_cascades: int, *, lambda_mix: float = 0.5) -> list[float]:
    """PSSM split distances. Mixes a logarithmic and a linear distribution
    via ``lambda_mix`` (0=linear, 1=log). Returns ``n_cascades + 1`` values
    starting at ``near`` and ending at ``far``."""
    splits = [near]
    for i in range(1, n_cascades):
        f = i / n_cascades
        log_split = near * (far / near) ** f
        lin_split = near + (far - near) * f
        splits.append(lambda_mix * log_split + (1.0 - lambda_mix) * lin_split)
    splits.append(far)
    return splits


# --------------------------------------------------------------- frustum corners
def view_frustum_corners_world(view_inv: np.ndarray, fov_rad: float, aspect: float,
                               near: float, far: float) -> np.ndarray:
    """8 world-space corners of the perspective frustum slice [near, far]."""
    th = math.tan(fov_rad * 0.5)
    nh = th * near; nw = nh * aspect
    fh = th * far;  fw = fh * aspect
    eye_corners = np.array([
        [-nw, -nh, -near, 1], [+nw, -nh, -near, 1],
        [+nw, +nh, -near, 1], [-nw, +nh, -near, 1],
        [-fw, -fh, -far,  1], [+fw, -fh, -far,  1],
        [+fw, +fh, -far,  1], [-fw, +fh, -far,  1],
    ], dtype=np.float64)
    world = (view_inv @ eye_corners.T).T
    return world[:, :3]


def _aabb_corners(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """8 corners of an axis-aligned box."""
    return np.array([
        [lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
        [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
        [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
        [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]],
    ], dtype=np.float64)


# --------------------------------------------------------------- light projection
def _fit_light_view_proj(
    corners_world: np.ndarray,
    light_dir: np.ndarray,
    z_padding: float,
    *,
    resolution: int | None,
    scene_bounds: tuple[np.ndarray, np.ndarray] | None,
    snap_texels: bool,
) -> tuple[np.ndarray, LightFrustum]:
    """Core fit. Returns ``(light_view_proj, LightFrustum)``.

    When ``scene_bounds`` (world AABB as ``(lo, hi)``) is given, the ortho
    XY range is intersected with the scene's light-space footprint and the
    Z range is taken from the scene (plus padding): every visible receiver
    is scene geometry and therefore stays covered, while empty frustum
    space no longer wastes shadow-map texels.

    When ``resolution`` is given and ``snap_texels`` is set, the XY bounds
    are snapped outward to whole-texel multiples so the texel grid is
    stable in world space (no shimmer under sub-texel camera motion).
    """
    light_dir = light_dir / (np.linalg.norm(light_dir) + 1e-9)
    centre = corners_world.mean(axis=0)
    eye = centre - light_dir * 50.0                                       # arbitrary, padded by ortho box anyway
    target = centre
    # Pick a stable up — avoid degenerate axis when light_dir // (0,1,0)
    if abs(light_dir[1]) > 0.999:
        up = np.array([0.0, 0.0, 1.0])
    else:
        up = np.array([0.0, 1.0, 0.0])
    view = look_at(eye, target, up)

    def to_light(pts: np.ndarray) -> np.ndarray:
        homog = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
        return (view @ homog.T).T[:, :3]

    # Project the corners into light view-space, fit an AABB.
    in_light = to_light(corners_world)
    lo = in_light.min(axis=0).copy()
    hi = in_light.max(axis=0).copy()

    if scene_bounds is not None:
        sb_lo, sb_hi = (np.asarray(b, dtype=np.float64) for b in scene_bounds)
        scene_light = to_light(_aabb_corners(sb_lo, sb_hi))
        s_lo = scene_light.min(axis=0)
        s_hi = scene_light.max(axis=0)
        # XY: intersect frustum slice footprint with the scene footprint.
        # Receivers are scene geometry inside the camera slice, so they lie
        # in both sets; a small margin guards rasterisation rounding.
        margin = 0.02 * np.maximum(hi[:2] - lo[:2], 1e-3)
        for ax in (0, 1):
            i_lo = max(lo[ax], s_lo[ax] - margin[ax])
            i_hi = min(hi[ax], s_hi[ax] + margin[ax])
            if i_hi > i_lo:                                # non-empty intersection
                lo[ax], hi[ax] = i_lo, i_hi
        # Z: the scene AABB contains every possible occluder and receiver,
        # so its light-space z range (plus padding) is sufficient.
        near_s = -s_hi[2] - z_padding
        far_s = -s_lo[2] + z_padding
        if far_s - near_s > 0.02:
            lo[2], hi[2] = s_lo[2], s_hi[2]

    # Snap XY bounds outward to whole texels (stable texel grid).
    if resolution is not None and resolution > 0 and snap_texels:
        for ax in (0, 1):
            texel = (hi[ax] - lo[ax]) / resolution
            if texel > 1e-9:
                lo[ax] = math.floor(lo[ax] / texel) * texel
                hi[ax] = math.ceil(hi[ax] / texel) * texel

    half_w = max(1e-3, 0.5 * (hi[0] - lo[0]))
    half_h = max(1e-3, 0.5 * (hi[1] - lo[1]))
    centre_xy = 0.5 * (lo[:2] + hi[:2])
    near = -hi[2] - z_padding
    far = -lo[2] + z_padding
    # A negative near is fine for an ortho projection (no division by near;
    # the rasteriser never near-clips ortho views because w == 1). Clamping
    # it to a small positive value would push geometry in front of the near
    # plane outside the NDC z range, breaking receiver coverage for
    # off-centre scenes. Only guard against a degenerate z window.
    if far - near < 0.02:
        far = near + 0.02
    proj = orthographic(half_w, half_h, near, far)
    # Re-centre the (possibly asymmetric) XY window on the ortho origin.
    vp = proj @ _translate(-centre_xy[0], -centre_xy[1], 0.0) @ view

    res = float(resolution) if resolution else 1.0
    info = LightFrustum(
        texel_size_x=float((hi[0] - lo[0]) / res),
        texel_size_y=float((hi[1] - lo[1]) / res),
        z_extent=float(far - near),
        ndc_per_world=float(2.0 / max(1e-9, far - near)),
    )
    return vp, info


def _translate(x: float, y: float, z: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    m[0, 3] = x; m[1, 3] = y; m[2, 3] = z
    return m


def fit_light_view_proj(corners_world: np.ndarray, light_dir: np.ndarray,
                        z_padding: float = 5.0) -> np.ndarray:
    """Build a view-proj matrix that snugly covers ``corners_world`` from
    the perspective of a directional light pointing along ``light_dir``."""
    vp, _ = _fit_light_view_proj(corners_world, light_dir, z_padding,
                                 resolution=None, scene_bounds=None,
                                 snap_texels=False)
    return vp


def build_cascades(
    camera_view_inv: np.ndarray,
    fov_deg: float,
    aspect: float,
    near: float,
    far: float,
    light_dir: tuple[float, float, float],
    n_cascades: int,
) -> list[tuple[np.ndarray, float, float]]:
    """Return a list of (light_view_proj_4x4, z_near, z_far) per cascade."""
    return [vp_zn_zf[:3] for vp_zn_zf in build_cascades_with_info(
        camera_view_inv, fov_deg, aspect, near, far, light_dir, n_cascades,
    )]


def build_cascades_with_info(
    camera_view_inv: np.ndarray,
    fov_deg: float,
    aspect: float,
    near: float,
    far: float,
    light_dir: tuple[float, float, float],
    n_cascades: int,
    *,
    resolution: int | None = None,
    scene_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    z_padding: float = 5.0,
    snap_texels: bool = True,
) -> list[tuple[np.ndarray, float, float, LightFrustum]]:
    """Like :func:`build_cascades` but also returns the fitted
    :class:`LightFrustum` metadata (texel size, NDC-per-world) needed for
    world-space bias computation. With ``resolution`` set, the light ortho
    frusta are snapped to whole texels (shimmer-free)."""
    fov_rad = math.radians(fov_deg)
    splits = cascade_splits(near, far, n_cascades)
    light_dir_np = np.asarray(light_dir, dtype=np.float64)
    out: list[tuple[np.ndarray, float, float, LightFrustum]] = []
    for i in range(n_cascades):
        n = splits[i]; f = splits[i + 1]
        corners = view_frustum_corners_world(camera_view_inv, fov_rad, aspect, n, f)
        vp, info = _fit_light_view_proj(corners, light_dir_np, z_padding,
                                        resolution=resolution,
                                        scene_bounds=scene_bounds,
                                        snap_texels=snap_texels)
        out.append((vp, n, f, info))
    return out


# --------------------------------------------------------------- depth bias
def ground_slope_texels(light_dir: tuple[float, float, float] | np.ndarray,
                        *, tan_max: float = 8.0) -> float:
    """Depth slope of a *horizontal* receiver (floor/ground) per world unit
    for a directional light, clamped to ``tan_max``.

    This is tan(theta) of the light against the up normal — i.e.
    ``|light_horizontal| / |light_vertical|`` — the worst plausible depth
    gradient a mostly-horizontal scene presents across a shadow texel.
    Scaling the slope-bias term by it makes the baked bias correct from
    overhead sun (≈0.5 texel) down to grazing golden-hour light (≈4+).
    """
    ld = np.asarray(light_dir, dtype=np.float64)
    n = np.linalg.norm(ld)
    if n < 1e-9:
        return 1.0
    ld = ld / n
    vertical = abs(float(ld[1]))
    horizontal = math.sqrt(max(0.0, 1.0 - vertical * vertical))
    return min(horizontal / max(vertical, 1e-3), float(tan_max))


def compute_receiver_bias_world(texel_size_world: float, *,
                                constant_texels: float = 0.2,
                                slope_texels: float = 1.0,
                                override_world: float | None = None) -> float:
    """World-space depth bias for one cascade.

    The default ``constant + slope`` model covers (a) raster/quantisation
    noise (constant term) and (b) the depth slope of receiver surfaces
    across one texel (slope term, analogous to a clamped slope-scaled bias
    plus a fractional-texel normal-offset). ``override_world`` (per-light
    or per-config) replaces the model with an explicit world-space value.
    """
    if override_world is not None:
        return float(override_world)
    return float(texel_size_world) * (float(constant_texels) + float(slope_texels))


def bake_depth_bias(depth: torch.Tensor, frustum: LightFrustum,
                    bias_world: float, *,
                    receiver_bias_ndc: float = 0.0) -> torch.Tensor:
    """Bake a world-space depth bias into a rasterised shadow map.

    Adding ``bias_ndc`` to every stored texel is equivalent to the receiver
    comparing with ``current_depth - (bias_ndc + receiver_bias_ndc)``. The
    CSM pass splits the total bias into a config-driven receiver-side
    constant (applied by the PBR pass via ``ShadowMap.receiver_bias_ndc``)
    and a baked slope term; when ``receiver_bias_ndc`` is given here, only
    the *excess* over it is baked, so the effective total is exactly
    ``bias_world`` in NDC — never a hidden hard-coded floor. Returns a new
    tensor; ``+inf`` (empty) texels are left untouched.
    """
    bias_ndc = float(bias_world) * frustum.ndc_per_world
    extra = max(0.0, bias_ndc - float(receiver_bias_ndc))
    out = depth.clone()
    if extra > 0.0:
        finite = torch.isfinite(out)
        out = torch.where(finite, out + extra, out)
    return out


def offset_along_normal(world_pos: torch.Tensor, normals: torch.Tensor,
                        texel_size_world: float, *,
                        offset_texels: float = 1.0) -> torch.Tensor:
    """Normal-offset bias helper: shift receiver positions along their
    (unit) world normal by ``offset_texels`` shadow texels before
    projecting into light space. This snaps receivers onto the shadow-map
    texel plane their occluder was rasterised at, removing self-shadow
    acne with at most ~1 texel of contact-shadow pull-in.
    """
    return world_pos + normals * (float(texel_size_world) * float(offset_texels))


# --------------------------------------------------------------- PCF
def pcf_sample(depth_map: torch.Tensor, uv: torch.Tensor, current_depth: torch.Tensor,
               *, bias: float = 0.005, radius: int = 1,
               normals: torch.Tensor | None = None,
               light_dir: tuple[float, float, float] | torch.Tensor | None = None,
               texel_size_world: float | None = None,
               ndc_per_world: float | None = None,
               slope_scale: float = 1.0,
               slope_tan_max: float = 4.0) -> torch.Tensor:
    """Percentage-closer filter sample.

    Args:
        depth_map: (H, W) shadow map depth (NDC).
        uv: (..., 2) sample positions in [0, 1].
        current_depth: (...,) depth of the shaded fragment in light NDC.
        bias: shadow acne bias (NDC). When ``normals`` + ``light_dir`` +
            ``texel_size_world`` + ``ndc_per_world`` are supplied, a
            receiver-side slope-scaled term (clamped at
            ``slope_tan_max``) is added per fragment.
        radius: 1 → 3x3, 2 → 5x5.
        normals: (..., 3) optional receiver world normals (unit).
        light_dir: optional direction *toward* the light (unit).
        texel_size_world: world units per shadow texel (slope term scale).
        ndc_per_world: NDC depth units per world unit (LightFrustum).
        slope_scale: multiplier for the slope-scaled bias term.
        slope_tan_max: clamp for tan(theta) in the slope term.

    Returns:
        (...,) light visibility ∈ [0, 1] — 1 = fully lit, 0 = fully shadowed.

    Texel addressing matches the rasteriser lattice: the rasteriser
    evaluates coverage at integer screen coordinates ``sx = uv.x * W`` /
    ``sy = (1 - uv.y) * H``, so texel ``i`` represents the sample point at
    coordinate ``i`` — round-to-nearest, NOT truncation (truncation reads
    the texel half a cell southwest and tears shadows along triangle
    edges).
    """
    h, w = depth_map.shape
    px = (uv[..., 0].clamp(0.0, 1.0) * w).round().long().clamp(0, w - 1)
    py = ((1.0 - uv[..., 1].clamp(0.0, 1.0)) * h).round().long().clamp(0, h - 1)

    # Receiver-side slope-scaled bias (optional, per fragment).
    bias_t: float | torch.Tensor = bias
    if (normals is not None and light_dir is not None
            and texel_size_world and ndc_per_world):
        ld = torch.as_tensor(light_dir, device=normals.device, dtype=normals.dtype)
        ld = ld / (torch.linalg.norm(ld) + 1e-9)
        ndotl = (normals * ld).sum(dim=-1).abs().clamp(min=1e-3, max=1.0)
        tan = ((1.0 - ndotl * ndotl).clamp(min=0.0)).sqrt() / ndotl
        tan = tan.clamp(max=slope_tan_max)
        bias_t = bias + tan * (float(texel_size_world) * float(ndc_per_world) * slope_scale)

    total = torch.zeros_like(current_depth)
    count = 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            yy = (py + dy).clamp(0, h - 1)
            xx = (px + dx).clamp(0, w - 1)
            sm = depth_map[yy, xx]
            lit = (current_depth - bias_t <= sm).to(current_depth.dtype)
            total = total + lit
            count += 1
    return total / count
