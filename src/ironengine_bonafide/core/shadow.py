"""Cascaded Shadow Map plumbing.

Splits the camera frustum into N depth slices, fits an orthographic light
projection around each slice, and exposes the resulting matrices + a
container ``ShadowMap`` that the PBR pass samples with PCF.

References:
  * "Cascaded Shadow Maps", NVIDIA whitepaper, 2007
  * Zhao Pan-style PSSM cascade splits (linear + log mix)

The actual depth raster is performed by the active backend's
``raster_depth`` helper. This module only does math.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from ironengine_bonafide.core.camera import look_at, orthographic


@dataclass(slots=True)
class ShadowMap:
    """One cascade's light-space matrix + depth texture."""
    light_view_proj: torch.Tensor                       # (4, 4) world -> light NDC
    depth: torch.Tensor                                 # (H, W) float32 NDC depth, +inf empty
    z_split_near: float                                 # camera-space near of this cascade
    z_split_far: float                                  # camera-space far of this cascade


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


# --------------------------------------------------------------- light projection
def fit_light_view_proj(corners_world: np.ndarray, light_dir: np.ndarray,
                        z_padding: float = 5.0) -> np.ndarray:
    """Build a view-proj matrix that snugly covers ``corners_world`` from
    the perspective of a directional light pointing along ``light_dir``."""
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

    # Project the corners into light view-space, fit an AABB, build ortho.
    homog = np.concatenate([corners_world, np.ones((corners_world.shape[0], 1))], axis=1)
    in_light = (view @ homog.T).T
    lo = in_light[:, :3].min(axis=0)
    hi = in_light[:, :3].max(axis=0)
    half_w = max(1e-3, 0.5 * (hi[0] - lo[0]))
    half_h = max(1e-3, 0.5 * (hi[1] - lo[1]))
    near = -hi[2] - z_padding
    far = -lo[2] + z_padding
    proj = orthographic(half_w, half_h, max(0.01, near), far)
    return proj @ view


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
    fov_rad = math.radians(fov_deg)
    splits = cascade_splits(near, far, n_cascades)
    light_dir_np = np.asarray(light_dir, dtype=np.float64)
    out: list[tuple[np.ndarray, float, float]] = []
    for i in range(n_cascades):
        n = splits[i]; f = splits[i + 1]
        corners = view_frustum_corners_world(camera_view_inv, fov_rad, aspect, n, f)
        vp = fit_light_view_proj(corners, light_dir_np)
        out.append((vp, n, f))
    return out


# --------------------------------------------------------------- PCF
def pcf_sample(depth_map: torch.Tensor, uv: torch.Tensor, current_depth: torch.Tensor,
               *, bias: float = 0.005, radius: int = 1) -> torch.Tensor:
    """Percentage-closer filter sample.

    Args:
        depth_map: (H, W) shadow map depth (NDC).
        uv: (..., 2) sample positions in [0, 1].
        current_depth: (...,) depth of the shaded fragment in light NDC.
        bias: shadow acne bias.
        radius: 1 → 3x3, 2 → 5x5.

    Returns:
        (...,) light visibility ∈ [0, 1] — 1 = fully lit, 0 = fully shadowed.
    """
    h, w = depth_map.shape
    px = (uv[..., 0].clamp(0.0, 1.0) * (w - 1)).long()
    py = ((1.0 - uv[..., 1].clamp(0.0, 1.0)) * (h - 1)).long()

    total = torch.zeros_like(current_depth)
    count = 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            yy = (py + dy).clamp(0, h - 1)
            xx = (px + dx).clamp(0, w - 1)
            sm = depth_map[yy, xx]
            lit = (current_depth - bias <= sm).to(current_depth.dtype)
            total = total + lit
            count += 1
    return total / count
