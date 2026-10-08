"""VSM shadow mode and AreaLight quadrature tests."""
from __future__ import annotations

import numpy as np
import torch

from ironengine_bonafide.api import (
    AreaLight,
    DirectionalLight,
    Engine,
    Mesh,
    PBRMaterial,
    PerspectiveCamera,
    RenderConfig,
    Scene,
    render,
)

_NRM_UP = np.array([[0, 1, 0]] * 4, dtype=np.float32)


def _floor() -> Mesh:
    return Mesh.from_arrays(
        positions=np.array([[-3, -0.5, -3], [3, -0.5, -3], [3, -0.5, 3],
                            [-3, -0.5, 3]], dtype=np.float32),
        indices=np.array([[0, 1, 2], [0, 2, 3]]),
        normals=_NRM_UP,
        material=PBRMaterial(albedo=(0.8, 0.8, 0.8), roughness=0.9))


def _occluder() -> Mesh:
    return Mesh.from_arrays(
        positions=np.array([[-0.6, 0.5, -0.6], [0.6, 0.5, -0.6], [0.6, 0.5, 0.6],
                            [-0.6, 0.5, 0.6]], dtype=np.float32),
        indices=np.array([[0, 1, 2], [0, 2, 3]]),
        normals=_NRM_UP,
        material=PBRMaterial(albedo=(0.6, 0.3, 0.2), roughness=0.9))


def _shadow_scene() -> Scene:
    return (Scene().add(_floor()).add(_occluder())
            .add(DirectionalLight(direction=(0.0, -1.0, 0.0), intensity=3.0)))


def _cam() -> PerspectiveCamera:
    return PerspectiveCamera(position=(0, 2.5, 4), look_at=(0, -0.2, 0), fov_deg=45)


def _floor_brightness_stats(out) -> tuple[float, float]:
    depth = out.depth
    vals = out.rgb[torch.isfinite(depth)].sum(-1)
    return float(vals.quantile(0.1)), float(vals.quantile(0.9))


def test_csm_and_vsm_both_shadow() -> None:
    cam = _cam()
    stats = {}
    for mode in ("off", "csm", "vsm"):
        out = render(Engine.cpu(), _shadow_scene(), cam,
                     RenderConfig(width=128, height=96, shadows=mode,
                                  output_color_space="linear",
                                  sensor_outputs=("rgb", "depth")))
        stats[mode] = _floor_brightness_stats(out)
    p10_off, p90_off = stats["off"]
    for mode in ("csm", "vsm"):
        p10, p90 = stats[mode]
        assert p10 < p10_off * 0.8, f"{mode}: no shadow (p10 {p10} vs {p10_off})"
        assert p90 > p90_off * 0.9, f"{mode}: lit area got darker too"


def test_vsm_penumbra_is_fractional() -> None:
    out = render(Engine.cpu(), _shadow_scene(), _cam(),
                 RenderConfig(width=128, height=96, shadows="vsm",
                              output_color_space="linear",
                              sensor_outputs=("rgb", "depth")))
    vals = out.rgb[torch.isfinite(out.depth)].sum(-1)
    lo, hi = float(vals.min()), float(vals.max())
    mid = vals[(vals > lo + 0.1 * (hi - lo)) & (vals < lo + 0.8 * (hi - lo))]
    # VSM's Chebyshev test yields soft penumbrae — brightness values that
    # are neither fully lit nor fully shadowed must exist along the edge.
    assert mid.numel() > 0


def _area_cam() -> PerspectiveCamera:
    return PerspectiveCamera(position=(0, 1.5, 3), look_at=(0, -0.5, 0), fov_deg=50)


def _floor_mean(out) -> float:
    return float(out.rgb[torch.isfinite(out.depth)].mean())


def test_area_light_lits_facing_side_only() -> None:
    cam = _area_cam()
    down = AreaLight(position=(0, 2, 0), normal=(0, -1, 0), extent=(2, 2),
                     intensity=6.0)
    up = AreaLight(position=(0, 2, 0), normal=(0, 1, 0), extent=(2, 2),
                   intensity=6.0)
    cfg = RenderConfig(width=128, height=96, output_color_space="linear",
                       sensor_outputs=("rgb", "depth"))
    lit = render(Engine.cpu(), Scene().add(_floor()).add(down), cam, cfg)
    away = render(Engine.cpu(), Scene().add(_floor()).add(up), cam, cfg)
    assert _floor_mean(lit) > _floor_mean(away) + 0.1


def test_area_light_wider_extent_softens_distribution() -> None:
    cam = _area_cam()
    small = AreaLight(position=(0, 2, 0), normal=(0, -1, 0), extent=(0.05, 0.05),
                      intensity=6.0)
    wide = AreaLight(position=(0, 2, 0), normal=(0, -1, 0), extent=(4.0, 4.0),
                     intensity=6.0)
    cfg = RenderConfig(width=128, height=96, output_color_space="linear",
                       sensor_outputs=("rgb", "depth"))
    s = render(Engine.cpu(), Scene().add(_floor()).add(small), cam, cfg)
    w = render(Engine.cpu(), Scene().add(_floor()).add(wide), cam, cfg)
    vs = s.rgb[torch.isfinite(s.depth)].sum(-1)
    vw = w.rgb[torch.isfinite(w.depth)].sum(-1)
    # The point-like light concentrates energy (higher peak); the wide
    # light spreads it (lower peak, both finite).
    assert float(vs.max()) > float(vw.max())
    assert torch.isfinite(w.rgb).all()
