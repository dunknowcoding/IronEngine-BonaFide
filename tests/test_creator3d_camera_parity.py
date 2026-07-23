"""creator3d shim: camera parity with 3DCreator's ``_orbit_mvp`` + tonemap
contract + flat-index acceptance (W18, tonemap contract, W19 call-site).

Uses stubbed option objects — never imports the real 3DCreator UI stack.
Upstream reference math (3DCreator rendering/api.py:119-145):

    centre   = positions.mean(axis=0)
    distance = max(0.5, extent * 1.4)          # extent = ‖bbox diagonal‖
    eye.y    = target.y + d * sin(+pitch)      # NOT sin(-pitch)
    near/far = 0.05 / 500.0 (fixed), fov 45°
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ironengine_bonafide.integrations import creator3d as shim


@dataclass
class _Opts:
    width: int = 64
    height: int = 48
    point_size: float = 4.0
    bg_color: tuple[float, float, float] = (0.05, 0.06, 0.10)
    light_dir: tuple[float, float, float] = (0.4, 0.8, 0.6)
    yaw_deg: float = 35.0
    pitch_deg: float = -20.0
    distance: float = 0.0
    target: Any = None


def test_camera_parity_default_orbit() -> None:
    """Eye position must match upstream `_orbit_mvp` exactly."""
    g = np.random.default_rng(3)
    positions = g.uniform(-0.5, 0.5, (100, 3)).astype(np.float64)
    opts = _Opts()                                              # yaw 35, pitch -20, auto distance
    cam = shim._camera_from_options(opts, positions)

    target = positions.mean(axis=0)
    extent = float(np.linalg.norm(positions.max(0) - positions.min(0)))
    d = max(0.5, extent * 1.4)
    yaw = math.radians(opts.yaw_deg)
    pitch = math.radians(opts.pitch_deg)
    eye = target + d * np.array([
        math.cos(pitch) * math.sin(yaw),
        math.sin(pitch),                                        # upstream: sin(+pitch)
        math.cos(pitch) * math.cos(yaw),
    ])
    np.testing.assert_allclose(cam.position, eye, atol=1e-12)
    np.testing.assert_allclose(cam.look_at, target, atol=1e-12)
    assert cam.fov_deg == 45.0
    assert cam.near == 0.05
    assert cam.far == 500.0
    # Regression guard: default pitch -20° must put the eye BELOW target.
    assert cam.position[1] < target[1]


def test_camera_parity_explicit_distance_and_target() -> None:
    positions = np.zeros((4, 3), dtype=np.float64)
    opts = _Opts(distance=7.5, target=(1.0, 2.0, 3.0), yaw_deg=0.0, pitch_deg=45.0)
    cam = shim._camera_from_options(opts, positions)
    eye = np.array([1.0, 2.0 + 7.5 * math.sin(math.pi / 4), 3.0 + 7.5 * math.cos(math.pi / 4)])
    np.testing.assert_allclose(cam.position, eye, atol=1e-12)


def test_camera_auto_frame_extent_factor() -> None:
    # Cube with diagonal √3 ≈ 1.732 → auto distance ≈ 2.425, not ×0.8.
    positions = np.array(list(np.ndindex(2, 2, 2)), dtype=np.float64)
    opts = _Opts()
    cam = shim._camera_from_options(opts, positions)
    d = float(np.linalg.norm(np.asarray(cam.position) - np.asarray(cam.look_at)))
    np.testing.assert_allclose(d, math.sqrt(3.0) * 1.4, atol=1e-12)


def test_to_creator_rgba_uses_display_ready_srgb_directly() -> None:
    """Tonemap contract: with output_color_space='sRGB', out.rgb is final
    sRGB. A mid-grey 0.5 must come back ≈128 — NOT ACES-lifted (~163 by
    the filmic curve would prove a second tonemap)."""
    class _Out:
        rgb = torch.full((4, 4, 3), 0.5, dtype=torch.float32)
        depth = torch.ones((4, 4), dtype=torch.float32)         # all geometry

    rgba = shim._to_creator_rgba(_Out(), _Opts())
    assert rgba.shape == (4, 4, 4)
    assert abs(int(rgba[0, 0, 0]) - 128) <= 1


def test_to_creator_rgba_fills_background_from_options() -> None:
    class _Out:
        rgb = torch.full((4, 4, 3), 0.5, dtype=torch.float32)
        depth = torch.full((4, 4), float("inf"), dtype=torch.float32)

    rgba = shim._to_creator_rgba(_Out(), _Opts())
    bg = (np.asarray(_Opts.bg_color) * 255).astype(np.uint8)
    np.testing.assert_array_equal(rgba[0, 0, :3], bg)
    assert rgba[0, 0, 3] == 255


def test_render_mesh_offscreen_accepts_flat_indices(monkeypatch) -> None:
    """W19 call-site: 3DCreator passes flat (T*3,) indices — the shim must
    reshape before Mesh.from_arrays (which requires (T, 3))."""
    captured: dict[str, np.ndarray] = {}

    class _FakeOut:
        rgb = torch.zeros((48, 64, 3), dtype=torch.float32)
        depth = torch.full((48, 64), float("inf"))

    def _fake_render(engine, scene, cam, cfg):                  # noqa: ANN001, ARG001
        captured["indices"] = scene.meshes[0].indices.detach().cpu().numpy()
        return _FakeOut()

    monkeypatch.setattr(shim, "render", _fake_render)
    monkeypatch.setattr(shim, "_engine", lambda: object())

    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    flat = np.array([0, 1, 2], dtype=np.uint32)                 # (T*3,) flat
    rgba = shim.render_mesh_offscreen(positions, flat, None, None, options=_Opts())
    assert captured["indices"].shape == (1, 3)
    assert rgba.shape == (48, 64, 4)
