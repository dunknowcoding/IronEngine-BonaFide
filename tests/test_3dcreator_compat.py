"""3DCreator integration test.

Three layers, increasing in fidelity:

  1. ``test_shim_signatures_match_creator``     — argspec parity check
  2. ``test_shim_renders_pointcloud_via_bonafide`` — call the shim adapters directly
  3. ``test_install_patches_creator_module``    — install/uninstall cycle

The slower 4th layer (drive 3DCreator's UI render loop and assert visual
parity) requires PySide6 + offscreen GL; gated on ``IRONENGINE_TEST_UI``.
"""
from __future__ import annotations

import inspect
import os
import sys

import numpy as np
import pytest

pytest.importorskip("ironengine_3d_creator", reason="3DCreator not on PYTHONPATH")

from ironengine_bonafide.api import Engine                     # noqa: E402
from ironengine_bonafide.integrations import creator3d as shim  # noqa: E402


# Force the CPU backend for all tests in this module — keeps results
# deterministic and skips CUDA-only paths.
@pytest.fixture(autouse=True)
def _cpu_engine() -> None:
    shim.set_engine(Engine.cpu())


def test_shim_signatures_match_creator() -> None:
    from ironengine_3d_creator.rendering.api import (
        render_mesh_offscreen as orig_mesh,
        render_points_offscreen as orig_points,
    )
    sig_orig = inspect.signature(orig_points)
    sig_shim = inspect.signature(shim.render_points_offscreen)
    assert list(sig_orig.parameters) == list(sig_shim.parameters), (
        f"signature drift: {sig_orig} vs {sig_shim}"
    )

    sig_orig_m = inspect.signature(orig_mesh)
    sig_shim_m = inspect.signature(shim.render_mesh_offscreen)
    assert list(sig_orig_m.parameters) == list(sig_shim_m.parameters)


def test_shim_renders_pointcloud_via_bonafide() -> None:
    """Render a tiny sphere of points through the shim — assert valid RGBA."""
    from ironengine_3d_creator.rendering.api import RenderOptions

    g = np.random.default_rng(0)
    n = 800
    phi = g.uniform(0, 2 * np.pi, n)
    cos_th = g.uniform(-1, 1, n)
    sin_th = np.sqrt(1 - cos_th ** 2)
    positions = np.stack(
        [sin_th * np.cos(phi), sin_th * np.sin(phi), cos_th], axis=1
    ).astype(np.float32) * 0.5
    colors = g.uniform(0, 1, (n, 3)).astype(np.float32)

    opts = RenderOptions(width=64, height=48, point_size=4.0,
                         bg_color=(0.05, 0.06, 0.10))
    rgba = shim.render_points_offscreen(positions, colors, options=opts)

    assert rgba.dtype == np.uint8
    assert rgba.shape == (48, 64, 4)
    assert rgba[..., 3].min() == 255                            # opaque
    # At least some non-background pixels exist.
    bg = (np.asarray(opts.bg_color) * 255).astype(np.uint8)
    matches_bg = np.all(rgba[..., :3] == bg, axis=-1)
    assert (~matches_bg).any(), "shim produced an all-background image"


def test_shim_renders_mesh_via_bonafide() -> None:
    from ironengine_3d_creator.rendering.api import RenderOptions

    positions = np.array([
        [-0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.5], [-0.5, 0.0, 0.0], [0.5, 0.0, 0.0],
    ], dtype=np.float32)
    indices = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    normals = np.array([[0, 0, 1]] * 6, dtype=np.float32)
    colors = np.array([[0.8, 0.2, 0.2]] * 6, dtype=np.float32)

    opts = RenderOptions(width=64, height=48, distance=3.0)
    rgba = shim.render_mesh_offscreen(positions, indices, normals, colors,
                                       options=opts)
    assert rgba.shape == (48, 64, 4)
    assert rgba[..., 3].min() == 255


def test_install_patches_creator_module() -> None:
    """install() must rebind the public symbols on the creator module."""
    import ironengine_3d_creator.rendering.api as creator_api

    original_pts = creator_api.render_points_offscreen
    original_msh = creator_api.render_mesh_offscreen
    try:
        shim.install()
        assert creator_api.render_points_offscreen is shim.render_points_offscreen
        assert creator_api.render_mesh_offscreen is shim.render_mesh_offscreen
    finally:
        # Restore manually instead of `shim.uninstall()` (importlib.reload
        # fights pytest's import system).
        creator_api.render_points_offscreen = original_pts
        creator_api.render_mesh_offscreen = original_msh


@pytest.mark.skipif(not os.environ.get("IRONENGINE_TEST_UI"),
                    reason="set IRONENGINE_TEST_UI=1 to run UI parity tests")
def test_ui_render_via_shim_smoke(qapp) -> None:                # type: ignore[no-untyped-def]
    """Drive the 3DCreator viewport's rendering API through the shim and
    assert the buffer it returns is valid (no GL crash)."""
    pytest.importorskip("PySide6")
    from ironengine_3d_creator.rendering.api import RenderOptions

    shim.install()
    g = np.random.default_rng(1)
    pts = (g.uniform(-1, 1, (500, 3)) * 0.4).astype(np.float32)
    cols = g.uniform(0, 1, (500, 3)).astype(np.float32)
    rgba = shim.render_points_offscreen(pts, cols, options=RenderOptions(
        width=128, height=96))
    assert rgba.shape == (96, 128, 4)
