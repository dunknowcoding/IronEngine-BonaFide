"""IronEngine-3DCreator drop-in shim.

Activates with a single line at app start:

    from ironengine_bonafide.integrations.creator3d import install
    install()

After that, every call to `ironengine_3d_creator.rendering.api.render_points_offscreen`
or `.render_mesh_offscreen` runs through BonaFide. 3DCreator's UI sees no
behavioural change.

The shim consumes 3DCreator's `RenderOptions` verbatim and translates each
field into a BonaFide `RenderConfig` + a `PerspectiveCamera` so the
authored yaw/pitch/distance preview math is preserved bit-for-bit.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ironengine_bonafide.api import (
    DirectionalLight,
    Engine,
    Mesh,
    PerspectiveCamera,
    PointCloud,
    RenderConfig,
    Scene,
    render,
)
from ironengine_bonafide.logging import logger

# Lazy single engine; switching backends is rare in editor lifetime.
_ENGINE: Engine | None = None


def _engine() -> Engine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = Engine.auto()
    return _ENGINE


def set_engine(engine: Engine) -> None:
    """Override the cached engine (e.g. force CPU for headless test runs)."""
    global _ENGINE
    _ENGINE = engine


# --------------------------------------------------------------- adapters
def _camera_from_options(opts: Any, positions: np.ndarray) -> PerspectiveCamera:
    """Mirror 3DCreator's preview-camera math.

    3DCreator orbits around the cloud centroid (or the user-set target),
    using yaw/pitch/distance. `distance=0` auto-frames from cloud extent.
    """
    target = (
        np.asarray(opts.target, dtype=np.float64)
        if opts.target is not None
        else positions.mean(axis=0).astype(np.float64)
    )
    # Upstream: `distance = opt.distance or _auto_frame(positions)[1]`
    # (rendering/api.py:263) — any falsy distance auto-frames.
    if opts.distance:
        d = float(opts.distance)
    else:
        # Auto-frame: bounding diagonal × 1.4 (upstream `_auto_frame`,
        # rendering/api.py:119-122), clamped to ≥ 0.5.
        if positions.size == 0:
            d = 3.0
        else:
            extent = float(np.linalg.norm(positions.max(0) - positions.min(0)))
            d = max(0.5, extent * 1.4)

    yaw = math.radians(opts.yaw_deg)
    pitch = math.radians(opts.pitch_deg)
    # Upstream `_orbit_mvp` (rendering/api.py:125-130): eye.y uses
    # sin(+pitch) — default pitch_deg=-20 orbits BELOW the target.
    eye = target + d * np.array([
        math.cos(pitch) * math.sin(yaw),
        math.sin(pitch),
        math.cos(pitch) * math.cos(yaw),
    ], dtype=np.float64)
    return PerspectiveCamera(
        position=tuple(eye.tolist()),                                    # type: ignore[arg-type]
        look_at=tuple(target.tolist()),                                  # type: ignore[arg-type]
        up=(0.0, 1.0, 0.0),
        fov_deg=45.0,
        near=0.05, far=500.0,           # upstream fixed near/far (api.py:138)
    )


def _config_from_options(opts: Any) -> RenderConfig:
    return RenderConfig(
        width=int(opts.width),
        height=int(opts.height),
        samples=1,
        aa="fxaa",
        output_dtype="uint8",
        output_color_space="sRGB",
        sensor_outputs=("rgb", "depth"),     # depth needed to mask background
        bloom=False,
        shadows="off",
        seed=0,
    )


def _light_from_options(opts: Any) -> DirectionalLight:
    return DirectionalLight(
        direction=tuple(-x for x in opts.light_dir),                     # type: ignore[arg-type]
        color=(1.0, 0.98, 0.95),
        intensity=2.5,
        cast_shadow=False,
    )


# --------------------------------------------------------------- public API
def render_points_offscreen(
    positions: np.ndarray,
    colors: np.ndarray,
    *,
    options: Any | None = None,
) -> np.ndarray:
    """Drop-in replacement for 3DCreator's ``render_points_offscreen``.

    Matches the upstream signature exactly: ``options`` is keyword-only.
    Returns a uint8 ``(H, W, 4)`` RGBA image with full opacity.
    """
    if options is None:
        options = _default_options()
    cloud = PointCloud.from_arrays(positions, colors, name="creator3d")
    cloud.point_size_px = float(getattr(options, "point_size", 4.0))
    scene = Scene().add(cloud).add(_light_from_options(options))
    cam = _camera_from_options(options, np.asarray(positions, dtype=np.float64))
    cfg = _config_from_options(options)
    out = render(_engine(), scene, cam, cfg)
    return _to_creator_rgba(out, options)


def render_mesh_offscreen(
    positions: np.ndarray,
    indices: np.ndarray,
    normals: np.ndarray | None,
    colors: np.ndarray | None,
    *,
    options: Any | None = None,
) -> np.ndarray:
    """Drop-in replacement for 3DCreator's ``render_mesh_offscreen``."""
    if options is None:
        options = _default_options()
    # 3DCreator passes FLAT (T*3,) indices (ReconstructedMesh.indices,
    # generation/reconstruct.py:30); reshape at this call site — see W19.
    indices = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    mesh = Mesh.from_arrays(
        positions=positions, indices=indices,
        normals=normals, colors=colors,
        name="creator3d",
    )
    scene = Scene().add(mesh).add(_light_from_options(options))
    cam = _camera_from_options(options, np.asarray(positions, dtype=np.float64))
    cfg = _config_from_options(options)
    out = render(_engine(), scene, cam, cfg)
    return _to_creator_rgba(out, options)


def _default_options() -> Any:
    """Build a default ``RenderOptions`` (lazy-import to avoid a hard dep)."""
    from ironengine_3d_creator.rendering.api import RenderOptions  # type: ignore[import-not-found]
    return RenderOptions()


def _to_creator_rgba(out: Any, options: Any) -> np.ndarray:
    """Convert RenderOutputs → uint8 RGBA the 3DCreator UI expects.

    Background pixels (depth = +inf) are filled with ``options.bg_color``
    so the output matches the upstream renderer's clear color. We guard
    against shape skew between rgb and depth (a neural-upscale pass can
    leave depth at native resolution while rgb gets upscaled).

    Tonemap contract: ``_config_from_options`` sets
    ``output_color_space="sRGB"`` so ``out.rgb`` is already final
    display-ready sRGB — convert directly, never re-apply ACES.
    """
    from ironengine_bonafide.integrations._display import srgb_to_uint8

    img = srgb_to_uint8(out.rgb)
    h, w = img.shape[:2]
    rgba = np.empty((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = img
    rgba[..., 3] = 255
    if out.depth is not None:
        depth_np = out.depth.detach().cpu().numpy()
        if depth_np.shape == (h, w):
            empty = ~np.isfinite(depth_np)
            bg = (np.asarray(options.bg_color, dtype=np.float32) * 255).clip(0, 255).astype(np.uint8)
            rgba[empty, :3] = bg
    return rgba


def install() -> None:
    """Monkey-patch 3DCreator's renderer entry points."""
    try:
        import ironengine_3d_creator.rendering.api as creator_api  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "ironengine_3d_creator is not importable. Install it on PYTHONPATH first."
        ) from exc
    creator_api.render_points_offscreen = render_points_offscreen          # type: ignore[assignment]
    creator_api.render_mesh_offscreen = render_mesh_offscreen              # type: ignore[assignment]
    logger.info("creator3d shim installed: 3DCreator now renders through BonaFide")


def uninstall() -> None:
    """Best-effort restore (forces a reload of 3DCreator's rendering.api)."""
    import importlib
    try:
        import ironengine_3d_creator.rendering.api as creator_api  # type: ignore[import-not-found]
    except ImportError:
        return
    importlib.reload(creator_api)
    logger.info("creator3d shim uninstalled")
