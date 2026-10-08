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
    wireframe: bool = False,
    skeleton: Any | None = None,
) -> np.ndarray:
    """Drop-in replacement for 3DCreator's ``render_mesh_offscreen``.

    ``wireframe=True`` renders triangle edges as 1-px lines instead of the
    filled mesh (upstream GL polygon-mode semantics). ``skeleton=(joints,
    bone_parents)`` draws the rig as a white line overlay with joint dots
    on top of the shaded mesh (bones connect ``joints[j]`` →
    ``joints[parents[j]]``; roots use parent ``-1``).
    """
    if options is None:
        options = _default_options()
    # 3DCreator passes FLAT (T*3,) indices (ReconstructedMesh.indices,
    # generation/reconstruct.py:30); reshape at this call site — see W19.
    indices = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    cam = _camera_from_options(options, np.asarray(positions, dtype=np.float64))
    cfg = _config_from_options(options)

    if wireframe:
        out = _render_wireframe(positions, indices, colors, cam, cfg)
        return _to_creator_rgba(out, options)

    mesh = Mesh.from_arrays(
        positions=positions, indices=indices,
        normals=normals, colors=colors,
        name="creator3d",
    )
    scene = Scene().add(mesh).add(_light_from_options(options))
    out = render(_engine(), scene, cam, cfg)
    if skeleton is not None:
        _overlay_skeleton(out, skeleton, cam, cfg)
    return _to_creator_rgba(out, options)


# ----------------------------------------------------------- wireframe
def _render_wireframe(positions: np.ndarray, indices: np.ndarray,
                      colors: np.ndarray | None,
                      cam: PerspectiveCamera, cfg: RenderConfig) -> Any:
    """Line rendering of a mesh's unique edges through ``raster_lines``,
    with the same background/tonemap contract as the filled path."""
    import torch

    from ironengine_bonafide.backends import torch_raster
    from ironengine_bonafide.core.color import aces_filmic, linear_to_srgb

    pos = torch.as_tensor(np.asarray(positions), dtype=torch.float32)
    edges = _unique_edges(indices)
    if colors is not None:
        col = torch.as_tensor(np.asarray(colors), dtype=torch.float32)
        if col.ndim == 2 and col.shape[0] != pos.shape[0] and col.shape[0] == indices.shape[0]:
            # Per-face colors → per-vertex (take the corner average).
            acc = torch.zeros_like(pos)
            cnt = torch.zeros((pos.shape[0], 1))
            for k in range(3):
                acc.index_add_(0, torch.as_tensor(indices[:, k]), col)
                cnt.index_add_(0, torch.as_tensor(indices[:, k]),
                               torch.ones((indices.shape[0], 1)))
            col = acc / cnt.clamp(min=1.0)
    else:
        col = None
    vp = cam.view_proj_torch(cfg.width / max(1, cfg.height))
    rgb, depth = torch_raster.raster_lines(
        pos, torch.as_tensor(edges), vp, cfg.width, cfg.height, colors=col,
    )
    # Match TonemapPass (cfg.output_color_space == "sRGB").
    mapped = linear_to_srgb(aces_filmic(rgb * cfg.exposure))
    return _FrameView(rgb=mapped, depth=depth)


def _unique_edges(indices: np.ndarray) -> np.ndarray:
    """(T, 3) triangles → (M, 2) unique undirected edge vertex pairs."""
    e = np.concatenate([indices[:, [0, 1]], indices[:, [1, 2]],
                        indices[:, [2, 0]]], axis=0)
    e.sort(axis=1)
    return np.unique(e, axis=0).astype(np.int64)


# ------------------------------------------------------------ skeleton
def _overlay_skeleton(out: Any, skeleton: Any, cam: PerspectiveCamera,
                      cfg: RenderConfig) -> None:
    """Draw the rig directly onto the display-ready frame.

    Skeleton lines are a UI annotation (not lit content), so they are drawn
    straight into the final sRGB buffer — white bones, orange joint dots —
    depth-tested against the rendered scene.
    """
    import torch

    from ironengine_bonafide.backends import torch_raster

    joints_np, parents = skeleton
    joints_np = np.asarray(joints_np, dtype=np.float32).reshape(-1, 3)
    if joints_np.shape[0] == 0 or out.depth is None:
        return
    segs: list[tuple[int, int]] = []
    for j, parent in enumerate(parents):
        if parent is None or int(parent) < 0 or int(parent) >= joints_np.shape[0]:
            continue
        segs.append((int(parent), j))
    vp = cam.view_proj_torch(cfg.width / max(1, cfg.height))
    joints = torch.as_tensor(joints_np)
    depth = out.depth
    if segs:
        _, ldepth = torch_raster.raster_lines(
            joints, torch.as_tensor(np.asarray(segs, dtype=np.int64)),
            vp, cfg.width, cfg.height, color=(1.0, 1.0, 1.0),
        )
        vis = ldepth < depth
        out.rgb[vis] = 1.0
    # Joint dots on top (small disks, orange in sRGB space).
    jrgb, jdepth = torch_raster.raster_points(
        joints, torch.full((joints_np.shape[0], 3), 1.0),
        vp, cfg.width, cfg.height, point_size_px=6.0,
    )
    vis = jdepth < depth
    orange = torch.tensor([1.0, 0.55, 0.0], dtype=out.rgb.dtype, device=out.rgb.device)
    out.rgb[vis] = orange


class _FrameView:
    """Minimal RenderOutputs stand-in for the shim's manual paths
    (``_to_creator_rgba`` only needs ``.rgb`` / ``.depth``)."""
    __slots__ = ("rgb", "depth", "color_space")

    def __init__(self, rgb: Any, depth: Any) -> None:
        self.rgb = rgb
        self.depth = depth
        self.color_space = "sRGB"


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
