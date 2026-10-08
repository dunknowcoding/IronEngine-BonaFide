"""Line rasterizer + 3DCreator wireframe/skeleton shim tests."""
from __future__ import annotations

import numpy as np
import torch

from ironengine_bonafide.backends.torch_raster import raster_lines
from ironengine_bonafide.integrations import creator3d


def test_raster_lines_draws_diagonal() -> None:
    # Two world-space points projected through identity view-proj → line
    # across the NDC square. With an identity matrix the clip w is 1
    # (ortho-like), which keeps the math exact.
    pos = torch.tensor([[-0.9, -0.9, 0.0], [0.9, 0.9, 0.0]], dtype=torch.float32)
    seg = torch.tensor([[0, 1]], dtype=torch.int64)
    vp = torch.eye(4)
    rgb, depth = raster_lines(pos, seg, vp, 32, 32, color=(1.0, 0.0, 0.0))
    hits = torch.isfinite(depth)
    assert int(hits.sum()) >= 28                          # ~30 px of diagonal
    ys, xs = torch.nonzero(hits, as_tuple=True)
    # Diagonal from (1.6, 30.4) to (30.4, 1.6): x + y ≈ 32 (screen y flipped).
    assert float(((ys + xs - 32).abs()).float().mean()) < 1.5
    assert bool((rgb[hits][:, 0] == 1.0).all())


def test_raster_lines_depth_orders_segments() -> None:
    pos = torch.tensor([
        [-0.5, 0.0, -0.5], [0.5, 0.0, -0.5],              # near segment
        [-0.5, 0.0, 0.5], [0.5, 0.0, 0.5],                # far segment
    ], dtype=torch.float32)
    seg = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
    colors = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    vp = torch.eye(4)
    rgb, depth = raster_lines(pos, seg, vp, 16, 16, colors=colors)
    mid = rgb[8, 8]
    # NDC z −0.5 wins over +0.5 → the red segment is on top.
    assert mid[0] == 1.0 and mid[2] == 0.0


def _opts() -> object:
    from ironengine_3d_creator.rendering.api import RenderOptions  # type: ignore[import-not-found]
    return RenderOptions(width=64, height=48)


def test_shim_wireframe_renders_edges() -> None:
    import pytest
    pytest.importorskip("ironengine_3d_creator")
    creator3d.set_engine(__import__("ironengine_bonafide.api", fromlist=["Engine"]).Engine.cpu())
    pos = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]],
                   dtype=np.float32)
    idx = np.array([0, 1, 2, 0, 2, 3], dtype=np.int64)
    rgba = creator3d.render_mesh_offscreen(
        pos, idx, None, None, options=_opts(), wireframe=True)
    assert rgba.shape == (48, 64, 4)
    # The quad boundary (2 NDC units ≈ 80+ px of edge) shows up as non-bg.
    bg = rgba[0, 0, :3].astype(int)
    non_bg = (np.abs(rgba[..., :3].astype(int) - bg).sum(axis=-1) > 10)
    assert 8 < int(non_bg.sum()) < 800                    # lines, not a fill


def test_shim_skeleton_overlay() -> None:
    import pytest
    pytest.importorskip("ironengine_3d_creator")
    creator3d.set_engine(__import__("ironengine_bonafide.api", fromlist=["Engine"]).Engine.cpu())
    pos = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]],
                   dtype=np.float32)
    idx = np.array([0, 1, 2, 0, 2, 3], dtype=np.int64)
    joints = np.array([[0, 0, 0.5], [0, 0.5, 0.5], [0.5, 0.5, 0.5]],
                      dtype=np.float32)
    parents = (-1, 0, 1)
    rgba = creator3d.render_mesh_offscreen(
        pos, idx, None, None, options=_opts(),
        skeleton=(joints, parents))
    assert rgba.shape == (48, 64, 4)
    # Orange joint dots must appear somewhere in the frame.
    orange = ((rgba[..., 0] > 200) & (rgba[..., 1] > 100) & (rgba[..., 1] < 180)
              & (rgba[..., 2] < 80))
    assert int(orange.sum()) > 0
