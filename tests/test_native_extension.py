"""Exercises the compiled ``bonafide_native`` CUDA kernels end-to-end.

Skipped automatically when the extension isn't built (``HAS_NATIVE`` is
False) or when CUDA isn't available. These tests run the *actual* CUDA
kernels on the GPU — they are the proof that the native build works,
not just that it imports.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from ironengine_bonafide.backends.cuda.native_bridge import HAS_NATIVE

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not HAS_NATIVE, reason="bonafide_native not built"),
]


# --------------------------------------------------------------- octree
def test_octree_build_and_visible() -> None:
    from ironengine_bonafide.backends.cuda.native_bridge import (
        octree_build, octree_visible,
    )

    g = np.random.default_rng(0)
    pts = torch.from_numpy(
        g.uniform(-1.0, 1.0, (20_000, 3)).astype(np.float32)
    ).cuda()

    handle = octree_build(pts, leaf_capacity=2048, max_depth=10)
    assert handle.n_nodes > 0
    assert handle.n_indices == pts.shape[0]

    # A camera far away with a generous SSE budget keeps coarse nodes →
    # fewer points; a tiny budget forces leaf expansion → most/all points.
    far = octree_visible(handle, eye=(0.0, 0.0, 50.0), fov_rad=0.8,
                         image_height=512, sse_budget_px=4.0,
                         n_max=pts.shape[0])
    near = octree_visible(handle, eye=(0.0, 0.0, 2.0), fov_rad=1.2,
                          image_height=2048, sse_budget_px=0.1,
                          n_max=pts.shape[0])
    assert far.numel() <= near.numel()
    assert near.numel() <= pts.shape[0]
    # Indices must be valid.
    if near.numel():
        assert int(near.min()) >= 0
        assert int(near.max()) < pts.shape[0]


# --------------------------------------------------------------- surfel
def test_surfel_estimate_normals_and_radii() -> None:
    from ironengine_bonafide.backends.cuda.native_bridge import surfel_estimate

    # A planar patch — every PCA normal should point along ±Z.
    g = np.random.default_rng(1)
    xy = g.uniform(-1.0, 1.0, (2000, 2)).astype(np.float32)
    pts = np.concatenate([xy, np.zeros((2000, 1), dtype=np.float32)], axis=1)
    pts_t = torch.from_numpy(pts).cuda()

    normals, radii = surfel_estimate(pts_t, k=12, radius_factor=1.5)
    assert normals.shape == (2000, 3)
    assert radii.shape == (2000,)
    # Normals should be unit-length.
    norms = torch.linalg.norm(normals, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-2)
    # For a Z=0 plane the normal's Z component dominates.
    assert float(normals[:, 2].abs().mean()) > 0.9
    # Radii are positive and finite.
    assert torch.all(radii > 0)
    assert torch.all(torch.isfinite(radii))


# --------------------------------------------------------------- splat
def test_splat_render_writes_pixels() -> None:
    from ironengine_bonafide.backends.cuda.native_bridge import splat_render

    # A single bright point at the origin, camera looking down -Z.
    pts = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device="cuda")
    cols = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32, device="cuda")
    # Simple orthographic-ish view_proj: identity maps origin to screen centre.
    view_proj = torch.eye(4, dtype=torch.float32, device="cuda")

    rgb, depth = splat_render(pts, cols, view_proj, width=64, height=64,
                              point_size_px=8.0, background=(0.0, 0.0, 0.0))
    assert rgb.shape == (64, 64, 3)
    assert depth.shape == (64, 64)
    # The red point must have stamped at least one pixel.
    assert float(rgb[..., 0].max()) > 0.5
    # Depth at written pixels is finite.
    assert torch.isfinite(depth).any()


# --------------------------------------------------------------- upload
def test_upload_async_round_trip() -> None:
    from ironengine_bonafide.backends.cuda.native_bridge import (
        upload_async, upload_sync,
    )

    host = torch.arange(4096, dtype=torch.float32)              # CPU, contiguous
    device = torch.empty(4096, dtype=torch.float32, device="cuda")
    n = upload_async(host, device, stream="test_xfer")
    upload_sync("test_xfer")
    assert n == host.numel() * 4                                # bytes transferred
    assert torch.allclose(device.cpu(), host)


# --------------------------------------------------------------- backend wiring
def test_cuda_backend_advertises_native_caps() -> None:
    from ironengine_bonafide.backends.cuda.backend import CudaBackend

    be = CudaBackend()
    for cap in ("native_octree", "native_surfel", "native_splat", "native_upload"):
        assert be.supports(cap), f"backend should advertise {cap} when HAS_NATIVE"
