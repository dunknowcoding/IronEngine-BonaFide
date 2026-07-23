"""Oriented surfel reconstruction.

Each retained point gets:
  * a normal estimated by PCA on its `k` nearest neighbours
  * a radius proportional to the local k-NN spacing × `radius_factor`

The result feeds the splat pass: instead of fixed-pixel disks, we draw
oriented disks whose world-space size compensates for hole density so
nearby points cover gaps without over-blurring sharp edges.

Pure-torch on CPU for tests; the CUDA backend can dispatch the same
code to GPU tensors transparently.
"""
from __future__ import annotations

import torch


def estimate_surfels(
    positions: torch.Tensor,            # (N, 3) float32
    *,
    k: int = 12,
    radius_factor: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (normals (N, 3), radii (N,)).

    Tries the native CUDA kernel first; falls back to the pure-torch
    O(N²) path otherwise. The torch path is fine up to ~50k points;
    above that, callers should chunk.
    """
    # Native fast path (when bonafide_native is built)
    if positions.is_cuda:
        try:
            from ironengine_bonafide.backends.cuda.native_bridge import (
                HAS_NATIVE,
            )
            from ironengine_bonafide.backends.cuda.native_bridge import (
                surfel_estimate as _native_surfel,
            )
            if HAS_NATIVE:
                return _native_surfel(positions, k=k, radius_factor=radius_factor)
        except Exception:                                                   # noqa: BLE001
            pass    # graceful fall-through to torch path

    n = positions.shape[0]
    if n == 0:
        return positions.new_zeros((0, 3)), positions.new_zeros((0,))
    if n <= k:
        # degenerate: every point neighbours every other
        nbrs = positions.unsqueeze(0).expand(n, n, 3)
    else:
        d2 = torch.cdist(positions, positions, p=2.0) ** 2
        # exclude self by setting the diagonal to +inf
        d2.fill_diagonal_(float("inf"))
        topk = d2.topk(k=k, largest=False)
        nbrs = positions[topk.indices]                           # (N, k, 3)
    centroids = nbrs.mean(dim=1, keepdim=True)
    centred = nbrs - centroids
    # PCA via SVD on the centred neighbourhood
    cov = centred.transpose(1, 2) @ centred                       # (N, 3, 3)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    normals = eigvecs[:, :, 0]                                     # smallest-eigenvalue axis
    # Spacing: mean distance to neighbours
    spacing = centred.norm(dim=2).mean(dim=1)                      # (N,)
    radii = spacing * radius_factor
    return normals, radii


def estimate_surfels_chunked(
    positions: torch.Tensor,
    *,
    k: int = 12,
    radius_factor: float = 1.5,
    chunk: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Memory-bounded variant for clouds > 50k points."""
    n = positions.shape[0]
    if n <= chunk:
        return estimate_surfels(positions, k=k, radius_factor=radius_factor)
    normals = positions.new_zeros((n, 3))
    radii = positions.new_zeros((n,))
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        d2 = torch.cdist(positions[start:end], positions, p=2.0) ** 2
        # mask self
        idxs = torch.arange(start, end, device=positions.device)
        d2[torch.arange(end - start), idxs] = float("inf")
        topk = d2.topk(k=min(k, n - 1), largest=False)
        nbrs = positions[topk.indices]
        centroids = nbrs.mean(dim=1, keepdim=True)
        centred = nbrs - centroids
        cov = centred.transpose(1, 2) @ centred
        _eig, eigvecs = torch.linalg.eigh(cov)
        normals[start:end] = eigvecs[:, :, 0]
        radii[start:end] = centred.norm(dim=2).mean(dim=1) * radius_factor
    return normals, radii
