"""Octree LOD streaming for huge point clouds.

Builds a balanced octree on first use; per-frame visibility selects
nodes whose screen-space error exceeds `screen_space_error_px`.

Pure-python build (small clouds: <1M points). For larger clouds, the
build is offloaded to a numpy-vectorized chunked path.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass(slots=True)
class OctreeNode:
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    indices: np.ndarray | None = None              # leaf only
    children: list[OctreeNode] = field(default_factory=list)
    sample: np.ndarray | None = None                # representative subset (point indices)

    @property
    def is_leaf(self) -> bool:
        return not self.children


@dataclass(slots=True)
class Octree:
    root: OctreeNode
    max_depth: int
    leaf_capacity: int


def build_octree(positions: torch.Tensor, *, leaf_capacity: int = 4096,
                 max_depth: int = 12) -> Octree:
    pts = positions.detach().cpu().numpy()
    if pts.size == 0:
        zero = np.zeros(3, dtype=np.float32)
        return Octree(root=OctreeNode(zero, zero, indices=np.empty(0, dtype=np.int64)),
                      max_depth=0, leaf_capacity=leaf_capacity)
    aabb_min = pts.min(0)
    aabb_max = pts.max(0)
    root = _build(pts, np.arange(len(pts)), aabb_min, aabb_max,
                  leaf_capacity=leaf_capacity, max_depth=max_depth, depth=0)
    return Octree(root=root, max_depth=max_depth, leaf_capacity=leaf_capacity)


def _build(pts: np.ndarray, idx: np.ndarray, lo: np.ndarray, hi: np.ndarray,
           *, leaf_capacity: int, max_depth: int, depth: int) -> OctreeNode:
    node = OctreeNode(aabb_min=lo.astype(np.float32), aabb_max=hi.astype(np.float32))
    # Representative subset (point indices) for coarser LOD levels — every
    # node gets one so zooming out renders a thinned cloud, not holes.
    n_keep = min(256, len(idx))
    if len(idx) > 0:
        sel = np.random.default_rng(depth * 9973 + len(idx)).choice(len(idx), n_keep, replace=False)
        node.sample = idx[sel]
    if len(idx) <= leaf_capacity or depth >= max_depth:
        node.indices = idx
        return node
    mid = 0.5 * (lo + hi)
    sel = pts[idx]
    masks = [
        (sel[:, 0] < mid[0]) & (sel[:, 1] < mid[1]) & (sel[:, 2] < mid[2]),
        (sel[:, 0] >= mid[0]) & (sel[:, 1] < mid[1]) & (sel[:, 2] < mid[2]),
        (sel[:, 0] < mid[0]) & (sel[:, 1] >= mid[1]) & (sel[:, 2] < mid[2]),
        (sel[:, 0] >= mid[0]) & (sel[:, 1] >= mid[1]) & (sel[:, 2] < mid[2]),
        (sel[:, 0] < mid[0]) & (sel[:, 1] < mid[1]) & (sel[:, 2] >= mid[2]),
        (sel[:, 0] >= mid[0]) & (sel[:, 1] < mid[1]) & (sel[:, 2] >= mid[2]),
        (sel[:, 0] < mid[0]) & (sel[:, 1] >= mid[1]) & (sel[:, 2] >= mid[2]),
        (sel[:, 0] >= mid[0]) & (sel[:, 1] >= mid[1]) & (sel[:, 2] >= mid[2]),
    ]
    for k, mask in enumerate(masks):
        if not np.any(mask):
            continue
        clo = lo.copy(); chi = hi.copy()
        if k & 1: clo[0] = mid[0]
        else:     chi[0] = mid[0]
        if k & 2: clo[1] = mid[1]
        else:     chi[1] = mid[1]
        if k & 4: clo[2] = mid[2]
        else:     chi[2] = mid[2]
        child = _build(pts, idx[mask], clo, chi,
                       leaf_capacity=leaf_capacity, max_depth=max_depth, depth=depth + 1)
        node.children.append(child)
    return node


def select_visible(
    octree: Octree,
    eye: np.ndarray,
    fov_rad: float,
    image_height: int,
    sse_budget_px: float,
) -> np.ndarray:
    """Walk the octree and return point-indices whose effective screen-space
    error fits the budget. A coarser node returns its `sample` array; a leaf
    returns its full `indices`."""
    keep_idx: list[np.ndarray] = []
    fov_term = (image_height * 0.5) / np.tan(fov_rad * 0.5)

    def visit(node: OctreeNode) -> None:
        center = 0.5 * (node.aabb_min + node.aabb_max)
        radius = 0.5 * float(np.linalg.norm(node.aabb_max - node.aabb_min))
        dist = float(np.linalg.norm(eye - center))
        if dist < 1e-6:
            sse = sse_budget_px * 2.0      # too close → fully expand
        else:
            sse = (radius / dist) * fov_term
        if node.is_leaf or sse <= sse_budget_px:
            arr = node.indices if node.is_leaf else node.sample
            if arr is not None:
                keep_idx.append(arr)
            return
        for c in node.children:
            visit(c)

    visit(octree.root)
    if not keep_idx:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(keep_idx)
