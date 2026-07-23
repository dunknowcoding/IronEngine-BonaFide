"""Persistent CUDA streams + workspace cache.

Two motivations:
  1. **Stream pool** — separate streams for "main render", "asset upload"
     and "background prefetch" so transfers and compute overlap instead
     of serializing on the default stream.
  2. **Workspace cache** — keep per-cloud / per-mesh GPU tensors alive
     across frames so we don't re-allocate gsplat parameters, intrinsic
     matrices, or LOD index buffers every render.

Both are no-ops on non-CUDA backends; the CPU / WGPU paths skip past
``with_stream(...)`` and ``ensure_workspace(...)`` cleanly.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

from ironengine_bonafide.logging import logger

# --------------------------------------------------------------- streams
_STREAMS: dict[str, torch.cuda.Stream] = {}


def stream(name: str) -> torch.cuda.Stream | None:
    """Return a named CUDA stream, creating it if needed.

    ``None`` if CUDA isn't available.
    """
    if not torch.cuda.is_available():
        return None
    s = _STREAMS.get(name)
    if s is None:
        s = torch.cuda.Stream()
        _STREAMS[name] = s
        logger.debug(f"created CUDA stream '{name}' (id={s.cuda_stream})")
    return s


@contextmanager
def with_stream(name: str) -> Iterator[None]:
    """Context manager that switches the current stream for the block."""
    s = stream(name)
    if s is None:
        yield
        return
    with torch.cuda.stream(s):
        yield


def synchronize_all() -> None:
    if not torch.cuda.is_available():
        return
    for s in _STREAMS.values():
        s.synchronize()


# --------------------------------------------------------------- workspace cache
@dataclass(slots=True)
class GsplatWorkspace:
    """Cached gsplat parameters for one PointCloud.

    Holds the per-Gaussian quaternions / scales / opacities that don't
    change between frames, the per-frame intrinsic / view matrices, and
    a stream handle.
    """
    quats: torch.Tensor
    scales: torch.Tensor
    opacities: torch.Tensor
    K: torch.Tensor | None = None              # set on first render at this resolution
    last_view: torch.Tensor | None = None
    n_points: int = 0


@dataclass(slots=True)
class WorkspaceCache:
    gsplat: dict[int, GsplatWorkspace] = field(default_factory=dict)

    def ensure_gsplat(self, cloud, device: str) -> GsplatWorkspace:        # type: ignore[no-untyped-def]
        """Cache by ``id(cloud)`` so repeated renders reuse the same buffer."""
        key = id(cloud)
        ws = self.gsplat.get(key)
        if ws is not None and ws.n_points == cloud.num_points:
            return ws
        n = cloud.num_points
        scales = torch.full((n, 3), 0.01, device=device)
        quats = torch.zeros((n, 4), device=device); quats[:, 0] = 1.0
        opacities = torch.ones(n, device=device)
        ws = GsplatWorkspace(quats=quats, scales=scales, opacities=opacities, n_points=n)
        self.gsplat[key] = ws
        return ws

    def evict(self, cloud) -> None:                                        # type: ignore[no-untyped-def]
        self.gsplat.pop(id(cloud), None)

    def clear(self) -> None:
        self.gsplat.clear()


# Module-level singleton (lifetime = process).
_GLOBAL_CACHE = WorkspaceCache()


def workspace_cache() -> WorkspaceCache:
    return _GLOBAL_CACHE


def reset_caches() -> None:
    _GLOBAL_CACHE.clear()
    _STREAMS.clear()
