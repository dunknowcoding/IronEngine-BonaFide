"""Bridge to the optional ``bonafide_native`` C++/CUDA extension.

The extension is built from ``native/`` via CMake (see ``native/README.md``).
When importable, it accelerates four hot paths:

  * Octree LOD walk + visibility    (``octree_build``, ``octree_visible``)
  * Surfel kNN + PCA normals        (``surfel_estimate``)
  * Disk splat raster               (``splat_render``)
  * Async pinned-host → device      (``upload_async``)

When **not** importable (the user hasn't built it, or runs CPU-only),
:data:`HAS_NATIVE` is ``False`` and callers fall back to their pure-Python
paths. The module guards every call with :func:`require` so a missing
``.pyd`` raises a clear, typed error rather than an ``AttributeError``.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import torch

from ironengine_bonafide.errors import CapabilityMissing
from ironengine_bonafide.logging import logger

_native: Any | None = None


def _register_cuda_dll_dirs() -> None:
    """Make the CUDA runtime DLLs discoverable to the extension loader.

    Python 3.8+ on Windows no longer searches ``PATH`` when resolving a
    ``.pyd``'s dependent DLLs — only directories registered via
    ``os.add_dll_directory()``. The extension links ``cudart64_*.dll``;
    we register, in priority order:

      1. ``torch/lib``                  (always present; bundles cudart)
      2. ``$CUDA_PATH/bin``             (system CUDA toolkit, if installed)
      3. every ``CUDA/v*/bin`` under the standard NVIDIA install root

    No-op on POSIX, where the dynamic loader uses RPATH / LD_LIBRARY_PATH.
    """
    if sys.platform != "win32":
        return
    candidates: list[Path] = []
    # 1. torch's bundled CUDA runtime — guaranteed to match the torch build.
    torch_lib = Path(torch.__file__).parent / "lib"
    if torch_lib.is_dir():
        candidates.append(torch_lib)
    # 2. CUDA_PATH / CUDA_HOME env var.
    for env_var in ("CUDA_PATH", "CUDA_HOME"):
        root = os.environ.get(env_var)
        if root and (Path(root) / "bin").is_dir():
            candidates.append(Path(root) / "bin")
    # 3. Standard Windows install location.
    nvidia_root = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
    if nvidia_root.is_dir():
        candidates.extend(sorted(p / "bin" for p in nvidia_root.glob("v*")
                                 if (p / "bin").is_dir()))
    seen: set[str] = set()
    for path in candidates:
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            os.add_dll_directory(str(path))
            logger.debug(f"native_bridge: registered DLL dir {path}")
        except (OSError, FileNotFoundError):
            pass


def _try_import() -> Any | None:
    spec = importlib.util.find_spec("bonafide_native")
    if spec is None:
        return None
    _register_cuda_dll_dirs()
    try:
        import bonafide_native as nat  # type: ignore[import-not-found]
        return nat
    except Exception as exc:                                    # noqa: BLE001
        logger.warning(f"bonafide_native present but failed to import: {exc}")
        return None


_native = _try_import()
HAS_NATIVE: bool = _native is not None
if HAS_NATIVE:
    logger.info("bonafide_native loaded — CUDA fast paths active")
else:
    logger.info("bonafide_native not available — using pure-Python CUDA paths")


def require(name: str) -> Any:
    if _native is None:
        raise CapabilityMissing(backend="cuda", capability=f"native:{name}")
    return getattr(_native, name)


# --------------------------------------------------------------- octree
class OctreeHandle:
    """Lifetime-managed wrapper around the native ``OctreeHandle``.

    The destructor calls ``octree_free`` so users don't leak GPU memory
    if they drop the reference.
    """
    __slots__ = ("_handle",)

    def __init__(self, handle: Any) -> None:
        self._handle = handle

    @property
    def n_nodes(self) -> int:
        return int(self._handle.n_nodes)

    @property
    def n_indices(self) -> int:
        return int(self._handle.n_indices)

    def __del__(self) -> None:
        try:
            if _native is not None and self._handle is not None:
                _native.octree_free(self._handle)
        except Exception:                                       # noqa: BLE001
            pass


def octree_build(positions: torch.Tensor, *, leaf_capacity: int = 4096,
                 max_depth: int = 12) -> OctreeHandle:
    fn = require("octree_build")
    pos = positions.detach().to(device="cuda", dtype=torch.float32).contiguous()
    return OctreeHandle(fn(pos, leaf_capacity, max_depth))


def octree_visible(h: OctreeHandle, eye: tuple[float, float, float],
                   fov_rad: float, image_height: int,
                   sse_budget_px: float, n_max: int) -> torch.Tensor:
    fn = require("octree_visible")
    out = torch.empty(n_max, dtype=torch.int32, device="cuda")
    n = fn(h._handle, list(eye), float(fov_rad), int(image_height),
           float(sse_budget_px), out)
    return out[:int(n)]


# --------------------------------------------------------------- surfel
def surfel_estimate(positions: torch.Tensor, *, k: int = 12,
                    radius_factor: float = 1.5) -> tuple[torch.Tensor, torch.Tensor]:
    fn = require("surfel_estimate")
    pos = positions.detach().to(device="cuda", dtype=torch.float32).contiguous()
    n = pos.shape[0]
    normals = torch.empty((n, 3), dtype=torch.float32, device="cuda")
    radii = torch.empty(n, dtype=torch.float32, device="cuda")
    fn(pos, int(k), float(radius_factor), normals, radii)
    return normals, radii


# --------------------------------------------------------------- splat
def splat_render(positions: torch.Tensor, colors: torch.Tensor,
                 view_proj: torch.Tensor, width: int, height: int,
                 point_size_px: float = 2.0,
                 background: tuple[float, float, float] = (0.0, 0.0, 0.0),
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    fn = require("splat_render")
    rgb = torch.empty((height, width, 3), dtype=torch.float32, device="cuda")
    rgb[:] = torch.tensor(background, dtype=torch.float32, device="cuda")
    depth = torch.full((height, width), float("inf"), dtype=torch.float32, device="cuda")
    fn(positions.contiguous(), colors.contiguous(),
       view_proj.contiguous().reshape(-1),
       int(width), int(height), float(point_size_px), rgb, depth)
    return rgb, depth


# --------------------------------------------------------------- upload
def upload_async(host: torch.Tensor, device: torch.Tensor, *, stream: str = "transfer") -> int:
    fn = require("upload_async")
    return int(fn(host.numpy(), device, stream))


def upload_sync(stream: str = "transfer") -> None:
    fn = require("upload_sync")
    fn(stream)
