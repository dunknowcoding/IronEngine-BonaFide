"""Backend ABC + capability registry.

A `Backend` is the only thing render passes ever talk to. Every backend
declares its capabilities up-front via `supports(name)`; a pass calls
`backend.require("gsplat")` to fail loudly when its dependency is missing.

Capability names are flat strings. Common ones:

  - "raster"         : differentiable triangle rasterization
  - "gsplat"         : 3D Gaussian Splatting
  - "surfel"         : oriented disk reconstruction
  - "shadow_csm"     : cascaded shadow maps
  - "warp_xpbd"      : NVIDIA Warp soft-body solver
  - "warp_flip"      : NVIDIA Warp fluid solver
  - "neural_field"   : tiny-cuda-nn / hash-grid MLPs
  - "vdb_volume"     : OpenVDB density grids
  - "neural_denoise" : OIDN-style learned denoiser
  - "neural_upscale" : DLSS-style learned super-res
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch


@dataclass(slots=True)
class BackendInfo:
    name: str
    device: str                      # e.g. "cuda:0", "cpu"
    capabilities: frozenset[str]
    version: str = "0.0.0"
    notes: str = ""


class Backend(ABC):
    """Base class for every render backend."""

    @property
    @abstractmethod
    def info(self) -> BackendInfo: ...

    @property
    def name(self) -> str:
        return self.info.name

    @property
    def device(self) -> str:
        return self.info.device

    def supports(self, capability: str) -> bool:
        return capability in self.info.capabilities

    def require(self, capability: str) -> None:
        if not self.supports(capability):
            from ironengine_bonafide.errors import CapabilityMissing
            raise CapabilityMissing(self.name, capability)

    @abstractmethod
    def empty(self, shape: tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor: ...

    @abstractmethod
    def zeros(self, shape: tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor: ...

    @abstractmethod
    def to_device(self, x: torch.Tensor) -> torch.Tensor: ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} device={self.device} caps={sorted(self.info.capabilities)}>"


# Back-compat alias — prefer ironengine_bonafide.errors.CapabilityMissing.
from ironengine_bonafide.errors import (
    CapabilityMissing as BackendCapabilityError,  # noqa: E402, F401
)


# --------------------------------------------------------------- discovery
@dataclass(slots=True)
class BackendDiscovery:
    cuda_available: bool = False
    wgpu_available: bool = False
    torch_cuda: bool = False
    torch_mps: bool = False
    notes: list[str] = field(default_factory=list)


def probe() -> BackendDiscovery:
    """Inspect which backends could be loaded right now. Pure read-only."""
    info = BackendDiscovery()
    try:
        import cupy as _cp  # type: ignore[import-not-found]  # noqa: F401
        try:
            import gsplat as _g  # type: ignore[import-not-found]  # noqa: F401
            import nvdiffrast.torch as _nvdr  # type: ignore[import-not-found]  # noqa: F401
            info.cuda_available = True
        except ImportError as exc:
            info.notes.append(f"CuPy present but gsplat/nvdiffrast missing: {exc}")
    except ImportError:
        info.notes.append("CuPy not installed; CUDA backend unavailable.")
    try:
        import wgpu as _wgpu  # type: ignore[import-not-found]  # noqa: F401
        info.wgpu_available = True
    except ImportError:
        info.notes.append("wgpu-py not installed; WGPU backend unavailable.")
    info.torch_cuda = bool(torch.cuda.is_available())
    info.torch_mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    return info
