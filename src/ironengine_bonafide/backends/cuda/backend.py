"""CUDA backend.

Wraps:
  * `gsplat`     for differentiable 3D Gaussian Splatting
  * `nvdiffrast` for differentiable triangle rasterization
  * NVIDIA `warp` for soft-body / particle / fluid solvers
  * `tiny-cuda-nn` for small neural fields

All four are imported lazily inside their respective wrapper modules so
the backend constructor can detect "CUDA available but accessory missing"
and report exactly which capability dropped.
"""
from __future__ import annotations

import torch

from ironengine_bonafide.core.backend import Backend, BackendInfo


class CudaBackend(Backend):
    def __init__(self) -> None:
        from ironengine_bonafide.errors import BackendUnavailable
        if not torch.cuda.is_available():
            raise BackendUnavailable("PyTorch built without CUDA support")
        caps: set[str] = {"raster", "splat", "surfel", "shadow_csm",
                          "neural_denoise", "neural_upscale"}
        notes: list[str] = []

        # Probe optional accessories
        try:
            import gsplat as _g  # type: ignore[import-not-found]  # noqa: F401
            caps.add("gsplat")
        except ImportError:
            notes.append("gsplat missing — falling back to splat raster")
        try:
            import nvdiffrast.torch as _nvdr  # type: ignore[import-not-found]  # noqa: F401
            caps.add("nvdiffrast")
        except ImportError:
            notes.append("nvdiffrast missing — mesh raster degraded")
        try:
            import warp as _wp  # type: ignore[import-not-found]  # noqa: F401
            caps.update({"warp_xpbd", "warp_flip", "warp_mpm"})
        except ImportError:
            notes.append("warp-lang missing — softbody/fluid degraded")
        try:
            import tinycudann as _tcnn  # type: ignore[import-not-found]  # noqa: F401
            caps.add("neural_field")
        except ImportError:
            notes.append("tiny-cuda-nn missing — neural completion degraded")
        try:
            from ironengine_bonafide.backends.cuda.native_bridge import HAS_NATIVE
            if HAS_NATIVE:
                caps.update({"native_octree", "native_surfel", "native_splat",
                             "native_upload"})
                notes.append("bonafide_native loaded")
            else:
                notes.append("bonafide_native missing — pure-Python fast paths in use")
        except Exception as exc:                                            # noqa: BLE001
            notes.append(f"native_bridge probe failed: {exc}")
            HAS_NATIVE = False

        # Honesty gate: without a real raster library (nvdiffrast or the
        # native extension) this "CUDA" backend would allocate every frame
        # target on the GPU and then round-trip all raster work over PCIe
        # to the CPU reference path — strictly slower than CpuBackend.
        # Refuse to construct so Engine.auto() lands on the CPU instead.
        if "nvdiffrast" not in caps and not HAS_NATIVE:
            raise BackendUnavailable(
                "CUDA device present but no raster library is importable "
                "(nvdiffrast and bonafide_native both missing). Constructing "
                "a CUDA backend here would PCIe-round-trip every raster "
                "call; use the CPU backend instead. "
                "Install the [cuda] extra to enable GPU rasterization."
            )

        device_idx = torch.cuda.current_device()
        device = f"cuda:{device_idx}"
        version = torch.version.cuda or "unknown"
        self._info = BackendInfo(
            name="cuda",
            device=device,
            capabilities=frozenset(caps),
            version=str(version),
            notes="; ".join(notes),
        )

    @property
    def info(self) -> BackendInfo:
        return self._info

    def empty(self, shape: tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.empty(shape, dtype=dtype, device=self.device)

    def zeros(self, shape: tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.zeros(shape, dtype=dtype, device=self.device)

    def to_device(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.device)

    # ===================================================== depth-only raster
    def raster_depth(
        self,
        positions: torch.Tensor,
        indices: torch.Tensor,
        view_proj: torch.Tensor,
        width: int,
        height: int,
    ) -> torch.Tensor:
        """Torch-on-device depth raster for the CSM shadow pass.

        Same vectorized implementation as the CPU reference path
        (:mod:`ironengine_bonafide.backends.torch_raster`), executed on
        this backend's device so shadows no longer skip with
        ``no_raster_depth`` on GPU builds."""
        from ironengine_bonafide.backends.torch_raster import raster_depth
        return raster_depth(
            positions.to(self.device), indices.to(self.device),
            view_proj.to(self.device), width, height,
        )
