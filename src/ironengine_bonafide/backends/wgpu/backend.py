"""WGPU backend — portable AMD/Intel/Apple path.

v0.1 establishes the device + capability surface. Actual WGSL compute /
graphics pipelines live next door (`splat.py`, `raster.py`, etc.) and
land progressively in 0.2. Until each pipeline ships, the backend
declares only the capabilities it can fulfil today (`raster`, `splat`)
through a torch-on-device fallback so passes degrade cleanly rather
than crash.

The wgpu device is held lazily — calling :meth:`device_info` triggers
adapter selection but no work is queued.
"""
from __future__ import annotations

import torch

from ironengine_bonafide.core.backend import Backend, BackendInfo


class WgpuBackend(Backend):
    _CAPS = frozenset({"raster", "splat", "surfel", "shadow_csm",
                       "neural_denoise", "neural_upscale"})

    def __init__(self) -> None:
        from ironengine_bonafide.errors import BackendUnavailable
        try:
            import wgpu  # type: ignore[import-not-found]
        except ImportError as exc:
            raise BackendUnavailable(
                "wgpu-py not installed. Add the [wgpu] extra: pip install -e .[wgpu]"
            ) from exc

        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self._wgpu_device = adapter.request_device_sync()
        backend_info = adapter.request_adapter_info_sync()
        # Try MPS / CUDA for the torch side; fall back to CPU.
        if torch.cuda.is_available():
            t_device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            t_device = "mps"
        else:
            t_device = "cpu"
        self._t_device = t_device
        self._info = BackendInfo(
            name="wgpu",
            device=t_device,
            capabilities=self._CAPS,
            version=str(getattr(backend_info, "device", "wgpu")),
            notes=(
                f"wgpu adapter='{getattr(backend_info, 'device', 'unknown')}' "
                f"backend='{getattr(backend_info, 'backend_type', '?')}'; "
                f"torch_device={t_device}. WGSL pipelines partial — falls back "
                f"to torch ops where shaders are not yet authored."
            ),
        )

    @property
    def info(self) -> BackendInfo:
        return self._info

    def empty(self, shape: tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.empty(shape, dtype=dtype, device=self._t_device)

    def zeros(self, shape: tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.zeros(shape, dtype=dtype, device=self._t_device)

    def to_device(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self._t_device)

    # ===================================================== depth-only raster
    def raster_depth(
        self,
        positions: torch.Tensor,
        indices: torch.Tensor,
        view_proj: torch.Tensor,
        width: int,
        height: int,
    ) -> torch.Tensor:
        """Torch-on-device depth raster for the CSM shadow pass (same
        vectorized implementation as the CPU reference path, executed on
        this backend's torch device)."""
        from ironengine_bonafide.backends.torch_raster import raster_depth
        return raster_depth(
            positions.to(self._t_device), indices.to(self._t_device),
            view_proj.to(self._t_device), width, height,
        )

    @property
    def wgpu_device(self) -> object:
        return self._wgpu_device
