"""Backend selection.

`auto_select(prefer)` picks the best Backend the host can construct.
The discovery order is documented in the module docstring of
:mod:`ironengine_bonafide.core.backend`.
"""
from __future__ import annotations

from typing import Literal

from ironengine_bonafide.core.backend import Backend, probe
from ironengine_bonafide.errors import BackendUnavailable
from ironengine_bonafide.logging import logger

DeviceHint = Literal["auto", "cuda", "wgpu", "cpu", "mps"]


def auto_select(prefer: DeviceHint = "auto") -> Backend:
    info = probe()
    order: tuple[str, ...]
    if prefer == "auto":
        order = ("cuda", "wgpu", "cpu")
    elif prefer == "mps":
        # Treat MPS as CPU-with-torch-MPS for now; full Metal path lives in WGPU.
        order = ("cpu",)
    else:
        order = (prefer, "cpu")

    for name in order:
        try:
            backend = _build(name)
            logger.info(f"Selected backend: {backend}")
            return backend
        except Exception as exc:
            logger.warning(f"Backend '{name}' unavailable: {exc}")
    raise BackendUnavailable(
        f"No backend could be constructed (probe={info}). "
        "Install [cuda] or [wgpu] extras, or rely on the CPU backend."
    )


def _build(name: str) -> Backend:
    if name == "cuda":
        from ironengine_bonafide.backends.cuda.backend import CudaBackend
        return CudaBackend()
    if name == "wgpu":
        from ironengine_bonafide.backends.wgpu.backend import WgpuBackend
        return WgpuBackend()
    if name == "cpu":
        from ironengine_bonafide.backends.cpu.backend import CpuBackend
        return CpuBackend()
    raise ValueError(f"Unknown backend: {name}")
