"""Backend auto-selection."""
from __future__ import annotations

from ironengine_bonafide.api import Engine
from ironengine_bonafide.backends.cpu.backend import CpuBackend
from ironengine_bonafide.core.backend import probe


def test_cpu_backend_always_constructs() -> None:
    e = Engine.cpu()
    assert isinstance(e.backend, CpuBackend)
    assert e.backend.name == "cpu"
    assert e.backend.supports("raster")
    assert e.backend.supports("splat")
    assert not e.backend.supports("gsplat")     # CPU never advertises this


def test_auto_falls_back_to_cpu() -> None:
    e = Engine.auto()
    # Either we have CUDA / wgpu (good) or we land on CPU (also good)
    assert e.backend.name in {"cuda", "wgpu", "cpu"}


def test_probe_runs_clean() -> None:
    p = probe()
    # Just ensures the probe never raises and returns a dataclass.
    assert hasattr(p, "cuda_available")
    assert hasattr(p, "wgpu_available")
