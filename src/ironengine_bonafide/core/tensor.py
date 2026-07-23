"""Tensor interop layer.

`as_torch(x)` and `as_numpy(x)` accept any of: numpy.ndarray, torch.Tensor,
or cupy.ndarray (when CuPy is installed). Conversion goes through DLPack
when both ends are GPU resident, avoiding a host roundtrip.

This is the only place in the codebase that has to know about the three
tensor libraries. Every other module talks to torch.Tensor.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _is_cupy(x: Any) -> bool:
    cls = type(x).__module__
    return cls.startswith("cupy")


def as_torch(x: Any, *, device: str | torch.device | None = None,
             dtype: torch.dtype | None = None) -> torch.Tensor:
    """Convert an arbitrary array-like into a torch.Tensor.

    Conversion rules:
      * numpy → torch.from_numpy (zero-copy on CPU)
      * cupy  → torch.utils.dlpack (zero-copy on GPU)
      * torch → returned as-is
    """
    if isinstance(x, torch.Tensor):
        out = x
    elif isinstance(x, np.ndarray):
        out = torch.from_numpy(np.ascontiguousarray(x))
    elif _is_cupy(x):
        # cupy → DLPack → torch
        from torch.utils.dlpack import from_dlpack
        out = from_dlpack(x.toDlpack())
    else:
        # last-ditch: try numpy first
        out = torch.as_tensor(np.asarray(x))

    if device is not None:
        out = out.to(device)
    if dtype is not None:
        out = out.to(dtype)
    return out


def as_numpy(x: Any) -> np.ndarray:
    """Convert torch.Tensor / cupy.ndarray / numpy.ndarray → numpy."""
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if _is_cupy(x):
        return x.get()  # type: ignore[no-any-return]
    return np.asarray(x)


def as_cupy(x: Any) -> Any:
    """Convert to cupy.ndarray on the current CUDA device. Raises if CuPy
    isn't installed — callers should gate this behind a backend check."""
    import cupy as cp  # type: ignore[import-not-found]
    if _is_cupy(x):
        return x
    if isinstance(x, torch.Tensor):
        return cp.fromDlpack(torch.utils.dlpack.to_dlpack(x))  # type: ignore[no-any-return]
    return cp.asarray(x)


def device_of(x: torch.Tensor) -> str:
    """Stringify a tensor's device, e.g. 'cuda:0' or 'cpu'."""
    return str(x.device)
