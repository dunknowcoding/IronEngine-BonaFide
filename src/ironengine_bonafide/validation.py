"""Argument validation helpers used at API boundaries.

Validators raise :class:`ironengine_bonafide.errors.ValidationError` with
actionable messages — never bare ``ValueError`` or ``AssertionError``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ironengine_bonafide.errors import ValidationError


def positive_int(value: int, *, name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValidationError(f"{name} must be a positive int, got {value!r}")
    return value


def non_negative_int(value: int, *, name: str) -> int:
    if not isinstance(value, int) or value < 0:
        raise ValidationError(f"{name} must be a non-negative int, got {value!r}")
    return value


def in_range(value: float, *, name: str, low: float, high: float) -> float:
    if not (low <= value <= high):
        raise ValidationError(f"{name}={value} must be in [{low}, {high}]")
    return value


def shape(t: torch.Tensor, *, name: str, ndim: int | None = None,
          last: int | None = None) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        raise ValidationError(f"{name} must be a torch.Tensor, got {type(t).__name__}")
    if ndim is not None and t.ndim != ndim:
        raise ValidationError(f"{name}: expected ndim={ndim}, got {t.ndim} (shape={tuple(t.shape)})")
    if last is not None and t.shape[-1] != last:
        raise ValidationError(f"{name}: expected last dim={last}, got {tuple(t.shape)}")
    return t


def dtype(t: torch.Tensor, *, name: str, allowed: tuple[torch.dtype, ...]) -> torch.Tensor:
    if t.dtype not in allowed:
        raise ValidationError(
            f"{name}: dtype {t.dtype} not in {allowed}"
        )
    return t


def file_exists(path: Any, *, name: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise ValidationError(f"{name}: file not found: {p}")
    if not p.is_file():
        raise ValidationError(f"{name}: not a file: {p}")
    return p


def one_of(value: Any, *, name: str, allowed: tuple[Any, ...]) -> Any:
    if value not in allowed:
        raise ValidationError(f"{name}={value!r} must be one of {allowed}")
    return value
