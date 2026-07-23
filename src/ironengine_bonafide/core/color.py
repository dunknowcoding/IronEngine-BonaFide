"""Color space utilities.

The engine works in linear HDR float internally. These helpers convert
to/from sRGB and uint8 for display / file IO.

Conventions:
  * `linear` — float32 linear-light, range [0, +inf)
  * `srgb`   — float32 sRGB-encoded, range [0, 1]
  * `uint8`  — uint8 sRGB, range [0, 255]
"""
from __future__ import annotations

import torch


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    """IEC 61966-2-1 sRGB transfer curve (gamma 2.4 with linear toe)."""
    a = 0.055
    return torch.where(
        x <= 0.0031308,
        12.92 * x,
        (1 + a) * torch.clamp(x, min=0.0) ** (1.0 / 2.4) - a,
    ).clamp_(0.0, 1.0)


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    a = 0.055
    return torch.where(
        x <= 0.04045,
        x / 12.92,
        ((x + a) / (1 + a)) ** 2.4,
    )


def to_uint8_srgb(linear: torch.Tensor) -> torch.Tensor:
    """linear-HDR → sRGB uint8 (standard display path)."""
    return (linear_to_srgb(linear) * 255.0 + 0.5).clamp_(0, 255).to(torch.uint8)


# --------- ACES tonemap (Narkowicz fit, fast & viewer-friendly) ----------
def aces_filmic(x: torch.Tensor) -> torch.Tensor:
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    return torch.clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


def tonemap_aces_to_srgb_uint8(linear: torch.Tensor, exposure: float = 1.0) -> torch.Tensor:
    """The standard 'show this on a display' path for an HDR linear frame."""
    return to_uint8_srgb(aces_filmic(linear * exposure))
