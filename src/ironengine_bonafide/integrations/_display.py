"""Shared display conversion for integration shims.

Tonemap contract (cross-repo): when ``RenderConfig.output_color_space ==
"sRGB"``, the ``TonemapPass`` output is *final, display-ready sRGB* in
``[0, 1]``. Integrations must convert that tensor straight to uint8 and
must **never** apply a second ACES curve (``to_aces_srgb_uint8``) on top —
doing so double-tonemaps and crushes highlights.
"""
from __future__ import annotations

import numpy as np
import torch


def srgb_to_uint8(rgb: torch.Tensor) -> np.ndarray:
    """Display-ready sRGB float tensor ``(H, W, 3)`` in [0, 1] → uint8."""
    return (
        rgb.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
