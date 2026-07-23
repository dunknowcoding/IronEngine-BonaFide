"""OpenEXR / HDR helpers — thin wrappers around imageio.

Returns float32 (H, W, 3) arrays in linear color space.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_image(path: Path) -> np.ndarray:
    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise RuntimeError("imageio required for EXR/HDR loading") from exc
    img = iio.imread(path)
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    else:
        img = np.asarray(img, dtype=np.float32)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img
