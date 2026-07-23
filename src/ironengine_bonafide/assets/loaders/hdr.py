"""Radiance HDR loader — alias of the EXR helper since imageio handles both."""
from __future__ import annotations

from ironengine_bonafide.assets.loaders.exr import load_image

__all__ = ["load_image"]
