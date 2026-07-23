"""KTX2 / Basis Universal texture loader.

Requires the `[formats]` extra (`pyktx`). Returns a numpy uint8 (H, W, 4)
RGBA array; transcoding happens inside pyktx so the engine sees a plain
texture buffer.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_texture(path: Path) -> np.ndarray:
    try:
        from pyktx import KtxTexture2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "pyktx required for KTX2. Install with: pip install -e .[formats]"
        ) from exc
    tex = KtxTexture2.create_from_named_file(str(path))
    # Transcode to RGBA8 if compressed
    if hasattr(tex, "needs_transcoding") and tex.needs_transcoding:
        tex.transcode_basis_u("RGBA32")
    pixels = np.frombuffer(tex.data, dtype=np.uint8).reshape(tex.base_height, tex.base_width, 4).copy()
    return pixels
