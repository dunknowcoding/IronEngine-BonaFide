"""Shared GLB fixture factory — a textured quad with all five PBR texture
slots, KHR_texture_transform, and KHR_materials_emissive_strength, built
byte-by-byte (no external assets needed).

Import from test modules as ``from _glb_factory import build_full_glb``
(pytest's prepend import mode puts this directory on ``sys.path``; do NOT
import via ``tests.`` — the tests directory is not a package).
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

iio = pytest.importorskip("imageio.v3", reason="imageio required to build PNG fixtures")

TEX_BASE = np.array(
    [[[220, 20, 20], [20, 200, 20]], [[220, 20, 20], [20, 200, 20]]], dtype=np.uint8)
TEX_NORMAL = np.full((2, 2, 3), (128, 128, 255), dtype=np.uint8)
TEX_MR = np.full((2, 2, 3), (0, 200, 0), dtype=np.uint8)           # rough≈0.78, metal 0
TEX_AO = np.full((2, 2, 3), 255, dtype=np.uint8)
TEX_EMISSIVE = np.full((2, 2, 3), (255, 64, 0), dtype=np.uint8)

_POS = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=np.float32)
_NRM = np.array([[0, 0, 1]] * 4, dtype=np.float32)
UV = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
_IDX = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint16)

# KHR_texture_transform baked at load: u' = u + 0.25.
TRANSFORM = {"offset": [0.25, 0.0], "scale": [1.0, 1.0], "rotation": 0.0}


def _png(pixels: np.ndarray) -> bytes:
    import io
    buf = io.BytesIO()
    iio.imwrite(buf, pixels, extension=".png")
    return buf.getvalue()


def build_full_glb(path: Path) -> None:
    """Write a quad GLB with all five texture slots + shared UV transform."""
    pngs = [_png(t) for t in (TEX_BASE, TEX_NORMAL, TEX_MR, TEX_AO,
                              TEX_EMISSIVE)]
    blob = _POS.tobytes() + _NRM.tobytes() + UV.tobytes() + _IDX.tobytes()
    views = [
        {"buffer": 0, "byteOffset": 0, "byteLength": _POS.nbytes},
        {"buffer": 0, "byteOffset": 48, "byteLength": _NRM.nbytes},
        {"buffer": 0, "byteOffset": 96, "byteLength": UV.nbytes},
        {"buffer": 0, "byteOffset": 128, "byteLength": _IDX.nbytes},
    ]
    images = []
    for png in pngs:
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(png)})
        blob += png
        images.append({"bufferView": len(views) - 1, "mimeType": "image/png"})

    def tex(i: int) -> dict:
        # Every slot shares the same transform → bakeable into the UVs.
        return {"index": i,
                "extensions": {"KHR_texture_transform": TRANSFORM}}

    doc = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": views,
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3",
             "max": [1.0, 1.0, 0.0], "min": [-1.0, -1.0, 0.0]},
            {"bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": 4, "type": "VEC2"},
            {"bufferView": 3, "componentType": 5123, "count": 6, "type": "SCALAR"},
        ],
        "images": images,
        "textures": [{"source": i} for i in range(5)],
        "materials": [{
            "name": "full",
            "pbrMetallicRoughness": {
                "baseColorTexture": tex(0),
                "metallicRoughnessTexture": tex(2),
                "roughnessFactor": 1.0,
                "metallicFactor": 0.0,
            },
            "normalTexture": tex(1),
            "occlusionTexture": tex(3),
            "emissiveTexture": tex(4),
            "emissiveFactor": [1.0, 1.0, 1.0],
            "extensions": {
                "KHR_materials_emissive_strength": {"emissiveStrength": 2.0},
            },
        }],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1,
                                                   "TEXCOORD_0": 2},
                                    "indices": 3, "material": 0}]}],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }

    json_bytes = json.dumps(doc).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    bin_bytes = blob + b"\x00" * ((4 - len(blob) % 4) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
    with path.open("wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, total))
        fh.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
        fh.write(json_bytes)
        fh.write(struct.pack("<II", len(bin_bytes), 0x004E4942))
        fh.write(bin_bytes)
