"""Full glTF texture-set loading: normal / metallic-roughness / occlusion /
emissive maps, KHR_texture_transform UV baking, and
KHR_materials_emissive_strength — plus a rendered check that the emissive
map reaches the frame.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pygltflib", reason="pygltflib required for glTF tests")
iio = pytest.importorskip("imageio.v3", reason="imageio required to build PNG fixtures")

from ironengine_bonafide.api import (  # noqa: E402
    Engine,
    PerspectiveCamera,
    RenderConfig,
    Scene,
    render,
)
from ironengine_bonafide.assets.loaders.gltf import load_primitives  # noqa: E402

_TEX_BASE = np.array(
    [[[220, 20, 20], [20, 200, 20]], [[220, 20, 20], [20, 200, 20]]], dtype=np.uint8)
_TEX_NORMAL = np.full((2, 2, 3), (128, 128, 255), dtype=np.uint8)
_TEX_MR = np.full((2, 2, 3), (0, 200, 0), dtype=np.uint8)         # rough≈0.78, metal 0
_TEX_AO = np.full((2, 2, 3), 255, dtype=np.uint8)
_TEX_EMISSIVE = np.full((2, 2, 3), (255, 64, 0), dtype=np.uint8)

_POS = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=np.float32)
_NRM = np.array([[0, 0, 1]] * 4, dtype=np.float32)
_UV = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
_IDX = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint16)

# KHR_texture_transform baked at load: u' = u + 0.25.
_TRANSFORM = {"offset": [0.25, 0.0], "scale": [1.0, 1.0], "rotation": 0.0}


def _png(pixels: np.ndarray) -> bytes:
    import io
    buf = io.BytesIO()
    iio.imwrite(buf, pixels, extension=".png")
    return buf.getvalue()


def _build_full_glb(path: Path) -> None:
    pngs = [_png(t) for t in (_TEX_BASE, _TEX_NORMAL, _TEX_MR, _TEX_AO,
                              _TEX_EMISSIVE)]
    blob = _POS.tobytes() + _NRM.tobytes() + _UV.tobytes() + _IDX.tobytes()
    views = [
        {"buffer": 0, "byteOffset": 0, "byteLength": _POS.nbytes},
        {"buffer": 0, "byteOffset": 48, "byteLength": _NRM.nbytes},
        {"buffer": 0, "byteOffset": 96, "byteLength": _UV.nbytes},
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
                "extensions": {"KHR_texture_transform": _TRANSFORM}}

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


@pytest.fixture()
def full_glb(tmp_path: Path) -> Path:
    p = tmp_path / "full_textured_quad.glb"
    _build_full_glb(p)
    return p


def test_all_five_map_slots_resolve(full_glb: Path) -> None:
    mat = load_primitives(full_glb)[0].mesh.material
    for slot in ("albedo_map", "normal_map", "metallic_roughness_map",
                 "ao_map", "emissive_map"):
        ref = getattr(mat, slot)
        assert ref is not None, f"{slot} not resolved"
        assert Path(ref).is_file(), f"{slot} cache file missing: {ref}"


def test_texture_bytes_roundtrip(full_glb: Path) -> None:
    mat = load_primitives(full_glb)[0].mesh.material
    pixels = np.asarray(iio.imread(mat.emissive_map))[..., :3]
    np.testing.assert_array_equal(pixels, _TEX_EMISSIVE)


def test_khr_texture_transform_bakes_uvs(full_glb: Path) -> None:
    prim = load_primitives(full_glb)[0]
    uvs = prim.mesh.uvs.detach().cpu().numpy()
    np.testing.assert_allclose(uvs[:, 0], _UV[:, 0] + 0.25, atol=1e-6)
    np.testing.assert_allclose(uvs[:, 1], _UV[:, 1], atol=1e-6)


def test_emissive_strength_scales_factor(full_glb: Path) -> None:
    mat = load_primitives(full_glb)[0].mesh.material
    np.testing.assert_allclose(mat.emissive, (2.0, 2.0, 2.0))


def test_render_samples_emissive_map(full_glb: Path) -> None:
    prim = load_primitives(full_glb)[0]
    scene = Scene().add(prim.mesh)                       # no lights on purpose
    cam = PerspectiveCamera(position=(0, 0, 3), look_at=(0, 0, 0), fov_deg=45)
    out = render(Engine.cpu(), scene, cam,
                 RenderConfig(width=96, height=64, output_color_space="sRGB"))
    rgb = out.rgb.cpu().numpy()
    center = rgb[16:48, 32:64]
    # Emissive texel (255, 64, 0) × factor 2 → strong red, faint green,
    # no blue; ambient alone could never produce this gap.
    assert center[..., 0].mean() > center[..., 2].mean() + 0.3
    assert center[..., 0].mean() > center[..., 1].mean() + 0.15
