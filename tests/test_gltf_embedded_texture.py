"""GLB embedded base-color texture decoding.

Builds a textured quad GLB by hand-crafting the container (same approach as
test_gltf_loader): geometry + a 2x2 PNG live in the GLB BIN chunk, the image
is referenced via ``bufferView``, and the material carries a
``baseColorTexture`` with no ``baseColorFactor``.

Left half of the texture is red, right half is green — so a rendered frame
can be checked spatially without depending on absolute lighting levels.
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
    DirectionalLight,
    Engine,
    PerspectiveCamera,
    RenderConfig,
    Scene,
    render,
)
from ironengine_bonafide.assets.loaders.gltf import (  # noqa: E402
    load_mesh,
    load_primitives,
)

# 2x2 RGB: left column red, right column green.
_TEX_PIXELS = np.array(
    [[[220, 20, 20], [20, 200, 20]], [[220, 20, 20], [20, 200, 20]]], dtype=np.uint8
)

_POS = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=np.float32)
_NRM = np.array([[0, 0, 1]] * 4, dtype=np.float32)
_UV = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
_IDX = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint16)


def _png_bytes() -> bytes:
    import io

    buf = io.BytesIO()
    iio.imwrite(buf, _TEX_PIXELS, extension=".png")
    return buf.getvalue()


def _build_textured_glb(path: Path, *, image_uri: str | None = None) -> None:
    """Quad with an embedded (bufferView) or external-URI baseColor texture."""
    png = _png_bytes()
    blob = _POS.tobytes() + _NRM.tobytes() + _UV.tobytes() + _IDX.tobytes()
    views = [
        {"buffer": 0, "byteOffset": 0, "byteLength": _POS.nbytes},
        {"buffer": 0, "byteOffset": 48, "byteLength": _NRM.nbytes},
        {"buffer": 0, "byteOffset": 96, "byteLength": _UV.nbytes},
        {"buffer": 0, "byteOffset": 128, "byteLength": _IDX.nbytes},
    ]
    if image_uri is None:
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(png)})
        blob += png
        image = {"bufferView": 4, "mimeType": "image/png"}
    else:
        image = {"uri": image_uri}

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
        "images": [image],
        "samplers": [{"magFilter": 9729, "minFilter": 9729}],
        "textures": [{"sampler": 0, "source": 0}],
        "materials": [
            {"name": "textured",
             "pbrMetallicRoughness": {
                 "baseColorTexture": {"index": 0},
                 "roughnessFactor": 1.0, "metallicFactor": 0.0}},
        ],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
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
        fh.write(struct.pack("<III", 0x46546C67, 2, total))          # magic, version, length
        fh.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))     # JSON chunk
        fh.write(json_bytes)
        fh.write(struct.pack("<II", len(bin_bytes), 0x004E4942))      # BIN chunk
        fh.write(bin_bytes)


@pytest.fixture()
def textured_glb(tmp_path: Path) -> Path:
    p = tmp_path / "textured_quad.glb"
    _build_textured_glb(p)
    return p


def test_embedded_texture_decodes_to_albedo_map(textured_glb: Path) -> None:
    prims = load_primitives(textured_glb)
    assert len(prims) == 1
    mat = prims[0].mesh.material
    assert mat.albedo_map is not None
    cached = Path(mat.albedo_map)
    assert cached.is_file()
    pixels = np.asarray(iio.imread(cached))[..., :3]
    np.testing.assert_array_equal(pixels, _TEX_PIXELS)


def test_albedo_factor_defaults_white_when_textured(textured_glb: Path) -> None:
    # glTF's default baseColorFactor is white and multiplies the texture;
    # falling back to the neutral gray would darken every textured asset.
    mat = load_primitives(textured_glb)[0].mesh.material
    np.testing.assert_allclose(mat.albedo, (1.0, 1.0, 1.0))


def test_external_uri_texture_resolves_relative_to_glb(tmp_path: Path) -> None:
    (tmp_path / "tex.png").write_bytes(_png_bytes())
    glb = tmp_path / "external_tex.glb"
    _build_textured_glb(glb, image_uri="tex.png")
    mat = load_mesh(glb).material
    assert mat.albedo_map == str(tmp_path / "tex.png")


def test_missing_external_uri_leaves_map_unset(tmp_path: Path) -> None:
    glb = tmp_path / "missing_tex.glb"
    _build_textured_glb(glb, image_uri="nope.png")
    mat = load_mesh(glb).material
    assert mat.albedo_map is None
    # baseColorFactor is absent in the JSON; pygltflib fills the glTF spec
    # default (white), which the loader honors.
    np.testing.assert_allclose(mat.albedo, (1.0, 1.0, 1.0))


def test_cpu_render_samples_embedded_texture(textured_glb: Path) -> None:
    mesh = load_mesh(textured_glb)
    assert mesh.material.albedo_map is not None
    scene = Scene().add(mesh).add(DirectionalLight(direction=(0.0, 0.0, -1.0),
                                                    intensity=3.0))
    cam = PerspectiveCamera(position=(0, 0, 3), look_at=(0, 0, 0), fov_deg=45)
    out = render(Engine.cpu(), scene, cam,
                 RenderConfig(width=96, height=64, output_color_space="sRGB"))
    rgb = out.rgb.cpu().numpy() if hasattr(out.rgb, "cpu") else np.asarray(out.rgb)
    # Quad fills the frame center; sample well inside it to dodge edges.
    left = rgb[16:48, 16:32]
    right = rgb[16:48, 64:80]
    assert left[..., 0].mean() > left[..., 1].mean() + 0.02     # red half
    assert right[..., 1].mean() > right[..., 0].mean() + 0.02   # green half
