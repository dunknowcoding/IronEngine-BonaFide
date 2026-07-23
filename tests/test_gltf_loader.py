"""GLB loader correctness: node world transforms, bufferView.byteStride,
multi-buffer GLBs, per-primitive materials, baseColor alpha, emissiveFactor.

Builds a synthetic GLB by hand-crafting the GLB container (pygltflib's
*writer* mangles multi-buffer files; the *reader* under test is fine):

* buffer 0 = GLB BIN chunk: prim0 vertices INTERLEAVED pos3|nrm3
  (byteStride=24) + uint16 indices.
* buffer 1 = ``data:`` URI: prim1 positions + uint32 indices.
* node 0 (mesh 0) carries TRS: translation (10,0,0), rotation +90° about Y,
  scale 2. node 1 (mesh 1) is untransformed.
* two distinct materials: alpha 0.5 + emissiveFactor on the first.
"""
from __future__ import annotations

import base64
import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pygltflib", reason="pygltflib required for glTF tests")

from ironengine_bonafide.assets.loaders.gltf import (  # noqa: E402
    load_mesh,
    load_primitives,
)

_PRIM0_POS = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
_PRIM0_NRM = np.array([[0.0, 0.0, 1.0]] * 3, dtype=np.float32)
_PRIM1_POS = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)


def _build_glb(path: Path) -> None:
    # --- buffer 0 (BIN): interleaved [pos3|nrm3] stride 24, then indices
    interleaved = np.empty((3, 6), dtype=np.float32)
    interleaved[:, :3] = _PRIM0_POS
    interleaved[:, 3:] = _PRIM0_NRM
    idx0 = np.array([0, 1, 2], dtype=np.uint16)
    blob = interleaved.tobytes() + idx0.tobytes()

    # --- buffer 1 (data: URI): positions + uint32 indices
    idx1 = np.array([0, 1, 2], dtype=np.uint32)
    buf1 = _PRIM1_POS.tobytes() + idx1.tobytes()
    uri1 = "data:application/octet-stream;base64," + base64.b64encode(buf1).decode()

    s = math.sqrt(0.5)
    doc = {
        "asset": {"version": "2.0"},
        "buffers": [
            {"byteLength": len(blob)},
            {"byteLength": len(buf1), "uri": uri1},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 72, "byteStride": 24},
            {"buffer": 0, "byteOffset": 72, "byteLength": 6},
            {"buffer": 1, "byteOffset": 0, "byteLength": 36},
            {"buffer": 1, "byteOffset": 36, "byteLength": 12},
        ],
        "accessors": [
            {"bufferView": 0, "byteOffset": 0, "componentType": 5126, "count": 3,
             "type": "VEC3", "max": [1.0, 1.0, 1.0], "min": [0.0, 0.0, 0.0]},
            {"bufferView": 0, "byteOffset": 12, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
            {"bufferView": 2, "byteOffset": 0, "componentType": 5126, "count": 3,
             "type": "VEC3", "max": [2.0, 2.0, 0.0], "min": [0.0, 0.0, 0.0]},
            {"bufferView": 3, "componentType": 5125, "count": 3, "type": "SCALAR"},
        ],
        "materials": [
            {"name": "red_half_alpha", "doubleSided": True,
             "emissiveFactor": [0.1, 0.2, 0.3],
             "pbrMetallicRoughness": {
                 "baseColorFactor": [1.0, 0.0, 0.0, 0.5],
                 "roughnessFactor": 0.4, "metallicFactor": 0.9}},
            {"name": "green_opaque",
             "pbrMetallicRoughness": {"baseColorFactor": [0.0, 1.0, 0.0, 1.0]}},
        ],
        "meshes": [
            {"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1},
                             "indices": 2, "material": 0}]},
            {"primitives": [{"attributes": {"POSITION": 3},
                             "indices": 4, "material": 1}]},
        ],
        "nodes": [
            {"mesh": 0, "translation": [10.0, 0.0, 0.0],
             "rotation": [0.0, s, 0.0, s], "scale": [2.0, 2.0, 2.0]},
            {"mesh": 1},
        ],
        "scenes": [{"nodes": [0, 1]}],
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
def glb_path(tmp_path: Path) -> Path:
    p = tmp_path / "constructed.glb"
    _build_glb(p)
    return p


def test_primitives_apply_node_world_transforms(glb_path: Path) -> None:
    prims = load_primitives(glb_path)
    assert len(prims) == 2
    pos0 = prims[0].mesh.positions.numpy()
    # (1,0,0) → scale 2 → (2,0,0) → rotY90 → (0,0,-2) → +t → (10,0,-2)
    np.testing.assert_allclose(pos0[0], [10.0, 0.0, -2.0], atol=1e-5)
    # (0,1,0) unaffected by Y rotation: → (0,2,0) → (10,2,0)
    np.testing.assert_allclose(pos0[1], [10.0, 2.0, 0.0], atol=1e-5)
    # Untransformed second mesh.
    np.testing.assert_allclose(prims[1].mesh.positions.numpy(), _PRIM1_POS, atol=1e-6)


def test_byte_stride_deinterleaves_normals(glb_path: Path) -> None:
    prims = load_primitives(glb_path)
    nrm = prims[0].mesh.normals
    assert nrm is not None
    # Normals rotated (not scaled/translated): (0,0,1) → rotY90 → (1,0,0).
    np.testing.assert_allclose(nrm.numpy(), np.array([[1.0, 0.0, 0.0]] * 3, dtype=np.float32),
                               atol=1e-5)


def test_multi_buffer_glb_reads_second_buffer(glb_path: Path) -> None:
    prims = load_primitives(glb_path)
    np.testing.assert_allclose(prims[1].mesh.positions.numpy()[1], [2.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_array_equal(prims[1].mesh.indices.numpy(), np.array([[0, 1, 2]]))


def test_each_primitive_keeps_its_material(glb_path: Path) -> None:
    prims = load_primitives(glb_path)
    m0 = prims[0].mesh.material
    assert m0.name == "red_half_alpha"
    np.testing.assert_allclose(m0.albedo, (1.0, 0.0, 0.0))
    assert m0.roughness == pytest.approx(0.4)
    assert m0.metallic == pytest.approx(0.9)
    np.testing.assert_allclose(m0.emissive, (0.1, 0.2, 0.3), atol=1e-7)
    assert prims[0].alpha == pytest.approx(0.5)
    assert prims[0].double_sided is True

    m1 = prims[1].mesh.material
    assert m1.name == "green_opaque"
    np.testing.assert_allclose(m1.albedo, (0.0, 1.0, 0.0))
    assert prims[1].alpha == pytest.approx(1.0)


def test_merged_load_mesh_uses_first_material_world_space(glb_path: Path) -> None:
    mesh = load_mesh(glb_path)
    assert mesh.material.name == "red_half_alpha"
    assert mesh.num_vertices == 6
    assert mesh.num_triangles == 2
    # Vertex order: prim0 (transformed) then prim1 (identity).
    np.testing.assert_allclose(mesh.positions.numpy()[0], [10.0, 0.0, -2.0], atol=1e-5)
    np.testing.assert_allclose(mesh.positions.numpy()[3], [0.0, 0.0, 0.0], atol=1e-6)
    # Indices of the second triangle are offset by 3.
    np.testing.assert_array_equal(mesh.indices.numpy()[1], np.array([3, 4, 5]))
