"""glTF / GLB loader.

Correctness notes (vs. the previous version):

* **Node world transforms are applied** — primitives are transformed by the
  composed node hierarchy of the default scene (previously everything was
  merged in mesh-local space).
* **``bufferView.byteStride`` is honored** — interleaved vertex buffers are
  de-interleaved row by row.
* **Multi-buffer GLBs work** — any buffer without a URI resolves to the GLB
  binary chunk; ``data:`` URIs and external files are also supported.
* **Every primitive keeps its own material** via :func:`load_primitives`
  (:func:`load_mesh` still merges for legacy callers; its docstring says the
  first primitive's material wins).
* **baseColor alpha and emissiveFactor are kept** — alpha rides on
  :class:`GltfPrimitive` because ``PBRMaterial`` has no opacity channel yet;
  emissiveFactor maps to ``PBRMaterial.emissive``.
* ``COLOR_0`` vertex colors are loaded when present.

Texture *maps* (baseColorTexture etc.) are still not sampled by any pass;
they remain unresolved.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ironengine_bonafide.core.material import PBRMaterial
from ironengine_bonafide.core.mesh import Mesh
from ironengine_bonafide.core.softbody import DollRig


@dataclass(slots=True)
class GltfPrimitive:
    """One glTF primitive as a world-space Mesh, plus the material fields
    ``PBRMaterial`` cannot yet represent."""
    mesh: Mesh
    alpha: float = 1.0                     # baseColorFactor[3]
    double_sided: bool = False


# --------------------------------------------------------------- accessors
_TYPE_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
_COMPONENT_DTYPE = {
    5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16,
    5125: np.uint32, 5126: np.float32,
}


def _read_accessor(gltf: object, buffers: list[bytes], accessor_idx: int) -> np.ndarray:
    acc = gltf.accessors[accessor_idx]                       # type: ignore[attr-defined]
    component = _COMPONENT_DTYPE[acc.componentType]
    count = int(acc.count)
    components = _TYPE_COUNT[acc.type]
    if acc.bufferView is None:
        # Accessor with no bufferView starts zero-filled (sparse overrides
        # are not supported yet).
        out = np.zeros((count, components), dtype=component)
        return out.reshape(count, components) if components > 1 else out.reshape(-1)
    view = gltf.bufferViews[acc.bufferView]                   # type: ignore[attr-defined]
    buf = buffers[view.buffer]
    offset = (view.byteOffset or 0) + (acc.byteOffset or 0)
    elem_bytes = components * np.dtype(component).itemsize
    stride = view.byteStride or elem_bytes
    if stride == elem_bytes:
        raw = np.frombuffer(buf, dtype=component, count=count * components, offset=offset)
    else:
        # Interleaved: copy each strided row, then reinterpret the packed
        # prefix as the component dtype. The last row only occupies
        # elem_bytes, so don't read a full stride past it.
        read_bytes = (count - 1) * stride + elem_bytes
        flat = np.frombuffer(buf, dtype=np.uint8, count=read_bytes, offset=offset)
        rows = np.empty((count, stride), dtype=np.uint8)
        if count > 1:
            rows[: count - 1] = flat[: (count - 1) * stride].reshape(count - 1, stride)
        rows[count - 1, :elem_bytes] = flat[(count - 1) * stride:]
        raw = np.ascontiguousarray(rows[:, :elem_bytes]).view(component).reshape(-1)
    out = raw.reshape(count, components) if components > 1 else raw.reshape(-1)
    # glTF `normalized` integer accessors (the standard encoding for COLOR_0
    # and quantized TEXCOORD_0) must be scaled to float: unsigned types map
    # to [0, 1], signed types to [-1, 1]. Without this, uint8 vertex colors
    # arrive as 0..255 "albedo" and blow every shaded pixel to white.
    if getattr(acc, "normalized", False) and np.issubdtype(component, np.integer):
        if np.issubdtype(component, np.unsignedinteger):
            scale = float(np.iinfo(component).max)
            out = out.astype(np.float32) / scale
        else:
            scale = float(np.iinfo(component).max)
            out = np.maximum(out.astype(np.float32) / scale, -1.0)
    return out


def _load_gltf(path: Path) -> tuple[object, list[bytes]]:
    try:
        import pygltflib
    except ImportError as exc:
        raise RuntimeError("pygltflib required to load glTF / GLB") from exc

    gltf = pygltflib.GLTF2().load(str(path))
    is_glb = path.suffix.lower() == ".glb"
    buffers: list[bytes] = []
    for b in gltf.buffers:
        uri = b.uri or ""
        if not uri:
            # URI-less buffer = the GLB binary chunk (multi-buffer GLBs
            # still have exactly one such buffer; extras use data: URIs).
            buffers.append(gltf.binary_blob() if is_glb else b"")
        elif uri.startswith("data:"):
            import base64
            _, _, payload = uri.partition(",")
            buffers.append(base64.b64decode(payload))
        else:
            buffers.append((path.parent / uri).read_bytes())
    return gltf, buffers


# --------------------------------------------------------------- node transforms
def _quat_to_mat3(q: Any) -> np.ndarray:
    """glTF node rotation quaternion (x, y, z, w) → 3x3 matrix."""
    x, y, z, w = (float(v) for v in np.asarray(q, dtype=np.float64).reshape(4))
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def _node_local_matrix(node: Any) -> np.ndarray:
    if node.matrix:
        return np.asarray(node.matrix, dtype=np.float64).reshape(4, 4).T
    m = np.eye(4, dtype=np.float64)
    rot = _quat_to_mat3(node.rotation or (0.0, 0.0, 0.0, 1.0))
    scale = np.asarray(node.scale or (1.0, 1.0, 1.0), dtype=np.float64)
    m[:3, :3] = rot * scale[None, :]
    m[:3, 3] = np.asarray(node.translation or (0.0, 0.0, 0.0), dtype=np.float64)
    return m


def _iter_mesh_nodes(gltf: object) -> list[tuple[int, np.ndarray]]:
    """Yield (mesh_index, world_matrix) for every node with a mesh in the
    default scene, composing ancestor transforms."""
    out: list[tuple[int, np.ndarray]] = []

    def walk(node_idx: int, parent_m: np.ndarray) -> None:
        node = gltf.nodes[node_idx]                           # type: ignore[attr-defined]
        m = parent_m @ _node_local_matrix(node)
        if node.mesh is not None:
            out.append((int(node.mesh), m))
        for child in node.children or []:
            walk(int(child), m)

    scenes = gltf.scenes                                      # type: ignore[attr-defined]
    if scenes:
        scene_idx = gltf.scene if gltf.scene is not None else 0  # type: ignore[attr-defined]
        for root in scenes[scene_idx].nodes or []:
            walk(int(root), np.eye(4, dtype=np.float64))
    else:
        # No scene graph: every mesh at identity.
        for i in range(len(gltf.meshes)):                     # type: ignore[attr-defined]
            out.append((i, np.eye(4, dtype=np.float64)))
    return out


# --------------------------------------------------------------- materials
def _material_for(gltf: object, mat_idx: int | None) -> tuple[PBRMaterial, float, bool]:
    """→ (PBRMaterial, baseColor alpha, double_sided)."""
    if mat_idx is None:
        return PBRMaterial(name="default"), 1.0, False
    m = gltf.materials[mat_idx]                               # type: ignore[attr-defined]
    pbr = m.pbrMetallicRoughness
    albedo = (0.8, 0.8, 0.8)
    alpha = 1.0
    roughness = 0.7
    metallic = 0.0
    if pbr is not None:
        if pbr.baseColorFactor is not None:
            albedo = tuple(float(c) for c in pbr.baseColorFactor[:3])  # type: ignore[assignment]
            alpha = float(pbr.baseColorFactor[3])
        if pbr.roughnessFactor is not None:
            roughness = float(pbr.roughnessFactor)
        if pbr.metallicFactor is not None:
            metallic = float(pbr.metallicFactor)
    emissive = (0.0, 0.0, 0.0)
    if m.emissiveFactor is not None:
        emissive = tuple(float(c) for c in m.emissiveFactor[:3])       # type: ignore[assignment]
    return (
        PBRMaterial(
            name=m.name or "default",
            albedo=albedo, roughness=roughness, metallic=metallic,
            emissive=emissive, two_sided=bool(m.doubleSided),
        ),
        alpha,
        bool(m.doubleSided),
    )


# --------------------------------------------------------------- public API
def load_primitives(path: Path) -> list[GltfPrimitive]:
    """Load every primitive of a glTF/GLB as a world-space Mesh with its
    own material (plus baseColor alpha on the wrapper record)."""
    gltf, buffers = _load_gltf(path)
    primitives: list[GltfPrimitive] = []

    for mesh_idx, world_m in _iter_mesh_nodes(gltf):
        rot = world_m[:3, :3]
        for prim in gltf.meshes[mesh_idx].primitives:         # type: ignore[attr-defined]
            attrs = prim.attributes
            pos = _read_accessor(gltf, buffers, attrs.POSITION).astype(np.float32)
            # Bake the node world transform into the geometry.
            pos = (pos.astype(np.float64) @ rot.T + world_m[:3, 3]).astype(np.float32)

            normals = None
            if attrs.NORMAL is not None:
                nrm = _read_accessor(gltf, buffers, attrs.NORMAL).astype(np.float64)
                nrm = nrm @ rot.T
                nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)
                normals = nrm.astype(np.float32)
            uvs = None
            if attrs.TEXCOORD_0 is not None:
                uvs = _read_accessor(gltf, buffers, attrs.TEXCOORD_0).astype(np.float32)
            colors = None
            color_idx = getattr(attrs, "COLOR_0", None)
            if color_idx is not None:
                col = _read_accessor(gltf, buffers, color_idx).astype(np.float32)
                if col.ndim == 2 and col.shape[1] == 4:
                    col = col[:, :3]                     # Mesh.colors is RGB
                colors = col
            if prim.indices is not None:
                idx = _read_accessor(gltf, buffers, prim.indices).astype(np.int64)
                idx = idx.reshape(-1, 3)
            else:
                idx = np.arange(pos.shape[0], dtype=np.int64).reshape(-1, 3)

            material, alpha, double_sided = _material_for(gltf, prim.material)
            primitives.append(GltfPrimitive(
                mesh=Mesh.from_arrays(
                    pos, idx, normals=normals, uvs=uvs, colors=colors,
                    material=material, name=path.stem,
                ),
                alpha=alpha,
                double_sided=double_sided,
            ))
    return primitives


def load_mesh(path: Path) -> Mesh:
    """Merge all primitives into one Mesh (world-space geometry).

    Legacy convenience — the first primitive's material wins and baseColor
    alpha is dropped. Use :func:`load_primitives` when materials matter.
    """
    prims = load_primitives(path)
    if not prims:
        return Mesh.from_arrays(
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.int64),
            name=path.stem,
        )
    positions: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    uvs: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    base = 0
    for p in prims:
        m = p.mesh
        pos = m.positions.detach().cpu().numpy()
        idx = m.indices.detach().cpu().numpy()
        positions.append(pos)
        indices.append(idx + base)
        base += pos.shape[0]
        if m.normals is not None:
            normals.append(m.normals.detach().cpu().numpy())
        if m.uvs is not None:
            uvs.append(m.uvs.detach().cpu().numpy())
        if m.colors is not None:
            colors.append(m.colors.detach().cpu().numpy())
    return Mesh.from_arrays(
        np.concatenate(positions, axis=0),
        np.concatenate(indices, axis=0) if indices else np.zeros((0, 3), dtype=np.int64),
        normals=np.concatenate(normals, axis=0) if len(normals) == len(prims) else None,
        uvs=np.concatenate(uvs, axis=0) if len(uvs) == len(prims) else None,
        colors=np.concatenate(colors, axis=0) if len(colors) == len(prims) else None,
        material=prims[0].mesh.material,
        name=path.stem,
    )


def load_rig(path: Path, *, stiffness: float = 0.8) -> DollRig:
    """Soft-body rig from a glTF/GLB. The mesh's vertices become particles
    and triangle edges become distance constraints. Skinning weights are
    not yet harvested — that lands with full skeleton support in 0.2."""
    mesh = load_mesh(path)
    pos = mesh.positions.cpu().numpy()
    idx = mesh.indices.cpu().numpy()
    edges_set: set[tuple[int, int]] = set()
    for tri in idx:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v in ((a, b), (b, c), (c, a)):
            edges_set.add((min(u, v), max(u, v)))
    edges = np.asarray(sorted(edges_set), dtype=np.int64)
    return DollRig.from_arrays(pos, edges, stiffness=stiffness, name=mesh.name)
