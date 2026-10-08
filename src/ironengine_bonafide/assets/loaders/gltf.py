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
  ``PBRMaterial.alpha`` (honored by the PBR pass when
  ``RenderConfig.transparency`` is on) and on :class:`GltfPrimitive` for
  legacy callers; emissiveFactor maps to ``PBRMaterial.emissive``.
* ``COLOR_0`` vertex colors are loaded when present.

Embedded textures are resolved for **all five PBR slots** — base color,
normal, metallic-roughness, occlusion, and emissive — whether carried in
the GLB binary chunk (``bufferView`` images), as ``data:`` URIs, or
referenced by a relative URI. Decoded bytes are written to a deterministic
temp-cache file and bound to the matching ``PBRMaterial`` map slot, which
the CPU PBR pass samples (factor values still multiply, per spec).
``KHR_texture_transform`` is honored by baking the UV affine
(offset/scale/rotation) into the primitive's UVs when every textured slot
of the material shares one transform; conflicting transforms keep the
base-color one and log a warning. KTX2 sources (``KHR_texture_basisu``)
require the ``[formats]`` extra and are skipped with a warning otherwise;
sampler wrap/filter modes are not applied (sampling always repeats with
bilinear filtering) and only ``TEXCOORD_0`` is used.
"""
from __future__ import annotations

import base64
import hashlib
import math
import tempfile
import urllib.parse
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


# --------------------------------------------------------------- textures
_TEX_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
}


def _cache_texture_bytes(raw: bytes, glb_path: Path, image_idx: int, mime: str) -> str:
    """Write embedded image bytes to a deterministic temp-cache file and
    return its path (``PbrPass`` loads texture maps from filesystem paths).

    The cache key covers the GLB path, image index, and the bytes themselves,
    so re-loading an unchanged GLB reuses the file and a changed texture never
    collides with a stale one.
    """
    ext = _TEX_EXT.get(mime.lower(), "." + mime.rsplit("/", 1)[-1].replace("x-", ""))
    key = hashlib.sha1(
        str(glb_path.resolve()).encode("utf-8") + b"|" + str(image_idx).encode() + b"|" + raw
    ).hexdigest()[:16]
    cache_dir = Path(tempfile.gettempdir()) / "ironengine_bonafide_glb_textures"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{glb_path.stem}_{image_idx}_{key}{ext}"
    if not out.is_file() or out.stat().st_size != len(raw):
        out.write_bytes(raw)
    return str(out)


def _texture_path(gltf: object, buffers: list[bytes], path: Path,
                  info: Any, *, slot: str) -> str | None:
    """Resolve any glTF texture-info (baseColor / normal / MR / occlusion /
    emissive) to a filesystem path the PBR pass can sample, or None when it
    cannot be resolved.

    Embedded images (GLB ``bufferView`` or ``data:`` URI) are decoded to a
    temp-cache file; relative URIs resolve against the GLB's folder. KTX2
    sources log a warning and return None (the ``[formats]`` extra ships a
    KTX2 loader for filesystem refs). Sampler wrap/filter settings are
    ignored — sampling always repeats with bilinear filtering.
    """
    if info is None or info.index is None:
        return None
    textures = getattr(gltf, "textures", None) or []
    images = getattr(gltf, "images", None) or []
    if info.index >= len(textures):
        return None
    src = textures[info.index].source
    if src is None or src >= len(images):
        return None
    img = images[src]
    mime = (img.mimeType or "").lower()
    if mime == "image/ktx2":
        from ironengine_bonafide.logging import logger
        logger.warning(f"gltf: KTX2 {slot} texture skipped (needs the [formats] extra)")
        return None
    if img.uri:
        if img.uri.startswith("data:"):
            header, _, payload = img.uri.partition(",")
            mime = header[len("data:"):].split(";")[0] or "image/png"
            try:
                raw = base64.b64decode(payload)
            except ValueError:
                return None
            return _cache_texture_bytes(raw, path, int(src), mime)
        candidate = path.parent / urllib.parse.unquote(img.uri)
        return str(candidate) if candidate.is_file() else None
    if img.bufferView is not None:
        view = gltf.bufferViews[img.bufferView]                   # type: ignore[attr-defined]
        buf = buffers[view.buffer]
        start = view.byteOffset or 0
        raw = bytes(buf[start:start + view.byteLength])
        return _cache_texture_bytes(raw, path, int(src), img.mimeType or "image/png")
    return None


def _base_color_texture_path(gltf: object, buffers: list[bytes], path: Path, pbr: Any) -> str | None:
    """Back-compat wrapper — resolves ``pbrMetallicRoughness.baseColorTexture``."""
    info = getattr(pbr, "baseColorTexture", None)
    return _texture_path(gltf, buffers, path, info, slot="baseColor")


# ------------------------------------------------------- KHR_texture_transform
@dataclass(slots=True)
class UvTransform:
    """``KHR_texture_transform`` affine: uv' = T(offset) · R(rotation) · S(scale) · uv."""
    offset: tuple[float, float] = (0.0, 0.0)
    scale: tuple[float, float] = (1.0, 1.0)
    rotation: float = 0.0

    def is_identity(self) -> bool:
        return (self.offset == (0.0, 0.0) and self.scale == (1.0, 1.0)
                and self.rotation == 0.0)

    def apply(self, uv: np.ndarray) -> np.ndarray:
        su, sv = self.scale
        c, s = math.cos(self.rotation), math.sin(self.rotation)
        x = uv[:, 0] * su
        y = uv[:, 1] * sv
        out = np.empty_like(uv)
        out[:, 0] = c * x - s * y + self.offset[0]
        out[:, 1] = s * x + c * y + self.offset[1]
        return out


def _uv_transform_of(info: Any) -> UvTransform | None:
    """Extract KHR_texture_transform from a texture-info, or None."""
    if info is None:
        return None
    ext = getattr(info, "extensions", None) or {}
    tr = ext.get("KHR_texture_transform")
    if not tr:
        return None
    if int(getattr(info, "texCoord", 0) or 0) != 0:
        # Only TEXCOORD_0 exists on our Mesh; alternate UV sets are dropped.
        return None
    return UvTransform(
        offset=tuple(float(v) for v in tr.get("offset", (0.0, 0.0))),  # type: ignore[arg-type]
        scale=tuple(float(v) for v in tr.get("scale", (1.0, 1.0))),    # type: ignore[arg-type]
        rotation=float(tr.get("rotation", 0.0)),
    )


# --------------------------------------------------------------- materials
def _material_for(
    gltf: object, mat_idx: int | None, buffers: list[bytes], path: Path,
) -> tuple[PBRMaterial, float, bool, UvTransform | None]:
    """→ (PBRMaterial, baseColor alpha, double_sided, uv_transform).

    All five texture slots are resolved; ``uv_transform`` is the shared
    KHR_texture_transform to bake into the primitive's UVs (None when no
    textured slot uses it).
    """
    if mat_idx is None:
        return PBRMaterial(name="default"), 1.0, False, None
    m = gltf.materials[mat_idx]                               # type: ignore[attr-defined]
    pbr = m.pbrMetallicRoughness
    albedo: tuple[float, float, float] | None = None
    alpha = 1.0
    roughness = 0.7
    metallic = 0.0
    albedo_map: str | None = None
    mr_map: str | None = None
    if pbr is not None:
        if pbr.baseColorFactor is not None:
            albedo = tuple(float(c) for c in pbr.baseColorFactor[:3])  # type: ignore[assignment]
            alpha = float(pbr.baseColorFactor[3])
        if pbr.roughnessFactor is not None:
            roughness = float(pbr.roughnessFactor)
        if pbr.metallicFactor is not None:
            metallic = float(pbr.metallicFactor)
        albedo_map = _texture_path(gltf, buffers, path,
                                   getattr(pbr, "baseColorTexture", None), slot="baseColor")
        mr_map = _texture_path(gltf, buffers, path,
                               getattr(pbr, "metallicRoughnessTexture", None),
                               slot="metallicRoughness")
    normal_map = _texture_path(gltf, buffers, path,
                               getattr(m, "normalTexture", None), slot="normal")
    ao_map = _texture_path(gltf, buffers, path,
                           getattr(m, "occlusionTexture", None), slot="occlusion")
    emissive_map = _texture_path(gltf, buffers, path,
                                 getattr(m, "emissiveTexture", None), slot="emissive")
    if albedo is None:
        # glTF's default baseColorFactor is white — it multiplies the texture.
        # Only use the neutral gray fallback when there is no texture at all.
        albedo = (1.0, 1.0, 1.0) if albedo_map else (0.8, 0.8, 0.8)
    emissive = (0.0, 0.0, 0.0)
    if m.emissiveFactor is not None:
        emissive = tuple(float(c) for c in m.emissiveFactor[:3])       # type: ignore[assignment]
    # KHR_materials_emissive_strength scales the whole emissive term; baking
    # it into the factor covers both the scalar and the texture paths
    # (emissive = factor × emissiveTexture × strength, per spec).
    m_ext = getattr(m, "extensions", None) or {}
    strength = (m_ext.get("KHR_materials_emissive_strength") or {}).get("emissiveStrength")
    if strength is not None:
        emissive = tuple(min(1e6, c * float(strength)) for c in emissive)  # type: ignore[assignment]

    # KHR_texture_transform: usable only when every textured slot agrees on
    # a single transform (our Mesh carries one UV set).
    transforms = [
        t for t in (
            _uv_transform_of(getattr(pbr, "baseColorTexture", None)) if pbr else None,
            _uv_transform_of(getattr(pbr, "metallicRoughnessTexture", None)) if pbr else None,
            _uv_transform_of(getattr(m, "normalTexture", None)),
            _uv_transform_of(getattr(m, "occlusionTexture", None)),
            _uv_transform_of(getattr(m, "emissiveTexture", None)),
        )
        if t is not None and not t.is_identity()
    ]
    uv_transform: UvTransform | None = None
    if transforms:
        first = transforms[0]
        same = all(
            t.offset == first.offset and t.scale == first.scale
            and t.rotation == first.rotation
            for t in transforms
        )
        if same:
            uv_transform = first
        else:
            from ironengine_bonafide.logging import logger
            logger.warning(
                "gltf: conflicting KHR_texture_transform across texture slots; "
                "baking the base-color transform only"
            )
            uv_transform = first

    return (
        PBRMaterial(
            name=m.name or "default",
            albedo=albedo, roughness=roughness, metallic=metallic,
            emissive=emissive, albedo_map=albedo_map,
            metallic_roughness_map=mr_map, normal_map=normal_map,
            ao_map=ao_map, emissive_map=emissive_map,
            two_sided=bool(m.doubleSided),
            alpha=alpha,
        ),
        alpha,
        bool(m.doubleSided),
        uv_transform,
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

            material, alpha, double_sided, uv_transform = _material_for(
                gltf, prim.material, buffers, path)
            if uv_transform is not None and uvs is not None:
                uvs = uv_transform.apply(uvs.astype(np.float64)).astype(np.float32)
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
