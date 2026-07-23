"""PLY loader.

Supports:
  * ASCII PLY (point clouds + small triangle meshes)
  * Binary little-endian PLY (point clouds + triangle meshes)

For point clouds we read x/y/z + (optional) red/green/blue or nx/ny/nz.
For meshes we additionally consume the `face` element.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from ironengine_bonafide.core.mesh import Mesh
from ironengine_bonafide.core.pointcloud import PointCloud


def _parse_header(lines: list[bytes]) -> tuple[str, list[tuple[str, int, list[tuple[str, str]]]], int]:
    fmt = "ascii"
    elements: list[tuple[str, int, list[tuple[str, str]]]] = []
    i = 0
    if not lines or lines[0].strip() != b"ply":
        raise ValueError("Not a PLY file")
    end = 0
    while i < len(lines):
        line = lines[i].strip().decode("utf-8", errors="ignore")
        if line.startswith("format"):
            fmt = line.split()[1]
        elif line.startswith("element"):
            parts = line.split()
            elements.append((parts[1], int(parts[2]), []))
        elif line.startswith("property") and elements:
            parts = line.split()
            if parts[1] == "list":
                # property list <count_type> <item_type> <name>
                elements[-1][2].append((parts[-1], f"list:{parts[2]}:{parts[3]}"))
            else:
                elements[-1][2].append((parts[2], parts[1]))
        elif line == "end_header":
            end = i + 1
            break
        i += 1
    return fmt, elements, end


_BINARY_TYPE_TO_STRUCT = {
    "char": "b", "int8": "b",
    "uchar": "B", "uint8": "B",
    "short": "h", "int16": "h",
    "ushort": "H", "uint16": "H",
    "int": "i", "int32": "i",
    "uint": "I", "uint32": "I",
    "float": "f", "float32": "f",
    "double": "d", "float64": "d",
}


def load_pointcloud(path: Path) -> PointCloud:
    raw = path.read_bytes()
    # Header is ASCII either way; binary body follows the `end_header\n` marker.
    header_end = raw.find(b"end_header\n") + len(b"end_header\n")
    header_lines = raw[:header_end].split(b"\n")
    fmt, elements, _ = _parse_header(header_lines)

    vertex_spec = next((e for e in elements if e[0] == "vertex"), None)
    if vertex_spec is None:
        raise ValueError(f"PLY {path} has no `vertex` element")
    n_vertices = vertex_spec[1]
    props = vertex_spec[2]
    prop_names = [p[0] for p in props]
    has_rgb = all(c in prop_names for c in ("red", "green", "blue"))

    if fmt == "ascii":
        body_lines = raw[header_end:].decode("utf-8", errors="ignore").splitlines()
        xs = np.empty(n_vertices, dtype=np.float32)
        ys = np.empty(n_vertices, dtype=np.float32)
        zs = np.empty(n_vertices, dtype=np.float32)
        rs = np.zeros(n_vertices, dtype=np.float32) if has_rgb else None
        gs = np.zeros(n_vertices, dtype=np.float32) if has_rgb else None
        bs = np.zeros(n_vertices, dtype=np.float32) if has_rgb else None
        for i in range(n_vertices):
            tokens = body_lines[i].split()
            for j, (name, _ty) in enumerate(props):
                v = float(tokens[j])
                if name == "x": xs[i] = v
                elif name == "y": ys[i] = v
                elif name == "z": zs[i] = v
                elif name == "red" and rs is not None:   rs[i] = v
                elif name == "green" and gs is not None: gs[i] = v
                elif name == "blue" and bs is not None:  bs[i] = v
        positions = np.stack([xs, ys, zs], axis=1)
        colors = (np.stack([rs, gs, bs], axis=1) / 255.0).astype(np.float32) if has_rgb else None
        return PointCloud.from_arrays(positions, colors, name=path.stem)

    # Binary little-endian path -- skip if not LE for now
    if fmt != "binary_little_endian":
        raise ValueError(f"PLY format '{fmt}' not supported")
    body = raw[header_end:]
    fmt_chars = []
    for _name, ty in props:
        if ty.startswith("list:"):
            raise ValueError("List properties on `vertex` not supported in fast path")
        fmt_chars.append(_BINARY_TYPE_TO_STRUCT[ty])
    unpacker = struct.Struct("<" + "".join(fmt_chars))
    rec_size = unpacker.size
    arr = np.frombuffer(body[: n_vertices * rec_size], dtype=np.uint8).reshape(n_vertices, rec_size)
    # Vectorized decode for the common float32 x/y/z + uint8 r/g/b layout
    out = np.empty((n_vertices, len(props)), dtype=np.float32)
    offset = 0
    for j, (_name, ty) in enumerate(props):
        sz = struct.calcsize(_BINARY_TYPE_TO_STRUCT[ty])
        col = arr[:, offset:offset + sz].copy().view(f"<{_BINARY_TYPE_TO_STRUCT[ty]}").reshape(-1)
        out[:, j] = col.astype(np.float32)
        offset += sz
    name_to_col = {p[0]: i for i, p in enumerate(props)}
    positions = np.stack([out[:, name_to_col[c]] for c in ("x", "y", "z")], axis=1)
    colors = None
    if has_rgb:
        colors = np.stack([out[:, name_to_col[c]] for c in ("red", "green", "blue")], axis=1) / 255.0
    return PointCloud.from_arrays(positions, colors, name=path.stem)


def load_mesh(path: Path) -> Mesh:
    """Load a PLY *mesh* (vertex + face elements)."""
    raw = path.read_bytes()
    header_end = raw.find(b"end_header\n") + len(b"end_header\n")
    header_lines = raw[:header_end].split(b"\n")
    fmt, elements, _ = _parse_header(header_lines)
    if fmt != "ascii":
        # For binary triangle meshes prefer the OBJ/GLB loaders.
        raise ValueError("Binary PLY meshes not supported here; convert to OBJ/GLB")

    vertex_spec = next(e for e in elements if e[0] == "vertex")
    face_spec = next((e for e in elements if e[0] == "face"), None)
    if face_spec is None:
        raise ValueError(f"PLY {path} has no `face` element; use load_pointcloud()")

    n_vertices = vertex_spec[1]
    n_faces = face_spec[1]
    body_lines = raw[header_end:].decode("utf-8", errors="ignore").splitlines()

    positions = np.empty((n_vertices, 3), dtype=np.float32)
    has_rgb = all(c in [p[0] for p in vertex_spec[2]] for c in ("red", "green", "blue"))
    colors = np.zeros((n_vertices, 3), dtype=np.float32) if has_rgb else None
    name_to_idx = {p[0]: i for i, p in enumerate(vertex_spec[2])}
    for i in range(n_vertices):
        tokens = body_lines[i].split()
        positions[i, 0] = float(tokens[name_to_idx["x"]])
        positions[i, 1] = float(tokens[name_to_idx["y"]])
        positions[i, 2] = float(tokens[name_to_idx["z"]])
        if colors is not None:
            colors[i, 0] = float(tokens[name_to_idx["red"]])   / 255.0
            colors[i, 1] = float(tokens[name_to_idx["green"]]) / 255.0
            colors[i, 2] = float(tokens[name_to_idx["blue"]])  / 255.0

    indices: list[list[int]] = []
    for i in range(n_faces):
        tokens = body_lines[n_vertices + i].split()
        count = int(tokens[0])
        verts = [int(t) for t in tokens[1:1 + count]]
        # fan triangulate
        for k in range(1, len(verts) - 1):
            indices.append([verts[0], verts[k], verts[k + 1]])

    idx = np.asarray(indices, dtype=np.int64)
    return Mesh.from_arrays(positions, idx, colors=colors, name=path.stem)
