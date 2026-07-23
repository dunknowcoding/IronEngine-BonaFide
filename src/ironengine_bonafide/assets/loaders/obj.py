"""OBJ mesh loader (positions + normals + uvs; triangulated faces)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ironengine_bonafide.core.mesh import Mesh


def load_mesh(path: Path) -> Mesh:
    positions: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    out_verts: list[list[float]] = []
    out_indices: list[list[int]] = []
    cache: dict[tuple[int, int, int], int] = {}

    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            tag = parts[0]
            if tag == "v":
                positions.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif tag == "vn":
                normals.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif tag == "vt":
                uvs.append((float(parts[1]), float(parts[2])))
            elif tag == "f":
                tri: list[int] = []
                for token in parts[1:]:
                    pieces = token.split("/")
                    pi = int(pieces[0]) - 1
                    ti = int(pieces[1]) - 1 if len(pieces) > 1 and pieces[1] else -1
                    ni = int(pieces[2]) - 1 if len(pieces) > 2 and pieces[2] else -1
                    key = (pi, ti, ni)
                    if key in cache:
                        idx = cache[key]
                    else:
                        p = positions[pi]
                        n = normals[ni] if ni >= 0 else (0.0, 1.0, 0.0)
                        t = uvs[ti] if ti >= 0 else (0.0, 0.0)
                        out_verts.append([p[0], p[1], p[2], n[0], n[1], n[2], t[0], t[1]])
                        idx = len(out_verts) - 1
                        cache[key] = idx
                    tri.append(idx)
                # fan triangulate
                for k in range(1, len(tri) - 1):
                    out_indices.append([tri[0], tri[k], tri[k + 1]])

    if not out_verts:
        raise ValueError(f"OBJ {path} has no vertices")
    arr = np.asarray(out_verts, dtype=np.float32)
    pos = arr[:, 0:3]
    nrm = arr[:, 3:6] if normals else None
    uv = arr[:, 6:8] if uvs else None
    idx = np.asarray(out_indices, dtype=np.int64)
    return Mesh.from_arrays(pos, idx, normals=nrm, uvs=uv, name=path.stem)
