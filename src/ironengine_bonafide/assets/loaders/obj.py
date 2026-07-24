"""OBJ mesh loader (positions + normals + uvs; triangulated faces).

``mtllib`` / ``usemtl`` are honored: the referenced ``.mtl`` libraries are
parsed (``newmtl`` / ``Kd`` / ``Ks`` / ``Ns`` / ``d`` / ``Tr`` / ``Ke`` /
``map_Kd``) and converted to :class:`PBRMaterial` records:

* ``Kd``      → ``albedo``
* ``Ns``      → ``roughness`` via the Blinn→GGX approximation
  ``sqrt(2 / (Ns + 2))`` (``Ns = 0`` → fully rough)
* ``Ke``      → ``emissive``
* ``map_Kd``  → ``albedo_map`` (path resolved against the .mtl directory;
  option flags like ``-s``/``-o`` are skipped, the last token wins)
* ``Ks`` / ``d`` / ``Tr`` are parsed but have no PBRMaterial slot; they are
  kept on the internal record only.

The :class:`Mesh` record carries a single material, so:

* exactly one referenced material → ``Mesh.material`` (colors untouched);
* several referenced materials  → the majority material (by face count)
  becomes ``Mesh.material`` and every face's ``Kd`` is additionally baked
  into per-vertex ``Mesh.colors`` so all parts keep their tint.

When no ``mtllib`` is present (or the .mtl file is missing/unparseable) the
behavior is exactly as before: default material, no baked colors.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ironengine_bonafide.core.material import PBRMaterial
from ironengine_bonafide.core.mesh import Mesh

_log = logging.getLogger(__name__)


@dataclass(slots=True)
class _MtlRecord:
    """Parsed .mtl material (superset of the PBRMaterial mapping)."""
    name: str
    kd: tuple[float, float, float] = (0.8, 0.8, 0.8)
    ks: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ns: float = 0.0
    d: float = 1.0
    ke: tuple[float, float, float] = (0.0, 0.0, 0.0)
    map_kd: str | None = None

    def to_pbr(self) -> PBRMaterial:
        roughness = float(np.clip(np.sqrt(2.0 / (self.ns + 2.0)), 0.0, 1.0))
        return PBRMaterial(
            name=self.name,
            albedo=self.kd,
            roughness=roughness,
            metallic=0.0,
            emissive=self.ke,
            albedo_map=self.map_kd,
        )


def _vec3(parts: list[str], default: tuple[float, float, float]) -> tuple[float, float, float]:
    try:
        return (float(parts[1]), float(parts[2]), float(parts[3]))
    except (IndexError, ValueError):
        return default


def _scalar(parts: list[str], default: float) -> float:
    try:
        return float(parts[1])
    except (IndexError, ValueError):
        return default


def load_mtl(path: Path) -> dict[str, _MtlRecord]:
    """Parse a Wavefront .mtl library into ``{name: _MtlRecord}``."""
    records: dict[str, _MtlRecord] = {}
    current: _MtlRecord | None = None
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            tag = parts[0].lower()
            if tag == "newmtl":
                name = " ".join(parts[1:]).strip() or f"mtl_{len(records)}"
                current = _MtlRecord(name=name)
                records[name] = current
            elif current is None:
                continue
            elif tag == "kd":
                current.kd = _vec3(parts, current.kd)
            elif tag == "ks":
                current.ks = _vec3(parts, current.ks)
            elif tag == "ke":
                current.ke = _vec3(parts, current.ke)
            elif tag == "ns":
                current.ns = max(_scalar(parts, current.ns), 0.0)
            elif tag == "d":
                current.d = _scalar(parts, current.d)
            elif tag == "tr":
                current.d = 1.0 - _scalar(parts, 1.0 - current.d)
            elif tag == "map_kd":
                # Texture options (-s/-o/-mm/-bm ...) precede the file name;
                # the last whitespace-separated token is the path.
                if len(parts) > 1:
                    ref = parts[-1]
                    tex = (path.parent / ref).resolve()
                    current.map_kd = str(tex)
    return records


def load_mesh(path: Path) -> Mesh:
    positions: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    out_verts: list[list[float]] = []
    out_vert_mat: list[str | None] = []               # material per deduped vertex
    out_indices: list[list[int]] = []
    out_index_mat: list[str | None] = []              # material per triangle
    cache: dict[tuple[int, int, int, str | None], int] = {}

    mtl_libs: dict[str, _MtlRecord] = {}
    current_mat: str | None = None

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
            elif tag == "mtllib":
                # One or more library file names follow the keyword.
                for lib_name in parts[1:]:
                    lib_path = (path.parent / lib_name).resolve()
                    if lib_path.is_file():
                        try:
                            mtl_libs.update(load_mtl(lib_path))
                        except OSError as exc:
                            _log.warning("OBJ %s: cannot read mtllib %s (%s)",
                                         path, lib_name, exc)
                    else:
                        _log.warning("OBJ %s: mtllib %s not found", path, lib_name)
            elif tag == "usemtl":
                name = " ".join(parts[1:]).strip()
                current_mat = name if name in mtl_libs else None
            elif tag == "f":
                tri: list[int] = []
                for token in parts[1:]:
                    pieces = token.split("/")
                    pi = int(pieces[0]) - 1
                    ti = int(pieces[1]) - 1 if len(pieces) > 1 and pieces[1] else -1
                    ni = int(pieces[2]) - 1 if len(pieces) > 2 and pieces[2] else -1
                    key = (pi, ti, ni, current_mat)
                    if key in cache:
                        idx = cache[key]
                    else:
                        p = positions[pi]
                        n = normals[ni] if ni >= 0 else (0.0, 1.0, 0.0)
                        t = uvs[ti] if ti >= 0 else (0.0, 0.0)
                        out_verts.append([p[0], p[1], p[2], n[0], n[1], n[2], t[0], t[1]])
                        out_vert_mat.append(current_mat)
                        idx = len(out_verts) - 1
                        cache[key] = idx
                    tri.append(idx)
                # fan triangulate
                for k in range(1, len(tri) - 1):
                    out_indices.append([tri[0], tri[k], tri[k + 1]])
                    out_index_mat.append(current_mat)

    if not out_verts:
        raise ValueError(f"OBJ {path} has no vertices")
    arr = np.asarray(out_verts, dtype=np.float32)
    pos = arr[:, 0:3]
    nrm = arr[:, 3:6] if normals else None
    uv = arr[:, 6:8] if uvs else None
    idx = np.asarray(out_indices, dtype=np.int64)

    # ---- material attachment ---------------------------------------------
    material: PBRMaterial | None = None
    colors = None
    used = [m for m in out_index_mat if m is not None]
    if used:
        unique = sorted(set(used))
        # Majority material by referenced face count (ties: first sorted).
        counts = {m: used.count(m) for m in unique}
        majority = max(unique, key=lambda m: counts[m])
        material = mtl_libs[majority].to_pbr()
        if len(unique) > 1:
            # Bake each part's Kd into per-vertex colors so the single-
            # material Mesh record keeps every part's tint.
            cols = np.ones((len(out_verts), 3), dtype=np.float32)
            for vi, mname in enumerate(out_vert_mat):
                if mname is not None:
                    cols[vi] = mtl_libs[mname].kd
            colors = cols

    return Mesh.from_arrays(
        pos, idx, normals=nrm, uvs=uv, colors=colors,
        material=material, name=path.stem,
    )
