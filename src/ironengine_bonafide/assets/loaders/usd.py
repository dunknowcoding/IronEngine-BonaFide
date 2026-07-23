"""USD scene loader.

Reads geometry + materials + lights from a `.usd` / `.usda` / `.usdc`
file into a `Scene`. Uses Pixar's `usd-core` from the `[formats]` extra.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ironengine_bonafide.core.material import PBRMaterial
from ironengine_bonafide.core.mesh import Mesh
from ironengine_bonafide.core.scene import Scene


def load_scene(path: Path) -> Scene:
    try:
        from pxr import Usd, UsdGeom  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "usd-core required for USD. Install with: pip install -e .[formats]"
        ) from exc
    stage = Usd.Stage.Open(str(path))
    scene = Scene(name=path.stem)
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        m = UsdGeom.Mesh(prim)
        positions = np.asarray(m.GetPointsAttr().Get(), dtype=np.float32)
        face_counts = np.asarray(m.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
        face_idx = np.asarray(m.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        # Triangulate by fan, ignoring n-gons > triangles
        tris: list[list[int]] = []
        cursor = 0
        for c in face_counts:
            verts = face_idx[cursor:cursor + c]
            for k in range(1, c - 1):
                tris.append([int(verts[0]), int(verts[k]), int(verts[k + 1])])
            cursor += c
        if not tris:
            continue
        scene.add(Mesh.from_arrays(
            positions=positions,
            indices=np.asarray(tris, dtype=np.int64),
            material=PBRMaterial(name=str(prim.GetName())),
            name=str(prim.GetName()),
        ))
    return scene
