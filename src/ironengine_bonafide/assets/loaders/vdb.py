"""OpenVDB volume loader.

Reads a `.vdb` density grid into a `Volume` asset. Requires `pyopenvdb`
from the `[formats]` extra.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ironengine_bonafide.core.volume import Volume


def load_volume(path: Path, *, grid_name: str | None = None) -> Volume:
    try:
        import openvdb as vdb  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "pyopenvdb required for VDB. Install with: pip install -e .[formats]"
        ) from exc
    grids = vdb.readAllGridMetadata(str(path))
    if not grids:
        raise ValueError(f"No grids in VDB file: {path}")
    name = grid_name or grids[0].name
    grid = vdb.read(str(path), name)
    bbox = grid.evalActiveVoxelBoundingBox()
    lo = np.asarray(bbox[0], dtype=np.int32)
    hi = np.asarray(bbox[1], dtype=np.int32)
    shape = (hi - lo + 1).astype(np.int32)
    arr = np.zeros((shape[0], shape[1], shape[2]), dtype=np.float32)
    grid.copyToArray(arr, ijk=tuple(lo.tolist()))
    voxel = grid.transform.voxelSize()[0]
    origin = (float(lo[0]) * voxel, float(lo[1]) * voxel, float(lo[2]) * voxel)
    return Volume.from_grid(arr, origin=origin, voxel_size=float(voxel))
