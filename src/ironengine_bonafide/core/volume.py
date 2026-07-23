"""Volume asset — fog, cloud, smoke densities.

Two construction paths:
  * `Volume.fog(density, color)` — uniform exponential fog
  * `Volume.from_vdb(path)` — OpenVDB grid (requires [formats] extra)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

Vec3 = tuple[float, float, float]


@dataclass(slots=True)
class Volume:
    kind: str                                              # "fog" | "vdb"
    density: float = 0.02                                  # uniform fog density (1/m)
    color: Vec3 = (0.7, 0.78, 0.86)
    height_falloff: float = 0.0                            # exp(-h * falloff)
    grid: torch.Tensor | None = None                       # (D, H, W) for VDB
    grid_origin: Vec3 = (0.0, 0.0, 0.0)
    grid_voxel_size: float = 0.1
    name: str = "volume"

    # ------------------------------------------------------------ ctors
    @classmethod
    def fog(
        cls,
        density: float = 0.02,
        color: Vec3 = (0.7, 0.78, 0.86),
        height_falloff: float = 0.0,
    ) -> Volume:
        return cls(kind="fog", density=density, color=color, height_falloff=height_falloff)

    @classmethod
    def from_vdb(cls, path: str | Path) -> Volume:
        from ironengine_bonafide.assets.loaders.vdb import load_volume
        return load_volume(Path(path))

    @classmethod
    def from_grid(
        cls,
        grid: np.ndarray | torch.Tensor,
        *,
        origin: Vec3 = (0.0, 0.0, 0.0),
        voxel_size: float = 0.1,
        color: Vec3 = (1.0, 1.0, 1.0),
    ) -> Volume:
        if isinstance(grid, np.ndarray):
            t = torch.from_numpy(np.ascontiguousarray(grid).astype(np.float32))
        else:
            t = grid.to(torch.float32)
        if t.ndim != 3:
            raise ValueError(f"Expected (D, H, W) volume, got {tuple(t.shape)}")
        return cls(
            kind="grid", density=1.0, color=color, grid=t,
            grid_origin=origin, grid_voxel_size=voxel_size,
        )
