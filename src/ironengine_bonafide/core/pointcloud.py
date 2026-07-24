"""PointCloud asset.

Carries N x 3 positions and (optionally) per-point colors / normals /
opacities. Builder methods chain so users can write:

    PointCloud.from_ply("scan.ply").with_lod().with_completion()

Each `with_*` returns a new PointCloud (or a wrapped version) carrying
config for the corresponding pass. The actual LOD octree / completion MLP
is built lazily by the relevant pass on first render.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(slots=True)
class PointCloud:
    positions: torch.Tensor                            # (N, 3) float32
    colors: torch.Tensor | None = None                  # (N, 3) float32 in [0, 1]
    normals: torch.Tensor | None = None                 # (N, 3)
    opacities: torch.Tensor | None = None               # (N,) — default 1.0
    name: str = "pointcloud"
    point_size_px: float = 2.0
    # When True the splat pass ignores ``point_size_px`` and infers a
    # world-space disk radius from the cloud's own density (mean inter-
    # point spacing), so kilometer- and millimeter-scale clouds both
    # render with sensible splat sizes. ``point_size_px`` stays the
    # manual override whenever this is False (the default).
    auto_point_size: bool = False
    # Per-pass enablement flags ----------------------------------------
    use_lod: bool = False
    use_completion: bool = False
    use_surfels: bool = False
    use_gsplat: bool = False
    # Cached per-pass artifacts (filled lazily) ------------------------
    _octree: Any = field(default=None, repr=False)
    _completion_prior: Any = field(default=None, repr=False)
    _surfel_radii: torch.Tensor | None = field(default=None, repr=False)
    _gsplat_params: dict[str, Any] | None = field(default=None, repr=False)

    # ------------------------------------------------------------ ctors
    @classmethod
    def from_arrays(
        cls,
        positions: np.ndarray | torch.Tensor,
        colors: np.ndarray | torch.Tensor | None = None,
        normals: np.ndarray | torch.Tensor | None = None,
        name: str = "pointcloud",
    ) -> PointCloud:
        pos = _to_tensor(positions, dim=2, last=3)
        col = _to_tensor(colors, dim=2, last=3) if colors is not None else None
        nrm = _to_tensor(normals, dim=2, last=3) if normals is not None else None
        return cls(positions=pos, colors=col, normals=nrm, name=name)

    @classmethod
    def from_ply(cls, path: str | Path) -> PointCloud:
        from ironengine_bonafide.assets.loaders.ply import load_pointcloud
        return load_pointcloud(Path(path))

    @classmethod
    def from_pcd(cls, path: str | Path) -> PointCloud:
        from ironengine_bonafide.assets.loaders.pcd import load_pointcloud
        return load_pointcloud(Path(path))

    @classmethod
    def from_generation_result(cls, result: Any) -> PointCloud:
        """Build from an IronEngine-3DCreator `GenerationResult`."""
        positions = np.asarray(result.positions, dtype=np.float32)
        colors = getattr(result, "colors", None)
        if colors is not None:
            colors = np.asarray(colors, dtype=np.float32)
            if colors.max() > 1.5:
                colors = colors / 255.0
        return cls.from_arrays(positions, colors, name=getattr(result, "label_names", "creator3d"))

    # --------------------------------------------------------- builders
    def with_lod(self, **kw: Any) -> PointCloud:
        return replace(self, use_lod=True, **{k: v for k, v in kw.items() if hasattr(self, k)})

    def with_completion(self, **kw: Any) -> PointCloud:
        return replace(self, use_completion=True, **{k: v for k, v in kw.items() if hasattr(self, k)})

    def with_surfels(self, **kw: Any) -> PointCloud:
        return replace(self, use_surfels=True, **{k: v for k, v in kw.items() if hasattr(self, k)})

    def with_gsplat(self, **kw: Any) -> PointCloud:
        return replace(self, use_gsplat=True, **{k: v for k, v in kw.items() if hasattr(self, k)})

    def with_auto_point_size(self, **kw: Any) -> PointCloud:
        """Enable density-inferred splat sizing (see ``auto_point_size``)."""
        return replace(self, auto_point_size=True, **{k: v for k, v in kw.items() if hasattr(self, k)})

    # --------------------------------------------------------- utilities
    @property
    def num_points(self) -> int:
        return int(self.positions.shape[0])

    def aabb(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.num_points == 0:
            zero = torch.zeros(3)
            return zero, zero
        return self.positions.min(dim=0).values, self.positions.max(dim=0).values

    def to(self, device: str | torch.device) -> PointCloud:
        return replace(
            self,
            positions=self.positions.to(device),
            colors=self.colors.to(device) if self.colors is not None else None,
            normals=self.normals.to(device) if self.normals is not None else None,
            opacities=self.opacities.to(device) if self.opacities is not None else None,
        )


# --------------------------------------------------------------- helpers
def _to_tensor(x: Any, dim: int, last: int) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x.to(torch.float32)
    else:
        t = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    if t.ndim != dim or t.shape[-1] != last:
        raise ValueError(f"Expected shape (N, {last}), got {tuple(t.shape)}")
    return t
