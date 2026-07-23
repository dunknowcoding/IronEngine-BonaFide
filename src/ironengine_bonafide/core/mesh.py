"""Mesh asset.

Stores indexed triangle geometry plus per-vertex normals / uvs / colors.
The layout is the one `nvdiffrast` consumes: float32 positions in (V, 3),
int64 indices in (T, 3).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ironengine_bonafide.core.material import PBRMaterial


@dataclass(slots=True)
class Mesh:
    positions: torch.Tensor                            # (V, 3) float32
    indices: torch.Tensor                              # (T, 3) int64
    normals: torch.Tensor | None = None                 # (V, 3)
    uvs: torch.Tensor | None = None                     # (V, 2)
    colors: torch.Tensor | None = None                  # (V, 3) per-vertex color
    material: PBRMaterial = field(default_factory=PBRMaterial)
    name: str = "mesh"

    # ------------------------------------------------------------ ctors
    @classmethod
    def from_arrays(
        cls,
        positions: np.ndarray | torch.Tensor,
        indices: np.ndarray | torch.Tensor,
        *,
        normals: np.ndarray | torch.Tensor | None = None,
        uvs: np.ndarray | torch.Tensor | None = None,
        colors: np.ndarray | torch.Tensor | None = None,
        material: PBRMaterial | None = None,
        name: str = "mesh",
    ) -> Mesh:
        pos = _vec(positions, last=3, dtype=torch.float32)
        idx = _vec(indices, last=3, dtype=torch.int64)
        return cls(
            positions=pos,
            indices=idx,
            normals=_vec(normals, last=3, dtype=torch.float32) if normals is not None else None,
            uvs=_vec(uvs, last=2, dtype=torch.float32) if uvs is not None else None,
            colors=_vec(colors, last=3, dtype=torch.float32) if colors is not None else None,
            material=material or PBRMaterial(),
            name=name,
        )

    @classmethod
    def from_obj(cls, path: str | Path) -> Mesh:
        from ironengine_bonafide.assets.loaders.obj import load_mesh
        return load_mesh(Path(path))

    @classmethod
    def from_glb(cls, path: str | Path) -> Mesh:
        from ironengine_bonafide.assets.loaders.gltf import load_mesh
        return load_mesh(Path(path))

    @classmethod
    def from_reconstructed(cls, recon: Any) -> Mesh:
        """Build from an IronEngine-3DCreator `ReconstructedMesh`.

        3DCreator stores ``indices`` as a flat ``(T*3,) uint32`` array
        (generation/reconstruct.py) — reshape to ``(T, 3)`` here.
        """
        indices = np.asarray(recon.indices, dtype=np.int64)
        if indices.ndim == 1:
            if indices.size % 3 != 0:
                raise ValueError(
                    f"flat indices length {indices.size} is not a multiple of 3"
                )
            indices = indices.reshape(-1, 3)
        return cls.from_arrays(
            positions=np.asarray(recon.positions, dtype=np.float32),
            indices=indices,
            normals=getattr(recon, "normals", None),
            name=getattr(recon, "source", "creator3d"),
        )

    # --------------------------------------------------------- builders
    def with_material(self, mat: PBRMaterial) -> Mesh:
        return replace(self, material=mat)

    def with_colors(self, colors: np.ndarray | torch.Tensor) -> Mesh:
        return replace(self, colors=_vec(colors, last=3, dtype=torch.float32))

    # --------------------------------------------------------- utilities
    @property
    def num_vertices(self) -> int:
        return int(self.positions.shape[0])

    @property
    def num_triangles(self) -> int:
        return int(self.indices.shape[0])

    def to(self, device: str | torch.device) -> Mesh:
        return replace(
            self,
            positions=self.positions.to(device),
            indices=self.indices.to(device),
            normals=self.normals.to(device) if self.normals is not None else None,
            uvs=self.uvs.to(device) if self.uvs is not None else None,
            colors=self.colors.to(device) if self.colors is not None else None,
        )


def _vec(x: Any, last: int, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x.to(dtype)
    else:
        t = torch.as_tensor(np.asarray(x), dtype=dtype)
    if t.ndim != 2 or t.shape[-1] != last:
        raise ValueError(f"Expected shape (N, {last}), got {tuple(t.shape)}")
    return t
