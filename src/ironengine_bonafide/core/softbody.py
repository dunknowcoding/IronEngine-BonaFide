"""DollRig — soft / non-rigid body wrapper.

Lightweight asset record for a soft-body / cloth / character rig. Heavy
lifting (Warp XPBD particle solver, skinning) lives in
`backends/cuda/softbody.py`. This dataclass just describes what the rig
is; the backend constructs a solver state on attach.

For users without `warp-lang` installed the CPU backend's softbody pass
runs a numpy XPBD reference (debug-only, slow).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(slots=True)
class DollRig:
    particles: torch.Tensor                            # (P, 3) rest positions
    edges: torch.Tensor                                # (E, 2) int64 distance constraints
    skin_weights: torch.Tensor | None = None           # (P, J) per-joint blend weights
    skeleton_parents: list[int] | None = None          # length-J parent indices
    masses: torch.Tensor | None = None                 # (P,)
    stiffness: float = 0.8
    damping: float = 0.05
    name: str = "doll"
    _solver_state: Any = field(default=None, repr=False)

    @classmethod
    def from_glb(cls, path: str | Path, *, stiffness: float = 0.8) -> DollRig:
        """Build from a glTF/GLB rigged mesh. Vertex positions become particles;
        triangle edges become distance constraints; skin weights become joint
        bindings."""
        from ironengine_bonafide.assets.loaders.gltf import load_rig
        return load_rig(Path(path), stiffness=stiffness)

    @classmethod
    def from_arrays(
        cls,
        particles: np.ndarray | torch.Tensor,
        edges: np.ndarray | torch.Tensor,
        *,
        masses: np.ndarray | torch.Tensor | None = None,
        stiffness: float = 0.8,
        name: str = "doll",
    ) -> DollRig:
        p = torch.as_tensor(np.asarray(particles), dtype=torch.float32)
        e = torch.as_tensor(np.asarray(edges), dtype=torch.int64)
        m = torch.as_tensor(np.asarray(masses), dtype=torch.float32) if masses is not None else None
        return cls(particles=p, edges=e, masses=m, stiffness=stiffness, name=name)

    def as_softbody(self, *, stiffness: float | None = None) -> DollRig:
        if stiffness is None:
            return self
        return replace(self, stiffness=float(stiffness))

    @property
    def num_particles(self) -> int:
        return int(self.particles.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edges.shape[0])
