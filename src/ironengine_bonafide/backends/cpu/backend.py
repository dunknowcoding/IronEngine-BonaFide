"""CPU reference backend.

Functional reference path used by CI and dev machines without a GPU.
The arithmetic is the same torch ops the CUDA path uses, just on CPU,
so cross-backend bugs surface cheaply. The rasterizers themselves live
in :mod:`ironengine_bonafide.backends.torch_raster` (device-agnostic);
this class is a thin delegator that pins the device to CPU.

Capabilities advertised:
  - ``"raster"``        — vectorized barycentric triangle raster
  - ``"splat"``         — vectorized screen-space disk splat
  - ``"surfel"``        — torch PCA on neighbour patches
  - ``"shadow_csm"``    — cascaded shadow depth raster
  - ``"neural_denoise"`` / ``"neural_upscale"`` — torch on CPU

Correctness notes (0.2):
  - Triangle attributes are interpolated perspective-correctly (1/w
    weighted barycentrics); triangles crossing the near plane are split
    instead of dropped.
  - The per-pixel depth resolve is deterministic
    (``scatter_reduce(amin)`` + first-candidate tiebreak).
  - Point disks scale with eye depth (``point_size_px / z``).
"""
from __future__ import annotations

import torch

from ironengine_bonafide.backends import torch_raster
from ironengine_bonafide.backends.torch_raster import GBuffer
from ironengine_bonafide.core.backend import Backend, BackendInfo


class CpuBackend(Backend):
    _CAPS = frozenset({
        "raster", "splat", "surfel", "shadow_csm",
        "neural_denoise", "neural_upscale",
    })

    def __init__(self) -> None:
        self._info = BackendInfo(
            name="cpu",
            device="cpu",
            capabilities=self._CAPS,
            version="reference",
            notes="Functional reference path. Use [cuda] extra for production speed.",
        )

    @property
    def info(self) -> BackendInfo:
        return self._info

    # --------------------------------------------------------- alloc
    def empty(self, shape: tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.empty(shape, dtype=dtype, device="cpu")

    def zeros(self, shape: tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.zeros(shape, dtype=dtype, device="cpu")

    def to_device(self, x: torch.Tensor) -> torch.Tensor:
        return x.to("cpu")

    # ===================================================== point splatting
    def raster_points(
        self,
        positions: torch.Tensor,           # (N, 3)
        colors: torch.Tensor,              # (N, 3)
        view_proj: torch.Tensor,           # (4, 4)
        width: int,
        height: int,
        point_size_px: float = 2.0,
        background: tuple[float, float, float] = (0.05, 0.06, 0.10),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project points to screen and stamp depth-tested disks.

        Disk size scales with eye depth (``point_size_px`` at 1 m).
        Returns (rgb HxWx3 float32 linear, depth HxW float32 NDC z).
        """
        return torch_raster.raster_points(
            positions.cpu(), colors.cpu(), view_proj.cpu(),
            width, height, point_size_px, background,
        )

    # ===================================================== depth-only raster
    def raster_depth(
        self,
        positions: torch.Tensor,
        indices: torch.Tensor,
        view_proj: torch.Tensor,
        width: int,
        height: int,
    ) -> torch.Tensor:
        """Depth-only triangle raster (HxW float32 NDC z, +inf empty).
        Used by the CSM shadow pass."""
        return torch_raster.raster_depth(
            positions.cpu(), indices.cpu(), view_proj.cpu(), width, height,
        )

    # ============================================== mesh raster (GBuffer)
    def raster_mesh_gbuffer(
        self,
        positions: torch.Tensor,           # (V, 3) world
        indices: torch.Tensor,             # (T, 3) int64
        colors: torch.Tensor,              # (V, 3) per-vertex albedo
        normals: torch.Tensor | None,      # (V, 3) world
        view_proj: torch.Tensor,           # (4, 4)
        width: int,
        height: int,
        *,
        uvs: torch.Tensor | None = None,       # (V, 2)
        tangents: torch.Tensor | None = None,  # (V, 3) world
    ) -> GBuffer:
        """GBuffer raster (albedo / world-pos / normal / uv / tangent /
        depth / mask) with perspective-correct interpolation. The PBR
        pass shades the GBuffer downstream."""
        return torch_raster.raster_mesh_gbuffer(
            positions.cpu(), indices.cpu(), colors.cpu(),
            normals.cpu() if normals is not None else None,
            view_proj.cpu(), width, height,
            uvs=uvs.cpu() if uvs is not None else None,
            tangents=tangents.cpu() if tangents is not None else None,
        )

    # ===================================================== mesh raster (shaded)
    def raster_mesh(
        self,
        positions: torch.Tensor,           # (V, 3)
        indices: torch.Tensor,             # (T, 3) int64
        colors: torch.Tensor,              # (V, 3) per-vertex
        normals: torch.Tensor | None,      # (V, 3)
        view_proj: torch.Tensor,           # (4, 4)
        width: int,
        height: int,
        light_dir: tuple[float, float, float] = (0.4, 0.8, 0.6),
        background: tuple[float, float, float] = (0.05, 0.06, 0.10),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorized barycentric raster + Lambert shading."""
        return torch_raster.raster_mesh(
            positions.cpu(), indices.cpu(), colors.cpu(),
            normals.cpu() if normals is not None else None,
            view_proj.cpu(), width, height, light_dir, background,
        )
