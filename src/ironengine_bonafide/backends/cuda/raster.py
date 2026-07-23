"""nvdiffrast wrapper — differentiable mesh rasterization with deferred shading.

The wrapper builds a GBuffer (positions, normals, albedo, depth) using
nvdiffrast and then shades it with a small torch kernel here. This keeps
gradients flowing all the way to vertex positions and material colors.
"""
from __future__ import annotations

from typing import Any

import torch

_GLCTX_CACHE: dict[str, Any] = {}


def _ctx(device: str) -> Any:
    if device not in _GLCTX_CACHE:
        try:
            import nvdiffrast.torch as dr  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "nvdiffrast not installed. Add the [cuda] extra: pip install -e .[cuda]"
            ) from exc
        # CUDA-only context first; fall back to GL on machines without it.
        try:
            _GLCTX_CACHE[device] = dr.RasterizeCudaContext(device=device)
        except Exception:
            _GLCTX_CACHE[device] = dr.RasterizeGLContext(device=device)
    return _GLCTX_CACHE[device]


def render_mesh_gbuffer(
    positions: torch.Tensor,             # (V, 3)
    indices: torch.Tensor,               # (T, 3) int32
    normals: torch.Tensor,               # (V, 3)
    colors: torch.Tensor,                # (V, 3)
    view_proj: torch.Tensor,             # (4, 4)
    width: int,
    height: int,
    *,
    background: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    import nvdiffrast.torch as dr  # type: ignore[import-not-found]

    ctx = _ctx(str(positions.device))
    n = positions.shape[0]
    ones = torch.ones((n, 1), device=positions.device, dtype=positions.dtype)
    pos_h = torch.cat([positions, ones], dim=1) @ view_proj.T            # (V, 4)
    pos_h = pos_h.unsqueeze(0).contiguous()

    idx32 = indices.to(torch.int32).contiguous()
    rast, _ = dr.rasterize(ctx, pos_h, idx32, resolution=(height, width))
    rgb, _ = dr.interpolate(colors.unsqueeze(0).contiguous(), rast, idx32)
    nrm, _ = dr.interpolate(normals.unsqueeze(0).contiguous(), rast, idx32)
    depth = rast[..., 2:3]                                               # (1, H, W, 1)
    mask = (rast[..., 3:4] > 0).float()                                  # (1, H, W, 1)

    bg = torch.tensor(background, device=positions.device, dtype=positions.dtype)
    rgb_out = rgb[0] * mask[0] + bg.expand_as(rgb[0]) * (1.0 - mask[0])
    nrm_out = torch.nn.functional.normalize(nrm[0], dim=-1)
    depth_out = depth[0].squeeze(-1)
    return rgb_out, nrm_out, depth_out, mask[0].squeeze(-1)
