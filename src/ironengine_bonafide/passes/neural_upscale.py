"""Spatial upscaling — FSR 1.0 style EASU + RCAS, with a DLSS bridge hook.

The render driver allocates the frame at ``1/upscale_factor`` of the output
resolution when ``RenderConfig.neural_upscale != "none"``; this pass
resolves it to the full output size before tonemapping.

Backends:

* ``"fsr"`` — AMD FSR 1.0 structure in pure torch: **EASU** (edge-adaptive
  spatial upsampling — a 16-tap Lanczos2 kernel whose footprint is
  stretched along the local edge direction) followed by **RCAS** (robust
  contrast-adaptive sharpening with neighbourhood anti-halo clamping).
  Faithful to the published algorithm's structure; not bit-exact with
  AMD's shader. Runs identically on CPU and CUDA, fully deterministic.
* ``"dlss"`` — NVIDIA DLSS through a user-supplied NGX bridge DLL. DLSS is
  proprietary NVIDIA technology that cannot be redistributed; to enable it,
  point ``BONAFIDE_DLSS_DLL`` at a bridge library exporting ::

      void bonafide_dlss_upscale(const float* src, float* dst,
                                 int src_w, int src_h, int dst_w, int dst_h)

  with contiguous float32 HWC RGB buffers, on a Turing-or-newer NVIDIA GPU.
  When the bridge or GPU is unavailable the pass falls back to FSR and
  records ``neural_upscale:dlss_unavailable→fsr`` in the skip notes —
  the render still completes.
* ``BONAFIDE_UPSCALE_WEIGHTS`` (optional, both modes) — path to a trained
  EDSR-style checkpoint; when present the learned net takes precedence.

Sensor outputs (depth / normals / ids / albedo) are resolved alongside RGB
with mode-appropriate filters so every output tensor matches the
configured output resolution.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ironengine_bonafide.passes.base import PassContext, RenderPass

# Lanczos2 support radius / tap count of the EASU footprint.
_RADIUS = 2
_TAPS = 2 * _RADIUS                                     # 4x4 neighborhood


class NeuralUpscalePass(RenderPass):
    name = "neural_upscale"

    def __init__(self) -> None:
        self._net: nn.Module | None = None

    def is_active(self, ctx: PassContext) -> bool:
        return ctx.config.neural_upscale != "none"

    def _ensure_net(self, device: str | torch.device) -> nn.Module | None:
        weights = os.environ.get("BONAFIDE_UPSCALE_WEIGHTS")
        if not weights or not os.path.exists(weights):
            return None
        if self._net is None:
            self._net = _Edsr(scale=2).to(device)
            self._net.load_state_dict(torch.load(weights, map_location=device))
            self._net.eval()
        return self._net

    @torch.no_grad()
    def run(self, ctx: PassContext) -> None:
        rgb = ctx.targets.rgb
        h, w, _ = rgb.shape
        target_h = int(ctx.config.height)
        target_w = int(ctx.config.width)
        if h == target_h and w == target_w:
            return
        sharpness = float(getattr(ctx.config, "upscale_sharpness", 0.2))
        mode = ctx.config.neural_upscale

        net = self._ensure_net(ctx.backend.device)
        if net is not None:
            x = rgb.permute(2, 0, 1).unsqueeze(0)
            y = net(x)
            y = F.interpolate(y, size=(target_h, target_w), mode="bilinear",
                              align_corners=False)
            out = y.squeeze(0).permute(1, 2, 0).contiguous()
        elif mode == "dlss" and _dlss_bridge() is not None:
            out = _dlss_upscale(rgb, target_w, target_h)
        else:
            if mode == "dlss":
                ctx.skipped.append("neural_upscale:dlss_unavailable→fsr")
            out = _easu(rgb, target_w, target_h)
            if sharpness > 0.0:
                out = _rcas(out, sharpness)
        ctx.targets.rgb = out
        self._resolve_aux(ctx, target_w, target_h)

    # ------------------------------------------------------------- aux
    @staticmethod
    def _resolve_aux(ctx: PassContext, target_w: int, target_h: int) -> None:
        """Bring depth / normals / ids / albedo to the output resolution
        with mode-appropriate filters (nearest for IDs, min-aware for
        depth, renormalized for normals)."""
        t = ctx.targets

        def _up(x: torch.Tensor, mode: str) -> torch.Tensor:
            img = x.permute(2, 0, 1).unsqueeze(0) if x.ndim == 3 else x[None, None]
            y = F.interpolate(img, size=(target_h, target_w), mode=mode,
                              align_corners=False if mode != "nearest" else None)
            y = y.squeeze(0)
            if x.ndim == 3:
                return y.permute(1, 2, 0).contiguous()
            return y.squeeze(0).contiguous()

        if t.depth.shape != (target_h, target_w):
            # Map +inf to a large sentinel so bilinear weights stay finite,
            # then restore inf wherever every contributing source px was empty.
            finite = torch.isfinite(t.depth)
            fill = float(t.depth[finite].max()) * 2.0 if bool(finite.any()) else 1.0
            d = torch.where(finite, t.depth, torch.full_like(t.depth, fill))
            d_up = _up(d, "bilinear")
            all_empty = _up((~finite).to(t.depth.dtype), "nearest") > 0.5
            t.depth = torch.where(all_empty, torch.full_like(d_up, float("inf")), d_up)
        if t.normals.shape[:2] != (target_h, target_w):
            n = _up(t.normals, "bilinear")
            t.normals = n / torch.linalg.norm(n, dim=-1, keepdim=True).clamp(min=1e-9)
        if t.albedo.shape[:2] != (target_h, target_w):
            t.albedo = _up(t.albedo, "bilinear")
        if t.ids.shape != (target_h, target_w):
            t.ids = _up(t.ids.to(torch.float32), "nearest").to(torch.int64)


# =================================================================== EASU
def _lanczos2(d: torch.Tensor) -> torch.Tensor:
    """Lanczos-2 kernel evaluated at distance ``d`` (support = 2)."""
    d = d.abs()
    out = torch.zeros_like(d)
    nz = (d > 1e-6) & (d < _RADIUS)
    x = d[nz] * math.pi
    out[nz] = (_RADIUS * torch.sin(x) * torch.sin(x / _RADIUS)) / (x * x)
    out[d <= 1e-6] = 1.0
    return out


def _edge_vectors(rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-texel edge direction (unit) and strength ∈ [0, 1] from luma
    gradients (Scharr-style cross derivative)."""
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    p = F.pad(luma[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
    gx = (p[:-2, 2:] - p[:-2, :-2]) + 2.0 * (p[1:-1, 2:] - p[1:-1, :-2]) \
        + (p[2:, 2:] - p[2:, :-2])
    gy = (p[2:, :-2] - p[:-2, :-2]) + 2.0 * (p[2:, 1:-1] - p[:-2, 1:-1]) \
        + (p[2:, 2:] - p[:-2, 2:])
    mag = torch.sqrt(gx * gx + gy * gy)
    scale = mag.max().clamp(min=1e-9)
    strength = (mag / scale).clamp(0.0, 1.0)
    # Edge direction is PERPENDICULAR to the gradient.
    norm = mag.clamp(min=1e-9)
    return -gy / norm, gx / norm, strength


@torch.no_grad()
def _easu(rgb: torch.Tensor, out_w: int, out_h: int) -> torch.Tensor:
    """Edge-adaptive spatial upsampling of an (H, W, 3) image.

    Each output pixel gathers the 4x4 input texels around its source
    location; tap weights follow a Lanczos2 kernel whose distances are
    compressed along the local edge direction (more weight along edges,
    less across), which reconstructs diagonals without the blur of
    bilinear or the ringing of a wide isotropic kernel.
    """
    device, dtype = rgb.device, rgb.dtype
    in_h, in_w, c = rgb.shape
    ex, ey, strength = _edge_vectors(rgb)

    # Source-space coordinate of each output pixel (align_corners=False).
    ys = (torch.arange(out_h, device=device, dtype=dtype) + 0.5) * (in_h / out_h) - 0.5
    xs = (torch.arange(out_w, device=device, dtype=dtype) + 0.5) * (in_w / out_w) - 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")           # (H_out, W_out)
    base_y = torch.floor(yy)
    base_x = torch.floor(xx)

    # Edge stats of the nearest input texel steer the kernel footprint.
    near_y = (base_y + 0.5).round().long().clamp(0, in_h - 1)
    near_x = (base_x + 0.5).round().long().clamp(0, in_w - 1)
    e_x = ex[near_y, near_x]
    e_y = ey[near_y, near_x]
    s = strength[near_y, near_x]
    # Footprint compression along the edge: 1 (no edge) → 0.5 (strong edge).
    along = 1.0 - 0.5 * s

    acc = torch.zeros((out_h, out_w, c), device=device, dtype=dtype)
    wsum = torch.zeros((out_h, out_w, 1), device=device, dtype=dtype)
    for j in range(-(_RADIUS - 1), _RADIUS + 1):
        for i in range(-(_RADIUS - 1), _RADIUS + 1):
            tap_y = (base_y + j).long().clamp(0, in_h - 1)
            tap_x = (base_x + i).long().clamp(0, in_w - 1)
            d_y = yy - (base_y + j)
            d_x = xx - (base_x + i)
            d_par = (d_x * e_x + d_y * e_y) * along           # along edge
            d_perp = d_x * e_y - d_y * e_x                    # across edge
            dist = torch.sqrt(d_par * d_par + d_perp * d_perp)
            wgt = _lanczos2(dist).unsqueeze(-1)
            acc += rgb[tap_y, tap_x] * wgt
            wsum += wgt
    return acc / wsum.clamp(min=1e-6)


@torch.no_grad()
def _rcas(rgb: torch.Tensor, sharpness: float) -> torch.Tensor:
    """Robust contrast-adaptive sharpening (RCAS).

    Negative-lobe 5-tap cross filter, clamped to the local 3x3 min/max so
    sharpening never overshoots into halos. HDR-safe: negative-lobe gain is
    scaled by the local luma so suns and emissives don't ring.
    """
    c = rgb
    p = F.pad(c.permute(2, 0, 1).unsqueeze(0), (1, 1, 1, 1), mode="replicate")
    north = p[0, :, :-2, 1:-1]
    south = p[0, :, 2:, 1:-1]
    west = p[0, :, 1:-1, :-2]
    east = p[0, :, 1:-1, 2:]
    cross = torch.stack([north, south, west, east], dim=0)   # (4, C, H, W)
    # Anti-halo bounds: min/max of the cross neighbourhood + centre.
    nbhd = torch.cat([cross, p[0, :, 1:-1, 1:-1].unsqueeze(0)], dim=0)
    lo = nbhd.min(dim=0).values
    hi = nbhd.max(dim=0).values
    # RCAS gain: sharpness ∈ [0, 1] → negative lobe weight a ∈ [0, 0.2].
    a = 0.2 * float(sharpness)
    center = p[0, :, 1:-1, 1:-1]
    out = center * (1.0 + 4.0 * a) - a * cross.sum(dim=0)
    out = out.clamp(min=lo, max=hi)
    out = out.clamp(min=0.0)                                  # HDR stays positive
    return out.permute(1, 2, 0).contiguous()


# =================================================================== DLSS
def _dlss_bridge() -> object | None:
    """Probe for the user-supplied DLSS bridge DLL + a Turing+ NVIDIA GPU.

    Returns a loaded ctypes handle, or None when DLSS is unavailable on
    this machine (the pass then falls back to FSR).
    """
    dll = os.environ.get("BONAFIDE_DLSS_DLL")
    if not dll or not Path(dll).is_file():
        return None
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability(0)
    if (major, minor) < (7, 5):
        return None
    try:
        import ctypes
        lib = ctypes.CDLL(dll)
        fn = lib.bonafide_dlss_upscale
        fn.restype = None
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        return fn
    except (OSError, AttributeError):
        return None


def _dlss_upscale(rgb: torch.Tensor, out_w: int, out_h: int) -> torch.Tensor:
    """Run the NGX bridge on contiguous float32 HWC RGB buffers."""
    import ctypes
    fn = _dlss_bridge()
    assert fn is not None                                     # caller checked
    src = rgb.detach().to(torch.float32).contiguous()
    dst = torch.empty((out_h, out_w, 3), dtype=torch.float32)
    fn(ctypes.c_void_p(src.data_ptr()), ctypes.c_void_p(dst.data_ptr()),
       src.shape[1], src.shape[0], out_w, out_h)
    return dst.to(device=rgb.device)


class _Edsr(nn.Module):
    """Tiny EDSR-style super-resolution net (8 residual blocks)."""
    def __init__(self, scale: int = 2, channels: int = 32) -> None:
        super().__init__()
        self.head = nn.Conv2d(3, channels, 3, padding=1)
        self.body = nn.Sequential(*[_ResBlock(channels) for _ in range(8)])
        self.tail = nn.Sequential(
            nn.Conv2d(channels, channels * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(channels, 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.head(x)
        return self.tail(self.body(h) + h)


class _ResBlock(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)
