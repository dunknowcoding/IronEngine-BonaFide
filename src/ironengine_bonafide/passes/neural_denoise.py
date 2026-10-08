"""Edge-aware denoiser — runs after geometry/AA, before tonemap.

Two paths, selected automatically:

* **Guided à-trous wavelet filter** (default, no weights needed) — an
  SVGF-style spatial filter: three à-trous iterations (step sizes 1, 2, 4)
  of a 3x3 kernel whose tap weights fall off with colour distance, normal
  disagreement, and depth discontinuity against the centre pixel. It
  removes Monte-Carlo / splat noise while preserving geometric edges,
  using only the GBuffers the engine already produces.
* **Learned U-Net** (optional) — when ``BONAFIDE_DENOISE_WEIGHTS`` points
  at a trained checkpoint, the micro U-Net below is used instead. Without
  weights the net is random and would degrade the image, so the à-trous
  filter is the fallback, not a silent no-op.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ironengine_bonafide.passes.base import PassContext, RenderPass

# À-trous schedule: dilation steps and per-step blend strength.
_ATROUS_STEPS = (1, 2, 4)
_SIGMA_COLOR = 0.6          # colour-distance falloff (linear HDR units)
_SIGMA_NORMAL = 0.5         # normal disagreement falloff (dot-product dist)
_SIGMA_DEPTH = 0.05         # relative depth falloff


def _weights_path() -> str | None:
    p = os.environ.get("BONAFIDE_DENOISE_WEIGHTS")
    return p if p and os.path.exists(p) else None


class NeuralDenoisePass(RenderPass):
    name = "neural_denoise"

    def __init__(self) -> None:
        self._net: nn.Module | None = None

    def is_active(self, ctx: PassContext) -> bool:
        return bool(ctx.config.neural_denoise)

    def _ensure_net(self, device: str | torch.device) -> nn.Module:
        if self._net is None:
            self._net = _MicroUNet().to(device)
            weights = _weights_path()
            if weights:
                self._net.load_state_dict(torch.load(weights, map_location=device))
            self._net.eval()
        return self._net

    @torch.no_grad()
    def run(self, ctx: PassContext) -> None:
        if _weights_path() is not None:
            net = self._ensure_net(ctx.backend.device)
            x = ctx.targets.rgb.permute(2, 0, 1).unsqueeze(0)      # (1, 3, H, W)
            h, w = x.shape[-2:]
            ph = (8 - h % 8) % 8
            pw = (8 - w % 8) % 8
            x_pad = nn.functional.pad(x, (0, pw, 0, ph), mode="replicate")
            y = net(x_pad)[..., :h, :w]
            ctx.targets.rgb = y.squeeze(0).permute(1, 2, 0).contiguous()
            return
        ctx.targets.rgb = atrous_denoise(
            ctx.targets.rgb, ctx.targets.normals, ctx.targets.depth,
        )


# ------------------------------------------------------------- à-trous
@torch.no_grad()
def atrous_denoise(rgb: torch.Tensor, normals: torch.Tensor,
                   depth: torch.Tensor) -> torch.Tensor:
    """Edge-aware à-trous wavelet denoising of an (H, W, 3) linear frame.

    ``normals`` (H, W, 3, world-space) and ``depth`` (H, W, NDC z, +inf on
    empty pixels) guide the filter so averaging stops at geometric and
    albedo boundaries. Deterministic on every backend.
    """
    h, w, c = rgb.shape
    device, dtype = rgb.device, rgb.dtype
    img = rgb.permute(2, 0, 1).unsqueeze(0)                   # (1, C, H, W)
    nrm = normals / torch.linalg.norm(normals, dim=-1, keepdim=True).clamp(min=1e-9)
    finite = torch.isfinite(depth)
    dfill = float(depth[finite].max()) if bool(finite.any()) else 1.0
    dep = torch.where(finite, depth, torch.full_like(depth, dfill))
    # 3x3 gaussian-ish base kernel.
    k = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
                     device=device, dtype=dtype) / 16.0

    out = img
    for step in _ATROUS_STEPS:
        padded = F.pad(out, (step, step, step, step), mode="replicate")
        nrm_p = F.pad(nrm.permute(2, 0, 1).unsqueeze(0),
                      (step, step, step, step), mode="replicate")
        dep_p = F.pad(dep.unsqueeze(0).unsqueeze(0),
                      (step, step, step, step), mode="replicate")
        acc = torch.zeros_like(out)
        wsum = torch.zeros((1, 1, h, w), device=device, dtype=dtype)
        center = out
        center_n = nrm.permute(2, 0, 1).unsqueeze(0)
        center_d = dep.unsqueeze(0).unsqueeze(0)
        for ky in range(3):
            for kx in range(3):
                ys, xs = ky * step, kx * step
                tap = padded[:, :, ys:ys + h, xs:xs + w]
                tap_n = nrm_p[:, :, ys:ys + h, xs:xs + w]
                tap_d = dep_p[:, :, ys:ys + h, xs:xs + w]
                # Colour distance (L2 across channels, HDR-scaled).
                dc = (tap - center).pow(2).sum(dim=1, keepdim=True).sqrt()
                w_c = torch.exp(-dc / _SIGMA_COLOR)
                # Normal distance: 1 - dot.
                dn = (1.0 - (tap_n * center_n).sum(dim=1, keepdim=True)).clamp(min=0.0)
                w_n = torch.exp(-dn / _SIGMA_NORMAL)
                # Relative depth distance.
                dd = ((tap_d - center_d).abs()
                      / center_d.abs().clamp(min=1e-3))
                w_d = torch.exp(-dd / _SIGMA_DEPTH)
                wgt = k[ky, kx] * w_c * w_n * w_d
                acc += tap * wgt
                wsum += wgt
        out = acc / wsum.clamp(min=1e-9)
    return out.squeeze(0).permute(1, 2, 0).contiguous()


class _MicroUNet(nn.Module):
    """Tiny U-Net (used only with a trained checkpoint)."""
    def __init__(self) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1, stride=2), nn.ReLU(inplace=True),
        )
        self.bottom = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 3, padding=1),
        )
        # Initialize to near-identity so an untrained net doesn't destroy frames
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_uniform_(m.weight, a=2.236)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        d = self.down(x)
        b = self.bottom(d)
        u = self.up(b)
        return torch.clamp(residual + 0.1 * u, 0.0, None)
