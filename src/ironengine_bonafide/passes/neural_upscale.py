"""Neural super-resolution (DLSS-like).

Renders at half-res internally, then bilinearly upscales + edge-sharpens
unless `BONAFIDE_UPSCALE_WEIGHTS` points at a learned model. Real
trained-model integration ships in 0.2 — this slot keeps the API stable.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ironengine_bonafide.passes.base import PassContext, RenderPass


class NeuralUpscalePass(RenderPass):
    name = "neural_upscale"

    def __init__(self) -> None:
        self._net: nn.Module | None = None

    def is_active(self, ctx: PassContext) -> bool:
        return ctx.config.neural_upscale != "none"

    def _ensure_net(self, device: str) -> nn.Module | None:
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
        target_h = ctx.config.height
        target_w = ctx.config.width
        if h == target_h and w == target_w:
            return
        net = self._ensure_net(ctx.backend.device)
        x = rgb.permute(2, 0, 1).unsqueeze(0)
        if net is not None:
            y = net(x)
            y = F.interpolate(y, size=(target_h, target_w), mode="bilinear", align_corners=False)
        else:
            # Lightweight fallback: bilinear upscale + soft sharpen
            y = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
            blurred = F.avg_pool2d(F.pad(y, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
            y = (y + 0.4 * (y - blurred)).clamp(min=0.0)
        ctx.targets.rgb = y.squeeze(0).permute(1, 2, 0).contiguous()


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
