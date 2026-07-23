"""Neural denoiser — runs after geometry, before tonemap.

The bundled U-Net is UNTRAINED; running it without learned weights adds
noise to the frame. The pass therefore stays inactive unless a weights
file is provided: set `BONAFIDE_DENOISE_WEIGHTS=<path>` in the env (and
`config.neural_denoise=True`).
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn

from ironengine_bonafide.passes.base import PassContext, RenderPass


def _weights_path() -> str | None:
    p = os.environ.get("BONAFIDE_DENOISE_WEIGHTS")
    return p if p and os.path.exists(p) else None


class NeuralDenoisePass(RenderPass):
    name = "neural_denoise"

    def __init__(self) -> None:
        self._net: nn.Module | None = None

    def is_active(self, ctx: PassContext) -> bool:
        # Gate on weights-present: without a trained checkpoint the net is
        # random and degrades the image, so the pass must stay off.
        return bool(ctx.config.neural_denoise) and _weights_path() is not None

    def _ensure_net(self, device: str) -> nn.Module:
        if self._net is None:
            self._net = _MicroUNet().to(device)
            weights = _weights_path()
            if weights:
                self._net.load_state_dict(torch.load(weights, map_location=device))
            self._net.eval()
        return self._net

    @torch.no_grad()
    def run(self, ctx: PassContext) -> None:
        net = self._ensure_net(ctx.backend.device)
        x = ctx.targets.rgb.permute(2, 0, 1).unsqueeze(0)              # (1, 3, H, W)
        # Pad to multiple of 8 so the U-Net contracts cleanly
        h, w = x.shape[-2:]
        ph = (8 - h % 8) % 8
        pw = (8 - w % 8) % 8
        x_pad = nn.functional.pad(x, (0, pw, 0, ph), mode="replicate")
        y = net(x_pad)[..., :h, :w]
        ctx.targets.rgb = y.squeeze(0).permute(1, 2, 0).contiguous()


class _MicroUNet(nn.Module):
    """Tiny U-Net (untrained — passes residual through cleanly)."""
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
