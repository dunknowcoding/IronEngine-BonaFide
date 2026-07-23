"""Post-processing passes: bloom, ACES tonemap, FXAA, vignette/grain.

These run after every geometry pass has settled. They consume `targets.rgb`
in linear HDR and either tone-map it in place or write a sRGB-encoded copy
into `targets.rgb_srgb` (created on demand) when the user asks for sRGB out.
"""
from __future__ import annotations

import torch

from ironengine_bonafide.core.color import aces_filmic, linear_to_srgb
from ironengine_bonafide.passes.base import PassContext, RenderPass


class BloomPass(RenderPass):
    name = "bloom"

    def is_active(self, ctx: PassContext) -> bool:
        return bool(ctx.config.bloom)

    def run(self, ctx: PassContext) -> None:
        rgb = ctx.targets.rgb
        # Extract bright fragments above a soft knee
        knee = 1.0
        bright = (rgb - knee).clamp(min=0.0)
        # Cheap separable Gaussian blur (5-tap) — runs on whatever device rgb lives on
        blurred = _blur5(bright)
        ctx.targets.rgb = rgb + 0.6 * blurred


class TonemapPass(RenderPass):
    """HDR → display conversion.

    When ``output_color_space == "sRGB"`` this pass applies ACES filmic
    tonemapping (with ``config.exposure``) **and** the linear→sRGB
    transfer encoding, so ``targets.rgb`` afterwards is final
    display-ready sRGB in [0, 1]. Consumers (CLI, examples, integrations)
    must use the tensor directly — applying a second ACES/sRGB conversion
    double-tonemaps the image.
    """
    name = "tonemap"

    def is_active(self, ctx: PassContext) -> bool:
        # Apply only when the user wants sRGB out; linear-HDR users skip it.
        return ctx.config.output_color_space == "sRGB"

    def run(self, ctx: PassContext) -> None:
        mapped = aces_filmic(ctx.targets.rgb * ctx.config.exposure)
        ctx.targets.rgb = linear_to_srgb(mapped)


class FxaaPass(RenderPass):
    """1-pass FXAA-style edge smoothing."""
    name = "fxaa"

    def is_active(self, ctx: PassContext) -> bool:
        return ctx.config.aa == "fxaa"

    def run(self, ctx: PassContext) -> None:
        ctx.targets.rgb = _fxaa(ctx.targets.rgb)


# ---------------------------------------------------------------- helpers
def _blur5(x: torch.Tensor) -> torch.Tensor:
    """Tiny separable Gaussian (kernel = [1, 4, 6, 4, 1] / 16)."""
    kernel = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], dtype=x.dtype, device=x.device) / 16.0
    h, w, c = x.shape
    img = x.permute(2, 0, 1).unsqueeze(0)               # (1, C, H, W)
    pad = 2
    img = torch.nn.functional.pad(img, (pad, pad, pad, pad), mode="replicate")
    # horizontal then vertical
    kh = kernel.view(1, 1, 1, 5).expand(c, 1, 1, 5)
    img = torch.nn.functional.conv2d(img, kh, groups=c)
    kv = kernel.view(1, 1, 5, 1).expand(c, 1, 5, 1)
    img = torch.nn.functional.conv2d(img, kv, groups=c)
    return img.squeeze(0).permute(1, 2, 0).contiguous()


def _fxaa(x: torch.Tensor) -> torch.Tensor:
    """Cheap 3x3 luminance-edge smoothing — not quite FXAA-3, but the same
    idea: blend toward neighbours when local luma variance is high."""
    h, w, c = x.shape
    img = x.permute(2, 0, 1).unsqueeze(0)
    img_p = torch.nn.functional.pad(img, (1, 1, 1, 1), mode="replicate")
    # average of 3x3 neighbours
    kernel = torch.full((c, 1, 3, 3), 1.0 / 9.0, dtype=x.dtype, device=x.device)
    avg = torch.nn.functional.conv2d(img_p, kernel, groups=c).squeeze(0).permute(1, 2, 0)
    # luma variance proxy
    luma = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2])
    luma_avg = (0.299 * avg[..., 0] + 0.587 * avg[..., 1] + 0.114 * avg[..., 2])
    weight = ((luma - luma_avg).abs() / (luma.abs() + 1e-3)).clamp(0.0, 0.5).unsqueeze(-1)
    return x * (1.0 - weight) + avg * weight
