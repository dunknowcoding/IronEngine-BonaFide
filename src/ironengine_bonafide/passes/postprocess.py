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
    """Threshold-driven separable-gaussian bloom.

    ``bloom_radius`` levels form a progressive gaussian pyramid (each level
    blurs the previous one wider), giving emissive surfaces — sun/moon
    discs, light discs, neon — a soft multi-scale halo. ``bloom_radius=1``
    with the default threshold/intensity reproduces the legacy single-pass
    behavior bit-for-bit.
    """
    name = "bloom"

    # Relative gain per pyramid level (level 0 must stay 1.0 — legacy).
    _LEVEL_GAINS = (1.0, 0.6, 0.35, 0.2)

    def is_active(self, ctx: PassContext) -> bool:
        return bool(ctx.config.bloom)

    def run(self, ctx: PassContext) -> None:
        cfg = ctx.config
        rgb = ctx.targets.rgb
        threshold = float(getattr(cfg, "bloom_threshold", 1.0))
        intensity = float(getattr(cfg, "bloom_intensity", 0.6))
        radius = int(getattr(cfg, "bloom_radius", 1))
        # Extract bright fragments above a soft knee
        bright = (rgb - threshold).clamp(min=0.0)
        if radius <= 1:
            # Legacy path — cheap separable Gaussian (5-tap), bit-identical
            # to the pre-pyramid implementation.
            ctx.targets.rgb = rgb + intensity * _blur5(bright)
            return
        blurred = bright
        halo = torch.zeros_like(rgb)
        for gain in self._LEVEL_GAINS[:radius]:
            blurred = _blur5(blurred)
            halo = halo + gain * blurred
        ctx.targets.rgb = rgb + intensity * halo


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
    """FXAA-style luma-edge blend.

    Per pixel, the local luma contrast (max − min over the cross
    neighbourhood) sets the blend weight toward the 3x3 box average —
    capped at 0.5, so staircase edges are smoothed toward their true
    sub-pixel position while flat regions (zero contrast) pass through
    bit-exact. Weight ∝ contrast/luma_max keeps HDR edges (sun disc,
    emissives) on the same footing as LDR ones.
    """
    h, w, c = x.shape
    img = x.permute(2, 0, 1).unsqueeze(0)
    img_p = torch.nn.functional.pad(img, (1, 1, 1, 1), mode="replicate")
    # average of 3x3 neighbours
    kernel = torch.full((c, 1, 3, 3), 1.0 / 9.0, dtype=x.dtype, device=x.device)
    avg = torch.nn.functional.conv2d(img_p, kernel, groups=c).squeeze(0).permute(1, 2, 0)
    # local luma contrast
    luma = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    lp = torch.nn.functional.pad(luma.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1),
                                 mode="replicate").squeeze(0).squeeze(0)
    n = lp[:-2, 1:-1]
    s = lp[2:, 1:-1]
    we = lp[1:-1, :-2]
    e = lp[1:-1, 2:]
    lmax = torch.maximum(torch.maximum(torch.maximum(n, s), torch.maximum(we, e)), luma)
    lmin = torch.minimum(torch.minimum(torch.minimum(n, s), torch.minimum(we, e)), luma)
    contrast = lmax - lmin
    weight = (contrast / (lmax.abs() + 1e-3)).clamp(0.0, 0.5)
    weight = torch.where(contrast > 1e-6, weight, torch.zeros_like(weight)).unsqueeze(-1)
    return x * (1.0 - weight) + avg * weight
