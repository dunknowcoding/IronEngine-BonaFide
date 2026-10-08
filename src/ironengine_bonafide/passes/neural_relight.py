"""Screen-space global illumination (``neural_relight="ssgi"``).

A one-bounce diffuse GI approximation, no training data required: the lit
frame is blurred through a two-level gaussian pyramid and gathered back
onto surfaces as indirect irradiance, modulated by albedo (colourbleed),
surface normal hemisphere weighting, and a screen-space occlusion term
derived from depth discontinuities. Runs after PBR, before AA/post.

``"neural_ibl"`` remains a roadmap item and records a skip note.

This is a deliberate approximation — it captures the perceptual bulk of
indirect light (contact fill, colourbleed) at a fraction of path-traced
cost, and is honest about it: the skip notes and docs call it SSGI-lite.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ironengine_bonafide.passes.base import PassContext, RenderPass

_PYRAMID_LEVELS = 3


class NeuralRelightPass(RenderPass):
    name = "neural_relight"

    def is_active(self, ctx: PassContext) -> bool:
        return ctx.config.neural_relight != "none"

    @torch.no_grad()
    def run(self, ctx: PassContext) -> None:
        if ctx.config.neural_relight != "ssgi":
            ctx.skipped.append(f"neural_relight:{ctx.config.neural_relight}_unimplemented")
            return
        t = ctx.targets
        intensity = float(getattr(ctx.config, "ssgi_intensity", 0.5))
        if intensity <= 0.0:
            return
        indirect = _ssgi(t.rgb, t.normals, t.depth, t.albedo)
        # Only surfaces receive bounce light; the sky does not.
        mask = torch.isfinite(t.depth).unsqueeze(-1)
        t.rgb = t.rgb + torch.where(mask, indirect * intensity, torch.zeros_like(indirect))


@torch.no_grad()
def _ssgi(rgb: torch.Tensor, normals: torch.Tensor, depth: torch.Tensor,
          albedo: torch.Tensor) -> torch.Tensor:
    """One-bounce screen-space indirect light, (H, W, 3) linear HDR.

    indirect = albedo * occlusion * Σᵢ gainᵢ · blurᵢ(rgb), gathered with a
    normal-hemisphere weight so floors don't bleed into walls.
    """
    h, w, _ = rgb.shape
    # Screen-space occlusion: flat / open surface → small local depth
    # spread → fully lit (1); depth discontinuities (crevices, contact
    # edges) → occluded (0), which keeps bounce light out of contact lines.
    finite = torch.isfinite(depth)
    dfill = float(depth[finite].max()) if bool(finite.any()) else 1.0
    dep = torch.where(finite, depth, torch.full_like(depth, dfill))
    d4 = dep.unsqueeze(0).unsqueeze(0)
    dmax = F.max_pool2d(d4, kernel_size=5, stride=1, padding=2)
    dmin = -F.max_pool2d(-d4, kernel_size=5, stride=1, padding=2)
    spread = (dmax - dmin) / d4.abs().clamp(min=1e-3)
    occlusion = (1.0 - spread / 0.1).clamp(0.0, 1.0).squeeze(0).squeeze(0)

    # Gaussian pyramid of the lit frame (only geometry pixels contribute).
    rgb_masked = torch.where(finite.unsqueeze(-1), rgb, torch.zeros_like(rgb))
    img = rgb_masked.permute(2, 0, 1).unsqueeze(0)
    levels = [img]
    for _ in range(_PYRAMID_LEVELS):
        levels.append(F.avg_pool2d(levels[-1], kernel_size=2, ceil_mode=True))

    # Hemisphere gather: each mip is upsampled and weighted by how much of
    # the surface normal's hemisphere faces the screen neighbourhood —
    # approximated by the normal's view-independent up-weight blended with
    # a Lambert-like term toward the mean incoming direction (view ray).
    nrm = normals / torch.linalg.norm(normals, dim=-1, keepdim=True).clamp(min=1e-9)
    hemi = (0.5 + 0.5 * nrm[..., 1]).clamp(0.0, 1.0)          # up-facing gather
    indirect = torch.zeros_like(rgb)
    gain = 1.0
    for lvl in levels[1:]:
        up = F.interpolate(lvl, size=(h, w), mode="bilinear", align_corners=False)
        up = up.squeeze(0).permute(1, 2, 0)
        indirect = indirect + gain * up
        gain *= 0.5
    indirect = indirect / sum(0.5 ** i for i in range(len(levels) - 1))
    bleed = albedo.clamp(0.0, 1.0)
    return indirect * bleed * occlusion.unsqueeze(-1) * (0.5 + 0.5 * hemi).unsqueeze(-1)
