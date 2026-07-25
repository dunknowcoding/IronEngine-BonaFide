"""Anti-aliasing dispatcher + SSAA resolve.

The actual implementations live next door (FXAA in postprocess.py for now;
TAA / SMAA stubs are kept here so the pass graph has a stable name).

``SsaaDownsamplePass`` resolves the full-scene supersampling requested via
``RenderConfig.ssaa``: the geometry passes render at ssaa× the output
resolution (see ``api.render._do_render``) and this pass reduces the frame
to the requested size *before* post passes (FXAA, bloom, tonemap) run —
the downsample therefore happens in linear HDR, which is the correct
colour space for averaging.
"""
from __future__ import annotations

import torch

from ironengine_bonafide.logging import logger
from ironengine_bonafide.passes.base import PassContext, RenderPass


class SsaaDownsamplePass(RenderPass):
    """Area-average resolve of an ssaa× frame to the config resolution."""
    name = "ssaa_downsample"

    def is_active(self, ctx: PassContext) -> bool:
        return int(getattr(ctx.config, "ssaa", 1)) > 1

    def run(self, ctx: PassContext) -> None:
        s = int(ctx.config.ssaa)
        t = ctx.targets
        t.rgb = area_downsample(t.rgb, s)
        t.normals = area_downsample(t.normals, s)
        t.albedo = area_downsample(t.albedo, s)
        t.depth = depth_downsample(t.depth, s)
        t.ids = ids_downsample(t.ids, s)


class TaaPass(RenderPass):
    name = "taa"

    def is_active(self, ctx: PassContext) -> bool:
        return ctx.config.aa == "taa"

    def run(self, ctx: PassContext) -> None:
        # TAA needs frame history we don't track yet; degrade to FXAA.
        from ironengine_bonafide.passes.postprocess import _fxaa
        logger.debug("TAA history not implemented yet; degrading to FXAA")
        ctx.targets.rgb = _fxaa(ctx.targets.rgb)


class SmaaPass(RenderPass):
    name = "smaa"

    def is_active(self, ctx: PassContext) -> bool:
        return ctx.config.aa == "smaa"

    def run(self, ctx: PassContext) -> None:
        from ironengine_bonafide.passes.postprocess import _fxaa
        logger.debug("SMAA not implemented yet; degrading to FXAA")
        ctx.targets.rgb = _fxaa(ctx.targets.rgb)


# ---------------------------------------------------------------- helpers
def area_downsample(x: torch.Tensor, s: int) -> torch.Tensor:
    """(H, W, C) → (H/s, W/s, C) area average (NOT nearest-neighbour)."""
    h, w, c = x.shape
    img = x.permute(2, 0, 1).unsqueeze(0)                    # (1, C, H, W)
    out = torch.nn.functional.avg_pool2d(img, kernel_size=s)
    return out.squeeze(0).permute(1, 2, 0).contiguous()


def depth_downsample(depth: torch.Tensor, s: int) -> torch.Tensor:
    """Min-pool depth so edge blocks keep the closest surface; blocks that
    are entirely empty stay +inf."""
    neg = (-depth).unsqueeze(0).unsqueeze(0)
    out = torch.nn.functional.max_pool2d(neg, kernel_size=s)
    return (-out).squeeze(0).squeeze(0).contiguous()


def ids_downsample(ids: torch.Tensor, s: int) -> torch.Tensor:
    """Nearest (block-centre) resolve for integer instance IDs — averaging
    IDs would be meaningless."""
    h, w = ids.shape
    blocks = ids.reshape(h // s, s, w // s, s)
    return blocks[:, s // 2, :, s // 2].contiguous()
