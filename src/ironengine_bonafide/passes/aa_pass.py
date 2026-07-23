"""Anti-aliasing dispatcher.

The actual implementations live next door (FXAA in postprocess.py for now;
TAA / SMAA stubs are kept here so the pass graph has a stable name).
"""
from __future__ import annotations

from ironengine_bonafide.logging import logger
from ironengine_bonafide.passes.base import PassContext, RenderPass


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
