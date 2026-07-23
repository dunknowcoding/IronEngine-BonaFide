"""Particle pass.

v0.1 stub — registers the pass slot so the engine knows to skip it
gracefully. Real implementation lives in `backends/cuda/particles.py`
behind the `warp_xpbd` capability gate and lands in 0.2.
"""
from __future__ import annotations

from ironengine_bonafide.passes.base import PassContext, RenderPass


class ParticlePass(RenderPass):
    name = "particles"

    def required_capabilities(self) -> tuple[str, ...]:
        return ("warp_xpbd",)

    def is_active(self, ctx: PassContext) -> bool:
        return False                       # no particle assets in scene model yet

    def run(self, ctx: PassContext) -> None:
        ctx.skipped.append("particles:not_implemented")
