"""Neural relighting — slot only in v0.1.

Real implementation (per-scene neural IBL or screen-space GI prior)
lands in 0.2 once we have the GBuffer + a training loop in
`training/`. Until then, this pass just records itself in the profile
report so users see it's planned.
"""
from __future__ import annotations

from ironengine_bonafide.passes.base import PassContext, RenderPass


class NeuralRelightPass(RenderPass):
    name = "neural_relight"

    def is_active(self, ctx: PassContext) -> bool:
        return ctx.config.neural_relight != "none"

    def run(self, ctx: PassContext) -> None:
        ctx.skipped.append(f"neural_relight:{ctx.config.neural_relight}_unimplemented")
