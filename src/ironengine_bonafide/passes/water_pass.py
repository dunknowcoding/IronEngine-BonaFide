"""Water pass — Gerstner waves + planar reflection.

Slot only in v0.1; a textured plane with screen-space reflection lands
in 0.2 once the GBuffer is fully populated.
"""
from __future__ import annotations

from ironengine_bonafide.passes.base import PassContext, RenderPass


class WaterPass(RenderPass):
    name = "water"

    def is_active(self, ctx: PassContext) -> bool:
        return False

    def run(self, ctx: PassContext) -> None:
        ctx.skipped.append("water:not_implemented")
