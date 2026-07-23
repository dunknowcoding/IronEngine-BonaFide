"""Hole-completion pass.

Trains (lazily, once per scene-load) a small MLP that predicts color
inside detected gaps. The pass writes synthesized samples directly into
the splat compositor's working buffers so the completion happens before
gsplat / surfel raster picks them up.

Practically: this v0.1 path is a placeholder that flags clouds with
`use_completion=True` and lets `splat_pass` densify cloud points using
the trained prior. The full hole-detection grid lands in 0.2.
"""
from __future__ import annotations

from ironengine_bonafide.passes.base import PassContext, RenderPass


class CompletionPass(RenderPass):
    name = "completion"

    def is_active(self, ctx: PassContext) -> bool:
        return any(c.use_completion for c in ctx.scene.pointclouds) and ctx.config.completion.enabled

    def run(self, ctx: PassContext) -> None:
        if not ctx.backend.supports("neural_field"):
            ctx.skipped.append("completion:no_neural_field")
            return
        from ironengine_bonafide.backends.cuda.completion import train_completion_prior

        for cloud in ctx.scene.pointclouds:
            if not cloud.use_completion or cloud._completion_prior is not None:
                continue
            if cloud.colors is None:
                continue
            cloud._completion_prior = train_completion_prior(
                cloud.positions.to(ctx.backend.device),
                cloud.colors.to(ctx.backend.device),
                width=ctx.config.completion.mlp_width,
                depth=ctx.config.completion.mlp_depth,
                iterations=int(getattr(ctx.config.completion, "iterations", 1000)),
                device=ctx.backend.device,
            )
