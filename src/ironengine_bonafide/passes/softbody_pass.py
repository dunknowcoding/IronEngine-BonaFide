"""Soft-body XPBD step pass.

For every :class:`DollRig` in the scene:

  1. Lazily build a :class:`WarpSolverState` and stash it on the rig
     (``rig._solver_state``).
  2. Step the solver one frame (``dt`` from config — default 1/60).
  3. Write the updated particle positions back into ``rig.particles``
     so downstream geometry passes see the deformed mesh.

When NVIDIA Warp isn't installed (or runs CPU-only), the solver
silently uses a NumPy XPBD fallback. The pass is always active so
softbodies still simulate even on the CPU backend.
"""
from __future__ import annotations

from ironengine_bonafide.passes.base import PassContext, RenderPass


class SoftBodyPass(RenderPass):
    name = "softbody"

    def required_capabilities(self) -> tuple[str, ...]:
        # Solver gracefully falls back to NumPy when warp_xpbd is missing.
        return ()

    def is_active(self, ctx: PassContext) -> bool:
        return bool(ctx.scene.softbodies)

    def run(self, ctx: PassContext) -> None:
        import torch

        from ironengine_bonafide.backends.cuda.softbody import build_state, step

        # dt — pin to 1/60 for v0.1; later: read from config.time once added.
        dt = float(getattr(ctx.config, "softbody_dt", 1.0 / 60.0))
        iterations = int(getattr(ctx.config, "softbody_iterations", 8))

        for rig in ctx.scene.softbodies:
            state = rig._solver_state
            if state is None:
                state = build_state(rig)
                rig._solver_state = state
            new_positions = step(state, dt=dt, iterations=iterations)
            # Sync back into the rig so geometry passes see deformation.
            rig.particles = new_positions.to(rig.particles.device).to(torch.float32)
