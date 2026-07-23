"""`render` and `render_differentiable` — the public render entry points.

The two share a single private ``_do_render`` driver that:

  1. seeds RNGs deterministically
  2. allocates ``FrameTargets`` on the backend's device
  3. runs the engine's pass list under capability gating
  4. constructs the user-visible :class:`RenderOutputs`

Lifecycle hooks fire at every interesting boundary (frame begin/end,
pass begin/end, on error). See :mod:`ironengine_bonafide.lifecycle`.
"""
from __future__ import annotations

import torch

from ironengine_bonafide import lifecycle
from ironengine_bonafide.api.engine import Engine
from ironengine_bonafide.api.outputs import RenderOutputs, _OutputTensor
from ironengine_bonafide.core.camera import Camera
from ironengine_bonafide.core.config import RenderConfig
from ironengine_bonafide.core.determinism import seed_everything
from ironengine_bonafide.core.profile import stopwatch
from ironengine_bonafide.core.scene import Scene
from ironengine_bonafide.errors import PassError
from ironengine_bonafide.logging import logger
from ironengine_bonafide.passes.base import FrameTargets, PassContext


def render(
    engine: Engine,
    scene: Scene,
    camera: Camera,
    config: RenderConfig | None = None,
) -> RenderOutputs:
    """Forward render. Returns a :class:`RenderOutputs` whose tensors live
    on the engine's device, with no autograd graph.

    For autograd-aware rendering use :func:`render_differentiable`.
    """
    cfg = config or RenderConfig()
    cfg.validate()
    with torch.no_grad():
        return _do_render(engine, scene, camera, cfg, differentiable=False)


def render_differentiable(
    engine: Engine,
    scene: Scene,
    camera: Camera,
    config: RenderConfig | None = None,
) -> RenderOutputs:
    """Differentiable render. Tensors carry ``grad_fn``; gradients flow
    into PointCloud / Mesh / Material parameters that have ``requires_grad=True``.
    """
    cfg = config or RenderConfig()
    cfg.differentiable = True
    cfg.validate()
    return _do_render(engine, scene, camera, cfg, differentiable=True)


def _do_render(
    engine: Engine,
    scene: Scene,
    camera: Camera,
    config: RenderConfig,
    *,
    differentiable: bool,
) -> RenderOutputs:
    if engine._closed:
        from ironengine_bonafide.errors import BonaFideError
        raise BonaFideError("Engine has been closed; construct a new one")
    seed_everything(config.seed)
    backend = engine.backend
    h, w = config.height, config.width
    device = backend.device

    targets = FrameTargets(
        rgb=torch.zeros((h, w, 3), dtype=torch.float32, device=device),
        depth=torch.full((h, w), float("inf"), dtype=torch.float32, device=device),
        normals=torch.zeros((h, w, 3), dtype=torch.float32, device=device),
        ids=torch.zeros((h, w), dtype=torch.int64, device=device),
        albedo=torch.zeros((h, w, 3), dtype=torch.float32, device=device),
    )
    aspect = w / max(1, h)

    ctx = PassContext(
        backend=backend, scene=scene, camera=camera,
        config=config, targets=targets, aspect=aspect,
    )

    report = Engine._active_profile()
    if report is None and config.profile:
        from ironengine_bonafide.core.profile import ProfileReport
        report = ProfileReport()

    lifecycle.fire("on_frame_begin",
                   engine=engine, scene=scene, camera=camera,
                   config=config, differentiable=differentiable)

    for ps in engine.passes:
        if not ps.is_active(ctx):
            ctx.skipped.append(f"{ps.name}:disabled")
            continue
        missing = [c for c in ps.required_capabilities() if not backend.supports(c)]
        if missing:
            ctx.skipped.append(f"{ps.name}:missing[{','.join(missing)}]")
            continue
        lifecycle.fire("on_pass_begin", engine=engine, pass_name=ps.name, ctx=ctx)
        try:
            if report is not None:
                with stopwatch(report, ps.name):
                    ps.run(ctx)
            else:
                ps.run(ctx)
        except Exception as exc:
            lifecycle.fire("on_error", engine=engine, pass_name=ps.name,
                           exception=exc, ctx=ctx)
            logger.exception(f"pass '{ps.name}' raised {type(exc).__name__}: {exc}")
            raise PassError(f"pass '{ps.name}' failed: {exc}") from exc
        lifecycle.fire("on_pass_end", engine=engine, pass_name=ps.name, ctx=ctx)

    rgb = _OutputTensor(targets.rgb)
    out = RenderOutputs(
        rgb=rgb,
        depth=targets.depth if "depth" in config.sensor_outputs else None,
        normals=targets.normals if "normals" in config.sensor_outputs else None,
        ids=targets.ids if "ids" in config.sensor_outputs else None,
        albedo=targets.albedo if "albedo" in config.sensor_outputs else None,
        color_space=config.output_color_space,
        profile=report,
        skipped_passes=ctx.skipped,
    )
    lifecycle.fire("on_frame_end", engine=engine, outputs=out)
    return out
