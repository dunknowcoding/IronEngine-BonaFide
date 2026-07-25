"""Screen-space crepuscular ("god") rays.

A cheap radial blur of the thresholded bright buffer, centred on the
projected position of the sun (the first DirectionalLight). Occluders
naturally carve the rays because only *visible* bright pixels feed the
blur. Runs after AA and before bloom so the rays themselves glow softly.

Opt-in via ``RenderConfig.god_rays``; the pass self-skips when the sun is
behind the camera or too far off-screen.
"""
from __future__ import annotations

import torch

from ironengine_bonafide.core.light import DirectionalLight
from ironengine_bonafide.passes.base import PassContext, RenderPass

# Skip when the projected sun sits further than this NDC distance outside
# the frame (rays from slightly off-screen suns still read well).
_NDC_MARGIN = 1.3


class GodRaysPass(RenderPass):
    name = "god_rays"

    def is_active(self, ctx: PassContext) -> bool:
        if not bool(getattr(ctx.config, "god_rays", False)):
            return False
        return any(isinstance(lt, DirectionalLight) for lt in ctx.scene.lights)

    def run(self, ctx: PassContext) -> None:
        cfg = ctx.config
        device = ctx.targets.rgb.device
        h, w, _ = ctx.targets.rgb.shape

        sun_dir = None
        for lt in ctx.scene.lights:
            if isinstance(lt, DirectionalLight):
                d = torch.tensor(lt.direction, dtype=torch.float32, device=device)
                sun_dir = -d / torch.linalg.norm(d).clamp(min=1e-9)
                break
        if sun_dir is None:                                     # pragma: no cover
            return

        # Project the sun (a point at infinity → homogeneous w = 0).
        vp = ctx.camera.view_proj_torch(ctx.aspect, device=device)
        dir_h = torch.cat([sun_dir, torch.zeros(1, dtype=torch.float32, device=device)])
        clip = vp @ dir_h
        if float(clip[3]) <= 1e-6:
            ctx.skipped.append("god_rays:sun_behind_camera")
            return
        ndc = clip[:2] / clip[3]
        if float(ndc[0].abs()) > _NDC_MARGIN or float(ndc[1].abs()) > _NDC_MARGIN:
            ctx.skipped.append("god_rays:sun_off_screen")
            return

        sx = float(ndc[0] * 0.5 + 0.5) * (w - 1)
        sy = float(0.5 - ndc[1] * 0.5) * (h - 1)

        samples = int(getattr(cfg, "god_ray_samples", 24))
        decay = float(getattr(cfg, "god_ray_decay", 0.95))
        intensity = float(getattr(cfg, "god_ray_intensity", 0.6))
        threshold = float(getattr(cfg, "bloom_threshold", 1.0))

        bright = (ctx.targets.rgb - threshold).clamp(min=0.0)
        yy, xx = torch.meshgrid(
            torch.arange(h, dtype=torch.float32, device=device),
            torch.arange(w, dtype=torch.float32, device=device),
            indexing="ij",
        )
        # March each pixel toward the sun, accumulating the bright buffer
        # with geometric decay — the classic screen-space radial blur.
        step_x = (sx - xx) / samples
        step_y = (sy - yy) / samples
        acc = torch.zeros_like(bright)
        weight = 1.0
        cx, cy = xx, yy
        for _ in range(samples):
            cx = cx + step_x
            cy = cy + step_y
            ix = cx.round().clamp(0, w - 1).long()
            iy = cy.round().clamp(0, h - 1).long()
            acc = acc + weight * bright[iy, ix]
            weight *= decay
        ctx.targets.rgb = ctx.targets.rgb + acc * (intensity / samples)
