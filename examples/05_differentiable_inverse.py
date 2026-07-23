"""Optimize a point cloud's colors against a target image (gradient demo).

Tiny synthetic example: a unit-cube of points whose colors start grey,
optimized to recreate a target frame with the differentiable splat path.
"""
from __future__ import annotations

import torch

from ironengine_bonafide.api import (
    Engine, PerspectiveCamera, PointCloud, RenderConfig, Scene, render_differentiable,
)
from ironengine_bonafide.training.losses import l2


def main() -> None:
    engine = Engine.auto()
    device = engine.backend.device
    g = torch.Generator(device="cpu").manual_seed(0)
    n = 4096
    positions = torch.rand((n, 3), generator=g) * 2.0 - 1.0
    cloud = PointCloud.from_arrays(positions, name="cube")
    cloud.colors = torch.full(positions.shape, 0.5, device=device).requires_grad_(True)
    cloud = cloud.to(device)
    cloud.use_gsplat = True

    target = torch.zeros((128, 128, 3), device=device)
    target[..., 0] = 0.8                                     # red target

    cam = PerspectiveCamera(position=(2.0, 1.0, 2.0), look_at=(0, 0, 0), fov_deg=45)
    cfg = RenderConfig(width=128, height=128, output_color_space="linear")

    opt = torch.optim.Adam([cloud.colors], lr=2e-2)
    scene = Scene().add(cloud)

    for it in range(30):
        out = render_differentiable(engine, scene, cam, cfg)
        loss = l2(out.rgb, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it % 5 == 0:
            print(f"iter={it} loss={float(loss):.4f}")
    print("final colors mean:", cloud.colors.mean(dim=0).tolist())


if __name__ == "__main__":
    main()
