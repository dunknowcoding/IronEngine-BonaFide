"""Soft-body solver smoke test.

Builds a tiny 4-particle tetrahedron with edge constraints; steps a few
frames; asserts particles fell under gravity but stopped at the ground
plane.
"""
from __future__ import annotations

import numpy as np
import torch

from ironengine_bonafide.api import (
    DirectionalLight, DollRig, Engine, PerspectiveCamera, RenderConfig,
    Scene, render,
)


def test_softbody_steps_under_gravity() -> None:
    pts = np.array([
        [0.0, 1.0, 0.0],
        [0.5, 1.5, 0.0],
        [-0.5, 1.5, 0.0],
        [0.0, 1.5, 0.5],
    ], dtype=np.float32)
    edges = np.array([
        [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3],
    ], dtype=np.int64)
    rig = DollRig.from_arrays(pts, edges, stiffness=0.6, name="tet")
    initial_y = rig.particles[:, 1].mean().item()

    scene = Scene().add(rig).add(DirectionalLight(intensity=1.0))
    cam = PerspectiveCamera(position=(2, 1.5, 2), look_at=(0, 0.5, 0))
    cfg = RenderConfig(width=32, height=24)

    engine = Engine.cpu()
    for _ in range(20):
        render(engine, scene, cam, cfg)

    final_y = rig.particles[:, 1].mean().item()
    assert final_y < initial_y, f"Particles should fall under gravity (y went {initial_y} -> {final_y})"
    assert (rig.particles[:, 1] >= -1e-3).all().item(), "Ground plane violated"
    assert isinstance(rig.particles, torch.Tensor)
