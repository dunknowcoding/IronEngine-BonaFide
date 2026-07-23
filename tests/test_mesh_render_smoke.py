"""Smoke render of a tiny triangle mesh on the CPU backend."""
from __future__ import annotations

import numpy as np

from ironengine_bonafide.api import (
    DirectionalLight, Engine, Mesh, PBRMaterial, PerspectiveCamera, RenderConfig,
    Scene, render,
)


def test_triangle_smoke(triangle_mesh: tuple[np.ndarray, np.ndarray]) -> None:
    positions, indices = triangle_mesh
    mesh = Mesh.from_arrays(positions, indices,
                            material=PBRMaterial(albedo=(1.0, 0.0, 0.0)))
    scene = Scene().add(mesh).add(DirectionalLight(direction=(0, -1, -0.3),
                                                    intensity=2.0))
    cam = PerspectiveCamera(position=(0, 0.5, 2.0), look_at=(0, 0.5, 0), fov_deg=60)
    cfg = RenderConfig(width=64, height=48)
    out = render(Engine.cpu(), scene, cam, cfg)
    # Triangle with red material should produce nonzero red channel
    assert float(out.rgb[..., 0].max()) > 0.0
