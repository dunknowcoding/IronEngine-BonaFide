"""RenderBundle save → load → reproduce."""
from __future__ import annotations

import numpy as np
import torch

from ironengine_bonafide.api import (
    DirectionalLight, Engine, Mesh, PerspectiveCamera, PointCloud,
    RenderConfig, Scene,
)
from ironengine_bonafide.bundle import RenderBundle


def test_bundle_round_trip(tmp_path) -> None:                                # type: ignore[no-untyped-def]
    g = np.random.default_rng(7)
    cloud = PointCloud.from_arrays(
        g.uniform(-0.5, 0.5, (500, 3)).astype(np.float32),
        g.uniform(0.0, 1.0, (500, 3)).astype(np.float32),
    )
    mesh = Mesh.from_arrays(
        positions=np.array([[-0.3, 0, 0], [0.3, 0, 0], [0, 0.6, 0]], dtype=np.float32),
        indices=np.array([[0, 1, 2]], dtype=np.int64),
    )
    scene = (Scene().add(cloud).add(mesh)
             .add(DirectionalLight(direction=(0, -1, 0), intensity=2.0)))
    cam = PerspectiveCamera(position=(2, 1, 2), look_at=(0, 0, 0))
    cfg = RenderConfig(width=64, height=48)

    bundle = RenderBundle.capture(scene, cam, cfg, seed=99)
    out_path = tmp_path / "case.bnf"
    bundle.save(out_path)
    assert out_path.exists()
    assert (tmp_path / "case.bnf.npz").exists()

    loaded = RenderBundle.load(out_path)
    assert loaded.seed == 99
    assert len(loaded.scene.meshes) == 1
    assert len(loaded.scene.pointclouds) == 1
    assert isinstance(loaded.config, RenderConfig)

    out = loaded.reproduce(Engine.cpu())
    assert out.rgb.shape == (48, 64, 3)
    assert isinstance(out.rgb, torch.Tensor)
