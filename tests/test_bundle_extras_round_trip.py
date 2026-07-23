"""Bundle round-trip for the previously-lost state (W25):
softbodies, scene.ibl, cloud flags + point_size_px, volume kinds."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ironengine_bonafide.api import (
    Mesh, PerspectiveCamera, PointCloud, RenderConfig, Scene,
)
from ironengine_bonafide.bundle import RenderBundle
from ironengine_bonafide.core.light import IBL
from ironengine_bonafide.core.softbody import DollRig
from ironengine_bonafide.core.volume import Volume


def _scene() -> Scene:
    g = np.random.default_rng(11)
    cloud = PointCloud.from_arrays(
        g.uniform(-1, 1, (50, 3)).astype(np.float32),
        g.uniform(0, 1, (50, 3)).astype(np.float32),
    )
    cloud.point_size_px = 5.5
    cloud.use_lod = True
    cloud.use_completion = True
    cloud.use_surfels = True

    mesh = Mesh.from_arrays(
        positions=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        indices=np.array([[0, 1, 2]], dtype=np.int64),
    )
    rig = DollRig.from_arrays(
        particles=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        edges=np.array([[0, 1], [1, 2]], dtype=np.int64),
        masses=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        stiffness=0.6, name="cloth",
    )
    rig.damping = 0.12
    grid = np.ones((2, 3, 4), dtype=np.float32) * 0.7
    volume = Volume.from_grid(grid, origin=(1.0, 2.0, 3.0), voxel_size=0.25,
                              color=(0.5, 0.6, 0.7))
    ibl = IBL(pixels=g.uniform(0, 1, (4, 8, 3)).astype(np.float32), intensity=1.7)

    return Scene().add(cloud).add(mesh).add(rig).add(volume).add(ibl)


def test_bundle_round_trip_extras(tmp_path: Path) -> None:
    bundle = RenderBundle.capture(_scene(), PerspectiveCamera(), RenderConfig(width=8, height=8))
    out = tmp_path / "extras.bnf"
    bundle.save(out)
    loaded = RenderBundle.load(out)

    # Point cloud flags + point size.
    c = loaded.scene.pointclouds[0]
    assert c.point_size_px == 5.5
    assert c.use_lod and c.use_completion and c.use_surfels
    assert not c.use_gsplat

    # Softbody restored with masses/stiffness/damping.
    assert len(loaded.scene.softbodies) == 1
    rig = loaded.scene.softbodies[0]
    assert rig.name == "cloth"
    assert rig.stiffness == 0.6
    assert rig.damping == 0.12
    np.testing.assert_allclose(rig.masses.numpy(), [1.0, 2.0, 3.0])
    assert rig.edges.shape == (2, 2)

    # Grid volume keeps its kind + grid + geometry metadata.
    v = loaded.scene.volumes[0]
    assert v.kind == "grid"
    assert v.grid is not None and v.grid.shape == (2, 3, 4)
    np.testing.assert_allclose(v.grid.numpy(), 0.7)
    assert v.grid_origin == (1.0, 2.0, 3.0)
    assert v.grid_voxel_size == 0.25
    assert v.color == (0.5, 0.6, 0.7)

    # IBL pixels + intensity restored.
    assert loaded.scene.ibl is not None
    assert loaded.scene.ibl.pixels is not None
    assert loaded.scene.ibl.pixels.shape == (4, 8, 3)
    assert loaded.scene.ibl.intensity == 1.7


def test_bundle_round_trip_fog_volume(tmp_path: Path) -> None:
    scene = Scene().add(Volume.fog(density=0.09, color=(0.1, 0.2, 0.3), height_falloff=0.4))
    bundle = RenderBundle.capture(scene, PerspectiveCamera(), RenderConfig(width=8, height=8))
    out = tmp_path / "fog.bnf"
    bundle.save(out)
    v = RenderBundle.load(out).scene.volumes[0]
    assert v.kind == "fog"
    assert v.density == 0.09
    assert v.color == (0.1, 0.2, 0.3)
    assert v.height_falloff == 0.4
