"""sim shim: sensor-pose TRS composition, +X→−Z convention conversion,
viewport size override, world resolution, Hierarchy chains, install gating.

Uses the real ``ironengine_sim`` component dataclasses (installed in the
test env) with fake component stores / RenderWorld doubles — no GL, no Qt.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from ironengine_bonafide.integrations import sim as shim

pytest.importorskip("ironengine_sim", reason="ironengine_sim not on PYTHONPATH")

from ironengine_sim.world.components import (  # noqa: E402
    Camera,
    Hierarchy,
    MeshRenderable,
    Transform,
)


# --------------------------------------------------------------- fakes
class _ComponentStore:
    def __init__(self) -> None:
        self._by_eid: dict[int, list[Any]] = {}

    def add(self, eid: int, comp: Any) -> None:
        self._by_eid.setdefault(eid, []).append(comp)

    @staticmethod
    def _match(comp: Any, cls: type) -> bool:
        return type(comp).__name__ == cls.__name__

    def iter_components(self, cls: type):
        for eid, comps in self._by_eid.items():
            for c in comps:
                if self._match(c, cls):
                    yield eid, c

    def entities_with(self, *clss: type):
        for eid, comps in self._by_eid.items():
            if all(any(self._match(c, cls) for c in comps) for cls in clss):
                yield eid

    def get(self, eid: int, cls: type):
        for c in self._by_eid.get(eid, []):
            if self._match(c, cls):
                return c
        return None

    def require(self, eid: int, cls: type):
        comp = self.get(eid, cls)
        if comp is None:
            raise KeyError(f"entity {eid} lacks {cls.__name__}")
        return comp


class _Graph:
    def __init__(self, store: _ComponentStore) -> None:
        self.components = store

    def alive(self, eid: int) -> bool:
        return eid in self.components._by_eid


def _world() -> tuple[Any, _ComponentStore]:
    store = _ComponentStore()
    assets = SimpleNamespace(get_mesh=lambda name: None, get_point_cloud=lambda name: None)
    return SimpleNamespace(graph=_Graph(store), assets=assets), store


# --------------------------------------------------------------- sensor pose
def test_sensor_pose_rotates_offset_and_converts_convention() -> None:
    s = math.sqrt(0.5)
    pose = Transform(
        position=np.array([10.0, 0.0, 0.0]),
        rotation=np.array([0.0, s, 0.0, s]),                    # +90° about Y
    )
    m = shim._sensor_pose(pose, (1.0, 0.0, 0.0))

    # Offset rotated into the body frame: rotY90 @ (1,0,0) = (0,0,-1).
    np.testing.assert_allclose(m[:3, 3], [10.0, 0.0, -1.0], atol=1e-12)

    rot = m[:3, :3]
    # BonaFide camera −Z must map onto Sim's sensor forward: rotY90 @ +X.
    forward = rot @ np.array([0.0, 0.0, -1.0])
    np.testing.assert_allclose(forward, [0.0, 0.0, -1.0], atol=1e-12)
    # Camera up stays world up.
    np.testing.assert_allclose(rot @ np.array([0.0, 1.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)
    # Camera right (+X) maps onto Sim sensor +Z, rotated by the body:
    # rotY90 @ (0,0,1) = (1,0,0).
    np.testing.assert_allclose(rot @ np.array([1.0, 0.0, 0.0]), [1.0, 0.0, 0.0], atol=1e-12)
    # Rotation block is orthonormal with det +1.
    np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rot) == pytest.approx(1.0, abs=1e-12)


def test_sensor_pose_identity() -> None:
    m = shim._sensor_pose(Transform(position=np.array([1.0, 2.0, 3.0])), (0.5, 0.0, 0.0))
    np.testing.assert_allclose(m[:3, 3], [1.5, 2.0, 3.0], atol=1e-12)
    # Identity body: camera forward −Z → Sim sensor forward +X.
    np.testing.assert_allclose(m[:3, :3] @ np.array([0.0, 0.0, -1.0]), [1.0, 0.0, 0.0], atol=1e-12)


def test_sensor_pose_matches_sim_sensor_camera() -> None:
    """Cross-check against Sim's own `_sensor_camera`: for a yaw-only body
    rotation, BonaFide's view matrix (inv(pose)) must send the same world
    points to the same eye coordinates as Sim's RenderCamera view matrix.

    Note: Sim's `_sensor_camera` builds its frame with WORLD up (no body
    roll), while `_sensor_pose` preserves the full body TRS including roll
    (the intended fix per the integration contract). The two constructions
    coincide exactly for yaw-only rotations, so the cross-check uses one.
    """
    from ironengine_sim.rendering.render_world import RenderWorld

    s = math.sqrt(0.5)
    pose = Transform(
        position=np.array([3.0, 1.0, -2.0]),
        rotation=np.array([0.0, s, 0.0, s]),                    # +90° about Y (yaw)
    )
    offset = (0.2, 0.0, 0.1)
    rc = RenderWorld._sensor_camera(None, pose, offset, 60.0, 64, 64, 0.05, 100.0)  # type: ignore[arg-type]
    sim_view = rc.view_matrix().astype(np.float64)

    bf_pose = shim._sensor_pose(pose, offset)
    bf_view = np.linalg.inv(bf_pose)

    g = np.random.default_rng(0)
    pts = g.uniform(-5, 5, (64, 3))
    hom = np.concatenate([pts, np.ones((64, 1))], axis=1)
    np.testing.assert_allclose(hom @ sim_view.T, hom @ bf_view.T, atol=1e-5)


# --------------------------------------------------------------- viewport size / world resolution
def test_viewport_size_honors_override() -> None:
    rw = SimpleNamespace(viewport_override_size=(640, 360))
    assert shim._viewport_size(rw) == (640, 360)
    assert shim._viewport_size(SimpleNamespace()) == (1280, 720)


def test_world_parts_prefers_scene_assets_then_pin() -> None:
    world, _ = _world()
    rw = SimpleNamespace(scene=world.graph, assets=world.assets)
    resolved = shim._world_parts(rw)
    assert resolved.graph is world.graph
    assert resolved.assets is world.assets
    pinned = SimpleNamespace(_world=world)
    assert shim._world_parts(pinned) is world
    with pytest.raises(RuntimeError, match="install_for_world"):
        shim._world_parts(SimpleNamespace())


# --------------------------------------------------------------- camera from world
def test_camera_from_render_camera_passthrough() -> None:
    world, _ = _world()
    rc = SimpleNamespace(
        position=np.array([1.0, 2.0, 3.0]), target=np.array([0.0, 0.0, 0.0]),
        up=np.array([0.0, 1.0, 0.0]), fov_deg=70.0, aspect=1.0,
        near=0.1, far=50.0,
    )
    cam = shim._camera_from_world(world, rc)
    np.testing.assert_allclose(cam.position, [1.0, 2.0, 3.0])
    assert cam.fov_deg == 70.0 and cam.near == 0.1 and cam.far == 50.0


def test_camera_from_entity_uses_transform_rotation_and_hierarchy() -> None:
    world, store = _world()
    s = math.sqrt(0.5)
    store.add(1, Transform(position=np.array([5.0, 0.0, 0.0])))
    store.add(2, Transform(
        position=np.array([0.0, 1.0, 0.0]),
        rotation=np.array([0.0, s, 0.0, s]),                    # +90° about Y
    ))
    store.add(2, Camera(fov_deg=75.0, near=0.2, far=80.0))
    store.add(2, Hierarchy(parent=1))
    cam = shim._camera_from_world(world, 2)
    # Eye = parent T ∘ child T: (5,1,0). Forward = rotY90 @ (0,0,-1) = (-1,0,0).
    np.testing.assert_allclose(cam.position, [5.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(cam.look_at) - np.asarray(cam.position), [-1.0, 0.0, 0.0], atol=1e-12)
    assert cam.fov_deg == 75.0 and cam.near == 0.2 and cam.far == 80.0


def test_camera_none_falls_back_to_sim_default_editor_camera() -> None:
    world, _ = _world()
    cam = shim._camera_from_world(world, None)
    np.testing.assert_allclose(cam.position, [3.5, 2.5, 3.5])
    np.testing.assert_allclose(cam.look_at, [0.0, 0.6, 0.0])
    assert cam.fov_deg == 55.0


# --------------------------------------------------------------- hierarchy baking
def test_scene_from_world_composes_hierarchy_chain() -> None:
    world, store = _world()
    verts = np.array([[1, 0, 0, 0, 0, 1, 0, 0]], dtype=np.float32)
    handle = SimpleNamespace(vertices=verts, indices=np.array([0, 0, 0], dtype=np.int64))
    world.assets = SimpleNamespace(
        get_mesh=lambda name: handle, get_point_cloud=lambda name: None)
    s = math.sqrt(0.5)
    store.add(1, Transform(position=np.array([10.0, 0.0, 0.0]),
                           rotation=np.array([0.0, s, 0.0, s])))
    store.add(2, MeshRenderable(mesh_id="tri"))
    store.add(2, Transform(position=np.array([0.0, 2.0, 0.0]), scale=np.array([2.0, 2.0, 2.0])))
    store.add(2, Hierarchy(parent=1))

    scene = shim._scene_from_world(world)
    assert len(scene.meshes) == 1
    # child TRS: (1,0,0)*2=(2,0,0), +t=(2,2,0); parent: rotY90→(0,2,-2), +t=(10,2,-2).
    np.testing.assert_allclose(scene.meshes[0].positions[0].numpy(), [10.0, 2.0, -2.0], atol=1e-5)


# --------------------------------------------------------------- install gating
def test_install_default_patches_sensors_only() -> None:
    from ironengine_sim.rendering.render_world import RenderWorld

    orig_viewport = RenderWorld.render_viewport
    orig_rgb = RenderWorld.render_sensor_rgb
    orig_depth = RenderWorld.render_sensor_depth
    try:
        shim.install()                                          # headless_only=True
        assert RenderWorld.render_viewport is orig_viewport     # untouched — editor safe
        assert RenderWorld.render_sensor_rgb is shim._patched_render_sensor_rgb
        assert RenderWorld.render_sensor_depth is shim._patched_render_sensor_depth
    finally:
        shim.uninstall()
    assert RenderWorld.render_viewport is orig_viewport
    assert RenderWorld.render_sensor_rgb is orig_rgb
    assert RenderWorld.render_sensor_depth is orig_depth


def test_install_full_viewport_opt_in() -> None:
    from ironengine_sim.rendering.render_world import RenderWorld

    orig_viewport = RenderWorld.render_viewport
    try:
        shim.install(headless_only=False)
        assert RenderWorld.render_viewport is shim._patched_render_viewport
    finally:
        shim.uninstall()
    assert RenderWorld.render_viewport is orig_viewport


def test_patched_viewport_uses_override_size_and_stashes_frame(monkeypatch) -> None:
    world, _ = _world()
    rw = SimpleNamespace(scene=world.graph, assets=world.assets,
                         viewport_override_size=(320, 240))
    captured: dict[str, Any] = {}

    class _Out:
        rgb = torch.full((240, 320, 3), 0.25, dtype=torch.float32)
        depth = None

    def _fake_render(engine, scene, cam, cfg):                  # noqa: ANN001, ARG001
        captured["cfg"] = cfg
        return _Out()

    monkeypatch.setattr(shim, "render", _fake_render)
    monkeypatch.setattr(shim, "_engine", lambda: object())

    shim._patched_render_viewport(rw, None)
    assert captured["cfg"].width == 320 and captured["cfg"].height == 240
    frame = rw._last_bonafide_frame
    assert frame.shape == (240, 320, 3) and frame.dtype == np.uint8
    assert abs(int(frame[0, 0, 0]) - 64) <= 1                   # 0.25 → ≈64, no ACES


def test_patched_sensor_rgb_pose_and_size(monkeypatch) -> None:
    world, _ = _world()
    rw = SimpleNamespace(scene=world.graph, assets=world.assets)
    captured: dict[str, Any] = {}

    class _Out:
        rgb = torch.zeros((12, 16, 3), dtype=torch.float32)
        depth = None

    def _fake_render(engine, scene, cam, cfg):                  # noqa: ANN001, ARG001
        captured["cam"] = cam
        return _Out()

    monkeypatch.setattr(shim, "render", _fake_render)
    monkeypatch.setattr(shim, "_engine", lambda: object())

    s = math.sqrt(0.5)
    pose = Transform(position=np.array([0.0, 0.0, 5.0]),
                     rotation=np.array([0.0, s, 0.0, s]))
    img = shim._patched_render_sensor_rgb(rw, "cam0", pose, (1.0, 0.0, 0.0), 16, 12, 60.0)
    assert img.shape == (12, 16, 3)
    cam_pose = captured["cam"].pose
    np.testing.assert_allclose(cam_pose[:3, 3], [0.0, 0.0, 4.0], atol=1e-9)


# --------------------------------------------------------------- soft bodies
def _mesh_handle(n_verts: int, indices: np.ndarray) -> Any:
    verts = np.zeros((n_verts, 8), dtype=np.float32)
    verts[:, 2] = 1.0                                       # flat sheet at z=1
    verts[:, 5] = 1.0                                       # normals +Z
    return SimpleNamespace(vertices=verts,
                           indices=np.asarray(indices, dtype=np.int64).ravel())


def _soft_world(n_verts: int, indices: np.ndarray,
                soft_positions: np.ndarray | None) -> tuple[Any, _ComponentStore]:
    store = _ComponentStore()
    handle = _mesh_handle(n_verts, indices)
    world = SimpleNamespace(
        graph=_Graph(store),
        assets=SimpleNamespace(get_mesh=lambda name: handle,
                               get_point_cloud=lambda name: None),
        physics=SimpleNamespace(
            get_soft_body=lambda eid: (
                SimpleNamespace(positions=soft_positions)
                if soft_positions is not None else None)),
    )
    store.add(1, MeshRenderable(mesh_id="cloth"))
    store.add(1, Transform(position=np.array([10.0, 0.0, 0.0])))
    return world, store


def test_softbody_deformed_positions_replace_static_mesh() -> None:
    """A live soft-body instance with a matching particle count supplies the
    rendered vertex positions (world-space, transform NOT re-applied)."""
    idx = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    deformed = np.array([[1.0, 2.0, 3.0], [2.0, 2.0, 3.0],
                         [2.0, 3.0, 4.0], [1.0, 3.0, 3.0]], dtype=np.float32)
    world, _ = _soft_world(4, idx, deformed)
    scene = shim._scene_from_world(world)
    assert len(scene.meshes) == 1
    np.testing.assert_allclose(scene.meshes[0].positions.numpy(), deformed, atol=1e-6)
    # Normals recomputed from the deformed quad (not the flat +Z source).
    n = scene.meshes[0].normals.numpy()
    assert not np.allclose(n[:, 2], 1.0), "normals must follow the deformation"
    np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-5)


def test_softbody_grid_retriangulated_on_count_mismatch() -> None:
    """Solver-resolution fallback (particle grid ≠ mesh verts): the cloth is
    re-triangulated as a regular grid so the simulated shape still renders."""
    idx = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)   # 4-vert mesh
    grid_pos = np.array([[x, 0.5, z] for z in (0.0, 1.0, 2.0) for x in (0.0, 1.0, 2.0)],
                        dtype=np.float32)                    # 3×3 XZ cloth grid
    world, _ = _soft_world(4, idx, grid_pos)
    scene = shim._scene_from_world(world)
    assert len(scene.meshes) == 1
    mesh = scene.meshes[0]
    assert mesh.positions.shape[0] == 9                      # 3×3 cloth grid
    assert mesh.indices.shape[0] == 8                        # 2×2 quads × 2 tris
    np.testing.assert_allclose(mesh.positions.numpy(), grid_pos, atol=1e-6)
    # Flat XZ-ish grid → unit normals, sensible shading.
    np.testing.assert_allclose(np.linalg.norm(mesh.normals.numpy(), axis=1),
                               1.0, atol=1e-5)


def test_softbody_absent_state_falls_back_to_static_mesh() -> None:
    """No live instance (or no physics world at all) → the statically baked
    mesh renders exactly as before."""
    idx = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

    world, _ = _soft_world(4, idx, None)                     # instance missing
    scene = shim._scene_from_world(world)
    np.testing.assert_allclose(scene.meshes[0].positions.numpy()[0],
                               [10.0, 0.0, 1.0], atol=1e-6)   # static + transform

    world, _ = _soft_world(4, idx, np.zeros((4, 3), dtype=np.float32))
    world.physics = None                                     # physics world gone
    scene = shim._scene_from_world(world)
    np.testing.assert_allclose(scene.meshes[0].positions.numpy()[0],
                               [10.0, 0.0, 1.0], atol=1e-6)

    world2, _ = _soft_world(4, idx, np.zeros((5, 3), dtype=np.float32))
    scene2 = shim._scene_from_world(world2)                  # odd count, no grid
    np.testing.assert_allclose(scene2.meshes[0].positions.numpy()[0],
                               [10.0, 0.0, 1.0], atol=1e-6)


def test_softbody_real_sim_world_end_to_end() -> None:
    """Real Sim objects: a World with a draped cloth (particles ≠ mesh verts
    via the resolution fallback) exposes its live particle state through the
    gc discovery path — no install_for_world pin required."""
    pytest.importorskip("ironengine_sim")
    from ironengine_sim.world.world import World

    w = World()
    eid = w.graph.spawn("towel") if hasattr(w.graph, "spawn") else None
    if eid is None:
        pytest.skip("SceneGraph.spawn not available in this Sim build")
    import numpy as _np

    from ironengine_sim.assets import MeshHandle
    from ironengine_sim.world.components import (
        MeshRenderable,
        SoftBody,
        Transform,
    )

    verts = _np.zeros((4, 8), dtype=_np.float32)
    verts[:, 2] = 1.0
    handle = MeshHandle(
        name="cloth", vertices=verts,
        indices=_np.array([0, 1, 2, 0, 2, 3], dtype=_np.int64),
        aabb_min=_np.zeros(3, dtype=_np.float32),
        aabb_max=_np.ones(3, dtype=_np.float32))
    w.assets.add_mesh(handle)
    w.graph.components.add(eid, MeshRenderable(mesh_id="cloth"))
    w.graph.components.add(eid, Transform(position=np.array([0.0, 1.0, 0.0])))
    w.graph.components.add(eid, SoftBody(
        kind="cloth", params={"resolution": 3, "size": 0.6}))
    inst = w.physics.add_soft_body(
        eid, w.graph.components.get(eid, SoftBody))
    assert inst is not None and inst.positions.shape[0] == 9

    # Resolve through the RenderWorld-shaped namespace (graph + assets only),
    # exactly what the patched sensor path sees without a pin.
    ns = SimpleNamespace(graph=w.graph, assets=w.assets)
    scene = shim._scene_from_world(ns)
    assert len(scene.meshes) == 1
    np.testing.assert_allclose(scene.meshes[0].positions.numpy(),
                               inst.positions, atol=1e-6)
