"""Sim bridge: TRS transform baking, dependency-free world double.

The fake component store matches the real Sim classes the bridge imports
by *name*, so this test never imports ``ironengine_sim`` itself.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np

from ironengine_bonafide.integrations.sim import _quat_to_mat3, _scene_from_world


# --------------------------------------------------------------- fakes
@dataclass
class Transform:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))
    scale: np.ndarray = field(default_factory=lambda: np.ones(3))


@dataclass
class MeshRenderable:
    mesh_id: str = ""
    visible: bool = True


class _ComponentStore:
    """Mimics Sim's ComponentStore; keys matched by class name."""

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


class _Assets:
    def __init__(self) -> None:
        self._meshes: dict[str, Any] = {}

    def add_mesh(self, name: str, handle: Any) -> None:
        self._meshes[name] = handle

    def get_mesh(self, name: str) -> Any:
        return self._meshes.get(name)

    def get_point_cloud(self, name: str) -> Any:
        return None


def _make_world() -> tuple[Any, _ComponentStore]:
    store = _ComponentStore()
    assets = _Assets()
    world = SimpleNamespace(
        graph=SimpleNamespace(components=store),
        assets=assets,
    )
    return world, store


def _triangle_handle() -> Any:
    # Interleaved [pos3 | normal3 | uv2], matching Sim's MeshHandle layout.
    verts = np.array([
        [1, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0],
    ], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.int64)
    return SimpleNamespace(vertices=verts, indices=indices)


# --------------------------------------------------------------- tests
def test_quat_to_mat3_rotate_y_90() -> None:
    s = math.sqrt(0.5)
    rot = _quat_to_mat3(np.array([0.0, s, 0.0, s]))        # +90° about Y
    v = rot @ np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(v, [0.0, 0.0, -1.0], atol=1e-12)


def test_mesh_vertices_baked_with_trs() -> None:
    world, store = _make_world()
    world.assets.add_mesh("tri", _triangle_handle())
    s = math.sqrt(0.5)
    store.add(7, MeshRenderable(mesh_id="tri"))
    store.add(7, Transform(
        position=np.array([1.0, 2.0, 3.0]),
        rotation=np.array([0.0, s, 0.0, s]),               # +90° about Y
        scale=np.array([2.0, 2.0, 2.0]),
    ))

    scene = _scene_from_world(world)
    assert len(scene.meshes) == 1
    mesh = scene.meshes[0]
    # Vertex (1,0,0): scale 2 → (2,0,0); rotY90 → (0,0,-2); +t → (1,2,1).
    np.testing.assert_allclose(
        mesh.positions[0].numpy(), [1.0, 2.0, 1.0], atol=1e-5)
    # Vertex (0,1,0) unaffected by Y rotation: scale → (0,2,0); +t → (1,4,3).
    np.testing.assert_allclose(
        mesh.positions[1].numpy(), [1.0, 4.0, 3.0], atol=1e-5)
    # Normal (0,0,1) rotated (not scaled) → (1,0,0).
    assert mesh.normals is not None
    np.testing.assert_allclose(
        mesh.normals[0].numpy(), [1.0, 0.0, 0.0], atol=1e-5)


def test_transform_cache_reuses_baked_geometry() -> None:
    world, store = _make_world()
    world.assets.add_mesh("tri", _triangle_handle())
    store.add(1, MeshRenderable(mesh_id="tri"))
    store.add(1, Transform(position=np.array([5.0, 0.0, 0.0])))
    first = _scene_from_world(world).meshes[0].positions
    second = _scene_from_world(world).meshes[0].positions
    torch_equal = np.array_equal(first.numpy(), second.numpy())
    assert torch_equal
    np.testing.assert_allclose(first[0].numpy(), [6.0, 0.0, 0.0], atol=1e-6)
