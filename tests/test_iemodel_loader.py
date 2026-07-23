"""iemodel/1 + iemodel/2 manifest loader tests (W20)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ironengine_bonafide.assets.loaders.iemodel import (
    load_creator_triple,
    load_iemodel,
    mesh_from_reconstructed,
)


def _v1_manifest() -> dict:
    return {
        "schema": "iemodel/1",
        "name": "chair",
        "generator": "ironengine-3d-creator 0.1.0",
        "created_utc": "2025-01-01T00:00:00+00:00",
        "units": "meters",
        "up_axis": "Y",
        "aabb_min": [-0.5, 0.0, -0.5],
        "aabb_max": [0.5, 1.0, 0.5],
        "bbox_size": [1.0, 1.0, 1.0],
        "shape": "chair",
        "material": {"name": "oak", "albedo": [0.6, 0.4, 0.2],
                     "roughness": 0.8, "metallic": 0.0},
        "physics": {"density_kg_m3": 700.0, "friction": 0.6,
                    "restitution": 0.1, "collider": "box", "dynamic": True},
        "mesh": {"path": "creator_model_1.glb", "format": "glb",
                 "vertices": 24, "faces": 12},
        "point_cloud": {"path": "creator_model_1.ply", "format": "ply", "points": 3},
        "spec": {"shape": "chair"},
    }


def _v2_manifest() -> dict:
    m = _v1_manifest()
    m["schema"] = "iemodel/2"
    m["materials"] = {
        "oak_seat": {"albedo": [0.6, 0.4, 0.2], "roughness": 0.8, "metallic": 0.0,
                     "density_kg_m3": 700.0, "friction": 0.6, "restitution": 0.1},
        "steel_legs": {"albedo": [0.7, 0.7, 0.75], "roughness": 0.3, "metallic": 1.0,
                       "density_kg_m3": 7850.0, "friction": 0.4, "restitution": 0.05},
    }
    m["parts"] = [
        {"label": "seat", "primitive": "box", "material": "oak_seat",
         "aabb_min": [-0.5, 0.4, -0.5], "aabb_max": [0.5, 0.5, 0.5],
         "solid_volume_m3": 0.05},
        {"label": "leg_fl", "primitive": "cylinder", "material": "steel_legs",
         "aabb_min": [-0.5, 0.0, -0.5], "aabb_max": [-0.45, 0.4, -0.45],
         "solid_volume_m3": 0.001},
    ]
    m["mesh"]["has_uvs"] = True
    m["mesh"]["has_vertex_colors"] = False
    m["mesh"]["analytic"] = True
    m["physics"]["solid_volume_m3"] = 0.054
    m["physics"]["mass_kg"] = 43.0
    return m


def _write_ascii_ply(path: Path) -> None:
    path.write_bytes(
        b"ply\nformat ascii 1.0\n"
        b"element vertex 3\n"
        b"property float x\nproperty float y\nproperty float z\n"
        b"property uchar red\nproperty uchar green\nproperty uchar blue\n"
        b"end_header\n"
        b"0 0 0 255 0 0\n1 0 0 0 255 0\n0 1 0 0 0 255\n"
    )


def test_load_v1_manifest(tmp_path: Path) -> None:
    p = tmp_path / "creator_model_1.iemodel.json"
    p.write_text(json.dumps(_v1_manifest()), encoding="utf-8")
    m = load_iemodel(p)
    assert m.schema == "iemodel/1"
    assert m.name == "chair"
    assert m.units == "meters" and m.up_axis == "Y"
    assert m.material.name == "oak"
    np.testing.assert_allclose(m.material.albedo, (0.6, 0.4, 0.2))
    assert m.material.roughness == pytest.approx(0.8)
    assert m.materials == {} and m.parts == []
    assert m.physics["density_kg_m3"] == pytest.approx(700.0)
    assert m.mesh_path == (tmp_path / "creator_model_1.glb").resolve()
    assert m.point_cloud_path == (tmp_path / "creator_model_1.ply").resolve()
    mesh_path, mat, physics, parts = m.as_tuple()
    assert mesh_path == m.mesh_path and mat is m.material
    assert physics is m.physics and parts == []


def test_load_v2_manifest(tmp_path: Path) -> None:
    p = tmp_path / "creator_model_1.iemodel.json"
    p.write_text(json.dumps(_v2_manifest()), encoding="utf-8")
    m = load_iemodel(p)
    assert m.schema == "iemodel/2"
    # v1 majority fallback still parsed.
    assert m.material.name == "oak"
    # v2 per-part materials with physics extras split out.
    assert set(m.materials) == {"oak_seat", "steel_legs"}
    steel = m.materials["steel_legs"]
    np.testing.assert_allclose(steel.albedo, (0.7, 0.7, 0.75))
    assert steel.metallic == pytest.approx(1.0)
    assert m.material_physics["steel_legs"]["density_kg_m3"] == pytest.approx(7850.0)
    assert m.material_physics["oak_seat"]["friction"] == pytest.approx(0.6)
    # Parts.
    assert len(m.parts) == 2
    seat = m.parts[0]
    assert seat.label == "seat" and seat.primitive == "box"
    assert seat.material == "oak_seat"
    np.testing.assert_allclose(seat.aabb_max, (0.5, 0.5, 0.5))
    assert seat.solid_volume_m3 == pytest.approx(0.05)
    # v2 physics extras pass through.
    assert m.physics["mass_kg"] == pytest.approx(43.0)


def test_load_rejects_unknown_schema(tmp_path: Path) -> None:
    p = tmp_path / "bad.iemodel.json"
    p.write_text(json.dumps({"schema": "iemodel/99"}), encoding="utf-8")
    with pytest.raises(ValueError, match="iemodel/99"):
        load_iemodel(p)


def test_creator_triple_ingests_ply_and_manifest(tmp_path: Path) -> None:
    _write_ascii_ply(tmp_path / "creator_model_1.ply")
    (tmp_path / "creator_model_1.iemodel.json").write_text(
        json.dumps(_v1_manifest()), encoding="utf-8")
    # Pass the .ply path — the loader locates the manifest by stem.
    triple = load_creator_triple(tmp_path / "creator_model_1.ply")
    assert triple.manifest.material.name == "oak"
    assert triple.cloud is not None
    assert triple.cloud.num_points == 3
    assert triple.mesh is None                                # .glb absent on disk


def test_mesh_from_reconstructed_reshapes_flat_indices() -> None:
    """W19 call-site guard: ReconstructedMesh.indices is flat (T*3,)."""
    class _Recon:
        positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        normals = np.array([[0, 0, 1]] * 3, dtype=np.float32)
        indices = np.array([0, 1, 2], dtype=np.uint32)          # flat
        source = "open3d_ball_pivot"

    mesh = mesh_from_reconstructed(_Recon())
    assert mesh.indices.shape == (1, 3)
    np.testing.assert_array_equal(mesh.indices.numpy(), np.array([[0, 1, 2]]))
