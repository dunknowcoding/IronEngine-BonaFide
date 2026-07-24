"""Regression: iemodel/3 manifests load tolerantly (3DCreator non-rigid export)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ironengine_bonafide.assets.loaders.iemodel import load_iemodel

# 1x1 white PNG (smallest valid file; only existence matters to the loader).
_PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d494844520000000100000001080600000"
    "01f15c4890000000d49444154789c626000010000050001a5f645400"
    "000000049454e44ae426082"
)


def _v3_manifest() -> dict:
    return {
        "schema": "iemodel/3",
        "name": "banner",
        "generator": "ironengine-3d-creator 0.3.0",
        "created_utc": "2025-06-01T00:00:00+00:00",
        "units": "meters",
        "up_axis": "Y",
        "aabb_min": [-1.0, 0.0, -0.05],
        "aabb_max": [1.0, 2.0, 0.05],
        "bbox_size": [2.0, 2.0, 0.1],
        "shape": "banner",
        "material": {"name": "canvas", "albedo": [0.9, 0.85, 0.7],
                     "roughness": 0.9, "metallic": 0.0},
        "materials": {
            "canvas": {"albedo": [0.9, 0.85, 0.7], "roughness": 0.9, "metallic": 0.0,
                       "density_kg_m3": 500.0, "friction": 0.8, "restitution": 0.05},
        },
        "parts": [
            {"label": "sheet", "primitive": "plane", "material": "canvas",
             "aabb_min": [-1.0, 0.0, -0.05], "aabb_max": [1.0, 2.0, 0.05],
             "solid_volume_m3": 0.0},
        ],
        "physics": {"density_kg_m3": 500.0, "friction": 0.8, "restitution": 0.05,
                    "collider": "box", "dynamic": True, "solid_volume_m3": 0.4,
                    "mass_kg": 0.2, "body_type": "soft"},
        "mesh": {"path": "creator_model_1.glb", "format": "glb",
                 "vertices": 4, "faces": 2, "has_uvs": True,
                 "has_vertex_colors": False, "analytic": True},
        "point_cloud": None,
        "spec": {"shape": "banner"},
        # ---- v3 non-rigid blocks (must be tolerated, not rejected) ----
        "soft_body": {"kind": "cloth", "resolution": [9, 9], "mass_kg": 0.2,
                      "stretch_stiffness": 0.9, "bend_stiffness": 0.2,
                      "damping": 0.1, "pin_indices": [0, 8]},
        "fracture": {"threshold_impulse": 5.0, "fragment_count": 12,
                     "pattern": "voronoi", "debris_material": "rubble"},
        "articulation": {"ragdoll": False, "joints": []},
        "cloth": {"width_m": 2.0, "height_m": 2.0},
        "some_future_block": {"anything": [1, 2, 3]},
    }


def _write(tmp_path: Path, manifest: dict) -> Path:
    p = tmp_path / "creator_model_1.iemodel.json"
    p.write_text(json.dumps(manifest), encoding="utf-8")
    return p


def test_v3_loads_with_nonrigid_blocks(tmp_path: Path) -> None:
    manifest = load_iemodel(_write(tmp_path, _v3_manifest()))
    assert manifest.schema == "iemodel/3"
    assert manifest.name == "banner"
    # physics.body_type flows through the generic physics dict.
    assert manifest.physics["body_type"] == "soft"
    # v2 fields intact.
    assert "canvas" in manifest.materials
    assert manifest.parts[0].label == "sheet"
    assert manifest.material_physics["canvas"]["density_kg_m3"] == pytest.approx(500.0)


def test_v3_textures_albedo_mapped(tmp_path: Path) -> None:
    tex_dir = tmp_path / "textures"
    tex_dir.mkdir()
    (tex_dir / "fabric_albedo_512px_s0.png").write_bytes(_PNG_1X1)
    m = _v3_manifest()
    m["textures"] = {
        "schema": "ietexture/1",
        "maps": {
            "fabric_albedo": {"file": "textures/fabric_albedo_512px_s0.png",
                              "kind": "fabric", "channel": "albedo",
                              "size": 512, "seed": 0, "tileable": True,
                              "format": "png"},
        },
        "assignments": [
            {"part": "sheet", "material": "canvas",
             "maps": {"albedo": "fabric_albedo"},
             "uv": {"wrap": "repeat", "scale": [1, 1]}},
        ],
    }
    manifest = load_iemodel(_write(tmp_path, m))
    expected = str((tex_dir / "fabric_albedo_512px_s0.png").resolve())
    assert manifest.material.albedo_map == expected          # majority
    assert manifest.materials["canvas"].albedo_map == expected  # per-part


def test_v3_textures_missing_file_ignored(tmp_path: Path) -> None:
    m = _v3_manifest()
    m["textures"] = {
        "schema": "ietexture/1",
        "maps": {"fabric_albedo": {"file": "textures/absent.png", "channel": "albedo"}},
        "assignments": [{"part": "sheet", "maps": {"albedo": "fabric_albedo"}}],
    }
    manifest = load_iemodel(_write(tmp_path, m))
    assert manifest.material.albedo_map is None
    assert manifest.materials["canvas"].albedo_map is None


def test_v3_without_textures_block(tmp_path: Path) -> None:
    manifest = load_iemodel(_write(tmp_path, _v3_manifest()))
    assert manifest.material.albedo_map is None


def test_v1_v2_still_load(tmp_path: Path) -> None:
    for schema in ("iemodel/1", "iemodel/2"):
        m = _v3_manifest()
        m["schema"] = schema
        for block in ("soft_body", "fracture", "articulation", "cloth",
                      "some_future_block", "textures"):
            m.pop(block, None)
        manifest = load_iemodel(_write(tmp_path, m))
        assert manifest.schema == schema


def test_unknown_schema_rejected(tmp_path: Path) -> None:
    m = _v3_manifest()
    m["schema"] = "iemodel/99"
    with pytest.raises(ValueError, match="unsupported iemodel schema"):
        load_iemodel(_write(tmp_path, m))
