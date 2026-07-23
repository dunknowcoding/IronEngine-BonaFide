"""``.iemodel.json`` manifest loader (3DCreator export sidecar).

Accepts **both** schema generations:

* ``iemodel/1`` — top-level ``material`` + ``physics`` only.
* ``iemodel/2`` — superset: adds a ``materials`` dict (per-part PBR +
  physics props), a ``parts`` list, ``mesh.has_uvs`` / ``has_vertex_colors``
  flags, and ``physics.solid_volume_m3`` / ``mass_kg``. The v1 top-level
  ``material`` (majority fallback) is still emitted by the writer, so it is
  the default ``IEModel.material`` for both schemas.

:func:`load_creator_triple` ingests a whole ``creator_model_*`` export —
manifest + sibling GLB mesh + sibling PLY point cloud — into BonaFide
assets with the manifest's PBR material applied.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ironengine_bonafide.core.material import PBRMaterial
from ironengine_bonafide.core.mesh import Mesh
from ironengine_bonafide.core.pointcloud import PointCloud

_SCHEMAS = ("iemodel/1", "iemodel/2")


@dataclass(slots=True)
class IEModelPart:
    """One entry of the v2 ``parts`` list."""
    label: str
    primitive: str
    material: str                                       # key into materials
    aabb_min: tuple[float, float, float]
    aabb_max: tuple[float, float, float]
    solid_volume_m3: float


@dataclass(slots=True)
class IEModel:
    """Parsed ``.iemodel.json`` manifest."""
    path: Path
    schema: str
    name: str
    units: str
    up_axis: str
    aabb_min: tuple[float, float, float]
    aabb_max: tuple[float, float, float]
    material: PBRMaterial                               # majority / v1 material
    materials: dict[str, PBRMaterial]                   # v2 per-part PBR (v1: {})
    material_physics: dict[str, dict[str, float]]       # v2 per-part physics extras
    physics: dict[str, Any]                             # top-level physics block
    parts: list[IEModelPart]                            # v2 parts (v1: [])
    mesh_path: Path | None                              # resolved absolute
    point_cloud_path: Path | None                       # resolved absolute
    spec: dict[str, Any] = field(default_factory=dict)

    def as_tuple(self) -> tuple[Path | None, PBRMaterial, dict[str, Any], list[IEModelPart]]:
        """(mesh_path, material, physics, parts) — the legacy unpack form."""
        return self.mesh_path, self.material, self.physics, self.parts


@dataclass(slots=True)
class CreatorModel:
    """A fully ingested ``creator_model_*`` export triple."""
    manifest: IEModel
    mesh: Mesh | None
    cloud: PointCloud | None


def _vec3(v: Any, default: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> tuple[float, float, float]:
    if v is None:
        return default
    arr = np.asarray(v, dtype=np.float64).reshape(-1)
    if arr.shape[0] != 3:
        raise ValueError(f"expected 3-vector, got {v!r}")
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _material_from_block(block: dict[str, Any] | None, name_fallback: str) -> PBRMaterial:
    if not block:
        return PBRMaterial(name=name_fallback)
    return PBRMaterial(
        name=str(block.get("name") or name_fallback),
        albedo=_vec3(block.get("albedo"), (0.8, 0.8, 0.8)),
        roughness=float(block.get("roughness", 0.7)),
        metallic=float(block.get("metallic", 0.0)),
    )


def load_iemodel(path: str | Path) -> IEModel:
    """Parse an ``iemodel/1`` or ``iemodel/2`` manifest."""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    schema = str(data.get("schema", ""))
    if schema not in _SCHEMAS:
        raise ValueError(f"unsupported iemodel schema: {schema!r} (expected one of {_SCHEMAS})")

    base = p.parent
    mesh_block = data.get("mesh") or None
    cloud_block = data.get("point_cloud") or None
    mesh_path = (base / mesh_block["path"]).resolve() if mesh_block and mesh_block.get("path") else None
    cloud_path = (base / cloud_block["path"]).resolve() if cloud_block and cloud_block.get("path") else None

    # v1 majority material is the default for both schemas.
    name = str(data.get("name") or p.stem.removesuffix(".iemodel"))
    material = _material_from_block(data.get("material"), name)

    # v2 per-part materials: PBR fields → PBRMaterial; physics extras split out.
    materials: dict[str, PBRMaterial] = {}
    material_physics: dict[str, dict[str, float]] = {}
    for mat_name, block in (data.get("materials") or {}).items():
        block = dict(block)
        extras = {}
        for key in ("density_kg_m3", "friction", "restitution"):
            if key in block:
                extras[key] = float(block.pop(key))
        materials[str(mat_name)] = _material_from_block(block, str(mat_name))
        if extras:
            material_physics[str(mat_name)] = extras

    parts = [
        IEModelPart(
            label=str(part.get("label", f"part_{i}")),
            primitive=str(part.get("primitive", "unknown")),
            material=str(part.get("material", "")),
            aabb_min=_vec3(part.get("aabb_min")),
            aabb_max=_vec3(part.get("aabb_max")),
            solid_volume_m3=float(part.get("solid_volume_m3", 0.0)),
        )
        for i, part in enumerate(data.get("parts") or [])
    ]

    return IEModel(
        path=p,
        schema=schema,
        name=name,
        units=str(data.get("units", "meters")),
        up_axis=str(data.get("up_axis", "Y")),
        aabb_min=_vec3(data.get("aabb_min")),
        aabb_max=_vec3(data.get("aabb_max")),
        material=material,
        materials=materials,
        material_physics=material_physics,
        physics=dict(data.get("physics") or {}),
        parts=parts,
        mesh_path=mesh_path,
        point_cloud_path=cloud_path,
        spec=dict(data.get("spec") or {}),
    )


def mesh_from_reconstructed(recon: Any, *, material: PBRMaterial | None = None,
                            name: str | None = None) -> Mesh:
    """Build a Mesh from a 3DCreator ``ReconstructedMesh``.

    Works around W19: ``ReconstructedMesh.indices`` is a FLAT ``(T*3,)``
    uint32 array while ``Mesh.from_reconstructed`` (core/mesh.py, not owned
    here) feeds it to a shape-checked helper and raises. The reshape is
    applied at this call site.
    """
    indices = np.asarray(recon.indices, dtype=np.int64).reshape(-1, 3)
    return Mesh.from_arrays(
        positions=np.asarray(recon.positions, dtype=np.float32),
        indices=indices,
        normals=getattr(recon, "normals", None),
        material=material,
        name=name or getattr(recon, "source", "creator3d") or "creator3d",
    )


def load_creator_triple(path: str | Path) -> CreatorModel:
    """Ingest a ``creator_model_*`` export: manifest + sibling GLB + PLY.

    ``path`` may point at the ``.iemodel.json`` itself or at any sibling of
    the triple (``.glb`` / ``.ply`` / stem) — the manifest is located by
    stem. The manifest's majority material is applied to the mesh when the
    GLB carries no material of its own.
    """
    p = Path(path)
    if p.suffix == ".json" or p.name.endswith(".iemodel.json"):
        manifest_path = p
    else:
        stem = p.stem if p.suffix else p.name
        manifest_path = p.parent / f"{stem}.iemodel.json"
    manifest = load_iemodel(manifest_path)

    mesh: Mesh | None = None
    if manifest.mesh_path is not None and manifest.mesh_path.exists():
        from ironengine_bonafide.assets.loaders.gltf import load_mesh
        mesh = load_mesh(manifest.mesh_path)
        if mesh.material.name == "default" and manifest.material.name != "default":
            mesh = mesh.with_material(manifest.material)

    cloud: PointCloud | None = None
    if manifest.point_cloud_path is not None and manifest.point_cloud_path.exists():
        from ironengine_bonafide.assets.loaders.ply import load_pointcloud
        cloud = load_pointcloud(manifest.point_cloud_path)

    return CreatorModel(manifest=manifest, mesh=mesh, cloud=cloud)
