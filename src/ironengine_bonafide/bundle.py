"""Render bundles — reproducible scene + camera + config snapshots.

A `.bnf` file is a JSON manifest plus a sibling `.npz` of all tensors. On
reproduce we rebuild the scene, restore tensor data, and re-render with
the same seed → bit-exact within a backend.

Limitations: light pass graphs (custom-registered passes) aren't captured;
re-create them via `Engine.with_passes(...)` after `load`.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ironengine_bonafide.api import Engine, RenderConfig, RenderOutputs, render
from ironengine_bonafide.core.camera import (
    Camera,
    OrthographicCamera,
    PerspectiveCamera,
    SensorCamera,
)
from ironengine_bonafide.core.light import IBL, light_from_dict, light_to_dict
from ironengine_bonafide.core.material import PBRMaterial
from ironengine_bonafide.core.mesh import Mesh
from ironengine_bonafide.core.particles import ParticleSystem
from ironengine_bonafide.core.pointcloud import PointCloud
from ironengine_bonafide.core.scene import Scene
from ironengine_bonafide.core.softbody import DollRig
from ironengine_bonafide.core.volume import Volume
from ironengine_bonafide.core.water import WaterSurface


@dataclass(slots=True)
class RenderBundle:
    scene: Scene
    camera: Camera
    config: RenderConfig
    seed: int = 0
    name: str = "bundle"

    # --------------------------------------------------------- capture
    @classmethod
    def capture(cls, scene: Scene, camera: Camera, config: RenderConfig, seed: int = 0) -> RenderBundle:
        return cls(scene=scene, camera=camera, config=config, seed=seed)

    # --------------------------------------------------------- save / load
    def save(self, path: str | Path) -> None:
        p = Path(path)
        npz_path = p.with_suffix(p.suffix + ".npz")
        manifest = {
            "schema": "bonafide.bundle/1",
            "name": self.name,
            "seed": self.seed,
            "config": self.config.to_dict(),
            "camera": _camera_to_dict(self.camera),
            "scene": _scene_manifest(self.scene, npz_path.name),
        }
        # Tensor payload
        payload: dict[str, np.ndarray] = {}
        for i, m in enumerate(self.scene.meshes):
            payload[f"mesh{i}_positions"] = m.positions.detach().cpu().numpy()
            payload[f"mesh{i}_indices"] = m.indices.detach().cpu().numpy()
            if m.normals is not None:
                payload[f"mesh{i}_normals"] = m.normals.detach().cpu().numpy()
            if m.uvs is not None:
                payload[f"mesh{i}_uvs"] = m.uvs.detach().cpu().numpy()
            if m.colors is not None:
                payload[f"mesh{i}_colors"] = m.colors.detach().cpu().numpy()
        for i, c in enumerate(self.scene.pointclouds):
            payload[f"pc{i}_positions"] = c.positions.detach().cpu().numpy()
            if c.colors is not None:
                payload[f"pc{i}_colors"] = c.colors.detach().cpu().numpy()
            if c.normals is not None:
                payload[f"pc{i}_normals"] = c.normals.detach().cpu().numpy()
            if c.opacities is not None:
                payload[f"pc{i}_opacities"] = c.opacities.detach().cpu().numpy()
        for i, sb in enumerate(self.scene.softbodies):
            payload[f"sb{i}_particles"] = sb.particles.detach().cpu().numpy()
            payload[f"sb{i}_edges"] = sb.edges.detach().cpu().numpy()
            if sb.masses is not None:
                payload[f"sb{i}_masses"] = sb.masses.detach().cpu().numpy()
        for i, v in enumerate(self.scene.volumes):
            if v.grid is not None:
                payload[f"vol{i}_grid"] = v.grid.detach().cpu().numpy()
        for i, ps in enumerate(self.scene.particles):
            payload[f"ps{i}_positions"] = ps.positions.detach().cpu().numpy()
            if ps.velocities is not None:
                payload[f"ps{i}_velocities"] = ps.velocities.detach().cpu().numpy()
            if ps.ages is not None:
                payload[f"ps{i}_ages"] = ps.ages.detach().cpu().numpy()
        if self.scene.ibl is not None and self.scene.ibl.pixels is not None:
            payload["ibl_pixels"] = np.asarray(self.scene.ibl.pixels, dtype=np.float32)
        np.savez_compressed(npz_path, **payload)
        p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> RenderBundle:
        p = Path(path)
        manifest = json.loads(p.read_text(encoding="utf-8"))
        npz_path = p.with_suffix(p.suffix + ".npz")
        data = dict(np.load(npz_path, allow_pickle=False))

        scene = Scene(name=manifest["scene"]["name"])
        for i, mspec in enumerate(manifest["scene"]["meshes"]):
            m = Mesh.from_arrays(
                positions=data[f"mesh{i}_positions"],
                indices=data[f"mesh{i}_indices"],
                normals=data.get(f"mesh{i}_normals"),
                uvs=data.get(f"mesh{i}_uvs"),
                colors=data.get(f"mesh{i}_colors"),
                material=PBRMaterial.from_dict(mspec["material"]),
                name=mspec["name"],
            )
            scene.add(m)
        for i, cspec in enumerate(manifest["scene"]["pointclouds"]):
            c = PointCloud.from_arrays(
                positions=data[f"pc{i}_positions"],
                colors=data.get(f"pc{i}_colors"),
                normals=data.get(f"pc{i}_normals"),
                name=cspec["name"],
            )
            op = data.get(f"pc{i}_opacities")
            if op is not None:
                c.opacities = torch.from_numpy(np.ascontiguousarray(op).astype(np.float32))
            c.point_size_px = float(cspec.get("point_size_px", 2.0))
            c.auto_point_size = bool(cspec.get("auto_point_size", False))
            c.use_lod = bool(cspec.get("use_lod", False))
            c.use_completion = bool(cspec.get("use_completion", False))
            c.use_surfels = bool(cspec.get("use_surfels", False))
            c.use_gsplat = bool(cspec.get("use_gsplat", False))
            scene.add(c)
        for i, vspec in enumerate(manifest["scene"].get("volumes", [])):
            scene.add(_volume_from_spec(i, vspec, data))
        for i, sspec in enumerate(manifest["scene"].get("softbodies", [])):
            rig = DollRig.from_arrays(
                particles=data[f"sb{i}_particles"],
                edges=data[f"sb{i}_edges"],
                masses=data.get(f"sb{i}_masses"),
                stiffness=float(sspec.get("stiffness", 0.8)),
                name=sspec.get("name", "doll"),
            )
            rig.damping = float(sspec.get("damping", 0.05))
            scene.add(rig)
        for wspec in manifest["scene"].get("waters", []):
            scene.add(_water_from_spec(wspec))
        for i, pspec in enumerate(manifest["scene"].get("particles", [])):
            scene.add(_particles_from_spec(i, pspec, data))
        for lspec in manifest["scene"].get("lights", []):
            scene.add(light_from_dict(dict(lspec)))                        # type: ignore[arg-type]
        ibl_spec = manifest["scene"].get("ibl")
        if ibl_spec is not None:
            scene.add(_ibl_from_spec(ibl_spec, data))

        camera = _camera_from_dict(manifest["camera"])
        config = RenderConfig.from_dict(manifest["config"])
        return cls(scene=scene, camera=camera, config=config,
                   seed=manifest.get("seed", 0), name=manifest.get("name", "bundle"))

    # --------------------------------------------------------- reproduce
    def reproduce(self, engine: Engine) -> RenderOutputs:
        cfg = self.config
        cfg.seed = self.seed
        return render(engine, self.scene, self.camera, cfg)


# --------------------------------------------------------------- helpers
def _camera_to_dict(c: Camera) -> dict[str, Any]:
    if isinstance(c, PerspectiveCamera):
        return {"kind": "perspective", **asdict(c)}
    if isinstance(c, OrthographicCamera):
        return {"kind": "orthographic", **asdict(c)}
    if isinstance(c, SensorCamera):
        d = asdict(c)
        d["pose"] = c.pose.tolist()
        return {"kind": "sensor", **d}
    raise TypeError(f"Unsupported camera type: {type(c).__name__}")


def _camera_from_dict(d: dict[str, Any]) -> Camera:
    kind = d.pop("kind")
    if kind == "perspective":
        return PerspectiveCamera(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})
    if kind == "orthographic":
        return OrthographicCamera(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})
    if kind == "sensor":
        d["pose"] = np.asarray(d["pose"], dtype=np.float64)
        return SensorCamera(**d)
    raise ValueError(f"Unknown camera kind: {kind}")


def _scene_manifest(scene: Scene, npz_filename: str) -> dict[str, Any]:
    return {
        "name": scene.name,
        "npz": npz_filename,
        "meshes": [{"name": m.name, "material": m.material.to_dict()} for m in scene.meshes],
        "pointclouds": [{
            "name": c.name,
            "point_size_px": c.point_size_px,
            "auto_point_size": c.auto_point_size,
            "use_lod": c.use_lod,
            "use_completion": c.use_completion,
            "use_surfels": c.use_surfels,
            "use_gsplat": c.use_gsplat,
        } for c in scene.pointclouds],
        "volumes": [_volume_to_spec(v) for v in scene.volumes],
        "waters": [_water_to_spec(w) for w in scene.waters],
        "particles": [_particles_to_spec(ps) for ps in scene.particles],
        "softbodies": [{
            "name": sb.name,
            "stiffness": sb.stiffness,
            "damping": sb.damping,
        } for sb in scene.softbodies],
        "ibl": _ibl_to_spec(scene.ibl) if scene.ibl is not None else None,
        "lights": [light_to_dict(lt) for lt in scene.lights],
    }


def _volume_to_spec(v: Volume) -> dict[str, Any]:
    return {
        "kind": v.kind,
        "density": v.density,
        "color": list(v.color),
        "height_falloff": v.height_falloff,
        "grid_origin": list(v.grid_origin),
        "grid_voxel_size": v.grid_voxel_size,
        "has_grid": v.grid is not None,
    }


def _volume_from_spec(i: int, spec: dict[str, Any], data: dict[str, np.ndarray]) -> Volume:
    """Restore a volume preserving its kind (fog / grid / vdb)."""
    grid = data.get(f"vol{i}_grid") if spec.get("has_grid") else None
    if grid is not None:
        return Volume(
            kind=str(spec.get("kind", "grid")),
            density=float(spec.get("density", 1.0)),
            color=tuple(spec.get("color", (1.0, 1.0, 1.0))),              # type: ignore[arg-type]
            height_falloff=float(spec.get("height_falloff", 0.0)),
            grid=torch.from_numpy(np.ascontiguousarray(grid).astype(np.float32)),
            grid_origin=tuple(spec.get("grid_origin", (0.0, 0.0, 0.0))),  # type: ignore[arg-type]
            grid_voxel_size=float(spec.get("grid_voxel_size", 0.1)),
        )
    return Volume.fog(
        density=float(spec.get("density", 0.02)),
        color=tuple(spec.get("color", (0.7, 0.78, 0.86))),                # type: ignore[arg-type]
        height_falloff=float(spec.get("height_falloff", 0.0)),
    )


def _ibl_to_spec(ibl: IBL) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "intensity": ibl.intensity,
        "has_pixels": ibl.pixels is not None,
    }
    if ibl.path is not None:
        spec["path"] = str(ibl.path)
    return spec


def _water_to_spec(w: WaterSurface) -> dict[str, Any]:
    return {
        "center": list(w.center), "normal": list(w.normal),
        "half_size": list(w.half_size),
        "wave_amplitude": w.wave_amplitude, "wave_length": w.wave_length,
        "wave_speed": w.wave_speed, "steepness": w.steepness, "time": w.time,
        "color": list(w.color), "scatter_color": list(w.scatter_color),
        "reflectivity": w.reflectivity, "name": w.name,
    }


def _water_from_spec(spec: dict[str, Any]) -> WaterSurface:
    return WaterSurface(
        center=tuple(spec.get("center", (0.0, 0.0, 0.0))),  # type: ignore[arg-type]
        normal=tuple(spec.get("normal", (0.0, 1.0, 0.0))),  # type: ignore[arg-type]
        half_size=tuple(spec.get("half_size", (20.0, 20.0))),  # type: ignore[arg-type]
        wave_amplitude=float(spec.get("wave_amplitude", 0.06)),
        wave_length=float(spec.get("wave_length", 1.7)),
        wave_speed=float(spec.get("wave_speed", 1.2)),
        steepness=float(spec.get("steepness", 0.5)),
        time=float(spec.get("time", 0.0)),
        color=tuple(spec.get("color", (0.02, 0.10, 0.14))),  # type: ignore[arg-type]
        scatter_color=tuple(spec.get("scatter_color", (0.05, 0.22, 0.24))),  # type: ignore[arg-type]
        reflectivity=float(spec.get("reflectivity", 1.0)),
        name=str(spec.get("name", "water")),
    )


def _particles_to_spec(ps: ParticleSystem) -> dict[str, Any]:
    return {
        "lifetime": ps.lifetime, "gravity": list(ps.gravity), "drag": ps.drag,
        "emitter_position": list(ps.emitter_position),
        "emitter_spread": ps.emitter_spread,
        "emitter_velocity": list(ps.emitter_velocity),
        "respawn": ps.respawn, "color": list(ps.color),
        "size_px": ps.size_px, "name": ps.name,
    }


def _particles_from_spec(i: int, spec: dict[str, Any],
                         data: dict[str, np.ndarray]) -> ParticleSystem:
    pos = data[f"ps{i}_positions"]
    vel = data.get(f"ps{i}_velocities")
    ps = ParticleSystem.from_arrays(
        pos, vel,
        name=str(spec.get("name", "particles")),
        lifetime=float(spec.get("lifetime", 2.0)),
        gravity=tuple(spec.get("gravity", (0.0, -9.8, 0.0))),  # type: ignore[arg-type]
        drag=float(spec.get("drag", 0.05)),
        emitter_position=tuple(spec.get("emitter_position", (0.0, 0.0, 0.0))),  # type: ignore[arg-type]
        emitter_spread=float(spec.get("emitter_spread", 0.3)),
        emitter_velocity=tuple(spec.get("emitter_velocity", (0.0, 1.5, 0.0))),  # type: ignore[arg-type]
        respawn=bool(spec.get("respawn", True)),
        color=tuple(spec.get("color", (1.0, 0.75, 0.35))),  # type: ignore[arg-type]
        size_px=float(spec.get("size_px", 3.0)),
    )
    ages = data.get(f"ps{i}_ages")
    if ages is not None:
        ps.ages = torch.from_numpy(np.ascontiguousarray(ages).astype(np.float32))
    return ps


def _ibl_from_spec(spec: dict[str, Any], data: dict[str, np.ndarray]) -> IBL:
    pixels = data.get("ibl_pixels") if spec.get("has_pixels") else None
    return IBL(
        path=Path(spec["path"]) if spec.get("path") else None,
        pixels=np.asarray(pixels, dtype=np.float32) if pixels is not None else None,
        intensity=float(spec.get("intensity", 1.0)),
    )
