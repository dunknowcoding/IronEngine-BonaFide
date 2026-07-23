"""PBR material record.

Mirrors the Sim package's `SurfaceMaterial` field set so scenes round-trip
between the two engines without information loss.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

Vec3 = tuple[float, float, float]


@dataclass(slots=True)
class PBRMaterial:
    name: str = "default"
    albedo: Vec3 = (0.8, 0.8, 0.8)
    roughness: float = 0.7
    metallic: float = 0.0
    ior: float = 1.45
    emissive: Vec3 = (0.0, 0.0, 0.0)
    # Map slots — names resolved via the asset mount.
    albedo_map: str | None = None
    metallic_roughness_map: str | None = None
    normal_map: str | None = None
    ao_map: str | None = None
    emissive_map: str | None = None
    height_map: str | None = None
    # Special effects (parity with Sim)
    sss_intensity: float = 0.0
    sss_tint: Vec3 = (1.0, 1.0, 1.0)
    two_sided: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "albedo": list(self.albedo),
            "roughness": self.roughness,
            "metallic": self.metallic,
            "ior": self.ior,
            "emissive": list(self.emissive),
            "albedo_map": self.albedo_map,
            "metallic_roughness_map": self.metallic_roughness_map,
            "normal_map": self.normal_map,
            "ao_map": self.ao_map,
            "emissive_map": self.emissive_map,
            "height_map": self.height_map,
            "sss_intensity": self.sss_intensity,
            "sss_tint": list(self.sss_tint),
            "two_sided": self.two_sided,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PBRMaterial:
        return cls(
            name=d.get("name", "default"),
            albedo=tuple(d.get("albedo", (0.8, 0.8, 0.8))),  # type: ignore[arg-type]
            roughness=float(d.get("roughness", 0.7)),
            metallic=float(d.get("metallic", 0.0)),
            ior=float(d.get("ior", 1.45)),
            emissive=tuple(d.get("emissive", (0.0, 0.0, 0.0))),  # type: ignore[arg-type]
            albedo_map=d.get("albedo_map"),
            metallic_roughness_map=d.get("metallic_roughness_map"),
            normal_map=d.get("normal_map"),
            ao_map=d.get("ao_map"),
            emissive_map=d.get("emissive_map"),
            height_map=d.get("height_map"),
            sss_intensity=float(d.get("sss_intensity", 0.0)),
            sss_tint=tuple(d.get("sss_tint", (1.0, 1.0, 1.0))),  # type: ignore[arg-type]
            two_sided=bool(d.get("two_sided", False)),
        )

    @classmethod
    def lookup(cls, name: str, lib: Any) -> PBRMaterial:
        """Look the material up in an `AssetLibrary` (assets.mount)."""
        spec = lib.material(name)
        if spec is None:
            return cls(name=name)
        return cls.from_dict(spec)
