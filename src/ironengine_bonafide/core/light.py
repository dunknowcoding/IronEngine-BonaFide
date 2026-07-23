"""Light primitives + IBL handle.

Lights are pure dataclasses; passes consume them. IBL holds the path /
loaded HDR pixels and lazily prefilters when a backend asks for it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

Vec3 = tuple[float, float, float]


@dataclass(slots=True)
class DirectionalLight:
    direction: Vec3 = (-0.4, -1.0, -0.3)
    color: Vec3 = (1.0, 0.98, 0.95)
    intensity: float = 3.0
    cast_shadow: bool = True

    @property
    def kind(self) -> str:
        return "directional"


@dataclass(slots=True)
class PointLight:
    position: Vec3 = (0.0, 2.0, 0.0)
    color: Vec3 = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    range: float = 10.0

    @property
    def kind(self) -> str:
        return "point"


@dataclass(slots=True)
class SpotLight:
    position: Vec3 = (0.0, 2.0, 0.0)
    direction: Vec3 = (0.0, -1.0, 0.0)
    color: Vec3 = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    range: float = 10.0
    inner_deg: float = 20.0
    outer_deg: float = 30.0

    @property
    def kind(self) -> str:
        return "spot"


@dataclass(slots=True)
class AreaLight:
    position: Vec3 = (0.0, 2.0, 0.0)
    normal: Vec3 = (0.0, -1.0, 0.0)
    extent: tuple[float, float] = (1.0, 1.0)
    color: Vec3 = (1.0, 1.0, 1.0)
    intensity: float = 1.0

    @property
    def kind(self) -> str:
        return "area"


@dataclass(slots=True)
class IBL:
    """Image-based light. Carries either a path to load on demand, or
    pre-loaded HDR pixels (H x W x 3 float32).
    """
    path: Path | None = None
    pixels: np.ndarray | None = None
    intensity: float = 1.0

    @property
    def kind(self) -> str:
        return "ibl"

    @classmethod
    def from_hdr(cls, path: str | Path, intensity: float = 1.0) -> IBL:
        return cls(path=Path(path), intensity=intensity)

    def load(self) -> np.ndarray:
        if self.pixels is not None:
            return self.pixels
        if self.path is None:
            raise ValueError("IBL has neither pixels nor path")
        try:
            import imageio.v3 as iio
            self.pixels = iio.imread(self.path).astype(np.float32)
        except Exception as exc:
            raise RuntimeError(f"Could not load HDR: {self.path}") from exc
        return self.pixels


Light = DirectionalLight | PointLight | SpotLight | AreaLight | IBL


def light_to_dict(lt: Light) -> dict[str, Any]:
    """Serialize a light record (used by render bundles)."""
    base = {"kind": lt.kind, **{k: getattr(lt, k) for k in lt.__slots__
                                if not k.startswith("_") and k not in {"pixels"}}}
    # Coerce Path to str.
    if isinstance(base.get("path"), Path):
        base["path"] = str(base["path"])
    # Tuple → list for JSON.
    for k, v in list(base.items()):
        if isinstance(v, tuple):
            base[k] = list(v)
    return base


def light_from_dict(d: dict[str, Any]) -> Light:
    kind = d.pop("kind")
    cls_map = {
        "directional": DirectionalLight,
        "point":       PointLight,
        "spot":        SpotLight,
        "area":        AreaLight,
        "ibl":         IBL,
    }
    cls = cls_map[kind]
    # Restore tuple fields.
    for k, v in list(d.items()):
        if isinstance(v, list) and k != "extent" or isinstance(v, list) and k == "extent":
            d[k] = tuple(v)
    if "path" in d and d["path"] is not None:
        d["path"] = Path(d["path"])
    return cls(**d)  # type: ignore[arg-type]
