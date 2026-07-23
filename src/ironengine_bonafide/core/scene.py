"""Scene — flat container of renderable assets + lights + volumes.

The container is intentionally non-hierarchical for v0.1; transforms live
on each asset. A future SceneNode tree slot is reserved but unused — that
indirection adds nothing for the headless render-to-tensor use case.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from ironengine_bonafide.core.light import (
    IBL,
    AreaLight,
    DirectionalLight,
    Light,
    PointLight,
    SpotLight,
)
from ironengine_bonafide.core.mesh import Mesh
from ironengine_bonafide.core.pointcloud import PointCloud
from ironengine_bonafide.core.softbody import DollRig
from ironengine_bonafide.core.volume import Volume

Renderable = Mesh | PointCloud | Volume | DollRig | Light

# Tuple of concrete light classes — narrower than `hasattr("kind")` and
# safe against accidental scene contamination.
_LIGHT_TYPES: tuple[type, ...] = (DirectionalLight, PointLight, SpotLight, AreaLight)


@dataclass(slots=True)
class Background:
    """Scene backdrop, painted by the SkyPass wherever depth is empty.

    ``mode`` is one of:

      * ``"gradient"`` — horizon → zenith blend by ray elevation, with a
        darker ground stop below the horizon (default; pleasant daylight
        instead of the old void-black).
      * ``"solid"``    — flat ``color``.
      * ``"envmap"``   — sample ``scene.ibl`` as an equirect panorama
        (falls back to the gradient when no IBL is set or it fails to load).

    Colors are linear HDR; ``intensity`` scales the result.
    """
    mode: str = "gradient"
    color: tuple[float, float, float] = (0.12, 0.14, 0.18)
    zenith_color: tuple[float, float, float] = (0.22, 0.38, 0.65)
    horizon_color: tuple[float, float, float] = (0.72, 0.76, 0.80)
    ground_color: tuple[float, float, float] = (0.32, 0.30, 0.28)
    intensity: float = 1.0


@dataclass(slots=True)
class Scene:
    meshes: list[Mesh] = field(default_factory=list)
    pointclouds: list[PointCloud] = field(default_factory=list)
    volumes: list[Volume] = field(default_factory=list)
    softbodies: list[DollRig] = field(default_factory=list)
    lights: list[Light] = field(default_factory=list)
    ibl: IBL | None = None
    background: Background | None = field(default_factory=Background)
    name: str = "scene"

    def add(self, item: Renderable) -> Scene:
        if isinstance(item, Mesh):
            self.meshes.append(item)
        elif isinstance(item, PointCloud):
            self.pointclouds.append(item)
        elif isinstance(item, Volume):
            self.volumes.append(item)
        elif isinstance(item, DollRig):
            self.softbodies.append(item)
        elif isinstance(item, IBL):
            self.ibl = item
        elif isinstance(item, Background):
            self.background = item
        elif isinstance(item, _LIGHT_TYPES):
            self.lights.append(item)  # type: ignore[arg-type]
        else:
            raise TypeError(f"Unsupported scene item: {type(item).__name__}")
        return self

    def __iadd__(self, item: Renderable) -> Scene:  # supports `scene += item`
        return self.add(item)

    def renderables(self) -> Iterator[Renderable]:
        yield from self.meshes
        yield from self.pointclouds
        yield from self.volumes
        yield from self.softbodies

    @property
    def is_empty(self) -> bool:
        return not (self.meshes or self.pointclouds or self.volumes or self.softbodies)

    def aabb(self) -> tuple[Any, Any] | None:
        """World AABB across all geometry, or None for an empty scene."""
        import torch
        mins: list[Any] = []
        maxs: list[Any] = []
        for m in self.meshes:
            if m.num_vertices:
                mins.append(m.positions.min(dim=0).values)
                maxs.append(m.positions.max(dim=0).values)
        for p in self.pointclouds:
            if p.num_points:
                lo, hi = p.aabb()
                mins.append(lo); maxs.append(hi)
        if not mins:
            return None
        return torch.stack(mins).min(dim=0).values, torch.stack(maxs).max(dim=0).values

    def to(self, device: str) -> Scene:
        return Scene(
            meshes=[m.to(device) for m in self.meshes],
            pointclouds=[p.to(device) for p in self.pointclouds],
            volumes=self.volumes,           # volumes carry their own grid lazily
            softbodies=self.softbodies,
            lights=self.lights,
            ibl=self.ibl,
            background=self.background,
            name=self.name,
        )
