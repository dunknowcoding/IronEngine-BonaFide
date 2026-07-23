"""Asset folder mount.

`mount(path)` walks a directory and indexes:

  textures/*   .png .jpg .jpeg .exr .hdr .ktx2
  meshes/*     .obj .glb .gltf .ply .pcd
  envmaps/*    .hdr .exr .ktx2
  volumes/*    .vdb
  scenes/*     .usd .usda .usdc .iesim.json .iecreator.json
  materials/*  .json   (PBRMaterial dicts)

Convention only — sub-folders are read by extension, not by name. Calling
code asks: `lib.texture("oak_albedo")`, `lib.mesh("chair.glb")`,
`lib.material("oak")`, etc.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ironengine_bonafide.logging import logger

_TEXTURE_EXT = {".png", ".jpg", ".jpeg", ".exr", ".hdr", ".ktx2"}
_MESH_EXT = {".obj", ".glb", ".gltf", ".ply", ".pcd"}
_VOL_EXT = {".vdb"}
_ENV_EXT = {".hdr", ".exr", ".ktx2"}
_SCENE_EXT = {".usd", ".usda", ".usdc", ".json"}


@dataclass(slots=True)
class AssetLibrary:
    root: Path
    textures: dict[str, Path] = field(default_factory=dict)
    meshes: dict[str, Path] = field(default_factory=dict)
    envmaps: dict[str, Path] = field(default_factory=dict)
    volumes: dict[str, Path] = field(default_factory=dict)
    scenes: dict[str, Path] = field(default_factory=dict)
    materials: dict[str, dict[str, Any]] = field(default_factory=dict)

    # ------------------------------------------------------------ getters
    def texture(self, name: str) -> Path | None:
        return self.textures.get(name) or self.textures.get(Path(name).stem)

    def mesh(self, name: str) -> Path | None:
        return self.meshes.get(name) or self.meshes.get(Path(name).stem)

    def envmap(self, name: str) -> Path | None:
        return self.envmaps.get(name) or self.envmaps.get(Path(name).stem)

    def volume(self, name: str) -> Path | None:
        return self.volumes.get(name) or self.volumes.get(Path(name).stem)

    def scene(self, name: str) -> Path | None:
        return self.scenes.get(name) or self.scenes.get(Path(name).stem)

    def material(self, name: str) -> dict[str, Any] | None:
        return self.materials.get(name) or self.materials.get(Path(name).stem)


def mount(root: str | Path) -> AssetLibrary:
    """Index a folder. Re-call to refresh after the user drops new files."""
    root_p = Path(root).expanduser().resolve()
    if not root_p.exists():
        logger.warning(f"asset mount: {root_p} does not exist")
        return AssetLibrary(root=root_p)
    lib = AssetLibrary(root=root_p)
    for path in root_p.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        # full-name + stem keys, so users can ask either way
        keys = (path.name, path.stem)
        if suffix in _TEXTURE_EXT and "envmaps" not in path.parts:
            for k in keys:
                lib.textures.setdefault(k, path)
        if suffix in _MESH_EXT:
            for k in keys:
                lib.meshes.setdefault(k, path)
        if suffix in _ENV_EXT and "envmaps" in path.parts:
            for k in keys:
                lib.envmaps.setdefault(k, path)
        if suffix in _VOL_EXT:
            for k in keys:
                lib.volumes.setdefault(k, path)
        if suffix in _SCENE_EXT:
            for k in keys:
                lib.scenes.setdefault(k, path)
        if suffix == ".json" and "materials" in path.parts:
            try:
                spec = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(spec, dict):
                    for k in keys:
                        lib.materials.setdefault(k, spec)
            except Exception as exc:
                logger.warning(f"asset mount: bad material JSON {path}: {exc}")
    logger.info(
        f"asset mount @ {root_p}: {len(lib.textures)} textures, "
        f"{len(lib.meshes)} meshes, {len(lib.envmaps)} envmaps, "
        f"{len(lib.volumes)} volumes, {len(lib.scenes)} scenes, "
        f"{len(lib.materials)} materials"
    )
    return lib
