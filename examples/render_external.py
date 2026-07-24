"""Render every downloaded model in ``external_assets/`` through BonaFide.

For each model the script:

  * loads it through BonaFide's own loaders (PLY mesh / point cloud, OBJ, GLB),
  * builds a procedural-sky HDR equirect (gradient + sun disc) used both as
    the scene IBL and as the ``envmap`` background,
  * auto-frames the camera on the model's bounding box,
  * lights it with a shadow-casting sun aligned to the sky's sun disc,
  * drops a neutral ground plane under the model to catch shadows,
  * renders 1280x720 sRGB PNGs into ``docs/gallery/``.

Run (from the repo root)::

    python examples/render_external.py                # all models
    python examples/render_external.py --only bunny   # just one
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ironengine_bonafide.api import (
    Background, DirectionalLight, Engine, IBL, Mesh, PBRMaterial,
    PerspectiveCamera, PointCloud, RenderConfig, Scene, render,
)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "external_assets"
OUT_DIR = ROOT / "docs" / "gallery"

# Direction FROM the scene TO the sun — shared by the sky IBL sun disc and
# the DirectionalLight (whose `direction` points from the sun down at the scene).
# Azimuth offset from the camera axis so cast shadows read in-frame.
SUN_DIR = np.array([0.80, 0.55, -0.35], dtype=np.float64)
SUN_DIR /= np.linalg.norm(SUN_DIR)


@dataclass
class ModelSpec:
    key: str                      # short id -> gallery filename
    path: Path
    kind: str                     # "ply_mesh" | "ply_points" | "obj" | "glb"
    title: str
    material: PBRMaterial | None = None   # override applied to every primitive
    point_size_px: float = 3.0
    view_dir: tuple[float, float, float] = (1.0, 0.42, 1.0)
    zoom: float = 1.0               # dist multiplier (<1 = tighter crop)


MODELS: list[ModelSpec] = [
    ModelSpec(
        key="bunny",
        path=ASSETS / "stanford_bunny.ply",
        kind="ply_mesh",
        title="Stanford Bunny (PLY mesh)",
        material=PBRMaterial(name="porcelain", albedo=(0.78, 0.74, 0.68),
                             roughness=0.55, metallic=0.0),
    ),
    ModelSpec(
        key="avocado",
        path=ASSETS / "Avocado.glb",
        kind="glb",
        title="Khronos Avocado (GLB)",
        # The glTF relies on a baseColorTexture (unsampled by the passes) so
        # the loader's factor is white — assign a believable avocado skin.
        material=PBRMaterial(name="avocado", albedo=(0.42, 0.52, 0.24),
                             roughness=0.6, metallic=0.0),
    ),
    ModelSpec(
        key="boombox",
        path=ASSETS / "BoomBox.glb",
        kind="glb",
        title="Khronos BoomBox (GLB)",
        # Same texture caveat as Avocado — the real BoomBox body is dark
        # charcoal plastic, which also reads better than texture-less white.
        material=PBRMaterial(name="boombox", albedo=(0.14, 0.13, 0.14),
                             roughness=0.35, metallic=0.05),
        view_dir=(1.0, 0.35, 1.0),
    ),
    ModelSpec(
        key="chest",
        path=ASSETS / "chest_gold.obj",
        kind="obj",
        title="KayKit Golden Chest (OBJ)",
        # OBJ loader ignores .mtl, so give the chest warm bronze wood.
        material=PBRMaterial(name="bronze_wood", albedo=(0.55, 0.38, 0.18),
                             roughness=0.5, metallic=0.35),
    ),
    ModelSpec(
        key="dolphins",
        path=ASSETS / "dolphins_colored.ply",
        kind="ply_points",
        title="Dolphins point cloud (colored ascii PLY)",
        point_size_px=22.0,
        zoom=0.68,
    ),
]


# ---------------------------------------------------------------- sky IBL
def make_sky_ibl(h: int = 128, w: int = 256) -> np.ndarray:
    """Procedural equirect sky: horizon->zenith gradient + warm sun disc.

    Row 0 of the equirect is the zenith, row h-1 the nadir; columns sweep the
    azimuth. The sun disc sits at ``SUN_DIR`` so it lines up with the
    DirectionalLight below.
    """
    elev = np.linspace(np.pi / 2, -np.pi / 2, h, dtype=np.float64)      # +90°..-90°
    azim = np.linspace(-np.pi, np.pi, w, endpoint=False, dtype=np.float64)
    ce, se = np.cos(elev), np.sin(elev)
    ca, sa = np.cos(azim), np.sin(azim)
    # ray directions (h, w, 3), matching core.envmap.equirect_sample's layout
    dirs = np.stack([
        ce[:, None] * ca[None, :],
        np.broadcast_to(se[:, None], (h, w)),
        ce[:, None] * sa[None, :],
    ], axis=-1)

    zenith = np.array([0.20, 0.36, 0.62])
    horizon = np.array([0.72, 0.74, 0.78])
    ground = np.array([0.36, 0.33, 0.30])
    t = np.clip(dirs[..., 1], -1.0, 1.0)                                # sin(elev)
    sky = horizon[None, :] * (1.0 - np.clip(t, 0, 1))[..., None] \
        + zenith[None, :] * np.clip(t, 0, 1)[..., None]
    below = ground[None, :] * (1.0 + np.clip(t, -1, 0))[..., None] \
        + horizon[None, :] * (-np.clip(t, -1, 0))[..., None]
    img = np.where((t >= 0)[..., None], sky, below)

    cosang = dirs @ SUN_DIR
    disc = np.clip((cosang - math.cos(math.radians(2.0)))
                   / (1.0 - math.cos(math.radians(2.0))), 0.0, 1.0)
    halo = np.clip((cosang - 0.85) / 0.15, 0.0, 1.0) ** 2
    img += disc[..., None] * np.array([18.0, 14.0, 9.0])
    img += halo[..., None] * np.array([0.9, 0.6, 0.3])
    return img.astype(np.float32)


# ------------------------------------------------------------ camera frame
def frame_camera(lo: torch.Tensor, hi: torch.Tensor, aspect: float,
                 fov_deg: float = 42.0,
                 view_dir: tuple[float, float, float] = (1.0, 0.42, 1.0),
                 zoom: float = 1.0,
                 ) -> PerspectiveCamera:
    lo_np = lo.cpu().numpy().astype(np.float64)
    hi_np = hi.cpu().numpy().astype(np.float64)
    center = (lo_np + hi_np) / 2.0
    radius = float(np.linalg.norm(hi_np - lo_np) / 2.0)
    radius = max(radius, 1e-6)

    fov_y = math.radians(fov_deg)
    fov_x = 2.0 * math.atan(math.tan(fov_y / 2.0) * aspect)
    dist = radius / math.sin(min(fov_y, fov_x) / 2.0) * 1.18 * zoom

    view_dir = np.asarray(view_dir, dtype=np.float64)
    view_dir /= np.linalg.norm(view_dir)
    pos = center + view_dir * dist
    return PerspectiveCamera(
        position=tuple(pos),
        look_at=tuple(center),
        fov_deg=fov_deg,
        near=max(dist - 3.0 * radius, radius * 1e-3, 1e-5),
        far=dist + 6.0 * radius,
    )


def ground_plane(y: float, radius: float) -> Mesh:
    """Subdivided grid ground — a single huge quad triggers CPU-rasterizer
    striping artifacts at grazing angles; 32x32 tiles raster cleanly."""
    s = radius * 3.5
    n = 32
    xs = np.linspace(-s, s, n + 1, dtype=np.float32)
    pos = np.array([[x, y, z] for z in xs for x in xs], dtype=np.float32)
    nrm = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (len(pos), 1))
    idx: list[list[int]] = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            idx += [[a, a + 1, a + n + 2], [a, a + n + 2, a + n + 1]]
    return Mesh.from_arrays(
        positions=pos,
        indices=np.asarray(idx, dtype=np.int64),
        normals=nrm,
        material=PBRMaterial(name="ground", albedo=(0.34, 0.33, 0.32),
                             roughness=0.9, metallic=0.0),
        name="ground",
    )


# ------------------------------------------------------------------ loaders
def load_spec(spec: ModelSpec) -> tuple[list[Mesh], PointCloud | None]:
    """→ (meshes, pointcloud) for the scene."""
    if spec.kind == "ply_mesh":
        from ironengine_bonafide.assets.loaders.ply import load_mesh
        mesh = load_mesh(spec.path)
    elif spec.kind == "obj":
        mesh = Mesh.from_obj(spec.path)
    elif spec.kind == "glb":
        from ironengine_bonafide.assets.loaders.gltf import load_primitives
        prims = load_primitives(spec.path)
        meshes = [p.mesh for p in prims]
        if spec.material is not None:
            meshes = [m.with_material(spec.material) for m in meshes]
        return meshes, None
    elif spec.kind == "ply_points":
        pc = PointCloud.from_ply(spec.path)
        # Real scans come in arbitrary units (dolphins span ~1000 units);
        # normalize to unit bbox radius so the splat/surfel screen-size
        # heuristics and the shadow/ground setup behave.
        center = pc.positions.mean(dim=0, keepdim=True)
        radius = (pc.positions - center).norm(dim=1).max().clamp(min=1e-9)
        pc.positions = (pc.positions - center) / radius
        pc = pc.with_lod().with_surfels()
        pc.point_size_px = spec.point_size_px
        return [], pc
    else:  # pragma: no cover - guarded by MODELS
        raise ValueError(spec.kind)
    if spec.material is not None:
        mesh = mesh.with_material(spec.material)
    return [mesh], None


# -------------------------------------------------------------------- main
def render_model(engine: Engine, spec: ModelSpec) -> Path:
    meshes, pc = load_spec(spec)

    scene = Scene()
    for m in meshes:
        scene.add(m)
    if pc is not None:
        scene.add(pc)
    aabb = scene.aabb()
    assert aabb is not None, f"{spec.key}: empty scene"
    lo, hi = aabb

    aspect = 1280 / 720
    cam = frame_camera(lo, hi, aspect, view_dir=spec.view_dir, zoom=spec.zoom)

    radius = float(torch.linalg.norm(hi - lo).item() / 2.0)
    scene.add(ground_plane(float(lo[1].item()), radius))
    scene.add(DirectionalLight(direction=tuple(-SUN_DIR), intensity=1.2,
                               cast_shadow=True))
    scene.add(IBL(pixels=make_sky_ibl(), intensity=0.4))
    scene.add(Background(mode="envmap"))

    cfg = RenderConfig(width=1280, height=720, samples=1,
                       output_color_space="sRGB", shadows="csm",
                       exposure=0.95,
                       shadow_map_resolution=1024,
                       shadow_bias_constant=1.5,
                       shadow_bias_slope=2.0)
    out = render(engine, scene, cam, cfg)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"{spec.key}.png"
    out.rgb.save(str(png), display_ready=out.color_space == "sRGB")

    kb = png.stat().st_size / 1024
    stats = (f"{spec.key}: {png.name} {kb:.0f} KiB  "
             f"bbox=[{lo[0]:.3g},{lo[1]:.3g},{lo[2]:.3g}].."
             f"[{hi[0]:.3g},{hi[1]:.3g},{hi[2]:.3g}]")
    if pc is not None:
        stats += f"  points={pc.num_points}"
    else:
        stats += f"  tris={sum(m.num_triangles for m in meshes)}"
    print(stats)
    return png


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", type=str, default=None,
                        help="render only the model with this key")
    args = parser.parse_args()

    engine = Engine.auto()
    print(f"engine backend: {type(engine.backend).__name__}")
    for spec in MODELS:
        if args.only and spec.key != args.only:
            continue
        render_model(engine, spec)


if __name__ == "__main__":
    main()
