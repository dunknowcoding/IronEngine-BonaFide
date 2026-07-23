# IronEngine-BonaFide — User Guide

> A complete walkthrough — installation, scenes, cameras, materials, the four
> point-cloud R&D paths, and how to plug into 3DCreator and Sim. Every example
> here is testable code.

---

## Table of Contents

1. [Installation](#1-installation)
2. [Hello, BonaFide](#2-hello-bonafide)
3. [Concepts](#3-concepts)
4. [Backends](#4-backends)
5. [Scenes & Cameras](#5-scenes--cameras)
6. [Point Clouds — the four-prong R&D](#6-point-clouds--the-four-prong-rd)
7. [Meshes & Materials](#7-meshes--materials)
8. [Volumes, Particles, Soft Bodies](#8-volumes-particles-soft-bodies)
9. [Lighting](#9-lighting)
10. [Differentiable Rendering](#10-differentiable-rendering)
11. [3DCreator Integration](#11-3dcreator-integration)
12. [IronEngine-Sim Integration](#12-ironengine-sim-integration)
13. [Render Bundles](#13-render-bundles)
14. [Profiling](#14-profiling)
15. [CLI](#15-cli)
16. [Troubleshooting](#16-troubleshooting)

---

## 1. Installation

```bash
conda activate IronEngineWorld
pip install -e .[all]
```

Extras:

| Extra        | Pulls                                               | When                                              |
|--------------|-----------------------------------------------------|---------------------------------------------------|
| `[cuda]`     | `cupy-cuda12x`, `warp-lang`, `gsplat`, `nvdiffrast` | NVIDIA GPU (recommended)                          |
| `[wgpu]`     | `wgpu`                                              | AMD / Intel / Apple GPU                            |
| `[formats]`  | `pyktx`, `openvdb`, `usd-core`, OpenEXR             | KTX2 / VDB / USD asset support                     |
| `[viewers]`  | `rerun-sdk`, `polyscope`                            | Optional viewers                                   |
| `[dev]`      | `pytest`, `pyright`, `ruff`                         | Development                                        |

Sanity check:

```python
import ironengine_bonafide
print(ironengine_bonafide.__version__)         # → "0.1.0"

from ironengine_bonafide.api import Engine
print(Engine.auto())                            # selects best available
```

---

## 2. Hello, BonaFide

```python
from ironengine_bonafide.api import (
    Engine, Scene, PointCloud, PerspectiveCamera, RenderConfig, render
)

engine = Engine.auto()
scene  = Scene().add(PointCloud.from_ply("scan.ply").with_lod().with_surfels())
cam    = PerspectiveCamera(position=(2, 1.5, 2), look_at=(0, 0.5, 0), fov_deg=45)
out    = render(engine, scene, cam, RenderConfig(width=1280, height=720,
                                                  output_color_space="sRGB"))
out.rgb.save("preview.png")
```

`out.rgb` is a `torch.Tensor` subclass with `.save()`, `.to_uint8_srgb()`,
`.to_aces_srgb_uint8()`, and `.to_sRGB()` helpers.

---

## 3. Concepts

```
┌──────────────────────────────── Scene ────────────────────────────────┐
│  meshes   pointclouds   volumes   softbodies   lights   ibl           │
└───────────────────────────────────────────────────────────────────────┘
                ▲                                            │
                │ render(engine, scene, cam, cfg)            │
                │                                            ▼
              Engine ──▶ Backend ──▶ Pass graph ──▶ FrameTargets
              (auto)     (cuda |     (shadow,        ↳ rgb / depth / normals
                          wgpu |      splat, pbr,      / ids / albedo
                          cpu)        post FX, …)
```

| Concept          | What it is                                                  |
|------------------|-------------------------------------------------------------|
| **Engine**       | Owns a backend + a configurable pass list                   |
| **Backend**      | CUDA / WGPU / CPU — declares its capabilities up-front       |
| **Pass**         | One step (shadow, splat, pbr, denoise, …) — capability-gated |
| **Scene**        | Flat container of renderable assets + lights + ibl           |
| **RenderConfig** | Single dataclass with every knob                              |
| **RenderOutputs**| RGB + depth + normals + ids + albedo as `torch.Tensor`s      |

---

## 4. Backends

Auto-selection picks **cuda → wgpu → cpu**:

```python
Engine.auto()                  # smart
Engine.cuda()                  # force NVIDIA path
Engine.wgpu()                  # force portable path (AMD / Intel / Apple)
Engine.cpu()                   # force CPU reference (CI / dev)
```

Each backend declares **capabilities** like `"raster"`, `"gsplat"`, `"warp_xpbd"`.
Passes ask `backend.supports("gsplat")` and degrade cleanly when it's missing.

```python
print(engine.backend.info)
# BackendInfo(name='cuda', device='cuda:0',
#             capabilities=frozenset({'raster','splat','gsplat','nvdiffrast',
#                                     'warp_xpbd', ...}),
#             version='12.4', notes='...')
```

---

## 5. Scenes & Cameras

```python
scene = (
    Scene()
      .add(PointCloud.from_ply("cloud.ply"))
      .add(Mesh.from_glb("model.glb").with_material(PBRMaterial(albedo=(0.8,0.5,0.2))))
      .add(DirectionalLight(direction=(-0.4,-1,-0.3), intensity=3))
      .add(IBL.from_hdr("studio.hdr"))
      .add(Volume.fog(density=0.02, color=(0.7,0.78,0.86)))
)

PerspectiveCamera(position=(2,1.5,2), look_at=(0,0.5,0), fov_deg=45)
OrthographicCamera(position=(0,5,0), look_at=(0,0,0), half_width=2, half_height=2)
SensorCamera(pose=np.eye(4), fov_deg=60.0)
```

Right-handed Y-up. Forward is `-Z` in eye space (matches Sim & 3DCreator).

---

## 6. Point Clouds — the four-prong R&D

```python
cloud = PointCloud.from_ply("scan.ply")        \
            .with_lod()                         \
            .with_surfels()                     \
            .with_completion()                  \
            .with_gsplat()
```

| Builder              | What it enables                                                                                 |
|----------------------|-------------------------------------------------------------------------------------------------|
| `.with_lod()`        | Octree LOD streaming. Per-frame visibility selects nodes by screen-space error.                 |
| `.with_surfels()`    | Each kept point becomes an oriented disk sized by k-NN spacing — seam-free dense clouds.        |
| `.with_completion()` | Trains a small hash-grid + MLP from dense regions, fills holes inside detected gaps.            |
| `.with_gsplat()`     | Differentiable 3D Gaussian Splatting via the `gsplat` library on CUDA backends.                 |

All four interoperate; toggle individually in `RenderConfig`:

```python
cfg = RenderConfig(
    gsplat=GsplatConfig(enabled=True, sigma_scale=1.0, densify=True),
    surfels=SurfelConfig(enabled=True, radius_factor=1.5),
    lod=LodConfig(enabled=True, screen_space_error_px=1.5),
    completion=CompletionConfig(enabled=True, mlp_width=64, mlp_depth=3),
)
```

---

## 7. Meshes & Materials

```python
mesh = Mesh.from_glb("model.glb").with_material(PBRMaterial(
    albedo=(0.85, 0.55, 0.30),
    roughness=0.45,
    metallic=0.10,
    normal_map="oak_normal",          # resolved against asset library
    albedo_map="oak_albedo",
    metallic_roughness_map="oak_mra",
    sss_intensity=0.0,
    two_sided=False,
))
```

CUDA path: `nvdiffrast` deferred shading with PBR + IBL. CPU path: barycentric
raster into a GBuffer, then Cook-Torrance GGX shading. Differentiable in either
case (CPU gradients flow through `colors`, CUDA gradients flow through
`positions` + `colors` + materials).

The PBR pass honors the scalar material fields on **both** backends:

- `roughness` (clamped to [0.045, 1], `alpha = roughness²`) drives the
  GGX/Trowbridge-Reitz normal distribution and Smith Schlick-GGX geometry term.
- `metallic` blends the Fresnel base reflectance `F0 = mix(0.04, albedo,
  metallic)` (0.04 is the dielectric baseline for `ior ≈ 1.45`) and scales the
  diffuse term by `(1 - metallic)` for energy conservation.
- `emissive` is added to the shaded result after lighting — it glows even with
  no lights in the scene.
- Ambient is a hemisphere model: `albedo · mix(ground, sky, 0.5 + 0.5·n.y) ·
  0.25`, so up-facing surfaces pick up the sky tint and down-facing surfaces
  the ground bounce.

Point clouds (`PointCloud` splats) are lit too: when the cloud carries
per-point `normals`, vertex colors are pre-shaded with Lambert `N·L` per scene
light plus a `0.25` ambient floor before splatting. Clouds without normals
keep their raw colors.

---

## 8. Volumes, Particles, Soft Bodies

```python
Volume.fog(density=0.02, color=(0.7, 0.78, 0.86))
Volume.from_vdb("clouds.vdb")          # requires [formats] extra
Volume.from_grid(my_density_array, voxel_size=0.1)

DollRig.from_glb("character.glb").as_softbody(stiffness=0.8)
DollRig.from_arrays(particles=verts, edges=edges, stiffness=0.7)
```

Particles + fluids land via NVIDIA Warp (XPBD / FLIP). Pass slots ship in v0.1
and gracefully skip when `warp-lang` isn't installed.

---

## 9. Lighting

```python
DirectionalLight(direction=(-0.4,-1,-0.3), color=(1,0.98,0.95), intensity=3)
PointLight(position=(0,2,0), color=(1,0.7,0.4), intensity=10, range=8)
SpotLight(position=(0,2,0), direction=(0,-1,0), inner_deg=20, outer_deg=30)
AreaLight(position=(0,2,0), normal=(0,-1,0), extent=(1,1), intensity=4)
IBL.from_hdr("studio_4k.hdr", intensity=1.2)
```

---

## 10. Differentiable Rendering

```python
from ironengine_bonafide.api import render_differentiable
from ironengine_bonafide.training.losses import l2

cloud.colors = cloud.colors.requires_grad_(True)
opt = torch.optim.Adam([cloud.colors], lr=1e-2)

for _ in range(200):
    out = render_differentiable(engine, scene, cam, cfg)
    loss = l2(out.rgb, target)
    opt.zero_grad(); loss.backward(); opt.step()
```

Helpers:

```python
from ironengine_bonafide.training import optimize_gsplat, train_completion_prior
optimize_gsplat(cloud, target=tgt, camera=cam, iterations=200)
prior = train_completion_prior(cloud.positions, cloud.colors, iterations=1000)
```

---

## 11. 3DCreator Integration

One-line install:

```python
from ironengine_bonafide.integrations.creator3d import install
install()
# 3DCreator's UI now renders through BonaFide. No 3DCreator code changed.
```

Behind the scenes, the shim monkey-patches:

```
ironengine_3d_creator.rendering.api.render_points_offscreen → BonaFide
ironengine_3d_creator.rendering.api.render_mesh_offscreen   → BonaFide
```

The shim mirrors the orbit-yaw-pitch-distance preview math 3DCreator's UI
authored, so the user sees identical framing.

Use the engine programmatically:

```python
from ironengine_bonafide.api import PointCloud
cloud = PointCloud.from_generation_result(creator_result)
```

---

## 12. IronEngine-Sim Integration

```python
from ironengine_bonafide.integrations.sim import install
install()                                      # patches RenderWorld
```

Or, when you've built the World programmatically:

```python
from ironengine_bonafide.integrations.sim import install_for_world
install_for_world(world)
```

Patched methods:
`RenderWorld.render_viewport`, `.render_sensor_rgb`, `.render_sensor_depth`.

Bridging semantics:

- **Transforms** — each entity's full TRS (`Transform.position`, xyzw
  `rotation` quaternion, `scale`) is baked into a transformed copy of the mesh
  / point-cloud geometry. Baked geometry is cached per `(asset, matrix)`, so
  static scenes don't re-transform every frame.
- **Point clouds** — Sim's `PointCloudAsset` component (`cloud_name`,
  `point_size`, `default_color`) maps to a BonaFide `PointCloud` resolved via
  `world.assets.get_point_cloud(name)`, with the same TRS bake applied.
- **Lights** — directional and point lights map 1:1; Sim **spot** lights are
  approximated as point lights (cone shaping is dropped).
- **Sensor depth** — `render_sensor_depth` returns **linear eye-space
  meters**, unprojected from the rasterizer's NDC z via the camera near/far
  (`2·near·far / (far + near − z·(far − near))`). Empty pixels read as `far`.

---

## 13. Render Bundles

Reproducibility-friendly snapshots of (scene + camera + config + seed):

```python
from ironengine_bonafide.bundle import RenderBundle
bundle = RenderBundle.capture(scene, cam, cfg, seed=42)
bundle.save("case.bnf")

# later, anywhere:
RenderBundle.load("case.bnf").reproduce(engine)
```

Bundles round-trip every dataclass tensor field via a sibling `.bnf.npz`
payload.

---

## 14. Profiling

```python
with engine.profile() as prof:
    out = render(engine, scene, cam, cfg)
print(prof.summary())                          # rich-formatted per-pass timings
```

Per-pass `cpu_ms` + (when CUDA is available) `gpu_alloc_mb` deltas, plus a
total. Skipped passes are listed in `out.skipped_passes`.

---

## 15. CLI

```bash
bonafide info                                  # show backend probe
bonafide render scene.json --out img.png       # JSON → PNG
bonafide render scene.json --out img.png --config render.json
bonafide bundle case.bnf  --out img.png        # reproduce a bundle
bonafide list-templates                        # bundled examples
```

`scene.json` schema:

```json
{
  "name": "demo",
  "pointclouds": [{"path": "scan.ply", "lod": true, "surfels": true}],
  "meshes":      [{"path": "model.glb"}],
  "lights":      [{"kind": "directional",
                   "direction": [-0.4, -1, -0.3], "intensity": 3.0}],
  "camera":      {"position": [2, 1.5, 2], "look_at": [0, 0.5, 0], "fov_deg": 45}
}
```

---

## 16. Troubleshooting

| Symptom                                                  | Likely cause / fix                                                                                  |
|----------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| `RuntimeError: gsplat not installed`                     | `pip install -e .[cuda]`                                                                             |
| `RuntimeError: nvdiffrast not installed`                 | Same — both ship in `[cuda]`                                                                         |
| `Engine.auto()` lands on CPU on a machine with NVIDIA GPU | Check `bonafide info` — likely cupy/gsplat/nvdiffrast aren't installed for the active CUDA toolkit  |
| Output is black                                          | Camera placement — try `look_at=(0,0,0)` and a wider FOV; also confirm the scene isn't empty        |
| Output is uniform fog color                              | `cfg.fog.enabled=True` and density too high; lower `fog.density` or disable                          |
| Differentiable render produces no gradient               | Check `requires_grad=True` on the optimized field, and that you used `render_differentiable`         |
| `BackendCapabilityError: backend 'cpu' does not support 'gsplat'` | Set `cloud.use_gsplat=False` for CPU testing, or run on the CUDA backend             |
| 3DCreator UI shows no change after `install()`           | Call `install()` *before* the first `render_*_offscreen` call; if it ran already, restart the app   |
