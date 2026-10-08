# IronEngine-BonaFide — User Guide

> A complete walkthrough — installation, scenes, cameras, materials, the four
> point-cloud R&D paths, and how to plug into 3DCreator. Every example
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
9A. [Visual Quality — AA, Upscaling, Denoise, SSGI, Bloom, God Rays, Glass, Sky Discs](#9a-visual-quality--aa-upscaling-denoise-ssgi-bloom-god-rays-glass-sky-discs)
10. [Differentiable Rendering](#10-differentiable-rendering)
11. [3DCreator Integration](#11-3dcreator-integration)
12. [Render Bundles](#12-render-bundles)
13. [Profiling](#13-profiling)
14. [CLI](#14-cli)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. Installation

```bash
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

Right-handed Y-up. Forward is `-Z` in eye space (matches 3DCreator).

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

Texture maps are sampled on the **CPU reference path** (the CUDA raster path
skips them and notes `pbr:texture_maps_cpu_path_only` in
`out.skipped_passes`). For file-format coverage — PLY/PCD/OBJ/GLB recipes and
embedded GLB textures — see [Rendering Files](RENDERING_FILES.md).

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

## 8. Volumes, Water, Particles, Soft Bodies

```python
Volume.fog(density=0.02, color=(0.7, 0.78, 0.86))
Volume.from_vdb("clouds.vdb")          # requires [formats] extra
Volume.from_grid(my_density_array, voxel_size=0.1)

DollRig.from_glb("character.glb").as_softbody(stiffness=0.8)
DollRig.from_arrays(particles=verts, edges=edges, stiffness=0.7)
```

Fog volumes render as single-scatter exponential fog (with optional height
falloff). Grid volumes — `Volume.from_grid`, and VDB grids once the
`[formats]` extra loads them — are **raymarched trilinearly** with
front-to-back emission/absorption; scene depth clips the march so geometry
occludes the volume and the volume occludes the sky.

### Water

```python
scene.add(WaterSurface(
    center=(0, 0, 0), half_size=(20, 20),
    wave_amplitude=0.06, wave_length=1.7, wave_speed=1.2, steepness=0.5,
    color=(0.02, 0.10, 0.14), scatter_color=(0.05, 0.22, 0.24),
))
```

`WaterSurface` renders a Gerstner-wave plane: analytic ray/plane hit,
three directional wave components for the surface normal, Schlick fresnel
between sky/IBL **reflection** (plus a sun glint from the first
`DirectionalLight`) and Beer-absorbed **refraction** of whatever is behind
the surface. `WaterSurface.time` advances by `RenderConfig.simulation_dt`
(default 1/60 s) on every `render()` — repeated calls animate the waves
deterministically. The surface writes depth/normals/albedo, so SSGI,
denoise, and fog treat it as scene geometry.

### Particles

```python
ps = ParticleSystem.fountain(
    256, emitter_position=(0, 0.2, 0), emitter_velocity=(0, 2.5, 0),
    lifetime=2.0, color=(1.0, 0.75, 0.35))
scene.add(ps)
```

CPU-simulated point sprites: gravity + drag + lifetime per
`simulation_dt` step, seeded emitter respawn (bit-exact given the same
`RenderConfig.seed`), age-faded depth-tested splats. Positions and
velocities round-trip through render bundles.

---

## 9. Lighting

```python
DirectionalLight(direction=(-0.4,-1,-0.3), color=(1,0.98,0.95), intensity=3)
PointLight(position=(0,2,0), color=(1,0.7,0.4), intensity=10, range=8)
SpotLight(position=(0,2,0), direction=(0,-1,0), inner_deg=20, outer_deg=30)
AreaLight(position=(0,2,0), normal=(0,-1,0), extent=(1,1), intensity=4)
IBL.from_hdr("studio_4k.hdr", intensity=1.2)
```

Directional lights cast shadows; pick the algorithm per render:

```python
RenderConfig(shadows="csm")   # cascaded shadow maps (default): texel-snapped
                              # frusta, world-space slope bias, 3x3 PCF
RenderConfig(shadows="vsm")   # variance shadow maps: blurred depth moments +
                              # Chebyshev test — soft penumbrae, no acne
RenderConfig(shadows="off")
```

`AreaLight` is a real rectangular emitter, not a point in disguise: the
rectangle (`extent`, facing `normal`) is sampled on a 3×3 grid with
inverse-square falloff and one-sided emission — wider extents give visibly
softer, more spread shading.

---

## 9A. Visual Quality — AA, Upscaling, Denoise, SSGI, Bloom, God Rays, Glass, Sky Discs

All of these are **opt-in config fields** (or default-off `Background` flags);
with every default untouched the renderer is bit-identical to before.

### Anti-aliasing

```python
RenderConfig(ssaa=4)          # render 4× larger, area-average down (linear HDR)
RenderConfig(aa="fxaa")       # default: luma-edge post blend after resolve
RenderConfig(aa="taa", taa_alpha=0.1, taa_jitter_frames=8)   # temporal AA
RenderConfig(aa="smaa")       # morphological AA (SMAA family)
```

- `ssaa` (1/2/4) is full-scene supersampling: geometry passes render at
  `ssaa×` resolution and the frame is area-averaged down **before** FXAA /
  bloom / tonemap (correct linear-space resolve). Depth is min-pooled, IDs
  use block-centre nearest.
- `aa="fxaa"` blends each pixel toward its 3×3 average with a weight ∝ the
  local luma contrast (capped 0.5); flat regions pass through bit-exact.
- `aa="taa"` is true temporal AA: the render driver applies a bounded-step
  sub-pixel jitter (8-phase sequence, ~0.35 px radius, ≤0.27 px steps) to
  the camera projection every frame, and the pass **reprojects** the
  history with per-pixel motion vectors (Catmull-Rom sampling — no
  diffusion blur) and **depth-rejects** it where the surface actually
  moved, so moving objects leave no ghost trail while static content
  converges to SSAA-class edges in ~8 frames. `taa_alpha` sets the
  current-frame weight. History resets automatically when the camera or
  resolution changes. Use it for stills and turntable sequences rendered
  through repeated `render()` calls on one `Engine`.
- `aa="smaa"` is a single-pass morphological AA in the SMAA family: luma
  edges are detected, measured by run length, and cross-blended (long
  staircase edges blend strongest, texture detail barely moves). It does
  not use Jimenez's precomputed area textures, so it is not bit-equal to
  reference SMAA.

### Upscaling (FSR 1.0 / DLSS)

```python
RenderConfig(neural_upscale="fsr", upscale_factor=2.0, upscale_sharpness=0.2)
RenderConfig(neural_upscale="dlss")   # needs an NGX bridge DLL, else → FSR
```

With upscaling enabled the engine renders internally at
`1/upscale_factor` of the output resolution and the upscale pass resolves
to full size before tonemapping (sensor outputs — depth/normals/ids/albedo
— are resolved alongside with mode-appropriate filters).

- `"fsr"` is an AMD FSR 1.0–style implementation in pure torch: **EASU**
  (edge-adaptive 16-tap Lanczos2 upsampling, anisotropic along edges) +
  **RCAS** (contrast-adaptive sharpening with anti-halo clamp). Faithful to
  the published algorithm's structure, deterministic, runs on every backend.
- `"dlss"` drives NVIDIA DLSS through a user-supplied bridge DLL pointed to
  by `BONAFIDE_DLSS_DLL` (DLSS is proprietary and cannot be redistributed —
  the bridge must export `bonafide_dlss_upscale(src, dst, sw, sh, dw, dh)`
  over float32 HWC RGB buffers, and a Turing-or-newer NVIDIA GPU must be
  present). When unavailable, the pass **honestly falls back to FSR** and
  records `neural_upscale:dlss_unavailable→fsr` in
  `RenderOutputs.skipped_passes`.
- `BONAFIDE_UPSCALE_WEIGHTS=<path>` overrides both with a trained
  EDSR-style checkpoint.
- `upscale_factor > 1` is mutually exclusive with `ssaa > 1` (both change
  the internal resolution).

### Denoising

```python
RenderConfig(neural_denoise=True)
```

An SVGF-style edge-aware à-trous wavelet filter (3 dilated iterations)
guided by the engine's own GBuffer — colour distance, world-normal
disagreement, and relative depth discontinuity all damp the blur, so noise
collapses while geometric edges stay put. No weights required. When
`BONAFIDE_DENOISE_WEIGHTS` points at a trained checkpoint, the bundled
micro U-Net is used instead.

### Screen-space GI (SSGI-lite)

```python
RenderConfig(neural_relight="ssgi", ssgi_intensity=0.5)
```

A one-bounce diffuse GI approximation: the lit frame is blurred through a
gaussian pyramid and gathered back as indirect irradiance, modulated by
albedo (colourbleed), a normal-hemisphere weight, and a depth-spread
occlusion term. Only geometry pixels receive bounce — the sky never does.
`neural_relight="neural_ibl"` remains roadmap and records a skip note.

### Bloom / glow

```python
RenderConfig(bloom=True, bloom_threshold=1.0, bloom_intensity=0.9, bloom_radius=3)
```

Pixels brighter than `bloom_threshold` (HDR knee) feed a progressive
separable-gaussian pyramid; `bloom_radius` (1–4) widens the halo,
`bloom_intensity` scales it. `bloom_radius=1` + defaults is the legacy
single-pass bloom, bit-for-bit.

### God rays

```python
RenderConfig(god_rays=True, god_ray_intensity=0.6, god_ray_samples=24,
             god_ray_decay=0.95)
```

Screen-space radial blur of the thresholded bright buffer, centred on the
projected sun (first `DirectionalLight`). Self-skips when the sun is behind
the camera or too far off-screen. Runs before bloom so the rays glow.

### Transparency (glass)

```python
RenderConfig(transparency=True)
PBRMaterial(albedo=(0.9, 0.95, 1.0), roughness=0.05, alpha=0.4)   # glass pane
cloud.opacities = torch.full((n,), 0.5)                            # soft splats
```

Two-pass draw: opaque meshes first (depth write), then `alpha < 1` meshes
sorted back-to-front — depth-tested against the opaque buffer, blended, and
**no depth / normals / ids writes** (glass never occludes). GLB
`baseColorFactor` alpha lands on `PBRMaterial.alpha`; point-cloud
`opacities` are honored on the CPU splat path. With `transparency=False`
alpha is ignored (legacy fully-opaque behavior).

### Sun / moon sky discs

```python
Background(sun_disc=True, sun_disc_intensity=40, sun_disc_radius_deg=1.0,
           sun_horizon_glow=0.3,
           moon_disc=True, moon_direction=(0.45, 0.35, -0.82),
           moon_disc_intensity=8)
```

The sun disc tracks the first `DirectionalLight`; the moon disc uses an
explicit direction. Both are HDR emitters painted by the sky pass
(gradient/envmap modes), so the default bloom pass halos them.
`sun_horizon_glow` adds a cheap warm forward-scatter band around the
horizon on the sun side.

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
authored, so the user sees identical framing. It matches the 3DCreator
0.2.0 API, including the optional kwargs:

```python
render_mesh_offscreen(pos, idx, normals, colors, options=opts,
                      wireframe=True)               # edges as 1-px lines
render_mesh_offscreen(pos, idx, normals, colors, options=opts,
                      skeleton=(joints, parents))   # white rig + orange joints
```

Use the engine programmatically:

```python
from ironengine_bonafide.api import PointCloud
cloud = PointCloud.from_generation_result(creator_result)
```

---

## 12. Render Bundles

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

## 13. Profiling

```python
with engine.profile() as prof:
    out = render(engine, scene, cam, cfg)
print(prof.summary())                          # rich-formatted per-pass timings
```

Per-pass `cpu_ms` + (when CUDA is available) `gpu_alloc_mb` deltas, plus a
total. Skipped passes are listed in `out.skipped_passes`.

---

## 14. CLI

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

## 15. Troubleshooting

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
