# IronEngine-BonaFide — API Reference

> Complete public surface. Every class, dataclass, and function with parameter
> tables and example calls. Internal helpers are intentionally omitted — see
> the source under `src/ironengine_bonafide/` for those.

**Conventions**

- Coordinate frame: right-handed, **Y-up**.
- Tensor shapes: PyTorch convention `(N, ...)` with the batch first when present.
- Frames: linear HDR `float32` in `(H, W, 3)`. Helpers convert to sRGB or uint8.

---

## Table of Contents

- [Top-level package](#top-level-package)
- [`Engine`](#engine)
- [`render` / `render_differentiable`](#render--render_differentiable)
- [`RenderConfig` and sub-configs](#renderconfig-and-sub-configs)
- [`RenderOutputs`](#renderoutputs)
- [`Scene`](#scene)
- [Cameras](#cameras)
- [Geometry assets — `Mesh`, `PointCloud`, `Volume`, `DollRig`](#geometry-assets)
- [Lights](#lights)
- [`PBRMaterial`](#pbrmaterial)
- [`AssetLibrary` & `mount_assets`](#assetlibrary--mount_assets)
- [`RenderBundle`](#renderbundle)
- [Color helpers](#color-helpers)
- [Determinism](#determinism)
- [Profiling](#profiling)
- [Backends](#backends)
- [Render passes](#render-passes)
- [Integrations](#integrations)
- [Training](#training)
- [CLI](#cli)

---

## Top-level package

`from ironengine_bonafide import *`

| Symbol                       | Kind        | Description                                       |
|------------------------------|-------------|---------------------------------------------------|
| `__version__`                | str         | Package version                                   |
| `Engine`                     | class       | Stateful engine instance                          |
| `render`                     | function    | Forward render → `RenderOutputs`                  |
| `render_differentiable`      | function    | Autograd render → `RenderOutputs` with `grad_fn`  |
| `Scene`                      | dataclass   | Renderable container                              |
| `PerspectiveCamera`          | dataclass   | Pinhole camera                                    |
| `OrthographicCamera`         | dataclass   | Ortho camera                                      |
| `SensorCamera`               | dataclass   | World-pose pinhole camera                         |
| `PointCloud`                 | dataclass   | Point cloud asset                                 |
| `Mesh`                       | dataclass   | Indexed-triangle mesh                             |
| `PBRMaterial`                | dataclass   | PBR material record                                |
| `Volume`                     | dataclass   | Fog / cloud / smoke densities                      |
| `DollRig`                    | dataclass   | Soft / non-rigid body                              |
| `DirectionalLight`           | dataclass   | Sun / directional light                            |
| `PointLight`                 | dataclass   | Point light                                        |
| `SpotLight`                  | dataclass   | Spot light                                         |
| `AreaLight`                  | dataclass   | Area light                                         |
| `IBL`                        | dataclass   | Image-based light                                  |
| `RenderConfig`               | dataclass   | Every render knob                                  |
| `RenderOutputs`              | dataclass   | RGB + depth + normals + ids + albedo               |
| `RenderBundle`               | dataclass   | Reproducible scene + camera + config snapshot      |
| `mount_assets`               | function    | Folder-mount asset library                         |

---

## `Engine`

`from ironengine_bonafide.api import Engine`

```python
class Engine:
    backend: Backend
    passes:  list[RenderPass]

    @classmethod
    def auto() -> Engine               # cuda → wgpu → cpu
    @classmethod
    def cuda() -> Engine
    @classmethod
    def wgpu() -> Engine
    @classmethod
    def cpu()  -> Engine

    def with_passes(passes: list[RenderPass]) -> Engine
    def profile() -> ContextManager[ProfileReport]    # `with engine.profile() as p:`
```

```python
engine = Engine.auto()
print(engine)                # <Engine backend=<CudaBackend ...> passes=[...]>
```

---

## `render` / `render_differentiable`

```python
def render(
    engine: Engine,
    scene: Scene,
    camera: Camera,
    config: RenderConfig | None = None,
) -> RenderOutputs

def render_differentiable(
    engine: Engine,
    scene: Scene,
    camera: Camera,
    config: RenderConfig | None = None,
) -> RenderOutputs
```

`render` runs under `torch.no_grad()`. `render_differentiable` does not, and
sets `config.differentiable = True` so passes that need a different code path
in autograd mode (e.g. `gsplat` densification, `nvdiffrast` antialias) take it.

---

## `RenderConfig` and sub-configs

```python
@dataclass(slots=True)
class RenderConfig:
    width:  int = 1280
    height: int = 720
    samples: int = 1
    output_dtype:        "uint8"|"float16"|"float32" = "float32"
    output_color_space:  "linear"|"sRGB" = "linear"
    sensor_outputs:      tuple[str, ...] = ("rgb",)   # any of "rgb","depth","normals","ids","albedo"
    device:              "auto"|"cuda"|"wgpu"|"cpu"|"mps" = "auto"
    vram_budget_mb:      float = 4096.0
    seed: int = 0
    shadows: "off"|"csm"|"vsm" = "csm"   # csm = cascaded PCF maps;
                                         # vsm = variance maps (soft penumbra)
    # --- anti-aliasing ---
    aa: "off"|"fxaa"|"taa"|"smaa" = "fxaa"
    ssaa: int = 1                     # full-scene supersample factor 1/2/4;
                                      # renders at ssaa× and area-averages down
                                      # (linear HDR) before post passes
    taa_alpha: float = 0.1            # TAA: weight of the CURRENT frame (0, 1]
    taa_jitter_frames: int = 8        # TAA: Halton jitter period (>= 2)
    # --- bloom / glow ---
    bloom:   bool = True
    bloom_threshold: float = 1.0      # HDR knee; only brighter pixels glow
    bloom_intensity: float = 0.6      # halo strength
    bloom_radius:    int = 1          # gaussian-pyramid levels 1-4 (1 = legacy)
    # --- god rays (opt-in) ---
    god_rays:         bool = False    # screen-space radial blur from the sun
    god_ray_intensity: float = 0.6
    god_ray_samples:  int = 24        # radial taps per pixel (1-128)
    god_ray_decay:    float = 0.95    # per-tap falloff in (0, 1]
    # --- transparency (opt-in) ---
    transparency: bool = False        # two-pass alpha blending for
                                      # PBRMaterial.alpha < 1 and point-cloud
                                      # opacities; depth test, no depth write
    exposure: float = 1.0
    gsplat:     GsplatConfig
    surfels:    SurfelConfig
    lod:        LodConfig
    completion: CompletionConfig
    fog:        FogConfig
    neural_denoise: bool = False      # edge-aware à-trous denoiser (GBuffer-guided);
                                      # uses BONAFIDE_DENOISE_WEIGHTS U-Net if present
    neural_upscale: "none"|"fsr"|"dlss" = "none"
    upscale_factor: float = 2.0       # output px per internal px, [1, 4]; the
                                      # engine renders at 1/factor internally
    upscale_sharpness: float = 0.2    # RCAS sharpening strength [0, 1]
    neural_relight: "none"|"ssgi"|"neural_ibl" = "none"
    ssgi_intensity: float = 0.5       # indirect bounce gain for "ssgi"
    simulation_dt: float = 1/60       # per-render frame step (water waves,
                                      # particle sim), seconds, > 0
    differentiable: bool = False
    profile: bool = False
```

Sub-configs:

```python
@dataclass(slots=True) class GsplatConfig:
    enabled: bool = True
    sigma_scale: float = 1.0
    densify: bool = True
    densify_grad_threshold: float = 0.0002
    sh_degree: int = 3

@dataclass(slots=True) class LodConfig:
    enabled: bool = True
    screen_space_error_px: float = 1.5
    max_chunks_in_vram: int = 256

@dataclass(slots=True) class CompletionConfig:
    enabled: bool = False
    mlp_width: int = 64
    mlp_depth: int = 3
    hash_levels: int = 16

@dataclass(slots=True) class SurfelConfig:
    enabled: bool = True
    radius_factor: float = 1.5

@dataclass(slots=True) class FogConfig:
    enabled: bool = False
    density: float = 0.02
    color: tuple[float, float, float] = (0.7, 0.78, 0.86)
    height_falloff: float = 0.1
```

IO:

```python
RenderConfig.from_dict(d)        → RenderConfig
RenderConfig.from_file(path)     → RenderConfig          # JSON or YAML
cfg.to_dict()                    → dict
cfg.to_file(path)                → None
```

---

## `RenderOutputs`

```python
@dataclass(slots=True)
class RenderOutputs:
    rgb:     _OutputTensor                       # (H, W, 3) linear HDR float32
    depth:   torch.Tensor | None = None          # (H, W) NDC z in [-1, 1], +inf where empty
                                                 # (unproject to linear meters for sensors)
    normals: torch.Tensor | None = None          # (H, W, 3) world-space
    ids:     torch.Tensor | None = None          # (H, W) int64 instance IDs
    albedo:  torch.Tensor | None = None          # (H, W, 3) GBuffer albedo
    profile: ProfileReport | None = None
    skipped_passes: list[str] = []
```

`_OutputTensor` is a `torch.Tensor` subclass with these helpers:

| Method                            | Returns                | Description                              |
|-----------------------------------|------------------------|------------------------------------------|
| `.to_sRGB()`                      | `torch.Tensor`         | Linear → sRGB float32                     |
| `.to_uint8_srgb()`                | `torch.Tensor` (uint8) | Linear → sRGB uint8                       |
| `.to_aces_srgb_uint8(exposure=1)` | `torch.Tensor` (uint8) | Linear → ACES → sRGB uint8                |
| `.save(path, exposure=1.0)`       | `None`                 | Save as PNG (ACES + sRGB)                 |

---

## `Scene`

```python
scene = Scene(name="demo")
scene.add(mesh).add(pointcloud).add(directional_light).add(ibl)
scene += volume                            # __iadd__ supported
for r in scene.renderables(): ...
scene.aabb()                               # (min, max) tensors or None
scene.to(device)                           # move all assets
```

Field summary:

| Field        | Type                |
|--------------|---------------------|
| `meshes`     | `list[Mesh]`        |
| `pointclouds`| `list[PointCloud]`  |
| `volumes`    | `list[Volume]`      |
| `softbodies` | `list[DollRig]`     |
| `lights`     | `list[Light]`       |
| `ibl`        | `IBL | None`        |

---

## Cameras

### `PerspectiveCamera`
```python
PerspectiveCamera(
    position: tuple[float×3] = (0, 1, 3),
    look_at:  tuple[float×3] = (0, 0, 0),
    up:       tuple[float×3] = (0, 1, 0),
    fov_deg:  float = 45.0,
    near:     float = 0.05,
    far:      float = 200.0,
    jitter_ndc: tuple[float, float] = (0.0, 0.0),   # sub-pixel NDC jitter,
)                                                   # driven by TAA; leave at 0
```

### `OrthographicCamera`
```python
OrthographicCamera(position, look_at, up=(0,1,0),
                   half_width=2.0, half_height=2.0, near=0.05, far=200,
                   jitter_ndc=(0.0, 0.0))
```

### `SensorCamera`
```python
SensorCamera(pose: np.ndarray (4,4), fov_deg=60, near=0.05, far=200,
             jitter_ndc=(0.0, 0.0))
```

All three expose:
```python
.view_matrix()          → np.ndarray (4, 4)
.proj_matrix(aspect)    → np.ndarray (4, 4)
.view_proj(aspect)      → np.ndarray (4, 4)
.view_proj_torch(aspect, device="cpu") → torch.Tensor (4, 4) float32
```

---

## Geometry assets

### `PointCloud`
```python
@dataclass(slots=True)
class PointCloud:
    positions:    torch.Tensor              # (N, 3) float32
    colors:       torch.Tensor | None       # (N, 3)
    normals:      torch.Tensor | None       # (N, 3)
    opacities:    torch.Tensor | None       # (N,)
    name:         str = "pointcloud"
    point_size_px: float = 2.0
    use_lod:        bool = False
    use_completion: bool = False
    use_surfels:    bool = False
    use_gsplat:     bool = False
```

Lighting: when `normals` is set, the splat pass pre-shades `colors` at the
vertex stage — Lambert `N·L` accumulated over the scene's directional / point /
spot / area lights (with the same attenuation as mesh shading) plus a `0.25`
ambient term. Without `normals`, raw `colors` are splatted unchanged.

Constructors:

| Method                                      | Description                                         |
|---------------------------------------------|-----------------------------------------------------|
| `PointCloud.from_arrays(positions, colors)` | From numpy / torch arrays                            |
| `PointCloud.from_ply(path)`                 | ASCII or binary little-endian PLY                    |
| `PointCloud.from_pcd(path)`                 | ASCII PCD                                            |
| `PointCloud.from_generation_result(result)` | From `ironengine_3d_creator.GenerationResult`        |

Builders (chainable, return new instance):

```python
.with_lod()         .with_completion()
.with_surfels()     .with_gsplat()
```

### `Mesh`
```python
@dataclass(slots=True)
class Mesh:
    positions:  torch.Tensor (V, 3)
    indices:    torch.Tensor (T, 3) int64
    normals:    torch.Tensor (V, 3) | None
    uvs:        torch.Tensor (V, 2) | None
    colors:     torch.Tensor (V, 3) | None
    material:   PBRMaterial = PBRMaterial()
    name:       str = "mesh"
```

Constructors:

| Method                                       | Description                                 |
|----------------------------------------------|---------------------------------------------|
| `Mesh.from_arrays(positions, indices, ...)`  | From numpy / torch arrays                    |
| `Mesh.from_obj(path)`                        | OBJ (positions / normals / uvs)              |
| `Mesh.from_glb(path)`                        | GLB / glTF — merged primitives, embedded baseColor textures decoded to `albedo_map` (see [Rendering Files](RENDERING_FILES.md)) |
| `Mesh.from_reconstructed(recon)`             | From 3DCreator's `ReconstructedMesh`         |

Builders: `.with_material(mat)`, `.with_colors(c)`.

### `Volume`
```python
Volume.fog(density=0.02, color=(0.7,0.78,0.86), height_falloff=0.0)
Volume.from_vdb(path)                       # requires [formats]
Volume.from_grid(grid: ndarray (D,H,W), origin=(0,0,0), voxel_size=0.1)
```

### `WaterSurface`
```python
WaterSurface(
    center=(0,0,0), normal=(0,1,0), half_size=(20,20),
    wave_amplitude=0.06, wave_length=1.7, wave_speed=1.2, steepness=0.5,
    time=0.0,                              # advances simulation_dt per render
    color=(0.02,0.10,0.14), scatter_color=(0.05,0.22,0.24),
    reflectivity=1.0,
)
```
Gerstner-wave plane: fresnel sky/IBL reflection + sun glint, Beer-absorbed
refraction; writes rgb/depth/normals/albedo.

### `ParticleSystem`
```python
ParticleSystem(positions (N,3), velocities=None, ages=None,
               lifetime=2.0, gravity=(0,-9.8,0), drag=0.05,
               emitter_position=(0,0,0), emitter_spread=0.3,
               emitter_velocity=(0,1.5,0), respawn=True,
               color=(1.0,0.75,0.35), size_px=3.0)
ParticleSystem.from_arrays(positions, velocities=None, **kw)
ParticleSystem.fountain(n=256, **kw)      # clustered at the emitter
```
Deterministic CPU sim (gravity/drag/lifetime, seeded respawn) + age-faded
depth-tested splats.

### `DollRig`
```python
DollRig.from_glb(path, stiffness=0.8)
DollRig.from_arrays(particles, edges, masses=None, stiffness=0.8)
.as_softbody(stiffness=...)                 # update stiffness
```

---

## Lights

```python
DirectionalLight(direction, color=(1,1,1), intensity=3, cast_shadow=True)
PointLight(position, color=(1,1,1), intensity=1.0, range=10)
SpotLight(position, direction, color, intensity, range, inner_deg=20, outer_deg=30)
AreaLight(position, normal, extent=(1,1), color, intensity)
IBL(path=None, pixels=None, intensity=1.0)
IBL.from_hdr(path, intensity=1.0)
```

Each carries a `.kind` property — `"directional"`, `"point"`, `"spot"`,
`"area"`, `"ibl"`.

---

## `PBRMaterial`

```python
@dataclass(slots=True)
class PBRMaterial:
    name:        str = "default"
    albedo:      tuple[float×3] = (0.8, 0.8, 0.8)
    roughness:   float = 0.7
    metallic:    float = 0.0
    ior:         float = 1.45
    emissive:    tuple[float×3] = (0, 0, 0)
    alpha:       float = 1.0          # opacity (glTF baseColorFactor alpha);
                                      # < 1 renders alpha-blended when
                                      # RenderConfig.transparency is on
    albedo_map:                str | None = None
    metallic_roughness_map:    str | None = None
    normal_map:                str | None = None
    ao_map:                    str | None = None
    emissive_map:              str | None = None
    height_map:                str | None = None
    sss_intensity:             float = 0.0
    sss_tint:                  tuple[float×3] = (1, 1, 1)
    two_sided:                 bool = False

    .to_dict()           → dict
    .from_dict(d)        → PBRMaterial
    .lookup(name, lib)   → PBRMaterial            # AssetLibrary lookup
```

Shading: `roughness` / `metallic` / `emissive` are honored by the PBR pass on
both backends via Cook-Torrance GGX (Trowbridge-Reitz D, Smith Schlick-GGX G,
Schlick F with `F0 = mix(0.04, albedo, metallic)`; roughness clamped to
[0.045, 1], `alpha = roughness²`). `emissive` is added after lighting.
`ior` is carried for round-trip parity with sibling engines; the dielectric F0
baseline is fixed at 0.04 (≈ ior 1.45). Map slots (`*_map`) are filesystem
paths (or asset-mount names) sampled by the PBR pass on the CPU path — see
[Rendering Files](RENDERING_FILES.md) for texture-map recipes.

---

## `AssetLibrary` & `mount_assets`

```python
from ironengine_bonafide.api import mount_assets
lib = mount_assets("F:/Arduino/Tiezhu/assets/")

lib.texture("oak_albedo")    → Path | None
lib.mesh("chair.glb")        → Path | None
lib.envmap("studio")         → Path | None
lib.volume("smoke")          → Path | None
lib.scene("demo")            → Path | None
lib.material("oak")          → dict | None
```

Folder convention:
```
<root>/textures/*    .png .jpg .exr .hdr .ktx2
<root>/meshes/*      .obj .glb .gltf .ply .pcd
<root>/envmaps/*     .hdr .exr .ktx2
<root>/volumes/*     .vdb
<root>/scenes/*      .usd / .iesim.json / .iecreator.json
<root>/materials/*   .json
```

---

## `RenderBundle`

```python
from ironengine_bonafide.bundle import RenderBundle

bundle = RenderBundle.capture(scene, camera, config, seed=42)
bundle.save("case.bnf")           # writes case.bnf + case.bnf.npz
RenderBundle.load("case.bnf").reproduce(engine)
```

---

## Color helpers

`from ironengine_bonafide.core.color import (...)`

```python
linear_to_srgb(x: torch.Tensor)          → torch.Tensor
srgb_to_linear(x: torch.Tensor)          → torch.Tensor
to_uint8_srgb(linear: torch.Tensor)      → torch.Tensor   uint8
aces_filmic(x: torch.Tensor)             → torch.Tensor   tonemapped
tonemap_aces_to_srgb_uint8(linear, exposure=1.0) → torch.Tensor uint8
```

---

## Determinism

`from ironengine_bonafide.core.determinism import (...)`

```python
seed_everything(seed: int)         # Python / numpy / torch / cuda
child(name: str, parent_seed: int) → int
torch_generator(seed, device="cpu") → torch.Generator
```

---

## Profiling

`from ironengine_bonafide.core.profile import (...)`

```python
@dataclass(slots=True) class PassTiming(name, cpu_ms, gpu_alloc_mb=0.0)
@dataclass(slots=True) class ProfileReport(timings: list[PassTiming], total_cpu_ms: float)
    .add(t)            .summary() → str (rich table)        .to_dict() → dict
```

---

## Backends

`from ironengine_bonafide.core.backend import Backend, BackendInfo, probe`

```python
@dataclass(slots=True) class BackendInfo(name, device, capabilities, version, notes)

class Backend(ABC):
    info: BackendInfo
    name, device: str
    supports(capability: str)  → bool
    require(capability: str)   → None    # raises BackendCapabilityError
    empty(shape, dtype=float32) → torch.Tensor
    zeros(shape, dtype=float32) → torch.Tensor
    to_device(x: torch.Tensor)  → torch.Tensor
```

`probe()` returns a `BackendDiscovery` dataclass with `cuda_available`,
`wgpu_available`, `torch_cuda`, `torch_mps`, `notes`.

Concrete backends:

| Class           | Module                                            |
|-----------------|---------------------------------------------------|
| `CudaBackend`   | `ironengine_bonafide.backends.cuda.backend`        |
| `WgpuBackend`   | `ironengine_bonafide.backends.wgpu.backend`        |
| `CpuBackend`    | `ironengine_bonafide.backends.cpu.backend`         |

---

## Render passes

`from ironengine_bonafide.passes.base import RenderPass, PassContext, FrameTargets`

```python
class RenderPass(ABC):
    name: str
    def required_capabilities() -> tuple[str, ...]
    def is_active(ctx: PassContext) -> bool
    def run(ctx: PassContext) -> None
```

Built-in passes (all in `ironengine_bonafide.passes.*`):

| Pass               | Module               | Required capability    | Notes |
|--------------------|----------------------|------------------------|-------|
| `SkyPass`          | `sky_pass`           |                        | solid / gradient / envmap backgrounds |
| `CsmShadowPass`    | `shadow`             | `shadow_csm`           | |
| `SoftBodyPass`     | `softbody_pass`      | `warp_xpbd`            | |
| `LodPass`          | `lod_pass`           |                        | |
| `CompletionPass`   | `completion_pass`    | (asks `neural_field`)  | |
| `SplatPass`        | `splat_pass`         | `splat`                | |
| `PbrPass`          | `pbr_pass`           | `raster`               | samples all five glTF texture slots incl. emissive maps |
| `ParticlePass`     | `particle_pass`      | `warp_xpbd`            | roadmap stub |
| `WaterPass`        | `water_pass`         |                        | roadmap stub |
| `VolumetricPass`   | `volumetric_pass`    |                        | uniform fog + trilinear grid raymarch |
| `NeuralRelightPass`| `neural_relight`     |                        | SSGI-lite one-bounce GI |
| `SsaaDownsamplePass`| `aa_pass`           |                        | linear-HDR area resolve |
| `FxaaPass`         | `postprocess`        |                        | |
| `TaaPass`          | `aa_pass`            |                        | Halton jitter + clamped history |
| `SmaaPass`         | `aa_pass`            |                        | morphological (run-length) AA |
| `GodRaysPass`      | `godrays_pass`       |                        | |
| `BloomPass`        | `postprocess`        |                        | |
| `NeuralDenoisePass`| `neural_denoise`     |                        | à-trous GBuffer-guided; U-Net with weights |
| `NeuralUpscalePass`| `neural_upscale`     |                        | FSR 1.0 EASU+RCAS; DLSS bridge hook |
| `TonemapPass`      | `postprocess`        |                        | |

Replace the default graph:

```python
engine = engine.with_passes([CsmShadowPass(), SplatPass(), TonemapPass()])
```

---

## Integrations

### 3DCreator

```python
from ironengine_bonafide.integrations.creator3d import install, uninstall, set_engine
install()                                       # patch 3DCreator's renderer
set_engine(Engine.cpu())                        # override the cached engine
uninstall()                                     # restore via importlib.reload
```

Adapter functions (you can call directly without monkey-patching):

```python
from ironengine_bonafide.integrations.creator3d import (
    render_points_offscreen, render_mesh_offscreen,
)
img_rgba = render_points_offscreen(positions, colors, options)   # (H, W, 4) uint8
img_rgba = render_mesh_offscreen(positions, indices, normals, colors, options)
```

---

## Training

```python
from ironengine_bonafide.training import optimize_gsplat, train_completion_prior
from ironengine_bonafide.training.losses import l1, l2, psnr, ssim

optimize_gsplat(
    cloud: PointCloud, *, target: torch.Tensor, camera: PerspectiveCamera,
    engine: Engine | None = None, iterations: int = 200, lr: float = 5e-3,
    width: int | None = None, height: int | None = None,
) -> PointCloud

train_completion_prior(
    positions: torch.Tensor, colors: torch.Tensor, *,
    width: int = 64, depth: int = 3, iterations: int = 1000, lr: float = 1e-3,
    device: str | torch.device = "cuda",
) -> CompletionPrior

CompletionPrior(positions: torch.Tensor) -> torch.Tensor    # call to predict
```

---

## CLI

```bash
bonafide info                        # show backend probe
bonafide render scene.json --out img.png [--config render.json] [--width 1920] [--height 1080]
bonafide bundle case.bnf  --out img.png
bonafide list-templates
```

`scene.json` schema is documented in [USER_GUIDE.md → CLI](USER_GUIDE.md#15-cli).
