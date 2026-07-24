# Rendering Files — point clouds & external 3D models

How to turn a file on disk into a rendered image with BonaFide. Every recipe
on this page was executed headless (CPU backend, `output_color_space="sRGB"`,
saved to PNG) against the current code before it was documented.

Related: [User Guide](USER_GUIDE.md) · [API Reference](API_REFERENCE.md)

---

## Supported formats

| Format | Loads as | Entry point | Notes |
|---|---|---|---|
| `.ply` | `PointCloud` | `PointCloud.from_ply(path)` | ASCII and `binary_little_endian`; vertex colors from `red`/`green`/`blue` properties |
| `.ply` | `Mesh` | `assets.loaders.ply.load_mesh(path)` | ASCII only; needs `vertex` + `face` elements; faces fan-triangulated; optional vertex colors |
| `.pcd` | `PointCloud` | `PointCloud.from_pcd(path)` | **ASCII only** — binary PCD raises `ValueError`; packed `rgb` float field decoded |
| `.obj` | `Mesh` | `Mesh.from_obj(path)` | `v` / `vn` / `vt` / `f`; faces fan-triangulated; **`.mtl` materials are not read** — assign a `PBRMaterial` in code |
| `.glb` / `.gltf` | `Mesh` | `Mesh.from_glb(path)` | Node world transforms baked in, interleaved `byteStride` buffers, multi-buffer, `data:` URIs, normalized `COLOR_0`; **embedded baseColor textures decoded** (see below) |
| `.hdr` / `.exr` | IBL environment | `IBL.from_hdr(path)` | Equirect panorama for image-based lighting and `envmap` backgrounds |
| `.png` / `.jpg` | Texture maps | `PBRMaterial(albedo_map="...")` | Any image format `imageio` reads; see [texture maps](#texture-maps) |
| `.iemodel.json` | `Mesh` / `PointCloud` | 3DCreator manifest sidecar | Schema versions `iemodel/1` and `iemodel/2` — see the API reference |
| KTX2 / VDB / USD | — | `[formats]` extra | Behind `pip install -e .[formats]` (`pyktx`, `openvdb`, `usd-core`) |

No FBX — convert to GLB first.

---

## The loader API

Two layers:

**Convenience classmethods** (what most code wants):

```python
from ironengine_bonafide.api import Mesh, PointCloud

cloud = PointCloud.from_ply("scan.ply")
cloud = PointCloud.from_pcd("scan.pcd")
mesh  = Mesh.from_obj("model.obj")
mesh  = Mesh.from_glb("model.glb")      # handles .gltf (JSON) files too
```

**Loader modules** (when you need control):

```python
from ironengine_bonafide.assets.loaders.gltf import load_mesh, load_primitives, load_rig
from ironengine_bonafide.assets.loaders.ply import load_pointcloud, load_mesh
from ironengine_bonafide.assets.loaders.obj import load_mesh
from ironengine_bonafide.assets.loaders.pcd import load_pointcloud
```

The important one is glTF's **`load_primitives()`**: `Mesh.from_glb()` /
`gltf.load_mesh()` *merge* all primitives into a single `Mesh` and the **first
primitive's material wins** (baseColor alpha is dropped). `load_primitives()`
returns one `GltfPrimitive` per primitive — each with its own world-space
`Mesh`, its own material, plus `alpha` and `double_sided` fields that
`PBRMaterial` cannot represent yet:

```python
from ironengine_bonafide.assets.loaders.gltf import load_primitives

prims = load_primitives("scene.glb")
for p in prims:
    print(p.mesh.material.name, p.mesh.material.albedo_map, p.alpha)
```

Custom formats can register into `LOADER_REGISTRY` (keyed by extension) from
`ironengine_bonafide.api`.

---

## Camera & light setup

Everything below uses the same rig:

```python
from ironengine_bonafide.api import (
    Background, DirectionalLight, Engine, IBL,
    PerspectiveCamera, RenderConfig, Scene, render,
)

engine = Engine.auto()          # cuda → wgpu → cpu fallback; Engine.cpu() to pin

cam = PerspectiveCamera(
    position=(2.5, 1.8, 2.5),   # right-handed, Y-up
    look_at=(0, 0, 0),
    fov_deg=45,
)

cfg = RenderConfig(width=1280, height=720, output_color_space="sRGB", samples=1)
```

Lights are added to the scene like geometry:

```python
sun = DirectionalLight(direction=(-0.4, -1.0, -0.3), intensity=3.0)
scene = Scene().add(mesh_or_cloud).add(sun)
```

Image-based lighting — from an `.hdr`/`.exr` file or from in-memory pixels —
plus an `envmap` background that shows the same panorama:

```python
scene = (Scene()
         .add(mesh)
         .add(IBL.from_hdr("envs/studio.hdr", intensity=1.0))   # or IBL(pixels=hdr_array)
         .add(Background(mode="envmap")))
```

Saving: with `output_color_space="sRGB"` the tensor is already display-ready,
so pass `display_ready=True` (no second ACES conversion):

```python
out = render(engine, scene, cam, cfg)
out.rgb.save("preview.png", display_ready=out.color_space == "sRGB")
```

---

## Recipes

### Point cloud from `.ply`

```python
from ironengine_bonafide.api import (
    DirectionalLight, Engine, PerspectiveCamera, PointCloud, RenderConfig,
    Scene, render,
)

engine = Engine.auto()
scene = (Scene()
         .add(PointCloud.from_ply("cloud.ply"))
         .add(DirectionalLight(direction=(-0.4, -1.0, -0.3), intensity=3.0)))
cam = PerspectiveCamera(position=(2.5, 1.8, 2.5), look_at=(0, 0, 0), fov_deg=45)
out = render(engine, scene, cam,
             RenderConfig(width=640, height=360, output_color_space="sRGB", samples=1))
out.rgb.save("cloud.png", display_ready=True)
```

Vertex colors are picked up automatically when the PLY carries
`red`/`green`/`blue` properties. Add `.with_lod().with_surfels()` after
`from_ply(...)` for large scans (see `examples/01_render_pointcloud.py`).

### Point cloud from `.pcd`

Same shape, different loader:

```python
scene = (Scene()
         .add(PointCloud.from_pcd("cloud.pcd"))
         .add(DirectionalLight(direction=(-0.4, -1.0, -0.3), intensity=3.0)))
out = render(engine, scene, cam, cfg)
out.rgb.save("cloud.png", display_ready=True)
```

The packed PCD `rgb` float field is decoded to per-point colors. Binary PCD
(`DATA binary` / `binary_compressed`) is **not** supported — convert to ASCII
or PLY.

### Mesh from `.obj`

```python
from ironengine_bonafide.api import Mesh, PBRMaterial

mesh = Mesh.from_obj("quad.obj").with_material(
    PBRMaterial(albedo=(0.85, 0.55, 0.3), roughness=0.45, metallic=0.0))
scene = (Scene().add(mesh)
         .add(DirectionalLight(direction=(0.0, 0.0, -1.0), intensity=3.0)))
out = render(engine, scene, cam, cfg)
out.rgb.save("mesh.png", display_ready=True)
```

`.mtl` files are ignored — materials always come from `with_material(...)` or
the default `PBRMaterial()`.

### Mesh from `.glb` (with embedded textures)

```python
mesh = Mesh.from_glb("model.glb")      # materials + embedded baseColor textures ride along
scene = (Scene().add(mesh)
         .add(DirectionalLight(direction=(0.0, 0.0, -1.0), intensity=3.0)))
out = render(engine, scene, cam, cfg)
out.rgb.save("model.png", display_ready=True)
```

**Embedded texture support.** When a glTF material has a
`baseColorTexture`, the loader resolves it to `PBRMaterial.albedo_map`:

- images embedded in the GLB binary chunk (`bufferView` images) are decoded to
  a deterministic temp-cache file (`<tempdir>/ironengine_bonafide_glb_textures/`,
  keyed by GLB path + image index + content hash);
- `data:` URI images are decoded the same way;
- relative-URI images resolve against the GLB's own folder.

`baseColorFactor` still multiplies the texture (per the glTF spec), and
`COLOR_0` vertex colors multiply both. glTF's default white factor is honored,
so textured assets are *not* darkened by a gray default.

For multi-material GLBs, use `load_primitives()` (above) — the merged
`Mesh.from_glb()` keeps only the first primitive's material.

### Mesh from `.gltf` (JSON)

`Mesh.from_glb()` also accepts plain `.gltf` JSON files, including external
buffer files and `data:` URI buffers:

```python
mesh = Mesh.from_glb("model.gltf")
```

### Texture maps supplied through the API

Any mesh with UVs can bind texture files directly — this is also how you
attach maps the GLB loader does not resolve (normal, metallic-roughness, AO):

```python
mesh = Mesh.from_obj("quad.obj").with_material(PBRMaterial(
    albedo=(1.0, 1.0, 1.0),          # multiplies the map
    roughness=0.8,
    albedo_map="textures/checker.png",
    # normal_map=..., metallic_roughness_map=..., ao_map=...
))
```

Map references are filesystem paths (or names resolvable via
`mount_assets(...)`). Color maps (`albedo_map`) are decoded sRGB → linear at
load; data maps are used as-is. Sampling is bilinear with repeat wrapping.

### IBL-lit render with environment background

```python
import numpy as np
from ironengine_bonafide.api import Background, IBL

sky = np.zeros((32, 64, 3), dtype=np.float32)   # equirect HDR pixels
sky[:16, :, 2] = 0.9                            # blue-ish dome
sky[16:, :] = (0.35, 0.30, 0.25)                # ground bounce

scene = (Scene()
         .add(mesh)
         .add(IBL(pixels=sky, intensity=1.0))
         .add(Background(mode="envmap")))
out = render(engine, scene, cam, cfg)
out.rgb.save("ibl.png", display_ready=True)
```

`IBL.from_hdr("file.hdr")` is the file-backed equivalent.

---

## Current limitations (precise)

- **Texture maps sample on the CPU reference path only.** On the CUDA
  (nvdiffrast) raster path maps are skipped and `pbr:texture_maps_cpu_path_only`
  appears in `out.skipped_passes`; shading falls back to material/vertex colors.
- **GLB loader resolves `baseColorTexture` only.** `normalTexture`,
  `metallicRoughnessTexture`, `occlusionTexture`, and `emissiveTexture`
  references inside GLBs are ignored — attach those maps via
  `PBRMaterial(...)` map slots instead.
- **KTX2 / Basis-compressed images are not decoded** (embedded or otherwise);
  sampler wrap/filter modes are ignored (always repeat + bilinear), and
  `KHR_texture_transform` is not applied.
- **PCD is ASCII-only**; binary PLY *meshes* are unsupported (binary PLY point
  clouds work); OBJ ignores `.mtl`.
- **Merged-GLB caveats:** `Mesh.from_glb()` drops baseColor alpha and keeps the
  first primitive's material. Use `load_primitives()` when either matters.
