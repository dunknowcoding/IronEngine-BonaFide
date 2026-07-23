# IronEngine-BonaFide — Performance Notes

> Honest performance positioning. Read this before benchmarking against
> Vulkan or PhysX.

## What we are

A pure-Python orchestration layer over best-in-class CUDA libraries:

- **gsplat** (Nerfstudio team) for 3D Gaussian Splatting
- **nvdiffrast** (NVIDIA) for differentiable triangle rasterization
- **NVIDIA Warp** for particle / fluid / cloth / soft-body solvers
- **tiny-cuda-nn** for fused MLPs and hash-grids

These are individually competitive with their C++/Vulkan counterparts; what
BonaFide adds is the glue that lets them share a scene, a camera, a config,
and a `RenderOutputs` tensor with autograd.

## What we are not

- A native engine.
- Faster than Vulkan or PhysX for any single op.
- A replacement for nerfstudio if you need a full training framework.
- A real-time game engine (no swapchain, no input, no asset hot-reload).

## Where the time goes

For a typical 1080p frame with a 1M-point cloud + a 50k-tri mesh on an
RTX 4080 (rough numbers from internal smoke tests):

| Stage                         | CUDA path | CPU reference |
|-------------------------------|-----------|---------------|
| Scene → backend transfer       | <1 ms     | <1 ms          |
| `SplatPass` (gsplat path)      | 4–8 ms    | minutes        |
| `PbrPass` (nvdiffrast path)    | 2–5 ms    | seconds        |
| `BloomPass` (torch convs)      | 1–2 ms    | 100 ms         |
| `TonemapPass`                  | <0.5 ms   | 5 ms           |
| `NeuralDenoisePass` (untrained)| 4–10 ms   | 200 ms         |
| Total                          | 15–30 ms  | many seconds   |

The CPU reference path is intentionally non-fast — it exists to keep tests
running on machines without a GPU and to provide a debug oracle.

## Memory

- Per-tensor: float32 frames at 1080p ≈ 24 MB (RGB + depth + normals + ids + albedo).
- Asset cache: LRU, configurable via `RenderConfig.vram_budget_mb`.
- gsplat: ~150 bytes per Gaussian (positions, scales, rotations, opacities, SH coeffs).
- Octree LOD: ~32 bytes per leaf node + 4 bytes per indexed point.

## Tuning recipes

**Faster previews** — drop samples, disable bloom + denoise + upscale,
render at 720p:

```python
RenderConfig(width=1280, height=720, samples=1, aa="off",
             bloom=False, neural_denoise=False, neural_upscale="none")
```

**Higher quality** — bump samples, enable everything:

```python
RenderConfig(width=1920, height=1080, samples=4, aa="taa",
             bloom=True, neural_denoise=True, neural_upscale="dlss",
             neural_relight="neural_ibl")
```

**Huge clouds** — enable LOD + surfels, leave gsplat off if VRAM is tight:

```python
cloud = PointCloud.from_ply(...).with_lod().with_surfels()
RenderConfig(lod=LodConfig(enabled=True, screen_space_error_px=2.0,
                           max_chunks_in_vram=128))
```

**Incomplete clouds** — train completion prior once, render many times:

```python
prior = train_completion_prior(cloud.positions, cloud.colors, iterations=2000)
cloud._completion_prior = prior
cloud.use_completion = True
```

**Bit-exact reproduction** — fix the seed and the backend:

```python
RenderConfig(seed=42)         # bit-exact within a backend
```

## What we will optimise next

- **Pre-built gsplat parameters** — today the splat pass synthesises identity
  quaternions and uniform scales when the cloud doesn't carry them. Caching
  surfel-derived gsplat params will cut per-frame setup by ~1 ms.
- **Pass fusion** — bloom + tonemap + AA can run as a single fused torch op.
- **Mesh batching** — N nvdiffrast calls collapse into one indexed draw when
  the meshes share material maps (0.2).
- **Persistent gsplat workspaces** — gsplat reallocates per call today;
  pinning a workspace tensor would cut allocator pressure.

## Benchmarking yourself

```python
with engine.profile() as prof:
    out = render(engine, scene, cam, cfg)
print(prof.summary())
```

For frame-rate measurements, render in a loop and divide:

```python
import time
N = 60
t0 = time.perf_counter()
for _ in range(N):
    render(engine, scene, cam, cfg)
torch.cuda.synchronize()
print(f"{N / (time.perf_counter() - t0):.1f} FPS")
```

Always synchronise CUDA before timing — `render(...)` returns to Python
before the GPU finishes.
