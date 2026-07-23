# IronEngine-BonaFide — Architecture Notes

> Layer-by-layer design rationale. Read this if you're touching the core,
> writing a new backend, or wondering why a thing is where it is.

## The six layers

```
┌──────────────────────────────────────────────────────────────────┐
│ L5  Integrations: 3DCreator shim · Sim shim · CLI · folder mount │
├──────────────────────────────────────────────────────────────────┤
│ L4  Public API:  Engine · Scene · render() / render_diff()       │
├──────────────────────────────────────────────────────────────────┤
│ L3  Render passes: shadow · gsplat · pbr · particles · fluid     │
│                    softbody · volumetric · neural FX · postfx    │
├──────────────────────────────────────────────────────────────────┤
│ L2  Scene graph + asset cache + material binding + asset mount   │
├──────────────────────────────────────────────────────────────────┤
│ L1  Kernel backends:  CUDAPath · WGPUPath · CPUPath              │
├──────────────────────────────────────────────────────────────────┤
│ L0  Tensor core: CuPy + PyTorch (DLPack interop), numpy fallback │
└──────────────────────────────────────────────────────────────────┘
```

**Imports flow downward only.** A pass may import core; core may import the
backend ABC. Backends never import passes; passes never import API. The CLI
and integrations import only the API.

## Why this shape

- **Backend ABC, not subclass dispatch.** Capabilities are flat strings
  (`"gsplat"`, `"warp_xpbd"`, `"raster"`). Passes ask `backend.supports(cap)`.
  This means we can add a new backend without touching any pass.
- **Passes are pure functions of `PassContext`.** They mutate the in-flight
  `FrameTargets` and nothing else. That makes the pass graph trivially
  reorderable and trivially mockable.
- **Data records are dataclasses, not classes with behaviour.** A `PointCloud`
  carries tensors and configuration flags; the *work* of rendering it lives in
  `SplatPass`. This split keeps assets cheap to serialise (`RenderBundle`) and
  simple to reason about.
- **Differentiable / non-differentiable share one path.** `render` wraps the
  same code in `torch.no_grad()`; `render_differentiable` does not. There is
  no separate "diff" pipeline.

## Pass scheduling

The default pass list in `_default_passes()` is:

```
shadow → softbody → completion → splat → pbr → particles → water →
volumetric → neural_relight → AA → bloom → neural_denoise →
neural_upscale → tonemap
```

A pass is **skipped** in three cases (and recorded in `out.skipped_passes`):

1. `is_active(ctx)` returns `False` — config or scene state disables it.
2. The active backend is missing one of `required_capabilities()`.
3. The pass raises (caught in 0.2; today they propagate).

## Backend matrix

| Capability                | CUDA                              | WGPU              | CPU                |
|---------------------------|-----------------------------------|-------------------|--------------------|
| `raster`                  | nvdiffrast deferred GBuffer        | WGSL forward PBR  | torch barycentric  |
| `splat`                   | screen-space disks (or gsplat)     | WGSL splat        | torch disks        |
| `gsplat`                  | gsplat 3DGS                        | —                 | —                  |
| `surfel`                  | torch PCA estimator                | torch             | torch              |
| `shadow_csm`              | nvdiffrast shadow                  | WGSL CSM          | placeholder        |
| `warp_xpbd`               | NVIDIA Warp                        | WGSL compute      | numpy XPBD (debug) |
| `warp_flip`               | NVIDIA Warp                        | —                 | —                  |
| `neural_field`            | tiny-cuda-nn                       | torch on device   | torch CPU          |
| `neural_denoise`          | torch UNet                         | torch UNet        | torch UNet         |
| `neural_upscale`          | torch EDSR                         | torch EDSR        | torch EDSR         |
| `vdb_volume`              | CuPy raymarch + OpenVDB            | WGSL raymarch     | numpy raymarch     |

## Where work happens

Roughly, each `render()` call does:

```
seed_everything(seed)
┌────────────────────────────────────────────────┐
│ build FrameTargets on the backend's device     │
│ build PassContext (backend, scene, cam, cfg)   │
│                                                │
│ for pass in engine.passes:                     │
│     if not pass.is_active(ctx): skip           │
│     if missing capabilities: skip              │
│     pass.run(ctx)             ← the actual GL  │
│                                                │
│ wrap targets.rgb in _OutputTensor              │
│ select sensor outputs requested by config      │
│ return RenderOutputs                           │
└────────────────────────────────────────────────┘
```

## Key invariants

1. `targets.rgb` always starts as black and accumulates linearly. Tonemap is
   the *last* pass and only fires when `output_color_space="sRGB"`.
2. Depth is **OpenGL-style NDC z in [-1, 1]** (+inf where empty), as the
   rasterizers write it. Sensor-facing consumers that need linear meters
   unproject via `2·near·far / (far + near − z·(far − near))` — the Sim
   shim's `render_sensor_depth` does this for you.
3. IDs start at `0` (background); each Mesh writes its `instance_id` ≥ 1.
4. Passes never resize `targets`. `NeuralUpscalePass` is the only pass
   permitted to change the output resolution, and it does so by replacing
   the rgb tensor — depth/normals/ids stay at native res.
5. `Engine` is per-thread; do not share across processes (CUDA contexts).

## Extending

- **New asset format**: add a loader to `assets/loaders/` that returns
  `Mesh` / `PointCloud` / `Volume`; mention it in `assets/__init__.py`.
- **New backend**: subclass `Backend` in `backends/<name>/backend.py`,
  declare capabilities, implement `empty/zeros/to_device`, and add a
  branch in `backends.__init__.auto_select`.
- **New pass**: subclass `RenderPass`, declare capabilities, mutate
  `ctx.targets`. Insert into the default graph or pass via
  `Engine.with_passes(...)`.
- **New camera**: dataclass with `view_matrix(...)`, `proj_matrix(...)`,
  `view_proj(...)`, `view_proj_torch(...)`. Register in `core/camera.py`
  type union.

## What's not in v0.1 — and why

- **No interactive viewer.** API-only. Plug `rerun-sdk` or `polyscope`.
- **No scene-graph hierarchy.** Flat lists fit the headless render-to-tensor
  use case and keep `RenderBundle` trivial.
- **No multi-GPU.** Single `Engine` per process; for batch rendering, spawn
  N processes each owning one Engine.
- **No CCD / advanced physics.** The render engine is render-only; physics
  belongs in IronEngine-Sim or NVIDIA Warp solvers.
- **No FBX.** Convert to GLB.
