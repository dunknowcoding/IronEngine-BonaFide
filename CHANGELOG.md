# Changelog

All notable changes to **IronEngine-BonaFide** will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **VSM shadow mode** (`shadows="vsm"`): variance shadow maps — blurred
  (E[z], E[z²]) moments per cascade + Chebyshev visibility with
  light-bleeding control; soft penumbrae, no acne, no baked bias.
- **Rectangular area lights**: `AreaLight.extent` is now honored via 3×3
  stratified emitter quadrature with inverse-square falloff and one-sided
  emission (previously shaded as a bare point, ignoring `extent`).
- **Motion-vector TAA**: history is reprojected per pixel (Catmull-Rom
  sampling — no diffusion blur) and depth-rejected on real motion, with
  sub-pixel-shift tolerance for jitter coverage changes; a bounded-step
  reordered 8-phase jitter sequence (~0.35 px radius) keeps every phase
  converged. Moving objects no longer leave ghost trails.
- **Gerstner water** (`WaterSurface` + `WaterPass`): analytic plane
  intersection, three directional wave components for the normal, Schlick
  fresnel between sky/IBL reflection (+ sun glint) and Beer-absorbed
  refraction; animates `time` by `RenderConfig.simulation_dt` per frame;
  writes rgb/depth/normals/albedo so SSGI/denoise/fog treat it as geometry.
- **CPU particle systems** (`ParticleSystem` + `ParticlePass`):
  deterministic gravity/drag/lifetime sim with seeded emitter respawn and
  age-faded depth-tested splats — replaces the warp-gated stub.
- `RenderConfig.simulation_dt` (default 1/60 s) drives water waves and
  the particle sim.
- Scene / render-bundle round-trip for `WaterSurface` and `ParticleSystem`
  (positions, velocities, ages, and every parameter).
- Tests: `test_shadows_vsm_arealight.py`, `test_water_particles.py` —
  210 passed, 1 skipped.

### Changed
- `ParticlePass` no longer requires the `warp_xpbd` capability (pure torch
  CPU/CUDA path).
- TAA default jitter is the bounded-step sequence (`_JITTER_SEQ_8`) instead
  of raw Halton order; `taa_jitter_frames != 8` still falls back to Halton.
- **Temporal anti-aliasing** (`aa="taa"`): the render driver applies a
  Halton-2,3 sub-pixel jitter to the camera projection every frame
  (`Camera.jitter_ndc`, honored by every geometry pass and the sky pass),
  and `TaaPass` blends into an exponential history (`taa_alpha`) with 3×3
  neighbourhood clamping; history auto-resets on camera/resolution change.
  Converges to SSAA-class edges in ~`taa_jitter_frames` (8) frames.
- **Morphological SMAA** (`aa="smaa"`): single-pass MLAA in the SMAA
  family — luma edge detection, run-length measurement, length-weighted
  symmetric cross-edge blending. Not bit-equal to reference SMAA (no
  precomputed area textures).
- **Spatial upscaling** (`neural_upscale`): the engine renders at
  `1/upscale_factor` internally and resolves to full size before tonemap.
  `"fsr"` = AMD FSR 1.0–style EASU (edge-adaptive 16-tap Lanczos2) + RCAS
  (anti-halo-clamped sharpening), pure torch on every backend; `"dlss"` =
  NVIDIA DLSS through a user `BONAFIDE_DLSS_DLL` NGX bridge (Turing+ GPU),
  with honest FSR fallback recorded in `skipped_passes`;
  `BONAFIDE_UPSCALE_WEIGHTS` EDSR checkpoint hook kept. Sensor outputs
  (depth / normals / ids / albedo) resolve alongside RGB.
- **Edge-aware denoiser** (`neural_denoise=True`): SVGF-style à-trous
  wavelet filter (steps 1/2/4) guided by colour distance, world-normal
  disagreement, and relative depth — works out of the box; the micro
  U-Net activates when `BONAFIDE_DENOISE_WEIGHTS` is provided.
- **SSGI-lite** (`neural_relight="ssgi"`): one-bounce screen-space GI —
  pyramid-blurred frame gathered as indirect irradiance with albedo
  colourbleed, hemisphere weighting, and depth-spread occlusion, masked to
  geometry pixels (`ssgi_intensity`).
- **glTF full texture set**: normal / metallic-roughness / occlusion /
  emissive maps resolve alongside baseColor (embedded, `data:`, and
  relative-URI sources); `KHR_texture_transform` baked into primitive UVs
  (conflicting per-slot transforms warn and keep the base-color one);
  `KHR_materials_emissive_strength` scales the emissive term; the PBR pass
  now samples **emissive maps** (factor × texture, sRGB-decoded). KTX2
  sources warn and stay behind the `[formats]` extra.
- **Grid volume raymarching**: `Volume.from_grid` densities render via a
  trilinear emission/absorption march (64 steps, AABB-clipped,
  depth-limited) so geometry occludes the volume and vice versa.
- Tests: `test_aa_passes.py`, `test_upscale.py`, `test_denoise_ssgi.py`,
  `test_volume_grid.py`, `test_gltf_full_textures.py`, `test_hardening.py`,
  `test_wireframe_skeleton.py` — 199 passed, 1 skipped.
- `raster_lines` — 1-px line-segment rasterizer (DDA at pixel resolution,
  perspective-correct depth, near-plane clip in clip space, per-vertex
  color lerp, deterministic amin + first-candidate resolve) in
  `backends.torch_raster`.
- 3DCreator shim: `render_mesh_offscreen` accepts `wireframe=` and
  `skeleton=` (3DCreator 0.2.0 API). Wireframe renders unique triangle
  edges through `raster_lines`; skeleton draws the rig as depth-tested
  white bone lines + orange joint dots on the shaded frame.
- CLI: `--backend` is now accepted both before and after the subcommand
  (`bonafide render scene.json --backend cpu` works).
- `bonafide_native` rebuilt and validated on the RTX 3090 with CUDA 13.3 +
  MSVC 14.44 (VS-bundled CMake 3.31 + Ninja). `build_native_win.bat` now
  auto-detects the active Python interpreter (override via
  `BONAFIDE_PYTHON`), discovers the system toolchain, and only sets the
  version-gap `NVCC_PREPEND_FLAGS` for CUDA < 12.4.

### Changed
- `RenderConfig` gains `taa_alpha`, `taa_jitter_frames`, `upscale_factor`,
  `upscale_sharpness`, and `ssgi_intensity`; `upscale_factor > 1` with
  `neural_upscale` is mutually exclusive with `ssaa > 1`
  (`ConfigurationError`).
- `PerspectiveCamera` / `OrthographicCamera` / `SensorCamera` gain
  `jitter_ndc` (default `(0, 0)` — bit-identical projections unless TAA
  drives it).
- `NeuralDenoisePass` is active whenever `neural_denoise=True` (à-trous
  default); previously it stayed off without a weights file.
- `WaterPass` / `ParticlePass` / `neural_relight="neural_ibl"` remain
  explicit roadmap stubs and record skip notes.

### Fixed
- Render bundles round-trip mesh **UVs** and point-cloud
  **normals / opacities / auto_point_size** — texture-mapped scenes now
  re-render bit-exact after save/load (UVs were silently dropped before,
  which also dropped every texture map).
- `SplatPass` moves cloud normals to the backend device before LOD
  subsetting (previously crashed on CUDA + LOD + CPU-resident clouds).
- TAA neighbourhood clamp uses max-pooling instead of `unfold` (~9× less
  transient memory at 720p).
- `light_from_dict` dropped a redundant boolean clause (no behavior change).
- Sky-pass ray directions now respect the TAA projection jitter, keeping
  the background aligned with jittered geometry.

- Cook-Torrance GGX specular shading in the PBR pass (CPU + CUDA):
  `PBRMaterial.roughness` / `metallic` / `emissive` are now honored —
  Trowbridge-Reitz D, Smith Schlick-GGX G, Schlick F with
  `F0 = mix(0.04, albedo, metallic)`, energy-conserving diffuse/specular mix,
  hemisphere ambient replacing the flat 15% term, emissive added after
  lighting.
- Point-cloud splat lighting: clouds with per-point `normals` are pre-shaded
  with Lambert `N·L` per scene light plus a 0.25 ambient term; clouds without
  normals keep raw colors.
- Sim integration: full TRS transform bridging (position + xyzw quaternion +
  scale) baked into mesh/point-cloud geometry with a per-`(asset, matrix)`
  cache; `PointCloudAsset` entities map to BonaFide `PointCloud`s; Sim spot
  lights bridge as point lights (cone shaping dropped).
- Tests: `test_pbr_specular.py`, `test_splat_lit.py`,
  `test_sim_integration_transforms.py`.
- Initial repository skeleton: `pyproject.toml`, `LICENSE`, `.gitignore`,
  CI workflow, issue/PR templates, `CHANGELOG.md`, `CONTRIBUTING.md`.
- Layered architecture (L0–L5) with backend ABC, render-pass framework,
  data model, and asset I/O.
- CPU reference backend (numpy + torch CPU) — small-resolution functional
  rendering for CI / GPU-less development.
- CUDA backend wrapping `gsplat` (3D Gaussian Splatting) and `nvdiffrast`
  (differentiable triangle raster).
- Public API `render(engine, scene, camera, config) -> RenderOutputs`
  returning RGB + depth + normals + IDs + albedo as `torch.Tensor`s.
- 3DCreator monkey-patch shim and IronEngine-Sim `RenderWorld` shim.
- Render bundles (`.bnf`) for reproducible scene + camera + config snapshots.
- **Native CUDA acceleration layer** (`bonafide_native`, C++/CUDA via
  nanobind): octree LOD walk, surfel kNN + PCA normals, disk-splat raster,
  async pinned-host upload. Built + validated on an RTX 3090
  (VS 2022 Build Tools + CUDA 11.7, Ninja generator).
- `scripts/build_native_win.bat` — one-command Windows native build that
  handles vcvars activation and the CUDA 11.7 ↔ MSVC 19.44 version-gap
  overrides.
- `scripts/build_native.py` — cross-platform build doctor: probes every
  prerequisite and emits an actionable fix per failing item.
- `cuda/native_bridge.py` registers CUDA runtime DLL directories via
  `os.add_dll_directory()` (required for `.pyd` loading on Python 3.8+
  Windows) and gracefully falls back to pure-Python paths when the
  extension isn't built.
- Test suites: `test_cuda_paths.py` (Python CUDA paths) and
  `test_native_extension.py` (compiled CUDA kernels) — 28 tests pass,
  1 UI test gated behind `IRONENGINE_TEST_UI`.

### Fixed
- Sim `render_sensor_depth` now returns linear eye-space **meters**
  (unprojected from the rasterizer's NDC z via near/far) instead of passing
  NDC depth through mislabeled as meters; depth labels in
  `RenderOutputs` / `FrameTargets` / docs corrected to NDC z in [-1, 1].

## [0.1.0] — TBD

Initial alpha. See `docs/USER_GUIDE.md` for capabilities matrix.
