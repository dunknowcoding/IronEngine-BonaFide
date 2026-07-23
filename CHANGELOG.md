# Changelog

All notable changes to **IronEngine-BonaFide** will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
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
