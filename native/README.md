# `bonafide_native` — C++ / CUDA acceleration layer

The Python engine works without this. When it's built and importable
(`import bonafide_native`), the CUDA backend transparently uses it for:

| Hot path                | Speed-up                                  |
|-------------------------|-------------------------------------------|
| Octree LOD walk         | ~50× over the NumPy traversal             |
| Surfel kNN + PCA normals| ~30-100× over `torch.cdist + eigh`        |
| Disk splat raster       | competitive with gsplat for sparse clouds |
| Async asset upload      | overlaps host→device with compute         |

> **Stale-build quarantine (W28):** the old `native/build/` tree was linked
> against a dead env path (`G_\Anaconda\...`) and never imported. It has
> been moved to `native/_stale_build_quarantined/` — do not install from
> it; rebuild fresh per below. The import guard in
> `backends/cuda/native_bridge.py` (`HAS_NATIVE=False` + warning) is the
> runtime safety net when no valid build exists.

---

## Prerequisites

Use the build-doctor to check everything in one shot:

```bash
python scripts/build_native.py --check
```

It probes and reports each item below. Fix any failing line and re-run.

| Requirement              | Linux                                | Windows                                                |
|--------------------------|--------------------------------------|--------------------------------------------------------|
| Python ≥ 3.11            | conda env (`IronEngineWorld`)        | conda env (`IronEngineWorld`)                          |
| CMake ≥ 3.24             | `apt install cmake` / `conda install cmake` | https://cmake.org/download/                  |
| CUDA toolkit ≥ 11.7      | NVIDIA `.run` installer or `apt`     | NVIDIA `.exe` installer                                |
| Host C++ compiler        | `g++` from `build-essential`         | **VS 2022 Build Tools** with the *Desktop development with C++* workload |
| nanobind                 | `pip install nanobind`               | `pip install nanobind`                                 |
| **CUDA ↔ VS integration** (Windows-only) | _(n/a)_         | The CUDA installer's *Visual Studio Integration* component must run **after** VS Build Tools is installed; otherwise the `.props` files don't land in `MSBuild/.../v170/BuildCustomizations` and CMake reports `No CUDA toolset found`. |

### Windows-specific gotchas

1. **Run from `x64 Native Tools Command Prompt for VS 2022`** (or activate via
   `vcvarsall.bat x64`) so `cl.exe` is on `PATH`.
2. **Verify VS-CUDA integration**:
   ```
   dir "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\MSBuild\Microsoft\VC\v170\BuildCustomizations\CUDA*.props"
   ```
   If empty, run the CUDA installer and tick *Visual Studio Integration*. Or
   manually copy the four files from
   `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\extras\visual_studio_integration\MSBuildExtensions\`
   into the `BuildCustomizations` directory.
3. **Multiple `cmake` installs**: the build-doctor will warn if it picks up a
   non-standard CMake (e.g. one shipped with STM32CubeCLT). Put your preferred
   `cmake` first on `PATH`.

---

## Build

### Windows (recommended) — `scripts\build_native_win.bat`

```bat
scripts\build_native_win.bat Release
```

This is the **validated** Windows path. It:

1. Activates the VS 2022 Build Tools x64 environment (`vcvarsall.bat x64`).
2. Puts a known-good CMake + Ninja first on `PATH` so a stray MinGW gcc or
   STM32CubeCLT toolchain doesn't win the compiler-detection race.
3. Builds with the **Ninja** generator — no VS `.props` CUDA integration
   required, sidestepping the "No CUDA toolset found" failure mode.
4. Exports `NVCC_PREPEND_FLAGS` to bridge the **CUDA 11.7 ↔ MSVC 19.44
   version gap** (two independent gates: nvcc's `host_config.h` check and
   the MSVC STL's `STL1002` assert). Honoured by every `nvcc` invocation,
   including CMake's compiler-ID probe.
5. Copies the resulting `.pyd` into the active env's `site-packages`.

> The clean long-term fix for the version gap is **CUDA ≥ 12.4** (which
> matches a modern MSVC). The override flags are safe for the small,
> STL-light kernels in this directory but are a stopgap, not a guarantee.

### Cross-platform — `scripts/build_native.py`

```bash
python scripts/build_native.py            # diagnose + build + install
python scripts/build_native.py --check    # diagnose only
python scripts/build_native.py --debug
```

The doctor probes every prerequisite, prints a PASS/FAIL report with an
actionable fix per failing line, and only then configures + builds.

### Manual CMake

```bash
cd native
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DPython_EXECUTABLE=$(which python)
cmake --build build --config Release -j
```

### Verify

```python
import bonafide_native
print(bonafide_native.__doc__)
# IronEngine-BonaFide native CUDA acceleration layer.
```

The Python `cuda/native_bridge.py` registers the CUDA runtime DLL
directories (`os.add_dll_directory` — required on Python 3.8+ Windows),
imports the extension, flips `HAS_NATIVE = True`, and the CUDA backend
starts advertising `native_octree`, `native_surfel`, `native_splat`,
`native_upload`.

If absent, the engine logs `bonafide_native not available — using
pure-Python CUDA paths` and continues — **the build is purely optional**.

Run the native kernel tests to confirm the GPU paths actually execute:

```bash
pytest tests/test_native_extension.py -v
```

---

## Layout

```
native/
├── CMakeLists.txt              top-level project
├── README.md                   this file
├── cmake/                      reusable Find*.cmake scripts
├── include/bonafide/
│   ├── api.hpp                 host-callable C++ surface
│   ├── octree.hpp              octree LOD walker
│   ├── surfel.hpp              kNN + PCA normals
│   ├── splat.hpp               disk-splat raster
│   └── upload.hpp              pinned-host async transfer helper
├── src/
│   ├── bindings.cpp            nanobind glue → bonafide_native
│   ├── octree.cu               CUDA octree kernels
│   ├── surfel.cu               CUDA kNN + PCA
│   ├── splat.cu                CUDA disk-splat raster + depth blend
│   └── upload.cpp              cudaMemcpyAsync + pinned-host helpers
└── python/
    └── bonafide_native/        thin Python re-export so users `import bonafide_native`
```
