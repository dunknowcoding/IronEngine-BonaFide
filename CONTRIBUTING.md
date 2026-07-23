# Contributing to IronEngine-BonaFide

Thanks for your interest. This document captures the conventions so PRs can ship quickly.

## Ground rules

1. **Layer discipline.** L4 (API) → L3 (passes) → L2 (scene/assets) → L1 (backend) →
   L0 (tensor). Imports only flow downward. No backend imports a pass; no pass imports
   `api.py`.
2. **Backend-agnostic passes.** A `RenderPass` asks `backend.supports(...)`; never
   `if isinstance(backend, CudaBackend)`.
3. **Data records are dataclasses.** `@dataclass(slots=True)`, JSON-serialisable,
   `from_dict` / `to_dict` round-trip required for anything that can land in a
   `RenderBundle`.
4. **Pure-Python orchestration only.** Never write a `setup.py` extension, a
   `Cython` module, a `.cpp` file, or vendor a binary blob. We *wrap* CUDA libraries;
   we don't author them.
5. **Strict typing.** `pyright --strict` clean. Public signatures use precise types
   (`torch.Tensor`, `np.ndarray`, `Backend`, etc.).

## Setting up

```bash
conda activate IronEngineWorld
git clone https://github.com/<you>/IronEngine-BonaFide.git
cd IronEngine-BonaFide
pip install -e .[all,dev]
pytest -q
```

## Adding a render pass

```python
# src/ironengine_bonafide/passes/my_pass.py
from ironengine_bonafide.passes.base import RenderPass, PassContext

class MyPass(RenderPass):
    name = "my_pass"

    def required_capabilities(self) -> tuple[str, ...]:
        return ("raster",)

    def run(self, ctx: PassContext) -> None:
        ctx.targets.rgb = ctx.backend.raster(...)
```

Then register it in `core/scene.py`'s default pass graph (or via
`Engine.with_passes(...)` for opt-in).

## Adding a backend

Implement `Backend` ABC in `backends/<name>/backend.py`. Cover at minimum:

- `device: str`, `supports(capability: str) -> bool`
- `to_tensor`, `from_tensor` (DLPack interop expected)
- `raster_mesh`, `splat_pointcloud`, `compute_shadow_map`

Add a discovery entry in `backends/__init__.py:auto_select()`.

## Adding an asset loader

```python
# src/ironengine_bonafide/assets/loaders/<format>.py
from pathlib import Path
from ironengine_bonafide.core.pointcloud import PointCloud   # or Mesh, Volume

def load(path: Path) -> PointCloud:
    ...
```

Register in `assets/loaders/__init__.py`.

## Style

- Python ≥ 3.11
- `ruff check` and `ruff format` clean
- `pyright --strict` clean
- Type hints on every public symbol
- `@dataclass(slots=True)` for data records
- NumPy/PyTorch tensors, never bare lists for numerical data

## PR checklist

- [ ] Public-API changes reflected in `docs/API_REFERENCE.md`
- [ ] User-facing changes reflected in `docs/USER_GUIDE.md` if relevant
- [ ] No new singletons or implicit registries
- [ ] Smoke test added for any new public symbol
- [ ] Render bundle round-trip stable for any new dataclass

## Testing on a machine without GPU

The CPU backend is functional and used for CI. Run tests with:

```bash
BONAFIDE_BACKEND=cpu pytest -q
```

The CUDA-only suites will skip themselves automatically when `cupy` / `gsplat` /
`nvdiffrast` aren't importable.
