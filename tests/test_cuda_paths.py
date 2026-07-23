"""End-to-end CUDA backend tests.

Skipped automatically when ``torch.cuda.is_available()`` is False (see
``conftest.py``). Exercises the Python CUDA paths through the engine —
backend selection, render, sensor outputs, optional native-bridge probe.
The native ``bonafide_native`` extension is a separate gate
(``HAS_NATIVE``) and is not required by these tests.

Since 0.2 the CUDA backend is honest about partial installs: without a
raster library (nvdiffrast or bonafide_native) ``CudaBackend()`` raises
``BackendUnavailable`` and ``Engine.cuda()``/``Engine.auto()`` fall back
to the CPU backend instead of PCIe-round-tripping every raster call.
Render tests that require real CUDA rasterization are gated on
``HAS_CUDA_RASTER``.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.cuda


def _has_cuda_raster_stack() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import nvdiffrast.torch  # type: ignore[import-not-found]  # noqa: F401
        return True
    except ImportError:
        pass
    from ironengine_bonafide.backends.cuda.native_bridge import HAS_NATIVE
    return bool(HAS_NATIVE)


HAS_CUDA_RASTER = _has_cuda_raster_stack()
needs_cuda_raster = pytest.mark.skipif(
    not HAS_CUDA_RASTER,
    reason="no CUDA raster stack (nvdiffrast / bonafide_native missing)",
)


# --------------------------------------------------------------- backend
def test_cuda_backend_constructs() -> None:
    from ironengine_bonafide.backends.cuda.backend import CudaBackend
    from ironengine_bonafide.backends.cuda.native_bridge import HAS_NATIVE

    if not HAS_CUDA_RASTER:
        # Honesty gate: no raster library → refuse to construct.
        from ironengine_bonafide.errors import BackendUnavailable
        with pytest.raises(BackendUnavailable):
            CudaBackend()
        return

    be = CudaBackend()
    assert be.device.startswith("cuda")
    assert be.supports("raster")
    assert be.supports("splat")
    # When no native build is present this MUST be False — otherwise we'd
    # claim capabilities we can't actually deliver.
    assert be.supports("native_splat") == HAS_NATIVE


def test_cuda_backend_has_torch_raster_depth() -> None:
    """Every backend provides raster_depth (torch-on-device) so the CSM
    shadow pass never silently skips with no_raster_depth."""
    if not HAS_CUDA_RASTER:
        pytest.skip("CUDA backend refuses to construct without a raster stack")
    from ironengine_bonafide.backends.cuda.backend import CudaBackend

    be = CudaBackend()
    positions = torch.tensor(
        [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32,
    )
    indices = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    from ironengine_bonafide.core.camera import PerspectiveCamera
    vp = torch.from_numpy(
        PerspectiveCamera(position=(0, 0.5, 2), look_at=(0, 0.5, 0)).view_proj(4 / 3)
    ).float()
    depth = be.raster_depth(positions, indices, vp, 32, 24)
    assert str(depth.device).startswith("cuda")
    assert torch.isfinite(depth).any()


def test_engine_cuda_picks_cuda() -> None:
    from ironengine_bonafide.api import Engine

    eng = Engine.cuda()
    if HAS_CUDA_RASTER:
        assert eng.backend.device.startswith("cuda")
    else:
        # Honest fallback: no raster stack → CPU instead of a fake CUDA.
        assert eng.backend.name == "cpu"


# --------------------------------------------------------------- pointcloud
@needs_cuda_raster
def test_pointcloud_renders_on_cuda() -> None:
    from ironengine_bonafide.api import (
        DirectionalLight, Engine, PerspectiveCamera, PointCloud, RenderConfig,
        Scene, render,
    )

    g = np.random.default_rng(0)
    pos = g.uniform(-0.5, 0.5, (4096, 3)).astype(np.float32)
    col = g.uniform(0.0, 1.0, (4096, 3)).astype(np.float32)

    scene = (
        Scene()
        .add(PointCloud.from_arrays(pos, col))
        .add(DirectionalLight(intensity=2.5))
    )
    cam = PerspectiveCamera(position=(2.0, 1.5, 2.0), look_at=(0, 0, 0), fov_deg=45)
    cfg = RenderConfig(width=128, height=96, output_color_space="sRGB")
    out = render(Engine.cuda(), scene, cam, cfg)

    assert out.rgb.is_cuda
    assert out.rgb.shape == (96, 128, 3)
    assert float(out.rgb.max()) > 0.0


# --------------------------------------------------------------- mesh
@needs_cuda_raster
def test_mesh_renders_on_cuda() -> None:
    from ironengine_bonafide.api import (
        DirectionalLight, Engine, Mesh, PerspectiveCamera, RenderConfig,
        Scene, render,
    )

    positions = np.array(
        [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32,
    )
    indices = np.array([[0, 1, 2]], dtype=np.int64)
    mesh = Mesh.from_arrays(positions, indices,
                             colors=np.full((3, 3), 0.7, dtype=np.float32))

    scene = Scene().add(mesh).add(DirectionalLight(intensity=2.0))
    cam = PerspectiveCamera(position=(0.0, 0.5, 2.0), look_at=(0, 0.5, 0), fov_deg=45)
    cfg = RenderConfig(width=64, height=48,
                        sensor_outputs=("rgb", "depth", "normals"))

    out = render(Engine.cuda(), scene, cam, cfg)
    assert out.rgb.is_cuda
    assert out.depth is not None and out.depth.is_cuda
    assert out.normals is not None and out.normals.is_cuda


# --------------------------------------------------------------- sensor outputs
@needs_cuda_raster
def test_all_sensor_outputs_on_cuda() -> None:
    from ironengine_bonafide.api import (
        Engine, Mesh, PerspectiveCamera, RenderConfig, Scene, render,
    )

    positions = np.array(
        [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32,
    )
    indices = np.array([[0, 1, 2]], dtype=np.int64)
    mesh = Mesh.from_arrays(positions, indices)

    scene = Scene().add(mesh)
    cam = PerspectiveCamera(position=(0.0, 0.5, 2.0), look_at=(0, 0.5, 0), fov_deg=45)
    cfg = RenderConfig(
        width=32, height=24,
        sensor_outputs=("rgb", "depth", "normals", "ids", "albedo"),
    )
    out = render(Engine.cuda(), scene, cam, cfg)
    assert out.rgb is not None and out.rgb.is_cuda
    assert out.depth is not None and out.depth.is_cuda
    assert out.normals is not None and out.normals.is_cuda
    assert out.ids is not None and out.ids.is_cuda and out.ids.dtype == torch.int64
    assert out.albedo is not None and out.albedo.is_cuda


# --------------------------------------------------------------- gsplat workspace cache
def test_gsplat_workspace_cache_is_persistent() -> None:
    from ironengine_bonafide.api import PointCloud
    from ironengine_bonafide.backends.cuda.streams import workspace_cache

    cache = workspace_cache()
    cache.clear()

    cloud = PointCloud.from_arrays(np.zeros((100, 3), dtype=np.float32))
    ws1 = cache.ensure_gsplat(cloud, "cuda:0")
    ws2 = cache.ensure_gsplat(cloud, "cuda:0")
    assert ws1 is ws2, "ensure_gsplat must reuse the per-cloud workspace"
    assert ws1.quats.is_cuda
    assert ws1.scales.shape == (100, 3)


# --------------------------------------------------------------- native bridge
def test_native_bridge_probe_does_not_crash() -> None:
    """Whether or not the .pyd is built, importing the bridge must succeed."""
    from ironengine_bonafide.backends.cuda.native_bridge import HAS_NATIVE
    assert HAS_NATIVE in (True, False)


def test_streams_named_pool_on_cuda() -> None:
    from ironengine_bonafide.backends.cuda.streams import stream

    s1 = stream("splat")
    s2 = stream("splat")
    assert s1 is s2, "named streams must be cached"
    s3 = stream("upload")
    assert s3 is not s1
