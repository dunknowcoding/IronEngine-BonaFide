"""Full glTF texture-set loading: normal / metallic-roughness / occlusion /
emissive maps, KHR_texture_transform UV baking, and
KHR_materials_emissive_strength — plus a rendered check that the emissive
map reaches the frame. The GLB fixture lives in ``tests/_glb_factory.py``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pygltflib", reason="pygltflib required for glTF tests")
iio = pytest.importorskip("imageio.v3", reason="imageio required to build PNG fixtures")

from _glb_factory import (  # noqa: E402
    TEX_EMISSIVE,
    UV,
    build_full_glb,
)

from ironengine_bonafide.api import (  # noqa: E402
    Engine,
    PerspectiveCamera,
    RenderConfig,
    Scene,
    render,
)
from ironengine_bonafide.assets.loaders.gltf import load_primitives  # noqa: E402


@pytest.fixture()
def full_glb(tmp_path: Path) -> Path:
    p = tmp_path / "full_textured_quad.glb"
    build_full_glb(p)
    return p


def test_all_five_map_slots_resolve(full_glb: Path) -> None:
    mat = load_primitives(full_glb)[0].mesh.material
    for slot in ("albedo_map", "normal_map", "metallic_roughness_map",
                 "ao_map", "emissive_map"):
        ref = getattr(mat, slot)
        assert ref is not None, f"{slot} not resolved"
        assert Path(ref).is_file(), f"{slot} cache file missing: {ref}"


def test_texture_bytes_roundtrip(full_glb: Path) -> None:
    mat = load_primitives(full_glb)[0].mesh.material
    pixels = np.asarray(iio.imread(mat.emissive_map))[..., :3]
    np.testing.assert_array_equal(pixels, TEX_EMISSIVE)


def test_khr_texture_transform_bakes_uvs(full_glb: Path) -> None:
    prim = load_primitives(full_glb)[0]
    uvs = prim.mesh.uvs.detach().cpu().numpy()
    np.testing.assert_allclose(uvs[:, 0], UV[:, 0] + 0.25, atol=1e-6)
    np.testing.assert_allclose(uvs[:, 1], UV[:, 1], atol=1e-6)


def test_emissive_strength_scales_factor(full_glb: Path) -> None:
    mat = load_primitives(full_glb)[0].mesh.material
    np.testing.assert_allclose(mat.emissive, (2.0, 2.0, 2.0))


def test_render_samples_emissive_map(full_glb: Path) -> None:
    prim = load_primitives(full_glb)[0]
    scene = Scene().add(prim.mesh)                       # no lights on purpose
    cam = PerspectiveCamera(position=(0, 0, 3), look_at=(0, 0, 0), fov_deg=45)
    out = render(Engine.cpu(), scene, cam,
                 RenderConfig(width=96, height=64, output_color_space="sRGB"))
    rgb = out.rgb.cpu().numpy()
    center = rgb[16:48, 32:64]
    # Emissive texel (255, 64, 0) × factor 2 → strong red, faint green,
    # no blue; ambient alone could never produce this gap.
    assert center[..., 0].mean() > center[..., 2].mean() + 0.3
    assert center[..., 0].mean() > center[..., 1].mean() + 0.15
