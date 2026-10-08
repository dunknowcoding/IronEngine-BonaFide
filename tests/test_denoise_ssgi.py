"""Denoiser (à-trous) and SSGI relight tests."""
from __future__ import annotations

import numpy as np
import torch

from ironengine_bonafide.api import (
    DirectionalLight,
    Engine,
    Mesh,
    PBRMaterial,
    PerspectiveCamera,
    RenderConfig,
    Scene,
    render,
)
from ironengine_bonafide.passes.neural_denoise import atrous_denoise


def _scene() -> Scene:
    pos = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]],
                   dtype=np.float32)
    idx = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    mesh = Mesh.from_arrays(pos, idx,
                            material=PBRMaterial(albedo=(0.7, 0.5, 0.3)))
    return Scene().add(mesh).add(
        DirectionalLight(direction=(0.0, 0.0, -1.0), intensity=3.0))


def _cam() -> PerspectiveCamera:
    return PerspectiveCamera(position=(0, 0, 3), look_at=(0, 0, 0), fov_deg=45)


def test_atrous_reduces_noise_and_keeps_edges() -> None:
    torch.manual_seed(0)
    clean = torch.zeros(32, 32, 3)
    clean[:, 16:] = 0.6
    normals = torch.zeros(32, 32, 3)
    normals[..., 2] = 1.0
    depth = torch.full((32, 32), 0.5)
    noisy = clean + 0.08 * torch.randn(32, 32, 3)
    out = atrous_denoise(noisy, normals, depth)
    # Flat-region variance collapses.
    var_noisy = float(noisy[:, 4].var())
    var_out = float(out[:, 4].var())
    assert var_out < var_noisy * 0.5
    # The hard edge survives: mean step across column 15→16 stays large.
    step = float((out[:, 16].mean() - out[:, 15].mean()).abs())
    assert step > 0.3


def test_atrous_deterministic() -> None:
    torch.manual_seed(1)
    img = torch.rand(24, 24, 3)
    normals = torch.zeros(24, 24, 3)
    normals[..., 1] = 1.0
    depth = torch.rand(24, 24)
    a = atrous_denoise(img, normals, depth)
    b = atrous_denoise(img, normals, depth)
    torch.testing.assert_close(a, b)


def test_denoise_pass_active_without_weights() -> None:
    out = render(Engine.cpu(), _scene(), _cam(),
                 RenderConfig(width=96, height=64, neural_denoise=True,
                              output_color_space="linear"))
    assert out.rgb.shape == (64, 96, 3)
    assert not any("denoise" in s for s in out.skipped_passes)


def test_ssgi_adds_indirect_on_geometry_only() -> None:
    # aa='off': FXAA's 3x3 blend would legitimately smear relit edge pixels
    # into neighbouring sky — we assert on the relight mask itself.
    base = dict(width=96, height=64, output_color_space="linear", bloom=False,
                aa="off", sensor_outputs=("rgb", "depth"))
    ref = render(Engine.cpu(), _scene(), _cam(), RenderConfig(**base))          # type: ignore[arg-type]
    out = render(Engine.cpu(), _scene(), _cam(),
                 RenderConfig(neural_relight="ssgi", ssgi_intensity=0.8, **base))  # type: ignore[arg-type]
    diff = out.rgb - ref.rgb
    finite = torch.isfinite(ref.depth) if ref.depth is not None else None
    assert float(out.rgb.sum()) > float(ref.rgb.sum())
    # Sky pixels (infinite depth) receive no bounce.
    assert ref.depth is not None
    sky = ~torch.isfinite(ref.depth)
    assert bool(sky.any())
    assert float(diff[sky].abs().max()) == 0.0
    assert finite is not None and float(diff[finite].abs().max()) > 0.0


def test_neural_ibl_records_skip_note() -> None:
    out = render(Engine.cpu(), _scene(), _cam(),
                 RenderConfig(width=64, height=48, neural_relight="neural_ibl"))
    assert "neural_relight:neural_ibl_unimplemented" in out.skipped_passes
