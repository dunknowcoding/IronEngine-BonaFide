"""Spatial upscaling tests — FSR 1.0 (EASU + RCAS) and the DLSS fallback.

The engine renders internally at ``1/upscale_factor`` of the output size
when ``neural_upscale != 'none'``; these tests pin the output contract
(shape, sensor buffers, determinism) and the honest DLSS fallback note.
"""
from __future__ import annotations

import numpy as np
import pytest
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
from ironengine_bonafide.errors import ConfigurationError
from ironengine_bonafide.passes.neural_upscale import _easu, _rcas


def _scene() -> Scene:
    pos = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]],
                   dtype=np.float32)
    idx = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    mesh = Mesh.from_arrays(pos, idx,
                            material=PBRMaterial(albedo=(0.8, 0.6, 0.3)))
    return Scene().add(mesh).add(
        DirectionalLight(direction=(0.0, 0.0, -1.0), intensity=3.0))


def _cam() -> PerspectiveCamera:
    return PerspectiveCamera(position=(0, 0, 3), look_at=(0, 0, 0), fov_deg=45)


def test_fsr_upscales_to_output_resolution() -> None:
    out = render(Engine.cpu(), _scene(), _cam(),
                 RenderConfig(width=192, height=128, neural_upscale="fsr",
                              upscale_factor=2.0, output_color_space="linear",
                              sensor_outputs=("rgb", "depth", "normals", "ids",
                                              "albedo")))
    assert out.rgb.shape == (128, 192, 3)
    assert out.depth is not None and out.depth.shape == (128, 192)
    assert out.normals is not None and out.normals.shape == (128, 192, 3)
    assert out.albedo is not None and out.albedo.shape == (128, 192, 3)
    assert out.ids is not None and out.ids.shape == (128, 192)
    # The quad still fills the center after upscaling.
    assert bool(torch.isfinite(out.depth[64, 96]))


def test_fsr_deterministic() -> None:
    cfg = RenderConfig(width=192, height=128, neural_upscale="fsr",
                       output_color_space="linear")
    a = render(Engine.cpu(), _scene(), _cam(), cfg).rgb
    b = render(Engine.cpu(), _scene(), _cam(), cfg).rgb
    torch.testing.assert_close(a, b)


def test_dlss_falls_back_to_fsr_without_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BONAFIDE_DLSS_DLL", raising=False)
    cfg = RenderConfig(width=192, height=128, neural_upscale="dlss",
                       output_color_space="linear")
    out = render(Engine.cpu(), _scene(), _cam(), cfg)
    assert "neural_upscale:dlss_unavailable→fsr" in out.skipped_passes
    assert out.rgb.shape == (128, 192, 3)


def test_easu_beats_bilinear_on_diagonal_edge() -> None:
    # Ground-truth high-res diagonal; the low-res input is its 2x
    # area-downsample.
    hi = torch.zeros(128, 128, 3)
    for y in range(128):
        hi[y, y // 2 + 16:] = 1.0
    lo = torch.nn.functional.avg_pool2d(
        hi.permute(2, 0, 1).unsqueeze(0), 2).squeeze(0).permute(1, 2, 0)
    easu = _easu(lo, 128, 128)
    bil = torch.nn.functional.interpolate(
        lo.permute(2, 0, 1).unsqueeze(0), size=(128, 128), mode="bilinear",
        align_corners=False).squeeze(0).permute(1, 2, 0)
    err_easu = float(((easu - hi) ** 2).mean())
    err_bil = float(((bil - hi) ** 2).mean())
    assert err_easu <= err_bil * 1.1


def test_rcas_sharpens_without_halo() -> None:
    img = torch.zeros(32, 32, 3)
    img[:, 16:] = 0.5
    out = _rcas(img, sharpness=1.0)
    # Contrast at the edge increases...
    edge_in = abs(float(img[16, 16, 0]) - float(img[16, 15, 0]))
    edge_out = abs(float(out[16, 16, 0]) - float(out[16, 15, 0]))
    assert edge_out >= edge_in
    # ...but never overshoots the local neighbourhood (anti-halo clamp).
    assert float(out.max()) <= 0.5 + 1e-6
    assert float(out.min()) >= 0.0


def test_upscale_and_ssaa_are_mutually_exclusive() -> None:
    with pytest.raises(ConfigurationError):
        RenderConfig(neural_upscale="fsr", upscale_factor=2.0, ssaa=2).validate()


def test_upscale_factor_bounds() -> None:
    with pytest.raises(ConfigurationError):
        RenderConfig(neural_upscale="fsr", upscale_factor=8.0).validate()
    with pytest.raises(ConfigurationError):
        RenderConfig(neural_upscale="fsr", upscale_sharpness=2.0).validate()
