"""TAA (temporal AA) and morphological SMAA tests.

TAA is exercised end-to-end: the render driver applies the Halton jitter to
the camera projection, and the pass accumulates into its clamped history.
The quality claim under test: after enough jittered frames, the TAA output
of a static scene is closer to an SSAA reference than any single
un-jittered frame is.
"""
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
from ironengine_bonafide.passes.aa_pass import _halton, _mlaa


def _scene() -> Scene:
    # A rotated quad gives long diagonal staircase edges.
    pos = np.array([[-1.4, -0.5, 0.0], [1.4, -0.5, 0.0],
                    [1.4, 0.5, 0.0], [-1.4, 0.5, 0.0]], dtype=np.float32)
    idx = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    mesh = Mesh.from_arrays(pos, idx,
                            material=PBRMaterial(albedo=(0.9, 0.4, 0.2),
                                                 roughness=0.9))
    return Scene().add(mesh).add(
        DirectionalLight(direction=(0.0, 0.0, -1.0), intensity=3.0))


def _cam() -> PerspectiveCamera:
    return PerspectiveCamera(position=(0.0, 0.0, 3.0), look_at=(0.0, 0.0, 0.0),
                             fov_deg=40.0)


def test_halton_sequence_in_unit_interval() -> None:
    vals = [_halton(i, 2) for i in range(1, 17)]
    assert all(0.0 <= v < 1.0 for v in vals)
    assert len(set(vals)) == len(vals)                       # no repeats


def test_taa_jittered_frames_differ_and_camera_restored() -> None:
    engine = Engine.cpu()
    cam = _cam()
    cfg = RenderConfig(width=96, height=64, aa="taa", output_color_space="linear")
    a = render(engine, _scene(), cam, cfg).rgb.clone()
    b = render(engine, _scene(), cam, cfg).rgb.clone()
    assert not torch.equal(a, b)                             # jitter applied
    assert cam.jitter_ndc == (0.0, 0.0)                      # restored after render


def test_taa_converges_toward_ssaa_reference() -> None:
    engine = Engine.cpu()
    cam = _cam()
    base = dict(width=96, height=64, output_color_space="linear", bloom=False)
    ref = render(engine, _scene(), _cam(),
                 RenderConfig(ssaa=4, aa="off", **base)).rgb            # type: ignore[arg-type]
    single = render(engine, _scene(), _cam(),
                    RenderConfig(aa="off", **base)).rgb                 # type: ignore[arg-type]
    cfg = RenderConfig(aa="taa", **base)                                # type: ignore[arg-type]
    out = None
    for _ in range(8):
        out = render(engine, _scene(), cam, cfg).rgb.clone()
    assert out is not None
    err_taa = float(((out - ref) ** 2).mean())
    err_single = float(((single - ref) ** 2).mean())
    assert err_taa < err_single


def test_taa_camera_move_resets_history() -> None:
    engine = Engine.cpu()
    cam = _cam()
    cfg = RenderConfig(width=96, height=64, aa="taa", output_color_space="linear")
    for _ in range(4):
        render(engine, _scene(), cam, cfg)
    cam.position = (0.5, 0.0, 3.0)                           # camera moved
    moved = render(engine, _scene(), cam, cfg).rgb.clone()
    # A fresh engine renders the same moved camera: with reset history and
    # jitter phase 0 the two must match exactly.
    fresh = render(Engine.cpu(), _scene(), cam, cfg).rgb.clone()
    torch.testing.assert_close(moved, fresh)


def test_taa_no_ghost_on_moving_object() -> None:
    """Depth-rejected reprojection: after the mesh jumps, the OLD location
    must show the background — not a smeared trail of the old position."""
    engine = Engine.cpu()
    cam = _cam()
    cfg = RenderConfig(width=96, height=64, aa="taa", output_color_space="linear",
                       bloom=False)
    scene = _scene()
    mesh = scene.meshes[0]
    for _ in range(6):                                       # build history
        render(engine, scene, cam, cfg)
    old_rgb = render(engine, scene, cam, cfg).rgb.clone()
    # Reference background value at the quad's old spot (quad removed).
    bg_scene = Scene().add(DirectionalLight(direction=(0.0, 0.0, -1.0), intensity=3.0))
    bg = render(engine, bg_scene, cam,
                RenderConfig(width=96, height=64, aa="off", output_color_space="linear",
                             bloom=False)).rgb
    # Jump the mesh far to the side.
    mesh.positions = mesh.positions + torch.tensor([2.0, 0.0, 0.0])
    out = render(engine, scene, cam, cfg).rgb.clone()
    # The old quad center was around frame center; the mesh moved 2 world
    # units right, so frame center now shows background.
    spot_old = old_rgb[32, 48]
    spot_new = out[32, 48]
    spot_bg = bg[32, 48]
    assert float((spot_new - spot_bg).abs().max()) < 0.05, (
        f"ghost trail at old location: {spot_new} vs bg {spot_bg}")
    assert float((spot_new - spot_old).abs().max()) > 0.05   # it really changed


def test_smaa_smooths_staircase_and_keeps_flats() -> None:
    # 16x16 image with a hard diagonal edge.
    img = torch.zeros(16, 16, 3)
    for y in range(16):
        x_edge = y // 2 + 2
        img[y, x_edge:] = 1.0
    out = _mlaa(img)
    # Flat interiors are untouched.
    torch.testing.assert_close(out[8, 12:], img[8, 12:])
    torch.testing.assert_close(out[3, :1], img[3, :1])
    # Along the staircase the luma gradient softens: the number of maximal
    # (full-step) transitions drops.
    luma_in = img[..., 0]
    luma_out = out[..., 0]
    hard_in = int(((luma_in[8, 1:] - luma_in[8, :-1]).abs() > 0.9).sum())
    hard_out = int(((luma_out[8, 1:] - luma_out[8, :-1]).abs() > 0.9).sum())
    assert hard_out < hard_in or (luma_out[8].diff().abs().max()
                                  < luma_in[8].diff().abs().max())
    # Energy is conserved per edge (symmetric blend).
    assert abs(float(out.mean()) - float(img.mean())) < 0.05


def test_smaa_pass_end_to_end() -> None:
    out = render(Engine.cpu(), _scene(), _cam(),
                 RenderConfig(width=96, height=64, aa="smaa",
                              output_color_space="linear"))
    assert out.rgb.shape == (64, 96, 3)
    assert torch.isfinite(out.rgb).all()
