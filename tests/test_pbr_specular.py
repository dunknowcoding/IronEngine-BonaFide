"""Cook-Torrance GGX specular + emissive shading on the CPU backend."""
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


def _quad_mesh(material: PBRMaterial) -> Mesh:
    positions = np.array(
        [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (4, 1))
    indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return Mesh.from_arrays(positions, indices, normals=normals, material=material)


def _render(material: PBRMaterial, lights: list) -> torch.Tensor:
    scene = Scene().add(_quad_mesh(material))
    for lt in lights:
        scene.add(lt)
    cam = PerspectiveCamera(position=(0, 0, 3.0), look_at=(0, 0, 0), fov_deg=45)
    cfg = RenderConfig(width=64, height=48)
    return render(Engine.cpu(), scene, cam, cfg).rgb


def test_specular_highlight_beats_diffuse_only() -> None:
    # Light travels -Z toward the +Z-facing quad; camera looks down -Z too,
    # so the half-vector aligns with the normal at the quad center.
    light = DirectionalLight(direction=(0, 0, -1), intensity=3.0, cast_shadow=False)
    spec_rgb = _render(
        PBRMaterial(albedo=(0.8, 0.8, 0.8), metallic=1.0, roughness=0.2),
        [light],
    )
    diff_rgb = _render(
        PBRMaterial(albedo=(0.8, 0.8, 0.8), metallic=0.0, roughness=1.0),
        [light],
    )
    spec_peak = float(spec_rgb.max())
    diff_peak = float(diff_rgb.max())
    assert spec_peak > diff_peak * 1.5, (
        f"expected a GGX highlight peak above the diffuse-only render "
        f"(spec={spec_peak:.3f}, diff={diff_peak:.3f})"
    )


def test_emissive_adds_in_darkness() -> None:
    rgb = _render(
        PBRMaterial(albedo=(0.1, 0.1, 0.1), emissive=(1.0, 0.0, 0.0)),
        lights=[],
    )
    assert float(rgb[..., 0].max()) >= 1.0 - 1e-3
    assert float(rgb[..., 0].max()) > float(rgb[..., 1].max())
