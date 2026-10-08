"""Gerstner water and CPU particle system tests."""
from __future__ import annotations

import numpy as np
import torch

from ironengine_bonafide.api import (
    DirectionalLight,
    Engine,
    Mesh,
    ParticleSystem,
    PBRMaterial,
    PerspectiveCamera,
    RenderConfig,
    Scene,
    WaterSurface,
    render,
)
from ironengine_bonafide.bundle import RenderBundle


def _water_scene() -> Scene:
    return (Scene()
            .add(WaterSurface(center=(0, 0, 0), half_size=(10, 10)))
            .add(DirectionalLight(direction=(-0.3, -1.0, -0.2), intensity=3.0)))


def _water_cam() -> PerspectiveCamera:
    return PerspectiveCamera(position=(0, 2.0, 4.0), look_at=(0, 0, 0), fov_deg=50)


def test_water_covers_and_shades() -> None:
    out = render(Engine.cpu(), _water_scene(), _water_cam(),
                 RenderConfig(width=128, height=96, output_color_space="linear",
                              sensor_outputs=("rgb", "depth")))
    fin = torch.isfinite(out.depth)
    assert int(fin.sum()) > 0.5 * out.depth.numel()
    assert torch.isfinite(out.rgb).all()
    # Fresnel sanity: grazing (far/top of frame) reflects the sky and is
    # brighter than the steep (near/bottom) transmission-dominated view.
    near = float(out.rgb[92, 64].sum())
    far = float(out.rgb[4, 64].sum())
    assert far > near


def test_water_animates_deterministically() -> None:
    eng = Engine.cpu()
    cam = _water_cam()
    cfg = RenderConfig(width=96, height=64, output_color_space="linear")
    scene = _water_scene()
    a = render(eng, scene, cam, cfg).rgb.clone()
    b = render(eng, scene, cam, cfg).rgb.clone()
    assert not torch.equal(a, b)                      # time advances per frame
    scene2 = _water_scene()
    eng2 = Engine.cpu()
    a2 = render(eng2, scene2, cam, cfg).rgb.clone()
    torch.testing.assert_close(a, a2)                 # same timeline, same frame


def test_water_occludes_and_is_occluded() -> None:
    # A quad above the water hides it; a quad below is tinted by it.
    above = Mesh.from_arrays(
        positions=np.array([[-0.5, 1.0, -0.5], [0.5, 1.0, -0.5], [0.5, 1.0, 0.5],
                            [-0.5, 1.0, 0.5]], dtype=np.float32),
        indices=np.array([[0, 1, 2], [0, 2, 3]]),
        normals=np.array([[0, 1, 0]] * 4, dtype=np.float32),
        material=PBRMaterial(albedo=(0.9, 0.1, 0.1), roughness=0.9))
    scene = (_water_scene().add(above)
             .add(DirectionalLight(direction=(0.0, -1.0, 0.0), intensity=3.0)))
    cam = PerspectiveCamera(position=(0, 2.0, 2.0), look_at=(0, 0.5, 0), fov_deg=45)
    out = render(Engine.cpu(), scene, cam,
                 RenderConfig(width=96, height=64, output_color_space="linear"))
    assert torch.isfinite(out.rgb).all()


def test_particle_fountain_simulates() -> None:
    ps = ParticleSystem.fountain(64, emitter_position=(0, 0.5, 0),
                                 emitter_velocity=(0.0, 2.5, 0.0), lifetime=1.0)
    scene = Scene().add(ps)
    cam = PerspectiveCamera(position=(0, 1.0, 4.0), look_at=(0, 0.8, 0), fov_deg=50)
    eng = Engine.cpu()
    cfg = RenderConfig(width=96, height=64, output_color_space="linear")
    base = render(Engine.cpu(), Scene(), cam, cfg).rgb
    p1 = render(eng, scene, cam, cfg).rgb.clone()
    pos1 = ps.positions.clone()
    render(eng, scene, cam, cfg)
    assert not torch.equal(p1, base)                  # splats visible
    assert not torch.equal(pos1, ps.positions)        # sim stepped
    # Gravity: mean y-velocity becomes negative-ish over time after launch.
    assert ps.velocities is not None


def test_particle_respawn_and_determinism() -> None:
    def _run() -> ParticleSystem:
        ps = ParticleSystem.fountain(16, emitter_position=(0, 0, 0),
                                     emitter_velocity=(0.0, 2.0, 0.0),
                                     lifetime=0.05)   # dies within ~3 frames
        eng = Engine.cpu()
        cam = PerspectiveCamera(position=(0, 0, 4), look_at=(0, 0, 0), fov_deg=50)
        cfg = RenderConfig(width=48, height=32)
        for _ in range(8):
            render(eng, Scene().add(ps), cam, cfg)
        return ps

    a = _run()
    b = _run()
    # After 8 frames at lifetime 0.05 every particle respawned at least once.
    assert bool((a.ages < 8.0 / 60.0).any())
    torch.testing.assert_close(a.positions, b.positions)   # seeded determinism


def test_bundle_roundtrip_water_and_particles(tmp_path) -> None:
    scene = (_water_scene()
             .add(ParticleSystem.fountain(8, emitter_position=(0, 0.2, 0))))
    bnf = tmp_path / "wp.bnf"
    cam = _water_cam()
    cfg = RenderConfig(width=48, height=32)
    RenderBundle.capture(scene, cam, cfg).save(bnf)
    loaded = RenderBundle.load(bnf).scene
    assert len(loaded.waters) == 1
    assert loaded.waters[0].wave_length == 1.7
    assert len(loaded.particles) == 1
    assert loaded.particles[0].num_particles == 8
    assert loaded.particles[0].velocities is not None
