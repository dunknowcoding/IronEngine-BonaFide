"""Default pass graph factory.

Order:

  sky → shadow → softbody-step → lod → completion → splat → pbr →
  particles → water → volumetric → neural_relight → AA → bloom →
  neural_denoise → neural_upscale → tonemap

Each pass is gated on its own ``is_active(ctx)`` and ``required_capabilities()``;
the engine skips passes whose checks fail and records the skip reason.
"""
from __future__ import annotations

from ironengine_bonafide.passes.aa_pass import SmaaPass, TaaPass
from ironengine_bonafide.passes.base import RenderPass
from ironengine_bonafide.passes.completion_pass import CompletionPass
from ironengine_bonafide.passes.lod_pass import LodPass
from ironengine_bonafide.passes.neural_denoise import NeuralDenoisePass
from ironengine_bonafide.passes.neural_relight import NeuralRelightPass
from ironengine_bonafide.passes.neural_upscale import NeuralUpscalePass
from ironengine_bonafide.passes.particle_pass import ParticlePass
from ironengine_bonafide.passes.pbr_pass import PbrPass
from ironengine_bonafide.passes.postprocess import BloomPass, FxaaPass, TonemapPass
from ironengine_bonafide.passes.shadow import CsmShadowPass
from ironengine_bonafide.passes.sky_pass import SkyPass
from ironengine_bonafide.passes.softbody_pass import SoftBodyPass
from ironengine_bonafide.passes.splat_pass import SplatPass
from ironengine_bonafide.passes.volumetric_pass import VolumetricPass
from ironengine_bonafide.passes.water_pass import WaterPass


def default_passes() -> list[RenderPass]:
    """Build a fresh list of pass instances for a new Engine."""
    return [
        SkyPass(),
        CsmShadowPass(),
        SoftBodyPass(),
        LodPass(),
        CompletionPass(),
        SplatPass(),
        PbrPass(),
        ParticlePass(),
        WaterPass(),
        VolumetricPass(),
        NeuralRelightPass(),
        FxaaPass(),
        TaaPass(),
        SmaaPass(),
        BloomPass(),
        NeuralDenoisePass(),
        NeuralUpscalePass(),
        TonemapPass(),
    ]
