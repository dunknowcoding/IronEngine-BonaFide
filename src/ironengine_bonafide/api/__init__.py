"""Public API surface — single import for users.

>>> from ironengine_bonafide.api import (
...     Engine, Scene, PerspectiveCamera, OrthographicCamera, SensorCamera,
...     PointCloud, Mesh, PBRMaterial, Volume, DollRig,
...     DirectionalLight, PointLight, SpotLight, AreaLight, IBL,
...     RenderConfig, RenderOutputs, render, render_differentiable,
...     mount_assets,
... )
"""
from __future__ import annotations

from typing import Any

from ironengine_bonafide._version import __version__

# ---- Engine + render entry points ---------------------------------------
from ironengine_bonafide.api.engine import Engine
from ironengine_bonafide.api.outputs import RenderOutputs, _OutputTensor  # noqa: F401
from ironengine_bonafide.api.passes_default import default_passes
from ironengine_bonafide.api.registry import (
    BACKEND_REGISTRY,
    LOADER_REGISTRY,
    PASS_REGISTRY,
)
from ironengine_bonafide.api.render import render, render_differentiable

# ---- Data model ----------------------------------------------------------
from ironengine_bonafide.assets.mount import AssetLibrary
from ironengine_bonafide.assets.mount import mount as _mount
from ironengine_bonafide.core.backend import Backend  # noqa: F401 — public re-export
from ironengine_bonafide.core.camera import (
    Camera,
    OrthographicCamera,
    PerspectiveCamera,
    SensorCamera,
)
from ironengine_bonafide.core.color import (
    aces_filmic,
    linear_to_srgb,
    srgb_to_linear,
    to_uint8_srgb,
    tonemap_aces_to_srgb_uint8,
)
from ironengine_bonafide.core.config import (
    CompletionConfig,
    FogConfig,
    GsplatConfig,
    LodConfig,
    RenderConfig,
    SurfelConfig,
)
from ironengine_bonafide.core.light import (
    IBL,
    AreaLight,
    DirectionalLight,
    PointLight,
    SpotLight,
)
from ironengine_bonafide.core.material import PBRMaterial
from ironengine_bonafide.core.mesh import Mesh
from ironengine_bonafide.core.pointcloud import PointCloud
from ironengine_bonafide.core.scene import Background, Scene
from ironengine_bonafide.core.softbody import DollRig
from ironengine_bonafide.core.volume import Volume

# ---- Pass framework ------------------------------------------------------
from ironengine_bonafide.passes.aa_pass import SmaaPass, TaaPass
from ironengine_bonafide.passes.base import FrameTargets, PassContext, RenderPass
from ironengine_bonafide.passes.completion_pass import CompletionPass
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


# --------------------------------------------------------------- assets
def mount_assets(root: Any) -> AssetLibrary:
    """Mount a folder as an :class:`AssetLibrary` (textures / meshes / volumes / scenes)."""
    return _mount(root)


__all__ = [
    # version
    "__version__",
    # cameras
    "Camera", "OrthographicCamera", "PerspectiveCamera", "SensorCamera",
    # data model
    "Background", "DollRig", "IBL", "Mesh", "PBRMaterial", "PointCloud", "Scene", "Volume",
    # lights
    "AreaLight", "DirectionalLight", "PointLight", "SpotLight",
    # config
    "CompletionConfig", "FogConfig", "GsplatConfig", "LodConfig",
    "RenderConfig", "SurfelConfig",
    # engine + render
    "Engine", "RenderOutputs", "render", "render_differentiable",
    # passes
    "BloomPass", "CompletionPass", "CsmShadowPass", "FxaaPass",
    "NeuralDenoisePass", "NeuralRelightPass", "NeuralUpscalePass",
    "ParticlePass", "PbrPass", "RenderPass", "SmaaPass", "SoftBodyPass",
    "SplatPass", "SkyPass", "TaaPass", "TonemapPass", "VolumetricPass", "WaterPass",
    "FrameTargets", "PassContext", "default_passes",
    # registries
    "BACKEND_REGISTRY", "LOADER_REGISTRY", "PASS_REGISTRY",
    # color
    "aces_filmic", "linear_to_srgb", "srgb_to_linear", "to_uint8_srgb",
    "tonemap_aces_to_srgb_uint8",
    # asset folders
    "AssetLibrary", "mount_assets",
]
