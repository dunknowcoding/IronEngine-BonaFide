"""IronEngine-BonaFide — differentiable, neural-first Python render engine."""
from __future__ import annotations

from ironengine_bonafide._version import __version__
from ironengine_bonafide.api import (
    IBL,
    AreaLight,
    DirectionalLight,
    DollRig,
    Engine,
    Mesh,
    OrthographicCamera,
    PBRMaterial,
    PerspectiveCamera,
    PointCloud,
    PointLight,
    RenderConfig,
    RenderOutputs,
    Scene,
    SensorCamera,
    SpotLight,
    Volume,
    mount_assets,
    render,
    render_differentiable,
)
from ironengine_bonafide.bundle import RenderBundle

__all__ = [
    "AreaLight", "DirectionalLight", "DollRig", "Engine", "IBL", "Mesh",
    "OrthographicCamera", "PBRMaterial", "PerspectiveCamera", "PointCloud",
    "PointLight", "RenderBundle", "RenderConfig", "RenderOutputs", "Scene",
    "SensorCamera", "SpotLight", "Volume", "__version__",
    "mount_assets", "render", "render_differentiable",
]
