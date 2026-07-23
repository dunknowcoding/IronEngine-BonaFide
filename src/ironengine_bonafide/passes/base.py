"""Render pass ABC + per-frame context.

A pass declares the capabilities it needs (e.g. "raster", "gsplat"). The
engine skips a pass whose capabilities aren't satisfied by the active
backend, recording the skip in the profile report.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch

from ironengine_bonafide.core.backend import Backend
from ironengine_bonafide.core.camera import Camera
from ironengine_bonafide.core.config import RenderConfig
from ironengine_bonafide.core.scene import Scene


@dataclass(slots=True)
class FrameTargets:
    """Per-frame in-flight buffers. Passes mutate these in-place."""
    rgb:     torch.Tensor                       # (H, W, 3) float32 linear HDR
    depth:   torch.Tensor                       # (H, W) float32 NDC z in [-1, 1], +inf where empty
    normals: torch.Tensor                       # (H, W, 3) float32 world-space
    ids:     torch.Tensor                       # (H, W) int64 instance ID; 0 = background
    albedo:  torch.Tensor                       # (H, W, 3) float32 GBuffer albedo
    shadow_maps: list = field(default_factory=list)   # list[ShadowMap] populated by CsmShadowPass


@dataclass(slots=True)
class PassContext:
    backend: Backend
    scene: Scene
    camera: Camera
    config: RenderConfig
    targets: FrameTargets
    aspect: float
    skipped: list[str] = field(default_factory=list)
    # Frame-local scratch space for cross-pass state (e.g. LodPass →
    # SplatPass index subsets) that must NOT mutate scene assets.
    frame_state: dict = field(default_factory=dict)


class RenderPass(ABC):
    """Single rendering step (shadow, splat, pbr, postprocess, …)."""
    name: str = "pass"

    def required_capabilities(self) -> tuple[str, ...]:
        return ()

    def is_active(self, ctx: PassContext) -> bool:
        """Return False to skip the pass for this frame (e.g. when the
        relevant config toggle is off)."""
        return True

    @abstractmethod
    def run(self, ctx: PassContext) -> None: ...
