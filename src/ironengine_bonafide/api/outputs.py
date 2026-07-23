"""`RenderOutputs` and the `_OutputTensor` torch.Tensor subclass.

The output tensor carries display-conversion helpers (sRGB / ACES / save)
so users don't have to import the color module separately.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ironengine_bonafide.core.color import (
    linear_to_srgb,
    to_uint8_srgb,
    tonemap_aces_to_srgb_uint8,
)
from ironengine_bonafide.core.profile import ProfileReport


class _OutputTensor(torch.Tensor):
    """torch.Tensor with display-conversion helpers.

    Helpers:
      * :meth:`to_sRGB`             — float32 sRGB-encoded
      * :meth:`to_uint8_srgb`        — uint8 sRGB
      * :meth:`to_aces_srgb_uint8`   — ACES tonemap → uint8 sRGB
      * :meth:`to_uint8_display`     — uint8 from ALREADY display-ready sRGB
                                       (use when ``output_color_space=="sRGB"``;
                                       applying ACES again double-tonemaps)
      * :meth:`save`                 — write a PNG via imageio
    """

    @staticmethod
    def __new__(cls, t: torch.Tensor) -> _OutputTensor:
        return torch.Tensor._make_subclass(cls, t, t.requires_grad)

    def to_sRGB(self) -> torch.Tensor:
        return linear_to_srgb(self)

    def to_uint8_srgb(self) -> torch.Tensor:
        return to_uint8_srgb(self)

    def to_aces_srgb_uint8(self, exposure: float = 1.0) -> torch.Tensor:
        return tonemap_aces_to_srgb_uint8(self, exposure=exposure)

    def to_uint8_display(self) -> torch.Tensor:
        """uint8 encode of values that are already display-ready sRGB in
        [0, 1] (i.e. after TonemapPass with ``output_color_space=="sRGB"``).
        No second tonemap / gamma is applied."""
        return (self.clamp(0.0, 1.0) * 255.0 + 0.5).to(torch.uint8)

    def save(self, path: str, exposure: float = 1.0, *,
             display_ready: bool = False) -> None:
        """Save as PNG (sRGB uint8).

        Pass ``display_ready=True`` when the tensor is already
        display-encoded sRGB (``output_color_space=="sRGB"`` renders);
        otherwise ACES + sRGB encoding is applied here."""
        import imageio.v3 as iio
        if display_ready:
            img = self.to_uint8_display().detach().cpu().numpy()
        else:
            img = self.to_aces_srgb_uint8(exposure).detach().cpu().numpy()
        iio.imwrite(path, img)


@dataclass(slots=True)
class RenderOutputs:
    """Result of a single :func:`render` call.

    Each tensor lives on the engine's device. Tensors that the user did
    not request via ``RenderConfig.sensor_outputs`` are ``None``.

    ``rgb`` is linear HDR float32 when ``color_space == "linear"``; when
    ``color_space == "sRGB"`` it is final display-ready sRGB in [0, 1]
    (ACES + sRGB encoding already applied by TonemapPass) — consumers
    must NOT apply a second conversion.
    """
    rgb: _OutputTensor                                  # (H, W, 3) see color_space
    depth: torch.Tensor | None = None                   # (H, W) NDC z in [-1, 1], +inf where empty
    normals: torch.Tensor | None = None                 # (H, W, 3) world-space
    ids: torch.Tensor | None = None                     # (H, W) int64 instance IDs
    albedo: torch.Tensor | None = None                  # (H, W, 3) GBuffer albedo
    color_space: str = "linear"                         # "linear" | "sRGB"
    profile: ProfileReport | None = None
    skipped_passes: list[str] = field(default_factory=list)
