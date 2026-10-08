"""Water surface asset — a Gerstner-wave plane rendered by WaterPass.

The surface is a finite rectangle on a plane (default: XZ at y=0, facing
+Y). Waves are evaluated analytically at the ray/plane intersection:
three directional Gerstner components perturb the surface normal (the
``steepness`` horizontal displacement of full Gerstner is folded into the
normal calculation, which is what matters visually at render scale).

``time`` is advanced by WaterPass each frame (``dt`` per render), so
repeated ``render()`` calls animate the waves deterministically.
"""
from __future__ import annotations

from dataclasses import dataclass

Vec3 = tuple[float, float, float]


@dataclass(slots=True)
class WaterSurface:
    # ---- geometry ------------------------------------------------------
    center: Vec3 = (0.0, 0.0, 0.0)
    normal: Vec3 = (0.0, 1.0, 0.0)               # plane facing
    half_size: tuple[float, float] = (20.0, 20.0)  # plane-rect extents (u, v)
    # ---- waves (Gerstner) ----------------------------------------------
    wave_amplitude: float = 0.06                 # metres, per component
    wave_length: float = 1.7                     # metres, primary component
    wave_speed: float = 1.2                      # phase speed multiplier
    steepness: float = 0.5                       # 0 = pure sine, 1 = sharp crests
    time: float = 0.0                            # animation clock (seconds)
    # ---- shading ---------------------------------------------------------
    color: Vec3 = (0.02, 0.10, 0.14)             # deep body colour (absorption)
    scatter_color: Vec3 = (0.05, 0.22, 0.24)     # shallow / crest scatter tint
    reflectivity: float = 1.0                    # fresnel scale (0 = matte)
    name: str = "water"
