"""Particle system asset — CPU-simulated point sprites.

A deterministic CPU particle set: positions/velocities/ages stepped by
ParticlePass each frame (``RenderConfig.simulation_dt`` per render) with
gravity, drag, lifetime expiry, and optional emitter respawn. Rendering
is a depth-tested splat per particle with an age-based fade.

Determinism: respawn randomness draws from a ``torch.Generator`` seeded
by ``config.seed`` at first step and advanced each step — same seed, same
trajectory, bit-exact within a backend.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

Vec3 = tuple[float, float, float]


@dataclass(slots=True)
class ParticleSystem:
    positions: torch.Tensor                          # (N, 3) float32 world
    velocities: torch.Tensor | None = None           # (N, 3) — zeros default
    ages: torch.Tensor | None = None                 # (N,) — zeros default
    lifetime: float = 2.0                            # seconds; dead at age > lifetime
    gravity: Vec3 = (0.0, -9.8, 0.0)
    drag: float = 0.05                               # velocity damping factor /s
    # ---- emitter (respawn) ----------------------------------------------
    emitter_position: Vec3 = (0.0, 0.0, 0.0)
    emitter_spread: float = 0.3                      # respawn position jitter (m)
    emitter_velocity: Vec3 = (0.0, 1.5, 0.0)         # respawn base velocity
    respawn: bool = True                             # dead particles re-emit
    # ---- rendering --------------------------------------------------------
    color: Vec3 = (1.0, 0.75, 0.35)
    size_px: float = 3.0
    name: str = "particles"
    # Internal RNG state (not serialized).
    _rng: torch.Generator | None = None

    def __post_init__(self) -> None:
        if self.velocities is None:
            self.velocities = torch.zeros_like(self.positions)
        if self.ages is None:
            self.ages = torch.zeros(self.positions.shape[0], dtype=torch.float32,
                                    device=self.positions.device)

    # ------------------------------------------------------------ ctors
    @classmethod
    def from_arrays(
        cls,
        positions: np.ndarray | torch.Tensor,
        velocities: np.ndarray | torch.Tensor | None = None,
        *,
        name: str = "particles",
        **kw: object,
    ) -> ParticleSystem:
        pos = torch.as_tensor(np.asarray(positions), dtype=torch.float32)
        if pos.ndim != 2 or pos.shape[-1] != 3:
            raise ValueError(f"Expected (N, 3) positions, got {tuple(pos.shape)}")
        vel = (torch.as_tensor(np.asarray(velocities), dtype=torch.float32)
               if velocities is not None else None)
        return cls(positions=pos, velocities=vel, name=name, **kw)  # type: ignore[arg-type]

    @classmethod
    def fountain(cls, n: int = 256, **kw: object) -> ParticleSystem:
        """N particles clustered at the emitter (spread = emitter_spread).

        The initial spread draws from a fixed-seed local generator, so two
        constructions start from identical positions (render-time respawn
        randomness is seeded separately by ``RenderConfig.seed``).
        """
        ep = kw.get("emitter_position", (0.0, 0.0, 0.0))
        spread = float(kw.get("emitter_spread", 0.3))
        base = torch.tensor(ep, dtype=torch.float32)
        g = torch.Generator().manual_seed(0)
        pos = base.unsqueeze(0).repeat(n, 1) \
            + spread * (torch.rand(n, 3, generator=g) - 0.5) * 2.0
        return cls(positions=pos, **kw)  # type: ignore[arg-type]

    @property
    def num_particles(self) -> int:
        return int(self.positions.shape[0])
