"""Camera primitives.

Right-handed Y-up convention. Forward is `-Z` in eye space (OpenGL style),
matching IronEngine-Sim and 3DCreator.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

Vec3 = tuple[float, float, float]


def _np_vec(v: Sequence[float], dim: int = 3) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    if arr.shape != (dim,):
        raise ValueError(f"Expected length-{dim} vector, got shape {arr.shape}")
    return arr


def look_at(eye: Vec3, target: Vec3, up: Vec3 = (0.0, 1.0, 0.0)) -> np.ndarray:
    """Build a 4x4 view matrix (right-handed, looking down -Z in eye space)."""
    e = _np_vec(eye); t = _np_vec(target); u = _np_vec(up)
    f = t - e
    f /= np.linalg.norm(f) + 1e-9
    s = np.cross(f, u); s /= np.linalg.norm(s) + 1e-9
    u2 = np.cross(s, f)
    m = np.eye(4, dtype=np.float64)
    m[0, :3] = s
    m[1, :3] = u2
    m[2, :3] = -f
    m[:3, 3] = -m[:3, :3] @ e
    return m


def perspective(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_deg) * 0.5)
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def orthographic(half_w: float, half_h: float, near: float, far: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = 1.0 / half_w
    m[1, 1] = 1.0 / half_h
    m[2, 2] = -2.0 / (far - near)
    m[2, 3] = -(far + near) / (far - near)
    return m


@dataclass(slots=True)
class PerspectiveCamera:
    position: Vec3 = (0.0, 1.0, 3.0)
    look_at: Vec3 = (0.0, 0.0, 0.0)
    up: Vec3 = (0.0, 1.0, 0.0)
    fov_deg: float = 45.0
    near: float = 0.05
    far: float = 200.0

    def view_matrix(self) -> np.ndarray:
        return look_at(self.position, self.look_at, self.up)

    def proj_matrix(self, aspect: float) -> np.ndarray:
        return perspective(self.fov_deg, aspect, self.near, self.far)

    def view_proj(self, aspect: float) -> np.ndarray:
        return self.proj_matrix(aspect) @ self.view_matrix()

    def view_proj_torch(self, aspect: float, device: str | torch.device = "cpu") -> torch.Tensor:
        return torch.from_numpy(self.view_proj(aspect)).to(device=device, dtype=torch.float32)


@dataclass(slots=True)
class OrthographicCamera:
    position: Vec3 = (0.0, 1.0, 3.0)
    look_at: Vec3 = (0.0, 0.0, 0.0)
    up: Vec3 = (0.0, 1.0, 0.0)
    half_width: float = 2.0
    half_height: float = 2.0
    near: float = 0.05
    far: float = 200.0

    def view_matrix(self) -> np.ndarray:
        return look_at(self.position, self.look_at, self.up)

    def proj_matrix(self, _aspect: float) -> np.ndarray:
        return orthographic(self.half_width, self.half_height, self.near, self.far)

    def view_proj(self, aspect: float) -> np.ndarray:
        return self.proj_matrix(aspect) @ self.view_matrix()

    def view_proj_torch(self, aspect: float, device: str | torch.device = "cpu") -> torch.Tensor:
        return torch.from_numpy(self.view_proj(aspect)).to(device=device, dtype=torch.float32)


@dataclass(slots=True)
class SensorCamera:
    """A robot/AR sensor camera. Pose is world-space; intrinsics are pinhole."""
    pose: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    fov_deg: float = 60.0
    near: float = 0.05
    far: float = 200.0

    def view_matrix(self) -> np.ndarray:
        return np.linalg.inv(self.pose)

    def proj_matrix(self, aspect: float) -> np.ndarray:
        return perspective(self.fov_deg, aspect, self.near, self.far)

    def view_proj(self, aspect: float) -> np.ndarray:
        return self.proj_matrix(aspect) @ self.view_matrix()

    def view_proj_torch(self, aspect: float, device: str | torch.device = "cpu") -> torch.Tensor:
        return torch.from_numpy(self.view_proj(aspect)).to(device=device, dtype=torch.float32)


Camera = PerspectiveCamera | OrthographicCamera | SensorCamera
