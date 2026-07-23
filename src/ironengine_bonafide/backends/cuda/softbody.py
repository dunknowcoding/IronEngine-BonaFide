"""NVIDIA Warp XPBD soft-body solver.

Distance constraints + gravity + ground plane. Each :class:`DollRig`
gets a persistent ``WarpSolverState`` cached against ``id(rig)`` so the
solver survives between frames.

If ``warp-lang`` isn't importable (or runs CPU-only without a GPU), the
solver falls back to a tiny pure-NumPy XPBD step so v0.1 stays
functional everywhere — the user just sees fewer iterations and slower
convergence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ironengine_bonafide.logging import logger

_WARP_INITIALIZED = False
_WARP_OK = False


def _ensure_warp() -> bool:
    global _WARP_INITIALIZED, _WARP_OK
    if _WARP_INITIALIZED:
        return _WARP_OK
    _WARP_INITIALIZED = True
    try:
        import warp as wp  # type: ignore[import-not-found]
        wp.init()
        _WARP_OK = True
        logger.info(f"NVIDIA Warp initialized (devices: {[str(d) for d in wp.get_devices()]})")
    except Exception as exc:                                   # noqa: BLE001
        logger.warning(f"warp-lang not usable: {exc}")
        _WARP_OK = False
    return _WARP_OK


@dataclass(slots=True)
class WarpSolverState:
    """Persistent solver state for one rig."""
    n_particles: int
    n_edges: int
    rest_lengths: np.ndarray                  # (E,)
    velocities: np.ndarray                    # (P, 3)
    inv_mass: np.ndarray                      # (P,)
    stiffness: float
    damping: float
    last_positions: np.ndarray                # (P, 3) — stored on host for restart
    warp_objects: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------- builders
def build_state(rig) -> WarpSolverState:                       # type: ignore[no-untyped-def]
    """Initialize per-rig solver state from rest geometry."""
    pts = rig.particles.detach().cpu().numpy().astype(np.float32)
    edges = rig.edges.detach().cpu().numpy().astype(np.int64)
    if rig.masses is not None:
        masses = rig.masses.detach().cpu().numpy().astype(np.float32)
    else:
        masses = np.ones(pts.shape[0], dtype=np.float32)
    inv_mass = np.where(masses > 0, 1.0 / masses, 0.0)

    diffs = pts[edges[:, 0]] - pts[edges[:, 1]]
    rest_lengths = np.linalg.norm(diffs, axis=1).astype(np.float32)

    state = WarpSolverState(
        n_particles=int(pts.shape[0]),
        n_edges=int(edges.shape[0]),
        rest_lengths=rest_lengths,
        velocities=np.zeros_like(pts),
        inv_mass=inv_mass,
        stiffness=float(rig.stiffness),
        damping=float(rig.damping),
        last_positions=pts.copy(),
    )
    # Always stash the int32 edge array — the NumPy fallback needs it.
    state.warp_objects["_numpy_edges"] = edges.astype(np.int64)

    if _ensure_warp():
        import warp as wp  # type: ignore[import-not-found]
        device = wp.get_preferred_device()
        state.warp_objects["device"] = device
        state.warp_objects["positions"] = wp.array(pts, dtype=wp.vec3, device=device)
        state.warp_objects["prev_positions"] = wp.array(pts.copy(), dtype=wp.vec3, device=device)
        state.warp_objects["velocities"] = wp.zeros(pts.shape[0], dtype=wp.vec3, device=device)
        state.warp_objects["inv_mass"] = wp.array(inv_mass, dtype=wp.float32, device=device)
        state.warp_objects["edges"] = wp.array(edges.astype(np.int32), dtype=wp.vec2i, device=device)
        state.warp_objects["rest_lengths"] = wp.array(rest_lengths, dtype=wp.float32, device=device)
    return state


# --------------------------------------------------------------- step
def step(state: WarpSolverState, dt: float = 1.0 / 60.0,
         iterations: int = 8, gravity: tuple[float, float, float] = (0.0, -9.81, 0.0),
         ground_y: float = 0.0) -> torch.Tensor:
    """Advance the solver one frame. Returns the updated particle
    positions as a torch.Tensor of shape ``(P, 3)``."""
    if _WARP_OK:
        return _step_warp(state, dt, iterations, gravity, ground_y)
    return _step_numpy(state, dt, iterations, gravity, ground_y)


def _step_warp(state: WarpSolverState, dt: float, iterations: int,
               gravity: tuple[float, float, float], ground_y: float) -> torch.Tensor:
    import warp as wp  # type: ignore[import-not-found]
    obj = state.warp_objects
    device = obj["device"]

    # Predictor: v += g·dt; x_pred = x + v·dt
    @wp.kernel
    def predict(pos: wp.array(dtype=wp.vec3),
                prev: wp.array(dtype=wp.vec3),
                vel: wp.array(dtype=wp.vec3),
                inv_m: wp.array(dtype=wp.float32),
                g: wp.vec3, dt: wp.float32):
        i = wp.tid()
        if inv_m[i] == 0.0:
            return
        prev[i] = pos[i]
        vel[i] = vel[i] + g * dt
        pos[i] = pos[i] + vel[i] * dt

    # Distance constraint solve (one Jacobi-style iteration)
    @wp.kernel
    def solve_distance(pos: wp.array(dtype=wp.vec3),
                       inv_m: wp.array(dtype=wp.float32),
                       edges: wp.array(dtype=wp.vec2i),
                       rest: wp.array(dtype=wp.float32),
                       stiff: wp.float32):
        e = wp.tid()
        a = edges[e][0]; b = edges[e][1]
        wa = inv_m[a]; wb = inv_m[b]
        if wa + wb == 0.0:
            return
        delta = pos[a] - pos[b]
        d = wp.length(delta)
        if d < 1e-6:
            return
        n = delta / d
        c = d - rest[e]
        denom = wa + wb
        lam = stiff * c / denom
        wp.atomic_sub(pos, a, n * (lam * wa))
        wp.atomic_add(pos, b, n * (lam * wb))

    # Ground plane + velocity update
    @wp.kernel
    def finalize(pos: wp.array(dtype=wp.vec3),
                 prev: wp.array(dtype=wp.vec3),
                 vel: wp.array(dtype=wp.vec3),
                 ground: wp.float32, damp: wp.float32, dt: wp.float32):
        i = wp.tid()
        if pos[i][1] < ground:
            pos[i] = wp.vec3(pos[i][0], ground, pos[i][2])
        vel[i] = (pos[i] - prev[i]) / dt * (1.0 - damp)

    g_vec = wp.vec3(*gravity)
    wp.launch(predict, dim=state.n_particles,
              inputs=[obj["positions"], obj["prev_positions"], obj["velocities"],
                      obj["inv_mass"], g_vec, float(dt)],
              device=device)
    for _ in range(iterations):
        wp.launch(solve_distance, dim=state.n_edges,
                  inputs=[obj["positions"], obj["inv_mass"], obj["edges"],
                          obj["rest_lengths"], float(state.stiffness)],
                  device=device)
    wp.launch(finalize, dim=state.n_particles,
              inputs=[obj["positions"], obj["prev_positions"], obj["velocities"],
                      float(ground_y), float(state.damping), float(dt)],
              device=device)
    out = obj["positions"].numpy()
    state.last_positions = out
    return torch.from_numpy(out).to(torch.float32)


def _step_numpy(state: WarpSolverState, dt: float, iterations: int,
                gravity: tuple[float, float, float], ground_y: float) -> torch.Tensor:
    """Pure NumPy XPBD step. Slow but correct for tests / no-GPU dev."""
    pos = state.last_positions.copy()
    prev = pos.copy()
    g = np.asarray(gravity, dtype=np.float32)

    # predictor
    state.velocities += g[None, :] * dt
    pos += state.velocities * dt

    # distance constraints — Jacobi-style vectorised relaxation.
    edge_array = _edges_cache(state)
    a = edge_array[:, 0]
    b = edge_array[:, 1]
    for _ in range(iterations):
        d_vec = pos[a] - pos[b]
        d = np.linalg.norm(d_vec, axis=1)
        valid = d > 1e-6
        n = d_vec[valid] / d[valid][:, None]
        c = d[valid] - state.rest_lengths[valid]
        wa = state.inv_mass[a[valid]]; wb = state.inv_mass[b[valid]]
        denom = (wa + wb).clip(min=1e-9)
        lam = state.stiffness * c / denom
        np.subtract.at(pos, a[valid], n * (lam * wa)[:, None])
        np.add.at(pos, b[valid], n * (lam * wb)[:, None])

    # ground
    pos[:, 1] = np.maximum(pos[:, 1], ground_y)
    state.velocities = (pos - prev) / dt * (1.0 - state.damping)
    state.last_positions = pos
    return torch.from_numpy(pos.astype(np.float32))


def _edges_cache(state: WarpSolverState) -> np.ndarray:
    """Reconstruct the (E, 2) edge array from the warp_objects cache or
    keep a NumPy copy on first use."""
    arr = state.warp_objects.get("_numpy_edges")
    if arr is not None:
        return arr
    # Fall back: derive from rest_lengths' size — but we need the original
    # edges. The Warp path always stashes them; here build from rig at
    # construction time. We attach lazily via build_state in the future.
    raise RuntimeError("NumPy XPBD path needs original edges in state.warp_objects['_numpy_edges']")
