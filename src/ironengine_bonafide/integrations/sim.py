"""IronEngine-Sim drop-in shim.

Activates with:

    from ironengine_bonafide.integrations.sim import install
    install()                       # sensor-only, SAFE for the editor
    install(headless_only=False)    # also replace render_viewport

**Safe default.** ``install(headless_only=True)`` (the default) patches ONLY
``RenderWorld.render_sensor_rgb`` / ``render_sensor_depth``. The viewport
keeps Sim's own ModernGL renderer, so installing the shim can never blind
the SceneEditor viewport. Passing ``headless_only=False`` additionally
replaces ``render_viewport``: BonaFide renders the frame and stashes the
uint8 RGB image on ``render_world._last_bonafide_frame`` — blitting it into
the Qt/ModernGL framebuffer is the caller's responsibility (opt into this
only for headless pipelines or if you drive the blit yourself).

Implementation notes
--------------------
The shim *monkey-patches* ``RenderWorld`` rather than subclassing, because
``RenderWorld`` is constructed inside ``World.__init__``. Original methods
are snapshotted on install and restored on uninstall.

World resolution uses ``RenderWorld.scene`` (the ``SceneGraph``) and
``RenderWorld.assets`` (the ``AssetManager``) — both are constructor
arguments of the real ``RenderWorld``, so no gc-walk is required.
``install_for_world(world)`` additionally pins ``world.render._world`` for
callers whose ``RenderWorld`` was rebuilt against different attributes.

Sensor convention: Sim's ``_sensor_camera`` (render_world.py:411-436) uses
a **+X-forward** sensor frame; BonaFide's ``SensorCamera`` uses
**-Z-forward** eye space. ``_sensor_pose`` composes the full body TRS
(rotation quaternion + position, local offset rotated into the body frame)
and then converts between the two conventions.
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import numpy as np

from ironengine_bonafide.api import (
    DirectionalLight,
    Engine,
    Mesh,
    PerspectiveCamera,
    PointLight,
    RenderConfig,
    Scene,
    SensorCamera,
    render,
)
from ironengine_bonafide.integrations._display import srgb_to_uint8
from ironengine_bonafide.logging import logger

_ORIGINALS: dict[str, Any] = {}
_ENGINE: Engine | None = None


def _engine() -> Engine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = Engine.auto()
    return _ENGINE


def set_engine(engine: Engine) -> None:
    """Override the cached engine (e.g. force CPU for headless test runs)."""
    global _ENGINE
    _ENGINE = engine


# --------------------------------------------------------------- scene build
# Cache of TRS-baked geometry keyed by (asset_name, matrix bytes) so static
# scenes don't re-transform every frame.
_XFORM_CACHE: dict[tuple[str, bytes], tuple[np.ndarray, np.ndarray | None]] = {}

# Sim's sensor frame is +X-forward / +Y-up / +Z-left; BonaFide's camera eye
# space is -Z-forward / +Y-up / +X-right. This change-of-basis maps
# BonaFide camera axes onto Sim sensor axes: C @ (0,0,-1) = (1,0,0),
# C @ (0,1,0) = (0,1,0), C @ (1,0,0) = (0,0,1).
_SENSOR_TO_CAMERA = np.array([
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
], dtype=np.float64)


def _quat_to_mat3(q: Any) -> np.ndarray:
    """xyzw quaternion → 3x3 rotation matrix (NumPy)."""
    x, y, z, w = (float(v) for v in np.asarray(q, dtype=np.float64).reshape(4))
    n = float(np.sqrt(x * x + y * y + z * z + w * w)) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def _trs_matrix(tx: Any) -> tuple[np.ndarray, np.ndarray]:
    """Sim Transform (position, xyzw quat, scale) → (4x4 TRS, 3x3 rotation)."""
    rot = _quat_to_mat3(tx.rotation)
    scale = np.asarray(tx.scale, dtype=np.float64).reshape(3)
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = rot * scale[None, :]                    # R @ diag(s)
    m[:3, 3] = np.asarray(tx.position, dtype=np.float64).reshape(3)
    return m, rot


def _sim_world_components() -> Any:
    """Sim's world component classes, or name-only stubs when Sim isn't
    installed.

    The bridge looks components up through the (duck-typed) world store;
    dependency-free world doubles — tests, minimal CI installs — match by
    class *name*, which the stubs preserve. A real Sim world can only exist
    when ``ironengine_sim`` is importable, so the fallback never shadows
    real classes in production use.
    """
    try:
        from ironengine_sim.world.components import (  # type: ignore[import-not-found]
            Hierarchy,
            Light,
            MeshRenderable,
            SurfaceMaterial,
            Transform,
        )
    except ImportError:
        class Hierarchy:
            pass

        class Light:
            pass

        class MeshRenderable:
            pass

        class SurfaceMaterial:
            pass

        class Transform:
            pass

    return SimpleNamespace(
        Hierarchy=Hierarchy, Light=Light, MeshRenderable=MeshRenderable,
        SurfaceMaterial=SurfaceMaterial, Transform=Transform,
    )


def _world_matrix(world: Any, eid: int) -> tuple[np.ndarray, np.ndarray]:
    """Compose the entity's full world transform, walking the Hierarchy
    parent chain (child matrix left-multiplied by each ancestor's TRS).

    Returns (4x4 world matrix, 3x3 pure-rotation chain for normals)."""
    sim = _sim_world_components()
    comps = world.graph.components
    m = np.eye(4, dtype=np.float64)
    rot = np.eye(3, dtype=np.float64)
    cur: int | None = eid
    seen: set[int] = set()
    while cur is not None and cur not in seen:
        seen.add(cur)
        tx = comps.get(cur, sim.Transform)
        if tx is not None:
            m_local, rot_local = _trs_matrix(tx)
            m = m_local @ m
            rot = rot_local @ rot
        hier = comps.get(cur, sim.Hierarchy)
        cur = hier.parent if hier is not None else None
    return m, rot


def _bake_transform(cache_key: str, positions: np.ndarray, normals: np.ndarray | None,
                    m4: np.ndarray, rot: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Bake a TRS matrix into vertex positions (and rotate normals).

    Non-uniform scale normal-matrix subtleties are ignored — normals are
    rotated by the pure rotation, acceptable for Sim scenes.
    """
    key = (cache_key, m4.tobytes())
    hit = _XFORM_CACHE.get(key)
    if hit is not None:
        return hit
    pos = (positions.astype(np.float64) @ m4[:3, :3].T + m4[:3, 3]).astype(np.float32)
    nrm = None
    if normals is not None:
        n = normals.astype(np.float64) @ rot.T
        n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
        nrm = n.astype(np.float32)
    _XFORM_CACHE[key] = (pos, nrm)
    return pos, nrm


# --------------------------------------------------------------- soft bodies
# id(SceneGraph) -> weakref to the owning PhysicsWorld (or the None marker).
_PHYSICS_BY_GRAPH: dict[int, Any] = {}


def _physics_from_world(world: Any) -> Any:
    """Resolve Sim's PhysicsWorld for a world-like namespace.

    A pinned (``install_for_world``) or hand-passed real ``World`` exposes
    ``.physics`` directly. The ``RenderWorld.scene``/``.assets`` fallback
    namespace doesn't, so the PhysicsWorld is discovered once via a bounded
    gc walk over live objects (``PhysicsWorld.scene is graph``) and cached
    weakly per scene graph. Returns None when no physics world exists —
    callers degrade to static meshes.
    """
    physics = getattr(world, "physics", None)
    if physics is not None:
        return physics
    graph = getattr(world, "graph", None)
    if graph is None:
        return None
    key = id(graph)
    if key in _PHYSICS_BY_GRAPH:
        return _deref(_PHYSICS_BY_GRAPH[key])
    found: Any = None
    try:
        import gc
        import weakref

        for obj in gc.get_objects():
            if type(obj).__name__ == "PhysicsWorld" and getattr(obj, "scene", None) is graph:
                try:
                    _PHYSICS_BY_GRAPH[key] = weakref.ref(obj)
                except TypeError:
                    _PHYSICS_BY_GRAPH[key] = obj          # unweakrefable double
                found = obj
                break
    except Exception:
        found = None
    if found is None:
        _PHYSICS_BY_GRAPH[key] = None
    return found


def _deref(entry: Any) -> Any:
    """Resolve a _PHYSICS_BY_GRAPH entry (weakref | strong | None)."""
    import weakref

    if isinstance(entry, weakref.ReferenceType):
        return entry()
    return entry


def _recompute_normals(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals for a (possibly deformed) triangle mesh."""
    pos = np.asarray(positions, dtype=np.float64)
    idx = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    nrm = np.zeros_like(pos)
    if idx.shape[0] > 0:
        v0 = pos[idx[:, 0]]; v1 = pos[idx[:, 1]]; v2 = pos[idx[:, 2]]
        face = np.cross(v1 - v0, v2 - v0)                    # area-weighted
        for corner in range(3):
            np.add.at(nrm, idx[:, corner], face)
    length = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = nrm / np.where(length > 1e-12, length, 1.0)
    return nrm.astype(np.float32)


def _cloth_grid_indices(n_particles: int) -> np.ndarray | None:
    """Regular r×r grid triangulation for XPBD cloth particles (row-major,
    ``idx = i * r + j`` with u along +X and v along +Z), wound for +Y
    normals at spawn. Returns None when ``n_particles`` isn't a square
    cloth grid (rope chains, torn bodies, …)."""
    r = int(round(math.sqrt(n_particles)))
    if r < 2 or r * r != n_particles:
        return None
    tris: list[tuple[int, int, int]] = []
    for i in range(r - 1):
        for j in range(r - 1):
            a = i * r + j
            b = a + 1
            c = a + r
            d = c + 1
            tris.append((a, c, b))
            tris.append((b, c, d))
    return np.asarray(tris, dtype=np.int64)


def _soft_override(world: Any, eid: int, n_static_verts: int,
                   ) -> tuple[np.ndarray, np.ndarray | None] | None:
    """Live soft-body state for an entity, if Sim simulates one.

    Returns ``(world_positions, grid_indices_or_None)``:
      * particle count == mesh vertex count → keep the authored topology
        (``grid_indices_or_None is None``), positions come from the solver;
      * otherwise a square cloth grid is re-triangulated from the solver
        particles (solver-resolution fallbacks where the authored mesh and
        the simulated grid diverge);
      * anything else (no physics, no instance, rope/odd counts) → None,
        i.e. render the static mesh.
    """
    physics = _physics_from_world(world)
    get_soft = getattr(physics, "get_soft_body", None)
    if not callable(get_soft):
        return None
    try:
        inst = get_soft(eid)
    except Exception:
        return None
    if inst is None:
        return None
    pos = getattr(inst, "positions", None)
    if pos is None:
        return None
    pos = np.asarray(pos, dtype=np.float32)
    if pos.ndim != 2 or pos.shape[1] != 3 or pos.shape[0] < 3:
        return None
    if pos.shape[0] == n_static_verts:
        return pos, None
    grid = _cloth_grid_indices(pos.shape[0])
    if grid is not None:
        return pos, grid
    return None


def _scene_from_world(world: Any) -> Scene:
    """Translate Sim's SceneGraph into a BonaFide Scene.

    Covers Mesh + PointCloud + Light. Full world transforms — own TRS
    composed with every Hierarchy ancestor — are baked into the geometry.

    Soft bodies: when the entity has a live XPBD soft-body instance in Sim's
    physics world (``PhysicsWorld.get_soft_body``), the instance's deformed
    particle positions (world-space) replace the statically baked mesh —
    matching vertex counts keep the authored topology (normals are
    recomputed from the deformed triangles); a cloth particle grid whose
    count no longer matches the mesh (e.g. solver-resolution fallback) is
    re-triangulated as a regular grid. When no Sim soft state exists (no
    physics world, solver unavailable, instance not spawned), the entity
    renders its static mesh exactly as before.
    """
    sim = _sim_world_components()
    Light, MeshRenderable, SurfaceMaterial, Transform = (
        sim.Light, sim.MeshRenderable, sim.SurfaceMaterial, sim.Transform,
    )
    try:
        from ironengine_sim.rendering.point_cloud_renderer import (  # type: ignore[import-not-found]
            PointCloudAsset,
        )
    except ImportError:
        PointCloudAsset = None  # type: ignore[assignment,misc]

    scene = Scene(name="sim")
    # Lights
    for eid, lt in world.graph.components.iter_components(Light):
        if lt.kind == "directional":
            scene.add(DirectionalLight(
                direction=lt.direction, color=lt.color, intensity=lt.intensity,
                cast_shadow=bool(lt.cast_shadow),
            ))
        elif lt.kind in ("point", "spot"):
            # Spot lights are approximated as point lights — cone shaping
            # (spot_inner_deg/spot_outer_deg) is dropped in the bridge.
            # Position honors the full Hierarchy chain.
            m4, _ = _world_matrix(world, eid)
            pos = tuple(m4[:3, 3].tolist())                          # type: ignore[arg-type]
            scene.add(PointLight(
                position=pos, color=lt.color, intensity=lt.intensity, range=lt.range,
            ))
    # Meshes — Sim's mesh data lives in AssetManager; resolve by mesh_id.
    for eid in world.graph.components.entities_with(MeshRenderable, Transform):
        rend = world.graph.components.require(eid, MeshRenderable)
        if not rend.visible:
            continue
        handle = world.assets.get_mesh(rend.mesh_id)
        if handle is None:
            continue
        # Sim mesh handles store interleaved [pos3 | normal3 | uv2]; recover.
        full = np.asarray(handle.vertices, dtype=np.float32)
        normals = full[:, 3:6] if full.shape[1] >= 6 else None
        # Bake the full world transform (own TRS + ancestors) — cached.
        m4, rot = _world_matrix(world, eid)
        positions, normals = _bake_transform(
            str(rend.mesh_id), full[:, :3], normals, m4, rot)
        # Sim 'indices' is flat triangle indices.
        idx = np.asarray(handle.indices, dtype=np.int64).reshape(-1, 3)
        # Material → BonaFide PBRMaterial. Entity-level SurfaceMaterial wins;
        # otherwise fall back to Sim's material library record for
        # rend.material_id (factory/builtin materials live there — without
        # this fallback everything rendered as the default flat grey).
        mat = world.graph.components.get(eid, SurfaceMaterial)
        if mat is None and getattr(rend, "material_id", None):
            try:
                from ironengine_sim.assets.material_library import (  # type: ignore[import-not-found]
                    BUILTIN_SURFACE_MATERIALS,
                )
                mat = BUILTIN_SURFACE_MATERIALS.get(rend.material_id)
            except Exception:
                mat = None
        from ironengine_bonafide.core.material import PBRMaterial
        bona_mat = PBRMaterial(
            albedo=tuple(mat.albedo) if mat else (0.7, 0.7, 0.7),         # type: ignore[arg-type]
            roughness=mat.roughness if mat else 0.7,
            metallic=mat.metallic if mat else 0.0,
            ior=mat.ior if mat else 1.45,
            emissive=tuple(mat.emissive) if mat else (0.0, 0.0, 0.0),     # type: ignore[arg-type]
        )
        # Vertex colors + UVs: Sim mesh handles may carry an (N, 4) float32
        # RGBA `colors` array (GLB/PLY imports — the 3DCreator analytic GLBs
        # always do). BonaFide wants (N, 3) RGB; pass through so imported
        # models keep their baked per-vertex albedo variation instead of
        # flattening to the scalar material albedo. UVs ride along when the
        # interleave carries them (harmless when maps are absent).
        vcolors = getattr(handle, "colors", None)
        if vcolors is not None:
            vcolors = np.ascontiguousarray(
                np.asarray(vcolors, dtype=np.float32)[:, :3])
        uvs = full[:, 6:8] if full.shape[1] >= 8 else None
        # Soft-body deformation: a live XPBD soft instance owns this
        # entity's shape. Its particle positions are world-space and replace
        # the statically baked mesh (bypassing the transform cache — soft
        # geometry is dynamic). Matching counts keep the authored topology;
        # a square cloth grid is re-triangulated (authored attributes that
        # no longer align are dropped). Normals are recomputed from the
        # deformed triangles. No Sim soft state → static mesh, unchanged.
        soft = _soft_override(world, eid, positions.shape[0])
        if soft is not None:
            soft_pos, grid_idx = soft
            positions = soft_pos
            if grid_idx is not None:
                idx = grid_idx
                vcolors = None
                uvs = None
            normals = _recompute_normals(positions, idx)
        scene.add(Mesh.from_arrays(
            positions=positions, indices=idx, normals=normals, uvs=uvs,
            colors=vcolors,
            material=bona_mat, name=str(eid),
        ))
    # Point clouds — PointCloudAsset component + AssetManager handle.
    if PointCloudAsset is not None:
        from ironengine_bonafide.core.pointcloud import PointCloud
        for eid in world.graph.components.entities_with(PointCloudAsset, Transform):
            pc = world.graph.components.require(eid, PointCloudAsset)
            if not getattr(pc, "visible", True):
                continue
            handle = world.assets.get_point_cloud(pc.cloud_name)
            if handle is None:
                continue
            m4, rot = _world_matrix(world, eid)
            positions, _ = _bake_transform(
                f"pc:{pc.cloud_name}", np.asarray(handle.positions, dtype=np.float32),
                None, m4, rot)
            colors = getattr(handle, "colors", None)
            if colors is None:
                colors = np.broadcast_to(
                    np.asarray(pc.default_color, dtype=np.float32),
                    (positions.shape[0], 3)).copy()
            cloud = PointCloud.from_arrays(
                positions, np.asarray(colors, dtype=np.float32),
                name=str(pc.cloud_name))
            cloud.point_size_px = float(getattr(pc, "point_size", 3.0))
            scene.add(cloud)
    return scene


# --------------------------------------------------------------- patched methods
def _patched_render_viewport(self: Any, camera: Any | None = None,
                             clear_color: tuple[float, float, float, float] = (0.05, 0.06, 0.08, 1.0)) -> None:
    """Replacement for RenderWorld.render_viewport.

    Matches the CURRENT upstream signature ``render_viewport(camera:
    RenderCamera | None = None, clear_color = ...)``. A passed RenderCamera
    is honored verbatim; an int is treated as a camera entity id (the
    legacy ``World.render_viewport(camera_entity)`` convention); None falls
    back to the world's Camera component + Transform, then to Sim's default
    editor camera.

    The frame is stashed as uint8 RGB on ``self._last_bonafide_frame``.
    This method is only installed when ``install(headless_only=False)`` —
    see the module docstring for why that is opt-in.
    """
    world = _world_parts(self)
    scene = _scene_from_world(world)
    cam = _camera_from_world(world, camera)
    width, height = _viewport_size(self)
    cfg = RenderConfig(
        width=width,
        height=height,
        output_color_space="sRGB",
    )
    out = render(_engine(), scene, cam, cfg)
    self._last_bonafide_frame = srgb_to_uint8(out.rgb)


def _patched_render_sensor_rgb(self: Any, camera_id: str, pose: Any, local_offset: Any,
                               width: int, height: int, fov_deg: float) -> np.ndarray:
    world = _world_parts(self)
    scene = _scene_from_world(world)
    pose4 = _sensor_pose(pose, local_offset)
    cam = SensorCamera(pose=pose4, fov_deg=fov_deg, near=0.05, far=200.0)
    cfg = RenderConfig(width=int(width), height=int(height),
                       output_color_space="sRGB", sensor_outputs=("rgb",))
    out = render(_engine(), scene, cam, cfg)
    return srgb_to_uint8(out.rgb)


def _patched_render_sensor_depth(self: Any, camera_id: str, pose: Any, local_offset: Any,
                                 width: int, height: int, fov_deg: float,
                                 near: float, far: float) -> np.ndarray:
    """Returns linear eye-space depth in **meters** (H x W float32).

    The rasterizer writes OpenGL-style NDC z in [-1, 1] (+inf where empty),
    so we unproject with the camera near/far:
    ``meters = 2·near·far / (far + near − z_ndc·(far − near))``.
    Empty pixels read as ``far``.
    """
    world = _world_parts(self)
    scene = _scene_from_world(world)
    pose4 = _sensor_pose(pose, local_offset)
    cam = SensorCamera(pose=pose4, fov_deg=fov_deg, near=near, far=far)
    cfg = RenderConfig(width=int(width), height=int(height),
                       sensor_outputs=("depth",))
    out = render(_engine(), scene, cam, cfg)
    if out.depth is None:
        return np.full((height, width), far, dtype=np.float32)
    ndc = out.depth.detach().cpu().numpy().astype(np.float64)
    finite = np.isfinite(ndc)
    meters = np.full(ndc.shape, float(far), dtype=np.float64)
    z = np.clip(ndc[finite], -1.0, 1.0)
    meters[finite] = (2.0 * near * far) / (far + near - z * (far - near))
    return meters.astype(np.float32)


# --------------------------------------------------------------- helpers
def _world_parts(render_world: Any) -> Any:
    """Resolve a world-like namespace (``.graph`` + ``.assets``).

    The real ``RenderWorld`` stores the ``SceneGraph`` as ``self.scene``
    and the ``AssetManager`` as ``self.assets`` — those are authoritative
    and need no gc tricks. ``install_for_world`` pins ``self._world`` for
    exotic setups; it wins when present.
    """
    pinned = getattr(render_world, "_world", None)
    if pinned is not None:
        return pinned
    graph = getattr(render_world, "scene", None)
    assets = getattr(render_world, "assets", None)
    if graph is not None and assets is not None:
        import types
        return types.SimpleNamespace(graph=graph, assets=assets)
    raise RuntimeError(
        "BonaFide sim shim could not resolve the scene graph — expected "
        "RenderWorld.scene/.assets, or call install_for_world(world) to pin "
        "the owning World explicitly."
    )


def _viewport_size(render_world: Any) -> tuple[int, int]:
    """Honor RenderWorld's real override attribute (render_world.py:471-476)."""
    override = getattr(render_world, "viewport_override_size", None)
    if override is not None:
        w, h = override
        return int(w), int(h)
    return 1280, 720


def install_for_world(world: Any) -> None:
    """Variant of `install()` that pins the World ahead of time.

    Useful for programmatic World construction; with the default
    ``RenderWorld.scene``/``.assets`` resolution this is rarely needed.
    Sensor-only patching stays the safe default here too.
    """
    world.render._world = world
    install()


def _camera_from_world(world: Any, camera: Any | None) -> PerspectiveCamera:
    """Build a BonaFide camera from whatever the caller passed.

    * ``RenderCamera`` (duck-typed: position/target/up/fov/near/far) →
      honored verbatim.
    * ``int`` → camera entity id; uses that entity's Camera component +
      Transform (Hierarchy-composed), looking down local −Z.
    * ``None`` → first entity with Camera + Transform, else Sim's default
      editor camera (render_world.py:1359-1368).
    """
    from ironengine_sim.world.components import (  # type: ignore[import-not-found]
        Camera,
        Transform,
    )
    # Duck-typed RenderCamera (avoids importing Sim's rendering stack here).
    if camera is not None and not isinstance(camera, int) and hasattr(camera, "position") \
            and hasattr(camera, "target"):
        return PerspectiveCamera(
            position=tuple(np.asarray(camera.position, dtype=np.float64).tolist()),  # type: ignore[arg-type]
            look_at=tuple(np.asarray(camera.target, dtype=np.float64).tolist()),     # type: ignore[arg-type]
            up=tuple(np.asarray(camera.up, dtype=np.float64).tolist()),              # type: ignore[arg-type]
            fov_deg=float(camera.fov_deg),
            near=float(camera.near), far=float(camera.far),
        )

    eid: int | None = None
    if isinstance(camera, int) and world.graph.alive(camera):
        eid = camera
    elif camera is None:
        for cand in world.graph.components.entities_with(Camera, Transform):
            eid = cand
            break
    if eid is not None:
        cam_comp = world.graph.components.get(eid, Camera)
        m4, rot = _world_matrix(world, eid)
        eye = m4[:3, 3]
        forward = rot @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
        fov = float(cam_comp.fov_deg) if cam_comp is not None else 60.0
        near = float(cam_comp.near) if cam_comp is not None else 0.05
        far = float(cam_comp.far) if cam_comp is not None else 200.0
        return PerspectiveCamera(
            position=tuple(eye.tolist()),                                # type: ignore[arg-type]
            look_at=tuple((eye + forward).tolist()),                     # type: ignore[arg-type]
            up=(0.0, 1.0, 0.0),
            fov_deg=fov, near=near, far=far,
        )
    # Sim's `_default_editor_camera` (render_world.py:1359-1368).
    return PerspectiveCamera(
        position=(3.5, 2.5, 3.5), look_at=(0.0, 0.6, 0.0),
        fov_deg=55.0, near=0.05, far=200.0,
    )


def _sensor_pose(pose: Any, local_offset: Any) -> np.ndarray:
    """Compose Sim's Transform + per-sensor local offset into a 4x4 pose.

    Full TRS: the body quaternion rotates the local offset into the body
    frame before translation (upstream `_sensor_camera`,
    render_world.py:411-436). The result is then converted from Sim's
    +X-forward sensor convention to BonaFide's -Z-forward camera
    convention via ``_SENSOR_TO_CAMERA``.
    """
    pos = np.asarray(getattr(pose, "position", (0, 0, 0)), dtype=np.float64).reshape(3)
    rot = _quat_to_mat3(getattr(pose, "rotation", (0.0, 0.0, 0.0, 1.0)))
    off = np.asarray(local_offset, dtype=np.float64).reshape(3)
    origin = pos + rot @ off
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = rot @ _SENSOR_TO_CAMERA
    m[:3, 3] = origin
    return m


# --------------------------------------------------------------- install
def install(headless_only: bool = True) -> None:
    """Monkey-patch ``RenderWorld`` entry points.

    ``headless_only=True`` (default, safe): patches ONLY the sensor methods
    (``render_sensor_rgb`` / ``render_sensor_depth``); the viewport keeps
    Sim's own renderer so the SceneEditor can never go black.

    ``headless_only=False``: additionally replaces ``render_viewport``.
    The BonaFide frame is exposed via ``render_world._last_bonafide_frame``
    — blitting to the Qt framebuffer is intentionally left to the caller.
    """
    try:
        from ironengine_sim.rendering.render_world import (
            RenderWorld,  # type: ignore[import-not-found]
        )
    except ImportError as exc:
        raise RuntimeError(
            "ironengine_sim is not importable. Install it on PYTHONPATH first."
        ) from exc
    patched: list[str] = []
    if not headless_only and "render_viewport" not in _ORIGINALS:
        _ORIGINALS["render_viewport"] = RenderWorld.render_viewport
        RenderWorld.render_viewport = _patched_render_viewport            # type: ignore[assignment]
        patched.append("render_viewport")
    for name, replacement in (
        ("render_sensor_rgb", _patched_render_sensor_rgb),
        ("render_sensor_depth", _patched_render_sensor_depth),
    ):
        if name in _ORIGINALS:
            continue
        original = getattr(RenderWorld, name, None)
        if original is None:
            continue
        _ORIGINALS[name] = original
        setattr(RenderWorld, name, replacement)
        patched.append(name)
    if patched:
        logger.info(
            f"sim shim installed ({'sensor-only' if headless_only else 'sensors + viewport'}): "
            f"patched {', '.join(patched)}"
        )
    else:
        logger.info("sim shim already installed")


def uninstall() -> None:
    if not _ORIGINALS:
        return
    from ironengine_sim.rendering.render_world import RenderWorld  # type: ignore[import-not-found]
    for name, original in _ORIGINALS.items():
        setattr(RenderWorld, name, original)
    _ORIGINALS.clear()
    logger.info("sim shim uninstalled")
