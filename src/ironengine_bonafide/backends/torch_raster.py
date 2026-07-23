"""Device-agnostic torch rasterizers.

Shared pure-torch implementation used by the CPU reference backend and by
the torch-on-device fallbacks of the CUDA / WGPU backends (shadow depth
raster). Every tensor is created on the input's device, so the same code
runs on CPU or GPU unchanged.

Conventions:

  * Right-handed, Y-up, forward -Z (same clip math as ``core.camera``).
  * NDC z in [-1, 1]; ``+inf`` marks an empty pixel.
  * Triangles crossing the near plane are *split* in clip space
    (``w = eps``) before the perspective divide instead of being dropped;
    per-vertex attributes are lerped at the clip points.
  * Attribute interpolation (color / normal / world-pos / uv / tangent)
    is perspective-correct: screen-space barycentrics are weighted by
    ``1/w`` and renormalized per pixel. NDC z stays affine in screen
    space, which is the correct depth to interpolate linearly.
  * Per-pixel depth resolve is deterministic: ``scatter_reduce(amin)``
    on the flattened depth buffer, then a first-candidate tiebreak for
    the attribute write — no ``argsort`` + duplicate ``index_put_``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

_NEAR_EPS = 1e-4
_CHUNK = 256


@dataclass(slots=True)
class GBuffer:
    """Per-pixel geometry/attribute buffers produced by the GBuffer raster."""
    albedo: torch.Tensor                    # (H, W, 3) vertex-color albedo
    world_pos: torch.Tensor                 # (H, W, 3)
    normal: torch.Tensor                    # (H, W, 3) unit world-space
    uv: torch.Tensor | None                 # (H, W, 2) or None
    tangent: torch.Tensor | None            # (H, W, 3) or None
    depth: torch.Tensor                     # (H, W) NDC z, +inf empty
    mask: torch.Tensor                      # (H, W) 1.0 where a triangle hit


# ---------------------------------------------------------------- clipping
def clip_near_plane(
    positions: torch.Tensor,                # (V, 3)
    indices: torch.Tensor,                  # (T, 3)
    view_proj: torch.Tensor,                # (4, 4)
    *,
    attrs: tuple[torch.Tensor, ...] = (),
    eps: float = _NEAR_EPS,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Split triangles crossing the near plane ``w = eps`` in clip space.

    Returns ``(positions, indices, attrs)`` possibly augmented with new
    vertices. Triangles fully behind the plane are dropped; triangles
    fully in front pass through unchanged (no copies are made in that
    common case).
    """
    device = positions.device
    n = positions.shape[0]
    ones = torch.ones((n, 1), dtype=torch.float32, device=device)
    clip = torch.cat([positions, ones], dim=1) @ view_proj.T
    inside = clip[:, 3] > eps
    t_in = inside[indices]                                        # (T, 3)
    n_in = t_in.sum(dim=1)
    if bool((n_in == 3).all()):
        return positions, indices, list(attrs)

    keep = indices[n_in == 3]
    cross = indices[(n_in == 1) | (n_in == 2)]
    new_pos: list[torch.Tensor] = []
    new_attr: list[list[torch.Tensor]] = [[] for _ in attrs]
    new_tris: list[list[int]] = []

    inside_l = inside.tolist()
    w_l = clip[:, 3].tolist()
    next_vid = n

    for k in range(cross.shape[0]):
        ids = [int(v) for v in cross[k].tolist()]
        ins = [i for i in ids if inside_l[i]]
        outs = [i for i in ids if not inside_l[i]]
        # Map each newly created (edge) vertex to a fresh index.
        edge_vid: dict[tuple[int, int], int] = {}

        def edge_vertex(i0: int, i1: int) -> int:
            nonlocal next_vid
            key = (i0, i1)
            if key in edge_vid:
                return edge_vid[key]
            w0, w1 = w_l[i0], w_l[i1]
            t = (eps - w0) / (w1 - w0)
            p0 = positions[i0].to(torch.float32)
            p1 = positions[i1].to(torch.float32)
            new_pos.append(p0 + t * (p1 - p0))
            for ai, attr in enumerate(attrs):
                new_attr[ai].append(attr[i0] + t * (attr[i1] - attr[i0]))
            edge_vid[key] = next_vid
            next_vid += 1
            return next_vid - 1

        if len(ins) == 1:
            a = ins[0]
            b, c = outs[0], outs[1]
            # Preserve winding order around the surviving vertex.
            order = _rotate_to_first(ids, a)
            a, b, c = order[0], order[1], order[2]
            ab = edge_vertex(a, b)
            ac = edge_vertex(a, c)
            new_tris.append([a, ab, ac])
        else:  # len(ins) == 2
            c = outs[0]
            order = _rotate_to_first(ids, c)      # c first: [c, a, b]
            c, a, b = order[0], order[1], order[2]
            bc = edge_vertex(b, c)
            ac = edge_vertex(a, c)
            new_tris.append([a, b, bc])
            new_tris.append([a, bc, ac])

    out_indices = keep
    if new_tris:
        tri_t = torch.tensor(new_tris, dtype=torch.long, device=device)
        out_indices = torch.cat([keep, tri_t], dim=0)
    out_pos = positions
    if new_pos:
        out_pos = torch.cat([positions, torch.stack(new_pos, dim=0)], dim=0)
    out_attrs: list[torch.Tensor] = []
    for ai, attr in enumerate(attrs):
        if new_attr[ai]:
            out_attrs.append(torch.cat([attr, torch.stack(new_attr[ai], dim=0)], dim=0))
        else:
            out_attrs.append(attr)
    return out_pos, out_indices, out_attrs


def _rotate_to_first(ids: list[int], first: int) -> list[int]:
    k = ids.index(first)
    return ids[k:] + ids[:k]


# ---------------------------------------------------------------- projection
def _project(
    positions: torch.Tensor,
    view_proj: torch.Tensor,
    width: int,
    height: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project world positions. Returns (sx, sy, ndc_z, clip_w)."""
    device = positions.device
    n = positions.shape[0]
    ones = torch.ones((n, 1), dtype=torch.float32, device=device)
    clip = torch.cat([positions, ones], dim=1) @ view_proj.T
    w = clip[:, 3]
    inv_w = 1.0 / w.clamp(min=1e-6)
    ndc = clip[:, :3] * inv_w.unsqueeze(1)
    sx = (ndc[:, 0] * 0.5 + 0.5) * width
    sy = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * height
    return sx, sy, ndc[:, 2], w


# ------------------------------------------------------- triangle scan core
def _scan_triangles(
    *,
    sx: torch.Tensor, sy: torch.Tensor, sz: torch.Tensor, w: torch.Tensor,
    indices: torch.Tensor,
    width: int,
    height: int,
    depth: torch.Tensor,                            # (H, W) output, updated
    attr_tris: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    out_bufs: list[torch.Tensor],                   # (H, W, C) output buffers
    mask: torch.Tensor | None = None,               # (H, W) optional
) -> None:
    """Barycentric triangle scan with perspective-correct attributes and a
    deterministic depth resolve. ``attr_tris`` holds per-triangle vertex
    attributes gathered as (T, C) tensors at corners a / b / c."""
    device = depth.device
    a = indices[:, 0]; b = indices[:, 1]; c = indices[:, 2]
    ax, ay, az = sx[a], sy[a], sz[a]
    bx, by, bz = sx[b], sy[b], sz[b]
    cx, cy, cz = sx[c], sy[c], sz[c]

    xmin = torch.minimum(torch.minimum(ax, bx), cx).clamp(min=0).floor().long()
    ymin = torch.minimum(torch.minimum(ay, by), cy).clamp(min=0).floor().long()
    xmax = torch.maximum(torch.maximum(ax, bx), cx).clamp(max=width - 1).ceil().long()
    ymax = torch.maximum(torch.maximum(ay, by), cy).clamp(max=height - 1).ceil().long()
    denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    # Clipped vertices sit exactly on w = eps (minus rounding); the
    # viability threshold must be looser than the clip plane.
    viable = ((xmax >= xmin) & (ymax >= ymin) & (denom.abs() > 1e-9)
              & (w[a] > _NEAR_EPS * 0.5) & (w[b] > _NEAR_EPS * 0.5)
              & (w[c] > _NEAR_EPS * 0.5))
    if not torch.any(viable):
        return
    sel = viable.nonzero(as_tuple=False).squeeze(1)

    wa, wb, wc = w[a], w[b], w[c]
    for start in range(0, sel.numel(), _CHUNK):
        chunk = sel[start:start + _CHUNK]
        _scan_chunk(
            chunk=chunk, ax=ax, ay=ay, az=az, bx=bx, by=by, bz=bz,
            cx=cx, cy=cy, cz=cz, wa=wa, wb=wb, wc=wc,
            xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, denom=denom,
            depth=depth, attr_tris=attr_tris, out_bufs=out_bufs,
            mask=mask, width=width, height=height, device=device,
        )


def _scan_chunk(
    *, chunk, ax, ay, az, bx, by, bz, cx, cy, cz, wa, wb, wc,
    xmin, ymin, xmax, ymax, denom, depth, attr_tris, out_bufs,
    mask, width, height, device,
) -> None:                                          # type: ignore[no-untyped-def]
    flat_xs: list[torch.Tensor] = []
    flat_ys: list[torch.Tensor] = []
    flat_tri: list[torch.Tensor] = []
    for k in range(chunk.numel()):
        t = int(chunk[k])
        x0 = int(xmin[t]); x1 = int(xmax[t])
        y0 = int(ymin[t]); y1 = int(ymax[t])
        if x1 < x0 or y1 < y0:
            continue
        ys = torch.arange(y0, y1 + 1, dtype=torch.float32, device=device)
        xs = torch.arange(x0, x1 + 1, dtype=torch.float32, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        flat_xs.append(gx.reshape(-1))
        flat_ys.append(gy.reshape(-1))
        flat_tri.append(torch.full((gx.numel(),), t, dtype=torch.long, device=device))
    if not flat_xs:
        return
    xs_f = torch.cat(flat_xs)
    ys_f = torch.cat(flat_ys)
    ti = torch.cat(flat_tri)

    inv_d = 1.0 / denom[ti]
    ws_a = ((by[ti] - cy[ti]) * (xs_f - cx[ti]) + (cx[ti] - bx[ti]) * (ys_f - cy[ti])) * inv_d
    ws_b = ((cy[ti] - ay[ti]) * (xs_f - cx[ti]) + (ax[ti] - cx[ti]) * (ys_f - cy[ti])) * inv_d
    ws_c = 1.0 - ws_a - ws_b
    tri_z = ws_a * az[ti] + ws_b * bz[ti] + ws_c * cz[ti]

    inside = (ws_a >= -1e-6) & (ws_b >= -1e-6) & (ws_c >= -1e-6)
    yi = ys_f.long(); xi = xs_f.long()
    flat = yi * width + xi
    dflat = depth.reshape(-1)
    keep = inside & (tri_z < dflat[flat])
    if not torch.any(keep):
        return
    xs_f = xs_f[keep]; ys_f = ys_f[keep]; ti = ti[keep]
    ws_a = ws_a[keep]; ws_b = ws_b[keep]; ws_c = ws_c[keep]
    tri_z = tri_z[keep]; flat = flat[keep]

    # ---- deterministic depth resolve: amin scatter + first-candidate tiebreak
    dflat.scatter_reduce_(0, flat, tri_z, reduce="amin", include_self=True)
    zb = dflat[flat]
    win = (tri_z <= zb).nonzero(as_tuple=False).squeeze(1)
    if win.numel() == 0:
        return
    pix = flat[win]
    pix_u, inv = torch.unique(pix, return_inverse=True)
    first = torch.full((pix_u.numel(),), win.numel(), dtype=torch.long, device=device)
    cand = torch.arange(win.numel(), dtype=torch.long, device=device)
    first.scatter_reduce_(0, inv, cand, reduce="amin", include_self=True)
    final = win[cand == first[inv]]

    ti_s = ti[final]
    # ---- perspective-correct barycentrics (1/w weighting)
    la = ws_a[final] / wa[ti_s]
    lb = ws_b[final] / wb[ti_s]
    lc = ws_c[final] / wc[ti_s]
    lsum = (la + lb + lc).clamp(min=1e-12)
    la = (la / lsum).unsqueeze(-1)
    lb = (lb / lsum).unsqueeze(-1)
    lc = (lc / lsum).unsqueeze(-1)

    flat_s = flat[final]
    for (at_a, at_b, at_c), buf in zip(attr_tris, out_bufs):
        vals = la * at_a[ti_s] + lb * at_b[ti_s] + lc * at_c[ti_s]
        bflat = buf.reshape(-1, buf.shape[-1])
        bflat.index_copy_(0, flat_s, vals)
    if mask is not None:
        mask.reshape(-1).index_copy_(
            0, flat_s, torch.ones(flat_s.numel(), dtype=mask.dtype, device=device),
        )


# ---------------------------------------------------------------- public API
def raster_depth(
    positions: torch.Tensor,
    indices: torch.Tensor,
    view_proj: torch.Tensor,
    width: int,
    height: int,
) -> torch.Tensor:
    """Depth-only triangle raster (H x W float32 NDC z, +inf where empty)."""
    device = positions.device
    positions = positions.to(torch.float32)
    indices = indices.to(torch.long)
    depth = torch.full((height, width), float("inf"), dtype=torch.float32, device=device)
    if indices.numel() == 0 or positions.numel() == 0:
        return depth
    positions, indices, _ = clip_near_plane(positions, indices, view_proj.to(torch.float32))
    if indices.numel() == 0:
        return depth
    sx, sy, sz, w = _project(positions, view_proj.to(torch.float32), width, height)
    _scan_triangles(
        sx=sx, sy=sy, sz=sz, w=w, indices=indices,
        width=width, height=height, depth=depth,
        attr_tris=[], out_bufs=[],
    )
    return depth


def raster_mesh_gbuffer(
    positions: torch.Tensor,            # (V, 3) world
    indices: torch.Tensor,              # (T, 3)
    colors: torch.Tensor,               # (V, 3) per-vertex albedo
    normals: torch.Tensor | None,       # (V, 3) world
    view_proj: torch.Tensor,            # (4, 4)
    width: int,
    height: int,
    *,
    uvs: torch.Tensor | None = None,    # (V, 2)
    tangents: torch.Tensor | None = None,  # (V, 3) world
) -> GBuffer:
    """GBuffer raster with perspective-correct attribute interpolation."""
    device = positions.device
    positions = positions.to(torch.float32)
    indices = indices.to(torch.long)
    colors = colors.to(torch.float32)
    if normals is None:
        normals = vertex_normals(positions, indices)
    else:
        normals = normals.to(torch.float32)
    if uvs is not None:
        uvs = uvs.to(torch.float32)
    if tangents is not None:
        tangents = tangents.to(torch.float32)

    albedo_buf = torch.zeros((height, width, 3), dtype=torch.float32, device=device)
    wpos_buf = torch.zeros((height, width, 3), dtype=torch.float32, device=device)
    nrm_buf = torch.zeros((height, width, 3), dtype=torch.float32, device=device)
    uv_buf = torch.zeros((height, width, 2), dtype=torch.float32, device=device) if uvs is not None else None
    tan_buf = torch.zeros((height, width, 3), dtype=torch.float32, device=device) if tangents is not None else None
    depth = torch.full((height, width), float("inf"), dtype=torch.float32, device=device)
    mask = torch.zeros((height, width), dtype=torch.float32, device=device)
    if indices.numel() == 0 or positions.numel() == 0:
        return GBuffer(albedo_buf, wpos_buf, nrm_buf, uv_buf, tan_buf, depth, mask)

    attrs: list[torch.Tensor] = [colors, normals]
    if uvs is not None:
        attrs.append(uvs)
    if tangents is not None:
        attrs.append(tangents)
    positions, indices, attrs = clip_near_plane(
        positions, indices, view_proj.to(torch.float32), attrs=tuple(attrs),
    )
    colors, normals = attrs[0], attrs[1]
    ai = 2
    if uvs is not None:
        uvs = attrs[ai]; ai += 1
    if tangents is not None:
        tangents = attrs[ai]; ai += 1
    if indices.numel() == 0:
        return GBuffer(albedo_buf, wpos_buf, nrm_buf, uv_buf, tan_buf, depth, mask)

    sx, sy, sz, w = _project(positions, view_proj.to(torch.float32), width, height)
    a = indices[:, 0]; b = indices[:, 1]; c = indices[:, 2]
    attr_tris: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = [
        (colors[a], colors[b], colors[c]),
        (positions[a], positions[b], positions[c]),
        (normals[a], normals[b], normals[c]),
    ]
    out_bufs = [albedo_buf, wpos_buf, nrm_buf]
    if uvs is not None and uv_buf is not None:
        attr_tris.append((uvs[a], uvs[b], uvs[c]))
        out_bufs.append(uv_buf)
    if tangents is not None and tan_buf is not None:
        attr_tris.append((tangents[a], tangents[b], tangents[c]))
        out_bufs.append(tan_buf)

    _scan_triangles(
        sx=sx, sy=sy, sz=sz, w=w, indices=indices,
        width=width, height=height, depth=depth,
        attr_tris=attr_tris, out_bufs=out_bufs, mask=mask,
    )
    hit = mask > 0.5
    if torch.any(hit):
        n = nrm_buf[hit]
        nrm_buf[hit] = n / torch.linalg.norm(n, dim=-1, keepdim=True).clamp(min=1e-9)
    return GBuffer(albedo_buf, wpos_buf, nrm_buf, uv_buf, tan_buf, depth, mask)


def raster_mesh(
    positions: torch.Tensor,            # (V, 3)
    indices: torch.Tensor,              # (T, 3)
    colors: torch.Tensor,               # (V, 3)
    normals: torch.Tensor | None,       # (V, 3)
    view_proj: torch.Tensor,            # (4, 4)
    width: int,
    height: int,
    light_dir: tuple[float, float, float] = (0.4, 0.8, 0.6),
    background: tuple[float, float, float] = (0.05, 0.06, 0.10),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Barycentric raster + Lambert shading. Returns (rgb, depth, normals)."""
    device = positions.device
    positions = positions.to(torch.float32)
    indices = indices.to(torch.long)
    colors = colors.to(torch.float32)
    if normals is None:
        normals = vertex_normals(positions, indices)
    else:
        normals = normals.to(torch.float32)

    bg = torch.tensor(background, dtype=torch.float32, device=device)
    rgb = bg.expand(height, width, 3).contiguous()
    depth = torch.full((height, width), float("inf"), dtype=torch.float32, device=device)
    nrm_buf = torch.zeros((height, width, 3), dtype=torch.float32, device=device)
    if indices.numel() == 0 or positions.numel() == 0:
        return rgb, depth, nrm_buf

    positions, indices, (colors, normals) = clip_near_plane(
        positions, indices, view_proj.to(torch.float32), attrs=(colors, normals),
    )
    if indices.numel() == 0:
        return rgb, depth, nrm_buf
    sx, sy, sz, w = _project(positions, view_proj.to(torch.float32), width, height)
    a = indices[:, 0]; b = indices[:, 1]; c = indices[:, 2]

    color_buf = torch.zeros((height, width, 3), dtype=torch.float32, device=device)
    _scan_triangles(
        sx=sx, sy=sy, sz=sz, w=w, indices=indices,
        width=width, height=height, depth=depth,
        attr_tris=[(colors[a], colors[b], colors[c]),
                   (normals[a], normals[b], normals[c])],
        out_bufs=[color_buf, nrm_buf],
    )
    hit = torch.isfinite(depth)
    if torch.any(hit):
        n = nrm_buf[hit]
        n = n / torch.linalg.norm(n, dim=-1, keepdim=True).clamp(min=1e-9)
        nrm_buf[hit] = n
        ldir = torch.tensor(light_dir, dtype=torch.float32, device=device)
        ldir = ldir / (torch.linalg.norm(ldir) + 1e-9)
        shade = (n @ ldir).clamp(min=0.0).unsqueeze(-1)
        rgb[hit] = color_buf[hit] * (0.2 + 0.8 * shade)
    return rgb, depth, nrm_buf


def raster_points(
    positions: torch.Tensor,            # (N, 3)
    colors: torch.Tensor,               # (N, 3)
    view_proj: torch.Tensor,            # (4, 4)
    width: int,
    height: int,
    point_size_px: float = 2.0,
    background: tuple[float, float, float] = (0.05, 0.06, 0.10),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Screen-space disk splat with perspective-scaled point size.

    ``point_size_px`` is the disk diameter at 1 m eye depth; the on-screen
    diameter scales as ``point_size_px / eye_depth`` (ortho cameras, whose
    clip w is ~1, keep a constant size). The depth resolve is
    deterministic (amin scatter + first-candidate tiebreak).
    """
    device = positions.device
    positions = positions.to(torch.float32)
    colors = colors.to(torch.float32)
    n = positions.shape[0]
    bg = torch.tensor(background, dtype=torch.float32, device=device)
    rgb = bg.expand(height, width, 3).contiguous()
    depth = torch.full((height, width), float("inf"), dtype=torch.float32, device=device)
    if n == 0:
        return rgb, depth

    sx, sy, sz, w = _project(positions, view_proj.to(torch.float32), width, height)
    px = sx.round().long()
    py = sy.round().long()
    valid = ((px >= 0) & (px < width) & (py >= 0) & (py < height) & (w > _NEAR_EPS))
    if not torch.any(valid):
        return rgb, depth
    px, py, z, c = px[valid], py[valid], sz[valid], colors[valid]
    eye_d = w[valid]

    # Perspective-scaled disk radius (capped to bound the kernel).
    size = (float(point_size_px) / eye_d.clamp(min=1e-3)).clamp(1.0, 32.0)
    radii = (size * 0.5).ceil().long()
    r_max = int(radii.max().item())
    if r_max <= 0:
        offsets = torch.tensor([[0, 0]], dtype=torch.long, device=device)
    else:
        yy, xx = torch.meshgrid(
            torch.arange(-r_max, r_max + 1, dtype=torch.long, device=device),
            torch.arange(-r_max, r_max + 1, dtype=torch.long, device=device),
            indexing="ij",
        )
        kmask = (yy * yy + xx * xx) <= r_max * r_max
        offsets = torch.stack([yy[kmask], xx[kmask]], dim=1)      # (K, 2)

    kx = offsets[:, 1]
    ky = offsets[:, 0]
    all_x = (px.unsqueeze(1) + kx.unsqueeze(0)).clamp_(0, width - 1)
    all_y = (py.unsqueeze(1) + ky.unsqueeze(0)).clamp_(0, height - 1)
    # Drop kernel samples outside this point's own radius.
    k_in = (kx * kx + ky * ky).unsqueeze(0) <= (radii * radii).unsqueeze(1)
    flat = (all_y * width + all_x).reshape(-1)
    all_z = z.unsqueeze(1).expand_as(all_x).reshape(-1)
    all_c = c.unsqueeze(1).expand(c.shape[0], all_x.shape[1], 3).reshape(-1, 3)
    k_in = k_in.reshape(-1)

    dflat = depth.reshape(-1)
    keep = k_in & (all_z < dflat[flat])
    if not torch.any(keep):
        return rgb, depth
    flat = flat[keep]; all_z = all_z[keep]; all_c = all_c[keep]

    dflat.scatter_reduce_(0, flat, all_z, reduce="amin", include_self=True)
    zb = dflat[flat]
    win = (all_z <= zb).nonzero(as_tuple=False).squeeze(1)
    if win.numel() == 0:
        return rgb, depth
    pix = flat[win]
    pix_u, inv = torch.unique(pix, return_inverse=True)
    first = torch.full((pix_u.numel(),), win.numel(), dtype=torch.long, device=device)
    cand = torch.arange(win.numel(), dtype=torch.long, device=device)
    first.scatter_reduce_(0, inv, cand, reduce="amin", include_self=True)
    final = win[cand == first[inv]]

    rgb.reshape(-1, 3).index_copy_(0, flat[final], all_c[final])
    return rgb, depth


# ------------------------------------------------------------------ helpers
def vertex_normals(positions: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Area-weighted per-vertex normals from face geometry."""
    v0 = positions[indices[:, 0]]
    v1 = positions[indices[:, 1]]
    v2 = positions[indices[:, 2]]
    face_n = torch.cross(v1 - v0, v2 - v0, dim=-1)
    accum = torch.zeros_like(positions)
    for k in range(3):
        accum.index_add_(0, indices[:, k], face_n)
    norm = torch.linalg.norm(accum, dim=-1, keepdim=True).clamp(min=1e-9)
    return accum / norm


def vertex_tangents(
    positions: torch.Tensor,            # (V, 3)
    indices: torch.Tensor,              # (T, 3)
    uvs: torch.Tensor,                  # (V, 2)
    normals: torch.Tensor,              # (V, 3)
) -> torch.Tensor:
    """Per-vertex tangent vectors (for tangent-space normal mapping).

    Tangents are solved per face from the UV Jacobian, accumulated with
    equal weights, then orthogonalized against the vertex normal
    (Gram–Schmidt). Degenerate UV faces contribute nothing.
    """
    v0 = positions[indices[:, 0]]
    v1 = positions[indices[:, 1]]
    v2 = positions[indices[:, 2]]
    t0 = uvs[indices[:, 0]]
    t1 = uvs[indices[:, 1]]
    t2 = uvs[indices[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    duv1 = t1 - t0
    duv2 = t2 - t0
    det = duv1[..., 0] * duv2[..., 1] - duv2[..., 0] * duv1[..., 1]
    safe = det.abs() > 1e-12
    r = torch.where(safe, 1.0 / torch.where(safe, det, torch.ones_like(det)), torch.zeros_like(det))
    face_t = (e1 * duv2[..., 1:2] - e2 * duv1[..., 1:2]) * r.unsqueeze(-1)
    accum = torch.zeros_like(positions)
    for k in range(3):
        accum.index_add_(0, indices[:, k], face_t)
    # Gram–Schmidt orthogonalize against the normal.
    ndotl = (accum * normals).sum(dim=-1, keepdim=True)
    tan = accum - normals * ndotl
    norm = torch.linalg.norm(tan, dim=-1, keepdim=True)
    fallback = torch.zeros_like(tan)
    fallback[..., 0] = 1.0
    tan = torch.where(norm > 1e-9, tan / norm.clamp(min=1e-9), fallback)
    return tan
