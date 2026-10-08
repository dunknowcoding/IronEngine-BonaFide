"""Anti-aliasing passes: SSAA resolve, TAA, and morphological AA.

* ``SsaaDownsamplePass`` — area-average resolve of an ssaa× frame back to
  the config resolution, in linear HDR (the correct space for averaging).
* ``TaaPass`` — real temporal anti-aliasing: a Halton-2,3 sub-pixel jitter
  is applied to the camera projection by the render driver (see
  ``api.render._do_render``), and this pass blends each frame into an
  exponential history with 3x3 neighbourhood clamping (variance-clipping
  style) to kill ghosting on static content. History resets whenever the
  camera, resolution, or jitter phase changes. Best for stills and
  sequences rendered through repeated ``render()`` calls on one Engine.
* ``SmaaPass`` — single-pass morphological AA in the SMAA family: luma
  edge detection, run-length measurement along detected edges, and
  length-weighted cross-edge blending. It does not use Jimenez's
  precomputed area textures, so it is not bit-equal to reference SMAA —
  but it is a true morphological filter, not an FXAA alias.
"""
from __future__ import annotations

import torch

from ironengine_bonafide.passes.base import PassContext, RenderPass


class SsaaDownsamplePass(RenderPass):
    """Area-average resolve of an ssaa× frame to the config resolution."""
    name = "ssaa_downsample"

    def is_active(self, ctx: PassContext) -> bool:
        return int(getattr(ctx.config, "ssaa", 1)) > 1

    def run(self, ctx: PassContext) -> None:
        s = int(ctx.config.ssaa)
        t = ctx.targets
        t.rgb = area_downsample(t.rgb, s)
        t.normals = area_downsample(t.normals, s)
        t.albedo = area_downsample(t.albedo, s)
        t.depth = depth_downsample(t.depth, s)
        t.ids = ids_downsample(t.ids, s)


class TaaPass(RenderPass):
    """Temporal AA with Halton jitter + reprojected, depth-rejected history.

    The render driver calls :meth:`next_jitter` before the pass list runs
    and applies the returned NDC offset to the camera; this pass then runs
    in the AA slot and temporally blends the jittered frame.

    Motion handling: the previous frame's history is **reprojected** with
    per-pixel motion vectors (current world position → previous frame's
    view-proj) and **depth-rejected** where the reprojected linear depth
    disagrees with the stored previous depth by more than 2% — so moving
    objects don't smear a ghost trail, while static content keeps its
    SSAA-class convergence. History resets when camera / resolution /
    jitter phase changes.
    """
    name = "taa"

    def __init__(self) -> None:
        self._history: torch.Tensor | None = None
        self._prev_vp: torch.Tensor | None = None
        self._prev_depth: torch.Tensor | None = None
        self._sig: tuple | None = None
        self._frame: int = 0

    # ------------------------------------------------------------ driver side
    def next_jitter(self, camera: object, width: int, height: int,
                    config: object) -> tuple[float, float]:
        """Advance the Halton sequence and return this frame's jitter in
        NDC units (applied to the camera projection by the render driver).

        Resets the sequence (and history) when the camera transform /
        intrinsics, the resolution, or the jitter period change.
        """
        aspect = width / max(1, height)
        try:
            vp = camera.view_proj(aspect)                      # type: ignore[attr-defined]
            sig = (width, height,
                   int(getattr(config, "taa_jitter_frames", 8)),
                   vp.tobytes())
        except Exception:                                       # noqa: BLE001
            sig = (width, height,
                   int(getattr(config, "taa_jitter_frames", 8)), None)
        if sig != self._sig:
            self._sig = sig
            self._frame = 0
            self._history = None
            self._prev_vp = None
            self._prev_depth = None
        period = int(getattr(config, "taa_jitter_frames", 8))
        idx = self._frame % period
        self._frame += 1
        if period == 8:
            # Bounded-step sequence (see _JITTER_SEQ_8) — the raw Halton
            # order spikes the reprojection error once per cycle.
            jx_px, jy_px = _JITTER_SEQ_8[idx]
        else:
            jx_px = _halton(idx + 1, 2) - 0.5
            jy_px = _halton(idx + 1, 3) - 0.5
        return (jx_px * 2.0 / max(1, width), jy_px * 2.0 / max(1, height))

    # ------------------------------------------------------------ pass side
    def is_active(self, ctx: PassContext) -> bool:
        return ctx.config.aa == "taa"

    def run(self, ctx: PassContext) -> None:
        cur = ctx.targets.rgb
        alpha = float(getattr(ctx.config, "taa_alpha", 0.1))
        device = cur.device
        vp = ctx.camera.view_proj_torch(ctx.aspect, device=device)
        if (self._history is None or self._history.shape != cur.shape
                or self._prev_vp is None or self._prev_depth is None):
            out = cur
        else:
            world = _world_pos_from_depth(ctx)              # (H, W, 3), 0 where empty
            prev_uv_g, prev_z_g, vis_g = _project_prev(world, self._prev_vp)
            # Sky has no world position: reproject by ray DIRECTION (a far
            # point through the pixel) instead of the origin fallback.
            far_pts, cam_pos = _far_points(ctx)
            prev_uv_s, _, vis_s = _project_prev(cam_pos + far_pts, self._prev_vp)
            is_sky = ~torch.isfinite(ctx.targets.depth)
            prev_uv = torch.where(is_sky.unsqueeze(-1), prev_uv_s, prev_uv_g)
            # History RGB: Catmull-Rom (no per-frame diffusion blur);
            # depth: bilinear (statistics, not structure).
            hist = _catmull_rom(self._history, prev_uv)
            prev_d = _bilinear(self._prev_depth.unsqueeze(-1),
                               prev_uv).squeeze(-1)
            # Depth rejection (motion): the point's linear depth in the
            # PREVIOUS frame vs the stored previous depth map — >2% off
            # means the surface moved (or disoccluded) and history is
            # stale there. SUB-PIXEL reprojection shifts (< 0.75 px) are
            # jitter-induced coverage changes, not motion — those accept
            # the clamped history instead of falling back to the raw
            # frame, which is what keeps every jitter phase converged.
            h, w = cur.shape[:2]
            yy, xx = torch.meshgrid(
                torch.arange(h, device=cur.device, dtype=cur.dtype),
                torch.arange(w, device=cur.device, dtype=cur.dtype),
                indexing="ij",
            )
            pix_uv = torch.stack([(xx + 0.5) / w, (yy + 0.5) / h], dim=-1)
            shift = ((prev_uv - pix_uv) * torch.tensor([w, h], device=cur.device,
                                                       dtype=cur.dtype)).norm(dim=-1)
            small_shift = shift <= 0.75
            cur_lin = _linear_eye_depth(ctx, prev_z_g)
            prev_lin = _linear_eye_depth(ctx, prev_d)
            both = torch.isfinite(prev_z_g) & torch.isfinite(prev_d)
            rel = ((cur_lin - prev_lin).abs()
                   / prev_lin.clamp(min=1e-3))
            depth_ok = both & (rel <= 0.02)
            valid_geo = vis_g & ~is_sky & (small_shift | depth_ok)
            valid_sky = vis_s & is_sky & (small_shift | ~torch.isfinite(prev_d))
            valid = valid_geo | valid_sky
            hist = torch.where(valid.unsqueeze(-1), hist, cur)
            lo, hi = _neighborhood_minmax(cur)
            hist = hist.clamp(min=lo, max=hi)
            out = alpha * cur + (1.0 - alpha) * hist
        ctx.targets.rgb = out
        self._history = out.clone()
        self._prev_vp = vp.clone()
        self._prev_depth = ctx.targets.depth.clone()


class SmaaPass(RenderPass):
    """Morphological AA (SMAA family): edge detect → run length → blend."""
    name = "smaa"

    def is_active(self, ctx: PassContext) -> bool:
        return ctx.config.aa == "smaa"

    def run(self, ctx: PassContext) -> None:
        ctx.targets.rgb = _mlaa(ctx.targets.rgb)


# ---------------------------------------------------------------- helpers
def area_downsample(x: torch.Tensor, s: int) -> torch.Tensor:
    """(H, W, C) → (H/s, W/s, C) area average (NOT nearest-neighbour)."""
    h, w, c = x.shape
    img = x.permute(2, 0, 1).unsqueeze(0)                    # (1, C, H, W)
    out = torch.nn.functional.avg_pool2d(img, kernel_size=s)
    return out.squeeze(0).permute(1, 2, 0).contiguous()


def depth_downsample(depth: torch.Tensor, s: int) -> torch.Tensor:
    """Min-pool depth so edge blocks keep the closest surface; blocks that
    are entirely empty stay +inf."""
    neg = (-depth).unsqueeze(0).unsqueeze(0)
    out = torch.nn.functional.max_pool2d(neg, kernel_size=s)
    return (-out).squeeze(0).squeeze(0).contiguous()


def ids_downsample(ids: torch.Tensor, s: int) -> torch.Tensor:
    """Nearest (block-centre) resolve for integer instance IDs — averaging
    IDs would be meaningless."""
    h, w = ids.shape
    blocks = ids.reshape(h // s, s, w // s, s)
    return blocks[:, s // 2, :, s // 2].contiguous()


def _halton(index: int, base: int) -> float:
    """Halton low-discrepancy sequence (1-based index), value in [0, 1)."""
    f = 1.0
    r = 0.0
    i = index
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


# 8-phase TAA jitter (centered sub-pixel offsets, px): the 8 Halton(2,3)
# points REORDERED so consecutive samples stay ~0.27 px apart, then scaled
# to a ~0.35 px radius. Two measurable failure modes are avoided:
# raw Halton order has a 0.78 px jump between samples 6 and 7 (every edge
# pixel fails depth rejection on that step and falls back to the raw
# frame), and a full ±0.5 px radius leaves visible per-phase error at the
# largest samples. This set keeps every phase converged.
_JITTER_SEQ_8: tuple[tuple[float, float], ...] = (
    (-0.1500,  0.2333),
    (-0.2625,  0.2333),
    (-0.2250, -0.0334),
    (-0.0750, -0.2333),
    ( 0.0000, -0.1000),
    ( 0.2250, -0.1667),
    ( 0.1500,  0.0334),
    ( 0.0750,  0.1667),
)


def _neighborhood_minmax(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """3x3 per-pixel min / max of an (H, W, C) tensor (replicate padding),
    via max-pooling — O(H·W·C) memory, not the O(9·H·W·C) of unfold."""
    img = x.permute(2, 0, 1).unsqueeze(0)                    # (1, C, H, W)
    hi = torch.nn.functional.max_pool2d(img, kernel_size=3, stride=1, padding=1)
    lo = -torch.nn.functional.max_pool2d(-img, kernel_size=3, stride=1, padding=1)
    return (lo.squeeze(0).permute(1, 2, 0).contiguous(),
            hi.squeeze(0).permute(1, 2, 0).contiguous())


# ------------------------------------------------------------- TAA helpers
def _world_pos_from_depth(ctx: PassContext) -> torch.Tensor:
    """Unproject the frame depth buffer to world positions (H, W, 3);
    zero where depth is empty. The camera's current (jittered) view-proj
    is used, so reprojection is consistent with the rendered frame."""
    depth = ctx.targets.depth
    h, w = depth.shape
    device = depth.device
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    ndc_x = (xx + 0.5) / w * 2.0 - 1.0
    ndc_y = 1.0 - ((yy + 0.5) / h * 2.0)
    z = torch.where(torch.isfinite(depth), depth, torch.zeros_like(depth))
    ndc = torch.stack([ndc_x, ndc_y, z, torch.ones_like(z)], dim=-1)
    vp = ctx.camera.view_proj_torch(ctx.aspect, device=device)
    inv = torch.linalg.inv(vp.double()).to(torch.float32)
    world_h = ndc.reshape(-1, 4) @ inv.T
    world = (world_h[:, :3] / world_h[:, 3:4].clamp(min=1e-6)).reshape(h, w, 3)
    return torch.where(torch.isfinite(depth).unsqueeze(-1), world,
                       torch.zeros_like(world))


def _project_prev(world: torch.Tensor, prev_vp: torch.Tensor,
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project world points into the PREVIOUS frame's view-proj.

    Returns (uv in [0, 1] (H, W, 2), ndc z (H, W), visibility mask (H, W)).
    """
    h, w, _ = world.shape
    homog = torch.cat([world, torch.ones((h, w, 1), device=world.device,
                                         dtype=world.dtype)], dim=-1)
    clip = homog.reshape(-1, 4) @ prev_vp.T
    w_clip = clip[:, 3:4].clamp(min=1e-6)
    ndc = (clip[:, :3] / w_clip).reshape(h, w, 3)
    uv = (ndc[..., :2] * 0.5 + 0.5)
    # Image-space v: row 0 is the TOP of the frame (ndc y = +1).
    uv = torch.stack([uv[..., 0], 1.0 - uv[..., 1]], dim=-1)
    vis = ((uv[..., 0] >= 0.0) & (uv[..., 0] <= 1.0)
           & (uv[..., 1] >= 0.0) & (uv[..., 1] <= 1.0)
           & (ndc[..., 2].abs() <= 1.0)
           & (clip[:, 3].reshape(h, w) > 1e-6))
    return uv, ndc[..., 2], vis


def _far_points(ctx: PassContext) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-pixel far-plane world points MINUS the camera position (i.e.
    scaled ray directions), (H, W, 3), plus the camera world position.

    Used to reproject sky pixels by direction rather than by a bogus
    world position.
    """
    depth = ctx.targets.depth
    h, w = depth.shape
    device = depth.device
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    ndc_x = (xx + 0.5) / w * 2.0 - 1.0
    ndc_y = 1.0 - ((yy + 0.5) / h * 2.0)
    z = torch.full((h, w), 0.999, device=device, dtype=torch.float32)
    ndc = torch.stack([ndc_x, ndc_y, z, torch.ones_like(z)], dim=-1)
    vp = ctx.camera.view_proj_torch(ctx.aspect, device=device)
    inv = torch.linalg.inv(vp.double()).to(torch.float32)
    far_h = ndc.reshape(-1, 4) @ inv.T
    far = (far_h[:, :3] / far_h[:, 3:4].clamp(min=1e-6)).reshape(h, w, 3)
    cam = ctx.camera
    if hasattr(cam, "position"):
        cam_pos = torch.tensor(cam.position, device=device, dtype=torch.float32)
    elif getattr(cam, "pose", None) is not None:
        cam_pos = torch.as_tensor(cam.pose[:3, 3]).to(device=device,
                                                      dtype=torch.float32)
    else:
        cam_pos = torch.zeros(3, device=device, dtype=torch.float32)
    return far - cam_pos, cam_pos


def _bilinear(img: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Bilinear-sample an (H, W, C) image at (H', W', 2) uv in [0, 1];
    clamps at the borders."""
    h, w = img.shape[0], img.shape[1]
    px = (uv[..., 0] * w - 0.5).clamp(0.0, w - 1.0)
    py = (uv[..., 1] * h - 0.5).clamp(0.0, h - 1.0)
    x0 = px.floor().long()
    y0 = py.floor().long()
    x1 = (x0 + 1).clamp(max=w - 1)
    y1 = (y0 + 1).clamp(max=h - 1)
    fx = (px - px.floor()).unsqueeze(-1)
    fy = (py - py.floor()).unsqueeze(-1)
    c00 = img[y0, x0]
    c01 = img[y0, x1]
    c10 = img[y1, x0]
    c11 = img[y1, x1]
    return ((c00 * (1 - fx) + c01 * fx) * (1 - fy)
            + (c10 * (1 - fx) + c11 * fx) * fy)


def _catmull_rom(img: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Catmull-Rom sample of an (H, W, C) image at (H', W', 2) uv in
    [0, 1] (4x4 taps, separable cubic). Used for TAA history: unlike
    bilinear it does not blur edges on every reprojection, which is what
    keeps the accumulated history sharp over many frames. Out-of-range
    taps clamp to the border; results are expected to be range-clamped
    downstream (the TAA neighbourhood clamp does this)."""
    h, w = img.shape[0], img.shape[1]
    px = (uv[..., 0] * w - 0.5).clamp(-0.5, w - 0.5)
    py = (uv[..., 1] * h - 0.5).clamp(-0.5, h - 0.5)
    x0 = px.floor()
    y0 = py.floor()
    fx = (px - x0)
    fy = (py - y0)

    def _w(t: torch.Tensor) -> tuple[torch.Tensor, ...]:
        t2 = t * t
        t3 = t2 * t
        return (
            -0.5 * t3 + t2 - 0.5 * t,
            1.5 * t3 - 2.5 * t2 + 1.0,
            -1.5 * t3 + 2.0 * t2 + 0.5 * t,
            0.5 * t3 - 0.5 * t2,
        )

    wx = _w(fx)
    wy = _w(fy)
    xi = [x0.long() + k - 1 for k in range(4)]
    yi = [y0.long() + k - 1 for k in range(4)]
    xi = [x.clamp(0, w - 1) for x in xi]
    yi = [y.clamp(0, h - 1) for y in yi]
    out = torch.zeros_like(img)
    for ky in range(4):
        row = torch.zeros_like(img)
        for kx in range(4):
            row = row + img[yi[ky], xi[kx]] * wx[kx].unsqueeze(-1)
        out = out + row * wy[ky].unsqueeze(-1)
    return out


def _linear_eye_depth(ctx: PassContext, ndc_z: torch.Tensor) -> torch.Tensor:
    """NDC z → linear eye depth in meters (same convention as the
    volumetric pass); +inf stays +inf."""
    from ironengine_bonafide.core.camera import OrthographicCamera
    near = float(getattr(ctx.camera, "near", 0.05))
    far = float(getattr(ctx.camera, "far", 200.0))
    z = torch.where(torch.isfinite(ndc_z), ndc_z, torch.ones_like(ndc_z))
    if isinstance(ctx.camera, OrthographicCamera):
        d = near + (z + 1.0) * 0.5 * (far - near)
    else:
        d = (2.0 * near * far) / (z * (near - far) + far + near)
    return torch.where(torch.isfinite(ndc_z), d, torch.full_like(d, float("inf")))


# ------------------------------------------------------------- MLAA (SMAA)
_LUMA = (0.299, 0.587, 0.114)
_EDGE_THRESHOLD = 0.1          # relative luma discontinuity
_MAX_RUN = 8                   # run length that maps to full blend weight


def _luma(rgb: torch.Tensor) -> torch.Tensor:
    w = torch.tensor(_LUMA, dtype=rgb.dtype, device=rgb.device)
    return (rgb * w).sum(dim=-1)


def _run_lengths(mask: torch.Tensor, along_rows: bool) -> torch.Tensor:
    """Length of the contiguous True run each True cell belongs to,
    measured along rows (``along_rows=True``) or columns."""
    m = mask if along_rows else mask.t()
    h, w = m.shape
    lengths = torch.zeros((h, w), dtype=torch.float32, device=mask.device)
    m_cpu = m.cpu().numpy()                                  # row runs: cheap
    for y in range(h):
        row = m_cpu[y]
        x = 0
        while x < w:
            if row[x]:
                x0 = x
                while x < w and row[x]:
                    x += 1
                lengths[y, x0:x] = float(x - x0)
            else:
                x += 1
    return lengths if along_rows else lengths.t()


def _blend_along_edges(rgb: torch.Tensor, edges: torch.Tensor,
                       horizontal: bool) -> torch.Tensor:
    """Cross-edge blend for one edge orientation.

    ``horizontal=True`` handles edges between vertically adjacent pixels
    (pixel (y, x) and (y+1, x)); runs are measured horizontally. The blend
    weight grows with run length: long straight staircases get smoothed,
    isolated texture detail barely moves.
    """
    h, w, _ = rgb.shape
    run = _run_lengths(edges, along_rows=horizontal)
    weight = (0.5 * run / _MAX_RUN).clamp(0.0, 0.5)
    if horizontal:
        above = rgb[:-1, :, :]
        below = rgb[1:, :, :]
        wgt = weight.unsqueeze(-1)                            # (H-1, W, 1)
        mix_a = above * (1.0 - wgt) + below * wgt
        mix_b = below * (1.0 - wgt) + above * wgt
        out = rgb.clone()
        out[:-1, :, :] = torch.where(edges.unsqueeze(-1), mix_a, above)
        out[1:, :, :] = torch.where(edges.unsqueeze(-1), mix_b, below)
    else:
        left = rgb[:, :-1, :]
        right = rgb[:, 1:, :]
        wgt = weight.unsqueeze(-1)                            # (H, W-1, 1)
        mix_l = left * (1.0 - wgt) + right * wgt
        mix_r = right * (1.0 - wgt) + left * wgt
        out = rgb.clone()
        out[:, :-1, :] = torch.where(edges.unsqueeze(-1), mix_l, left)
        out[:, 1:, :] = torch.where(edges.unsqueeze(-1), mix_r, right)
    return out


def _mlaa(rgb: torch.Tensor) -> torch.Tensor:
    """Single-pass morphological AA on an (H, W, 3) image.

    1. Luma discontinuity edges between orthogonal neighbours, thresholded
       relative to the frame's luma range (HDR-safe).
    2. For each edge, the contiguous run length along the edge direction
       sets the blend weight (long staircase edges blend strongest).
    3. Both sides of the edge move toward each other symmetrically, so the
       filter is energy-preserving and does not shift edges.
    """
    if rgb.shape[0] < 2 or rgb.shape[1] < 2:
        return rgb
    luma = _luma(rgb)
    scale = (luma.max() - luma.min()).clamp(min=1e-6)
    thr = _EDGE_THRESHOLD * scale
    # Edge between a pixel and its vertical neighbour → horizontal staircase.
    h_edges = (luma[1:, :] - luma[:-1, :]).abs() > thr
    # Edge between a pixel and its horizontal neighbour → vertical staircase.
    v_edges = (luma[:, 1:] - luma[:, :-1]).abs() > thr
    out = _blend_along_edges(rgb, h_edges, horizontal=True)
    out = _blend_along_edges(out, v_edges, horizontal=False)
    return out
