"""Equirectangular environment-map sampling helpers.

Shared by :class:`~ironengine_bonafide.passes.sky_pass.SkyPass`
(background) and :class:`~ironengine_bonafide.passes.pbr_pass.PbrPass`
(image-based lighting). Maps are ``(H, W, 3)`` float32 linear-HDR
equirect panoramas; directions are unit vectors.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def equirect_sample(env: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Bilinear-sample an equirect panorama at unit directions.

    ``env`` is (H, W, C); ``dirs`` is (..., 3). Longitude wraps, latitude
    clamps. Convention: u = atan2(z, x) / 2π + 0.5, v = 0.5 − asin(y) / π.
    """
    x = dirs[..., 0]
    y = dirs[..., 1].clamp(-1.0, 1.0)
    z = dirs[..., 2]
    u = torch.atan2(z, x) * (1.0 / (2.0 * torch.pi)) + 0.5
    v = 0.5 - torch.asin(y) * (1.0 / torch.pi)
    h, w = env.shape[0], env.shape[1]
    # Pixel-center coordinates, wrap in u, clamp in v.
    px = (u * w - 0.5) % w
    py = (v * h - 0.5).clamp(0.0, h - 1.0)

    x0 = px.floor().long() % w
    x1 = (x0 + 1) % w
    y0 = py.floor().long()
    y1 = (y0 + 1).clamp(max=h - 1)
    fx = (px - px.floor()).unsqueeze(-1)
    fy = (py - py.floor()).unsqueeze(-1)

    c00 = env[y0, x0]
    c01 = env[y0, x1]
    c10 = env[y1, x0]
    c11 = env[y1, x1]
    top = c00 * (1.0 - fx) + c01 * fx
    bot = c10 * (1.0 - fx) + c11 * fx
    return top * (1.0 - fy) + bot * fy


def build_mip_chain(env: torch.Tensor, *, min_h: int = 4, max_levels: int = 7) -> list[torch.Tensor]:
    """Average-pooled mip pyramid of an equirect map (level 0 = full res).

    Coarser levels approximate prefiltered (blurred) environment lighting;
    used for roughness-based specular and diffuse irradiance lookups.
    """
    chain = [env]
    cur = env
    while cur.shape[0] > min_h and len(chain) < max_levels:
        img = cur.permute(2, 0, 1).unsqueeze(0)             # (1, C, H, W)
        h, w = img.shape[-2:]
        kh = 2 if h % 2 == 0 else 1
        kw = 2 if w % 2 == 0 else 1
        if kh == 1 and kw == 1:
            break
        img = F.avg_pool2d(img, kernel_size=(kh, kw), stride=(kh, kw))
        cur = img.squeeze(0).permute(1, 2, 0).contiguous()
        chain.append(cur)
    return chain


def sample_mip_blended(
    mips: list[torch.Tensor],
    dirs: torch.Tensor,
    level: torch.Tensor | float,
) -> torch.Tensor:
    """Sample the mip pyramid at a fractional level (trilinear-ish).

    ``level`` is a scalar or a (...,) tensor in [0, len(mips) − 1]; the
    two nearest levels are sampled and linearly blended.
    """
    n = len(mips)
    if n == 1:
        return equirect_sample(mips[0], dirs)
    lvl = torch.as_tensor(level, dtype=torch.float32, device=dirs.device)
    lvl = lvl.clamp(0.0, n - 1.0)
    l0 = lvl.floor().long().clamp(max=n - 1)
    l1 = (l0 + 1).clamp(max=n - 1)
    frac = (lvl - l0.float()).unsqueeze(-1)

    if l0.ndim == 0:
        c0 = equirect_sample(mips[int(l0)], dirs)
        c1 = equirect_sample(mips[int(l1)], dirs)
        return c0 * (1.0 - frac) + c1 * frac

    # Per-element level: sample every level touched and select by mask.
    out = torch.zeros((*dirs.shape[:-1], mips[0].shape[-1]),
                      dtype=dirs.dtype, device=dirs.device)
    for li in torch.unique(l0).tolist():
        li = int(li)
        m = (l0 == li)
        c0 = equirect_sample(mips[li], dirs[m])
        c1 = equirect_sample(mips[min(li + 1, n - 1)], dirs[m])
        out[m] = c0 * (1.0 - frac[m]) + c1 * frac[m]
    return out
