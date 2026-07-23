"""Common image-space losses for differentiable rendering."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).abs().mean()


def l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).pow(2).mean()


def psnr(a: torch.Tensor, b: torch.Tensor, *, data_range: float = 1.0) -> torch.Tensor:
    mse = (a - b).pow(2).mean()
    if float(mse) <= 0:
        return torch.tensor(99.0)
    return 10.0 * torch.log10((data_range ** 2) / mse)


def ssim(a: torch.Tensor, b: torch.Tensor, *, window: int = 11) -> torch.Tensor:
    """Simplified SSIM in 0..1 — single-scale, gray-converted."""
    if a.ndim == 3:
        a = a.permute(2, 0, 1).unsqueeze(0)
    if b.ndim == 3:
        b = b.permute(2, 0, 1).unsqueeze(0)
    if a.shape[1] == 3:
        a = (0.2989 * a[:, 0] + 0.5870 * a[:, 1] + 0.1140 * a[:, 2]).unsqueeze(1)
        b = (0.2989 * b[:, 0] + 0.5870 * b[:, 1] + 0.1140 * b[:, 2]).unsqueeze(1)

    pad = window // 2
    kernel = torch.ones(1, 1, window, window, device=a.device) / (window * window)
    mu_a = F.conv2d(a, kernel, padding=pad)
    mu_b = F.conv2d(b, kernel, padding=pad)
    sa = F.conv2d(a * a, kernel, padding=pad) - mu_a ** 2
    sb = F.conv2d(b * b, kernel, padding=pad) - mu_b ** 2
    sab = F.conv2d(a * b, kernel, padding=pad) - mu_a * mu_b
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (sa + sb + c2))
    return s.mean()
