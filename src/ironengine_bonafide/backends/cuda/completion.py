"""Neural hole completion (per-scene prior).

Trains a small hash-grid + MLP on the dense regions of an input cloud and
samples it inside detected holes to predict (color, density) — turning
incomplete scans into seam-free renders.

When `tiny-cuda-nn` is available we use its hash encoding + fused MLP.
Otherwise we degrade to a torch-native MLP with a positional encoding,
which is much slower but produces equivalent results for testing.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(slots=True)
class CompletionPrior:
    """Lightweight learned representation. Call `__call__(positions)` to
    predict per-point colors."""
    network: nn.Module
    aabb_min: torch.Tensor
    aabb_max: torch.Tensor

    def __call__(self, positions: torch.Tensor) -> torch.Tensor:
        norm = (positions - self.aabb_min) / (self.aabb_max - self.aabb_min + 1e-9)
        return self.network(norm).clamp(0.0, 1.0)


# --------------------------------------------------------------- training
def train_completion_prior(
    positions: torch.Tensor,                # (N, 3)
    colors: torch.Tensor,                   # (N, 3) in [0, 1]
    *,
    width: int = 64,
    depth: int = 3,
    iterations: int = 1000,
    lr: float = 1e-3,
    device: str | torch.device = "cuda",
) -> CompletionPrior:
    aabb_min = positions.min(0).values.to(device)
    aabb_max = positions.max(0).values.to(device)

    net = _build_network(width=width, depth=depth, device=device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    pos = positions.to(device)
    col = colors.to(device)
    norm_pos = (pos - aabb_min) / (aabb_max - aabb_min + 1e-9)

    n = pos.shape[0]
    bs = min(8192, n)
    for _ in range(iterations):
        idx = torch.randint(0, n, (bs,), device=device)
        pred = net(norm_pos[idx])
        loss = (pred - col[idx]).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return CompletionPrior(network=net, aabb_min=aabb_min, aabb_max=aabb_max)


def _build_network(*, width: int, depth: int, device: str | torch.device) -> nn.Module:
    """Hash-grid + MLP if `tiny-cuda-nn` is available, else MLP+positional encoding."""
    try:
        import tinycudann as tcnn  # type: ignore[import-not-found]
        config = {
            "encoding": {
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.5,
            },
            "network": {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": width,
                "n_hidden_layers": depth,
            },
        }
        return tcnn.NetworkWithInputEncoding(3, 3, config["encoding"], config["network"]).to(device)
    except ImportError:
        return _MlpWithPositional(width=width, depth=depth, n_freqs=10).to(device)


class _MlpWithPositional(nn.Module):
    def __init__(self, *, width: int, depth: int, n_freqs: int) -> None:
        super().__init__()
        self.n_freqs = n_freqs
        in_dim = 3 + 3 * 2 * n_freqs
        layers: list[nn.Module] = [nn.Linear(in_dim, width), nn.ReLU(inplace=True)]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.ReLU(inplace=True)]
        layers += [nn.Linear(width, 3), nn.Sigmoid()]
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NeRF-style positional encoding
        feats = [x]
        for f in range(self.n_freqs):
            scale = 2.0 ** f
            feats.append(torch.sin(x * scale))
            feats.append(torch.cos(x * scale))
        return self.body(torch.cat(feats, dim=-1))
