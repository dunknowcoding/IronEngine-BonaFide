"""Seed plumbing.

Reproducibility contract: with the same backend and the same `RenderConfig.seed`,
two `render()` calls produce byte-identical outputs. Cross-backend results may
differ by floating-point reduction order.

Use `seed_everything(seed)` once at the start of a deterministic run, then
spawn child seeds with `child(name, parent_seed)` for each subsystem so
independent passes don't share an RNG stream.
"""
from __future__ import annotations

import hashlib
import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, Torch (CPU + CUDA) RNGs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed & 0xFFFF_FFFF)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def child(name: str, parent_seed: int) -> int:
    """Derive a stable child seed from a name + parent. The same name always
    produces the same child seed, so passes are independently reproducible."""
    h = hashlib.blake2b(f"{parent_seed}:{name}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") & 0x7FFF_FFFF_FFFF_FFFF


def torch_generator(seed: int, device: str | torch.device = "cpu") -> torch.Generator:
    """Build a torch.Generator pre-seeded for one subsystem."""
    g = torch.Generator(device=str(device))
    g.manual_seed(int(seed))
    return g
