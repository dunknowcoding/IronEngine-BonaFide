"""pytest configuration for BonaFide.

Skip CUDA-only suites when CUDA isn't available; provide a small set of
deterministic asset fixtures that live in `examples/assets/` once we add
sample PLY/GLB files. For now, fixtures synthesise tiny in-memory data.

Mark a test with ``@pytest.mark.cuda`` to declare it CUDA-only — it's
skipped automatically when ``torch.cuda.is_available()`` is False or when
``BONAFIDE_BACKEND=cpu`` is in the environment.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "cuda: requires a CUDA-capable PyTorch + GPU")


@pytest.fixture
def cube_pointcloud() -> tuple[np.ndarray, np.ndarray]:
    g = np.random.default_rng(42)
    positions = g.uniform(-0.5, 0.5, size=(2000, 3)).astype(np.float32)
    colors = g.uniform(0.0, 1.0, size=(2000, 3)).astype(np.float32)
    return positions, colors


@pytest.fixture
def triangle_mesh() -> tuple[np.ndarray, np.ndarray]:
    positions = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 1.0, 0.0]],
                         dtype=np.float32)
    indices = np.array([[0, 1, 2]], dtype=np.int64)
    return positions, indices


def pytest_collection_modifyitems(config: object, items: list[pytest.Item]) -> None:  # noqa: ARG001
    cuda_off = (not torch.cuda.is_available()) or os.environ.get("BONAFIDE_BACKEND") == "cpu"
    if not cuda_off:
        return
    skip_cuda = pytest.mark.skip(
        reason="torch.cuda not available or BONAFIDE_BACKEND=cpu",
    )
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip_cuda)
