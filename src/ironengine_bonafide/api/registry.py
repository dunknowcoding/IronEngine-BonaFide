"""Backend / pass / loader registries.

Each registry is a :class:`ironengine_bonafide.registry.Registry` instance
with the type narrowed to the kind of entry it holds. Registration calls
should happen at module import time (see ``backends/__init__``,
``passes/__init__``).
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ironengine_bonafide.core.backend import Backend
from ironengine_bonafide.passes.base import RenderPass
from ironengine_bonafide.registry import Registry

# Concrete Backend factories: ``"cuda" → CudaBackend``-like callable.
BACKEND_REGISTRY: Registry[Callable[[], Backend]] = Registry("backend")

# RenderPass factories. Default passes register on import, plug-ins via
# ``passes.register("name", PassClass)``.
PASS_REGISTRY: Registry[Callable[[], RenderPass]] = Registry("pass")

# Asset loaders keyed by extension (``"ply" → load_pointcloud``).
LOADER_REGISTRY: Registry[Callable[..., Any]] = Registry("loader")
