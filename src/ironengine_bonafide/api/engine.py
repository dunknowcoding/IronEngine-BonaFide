"""`Engine` — stateful renderer instance.

An Engine owns a backend, a configurable list of passes, and a profile
context-manager. It does not own scenes or cameras (those are passed in
per render call). Engines are not thread-safe; spawn one per process.
"""
from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager

from ironengine_bonafide import lifecycle
from ironengine_bonafide.api.passes_default import default_passes
from ironengine_bonafide.backends import auto_select
from ironengine_bonafide.core.backend import Backend
from ironengine_bonafide.core.profile import ProfileReport
from ironengine_bonafide.passes.base import RenderPass

_PROFILE_TOKEN: contextvars.ContextVar[ProfileReport | None] = contextvars.ContextVar(
    "_PROFILE_TOKEN", default=None,
)


class Engine:
    """Stateful engine instance.

    Pickle-unfriendly (holds GPU handles). One per process; spawn many
    processes for batch rendering.
    """

    __slots__ = ("backend", "passes", "_closed")

    def __init__(self, backend: Backend, passes: list[RenderPass] | None = None) -> None:
        self.backend = backend
        self.passes: list[RenderPass] = passes or default_passes()
        self._closed = False
        lifecycle.fire("on_engine_start", engine=self)

    # --------------------------------------------------------- factories
    @classmethod
    def auto(cls) -> Engine:
        return cls(auto_select("auto"))

    @classmethod
    def cuda(cls) -> Engine:
        return cls(auto_select("cuda"))

    @classmethod
    def wgpu(cls) -> Engine:
        return cls(auto_select("wgpu"))

    @classmethod
    def cpu(cls) -> Engine:
        return cls(auto_select("cpu"))

    # --------------------------------------------------------- builders
    def with_passes(self, passes: list[RenderPass]) -> Engine:
        """Return a new Engine that shares this backend but uses ``passes``.

        Useful for testing custom pass orderings or building bespoke
        pipelines (e.g. a depth-only pass list for sensor capture).
        """
        return Engine(self.backend, passes)

    # --------------------------------------------------------- profile
    @contextmanager
    def profile(self) -> Iterator[ProfileReport]:
        """Context manager that captures per-pass timings into a
        :class:`ProfileReport`."""
        report = ProfileReport()
        token = _PROFILE_TOKEN.set(report)
        try:
            yield report
        finally:
            _PROFILE_TOKEN.reset(token)

    @staticmethod
    def _active_profile() -> ProfileReport | None:
        return _PROFILE_TOKEN.get()

    # --------------------------------------------------------- lifecycle
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        lifecycle.fire("on_engine_close", engine=self)

    def __enter__(self) -> Engine:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<Engine backend={self.backend} passes={[p.name for p in self.passes]}>"
