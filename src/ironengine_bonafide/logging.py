"""Logging + progress wiring.

Loguru drives the structured log surface; rich.progress drives the
progress bars used for long renders / training runs. Both are optional —
if either dependency is missing we fall back to stdlib :mod:`logging`
and print-style progress, so the engine still imports.
"""
from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


def _build_logger() -> Any:
    try:
        from loguru import logger as _logger
        _logger.remove()
        level = os.environ.get("BONAFIDE_LOG_LEVEL", "INFO").upper()
        _logger.add(
            sys.stderr,
            level=level,
            format=("<green>{time:HH:mm:ss}</green> | "
                    "<level>{level: <8}</level> | "
                    "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"),
        )
        return _logger
    except ImportError:
        import logging as _logging
        _logging.basicConfig(
            level=os.environ.get("BONAFIDE_LOG_LEVEL", "INFO"),
            format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
        )
        return _logging.getLogger("ironengine_bonafide")


logger = _build_logger()


@contextmanager
def progress(description: str, total: int | None = None) -> Iterator[Any]:
    """Yield a progress handle. Call `handle.update(n=1)` per tick.

    Falls back to a no-op handle if `rich` isn't installed.
    """
    try:
        from rich.progress import Progress
    except ImportError:
        class _Noop:
            def update(self, n: int = 1, **_: Any) -> None: ...
        yield _Noop()
        return
    with Progress() as bar:
        task_id = bar.add_task(description, total=total)
        class _Handle:
            def update(self, n: int = 1, **kw: Any) -> None:
                bar.update(task_id, advance=n, **kw)
        yield _Handle()
