"""Engine lifecycle hooks.

Subscribers register callables for one of these events:

  * ``"on_engine_start"``  — fired once when an Engine is constructed
  * ``"on_frame_begin"``   — fired at the start of every render(...)
  * ``"on_pass_begin"``    — fired before each pass run
  * ``"on_pass_end"``      — fired after each pass run
  * ``"on_frame_end"``     — fired at the end of every render(...)
  * ``"on_error"``          — fired when a pass raises (always called before re-raise)
  * ``"on_engine_close"``  — fired on Engine.close()

Hooks run synchronously on the rendering thread. They get whatever the
emitting site sends as kwargs — see signatures in ``api.engine.Engine``.

Use case: profilers, benchmarks, debug overlays, telemetry.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

from ironengine_bonafide.errors import ValidationError
from ironengine_bonafide.logging import logger

HookCallable = Callable[..., None]

_HOOKS: dict[str, list[HookCallable]] = defaultdict(list)
_VALID_EVENTS = frozenset({
    "on_engine_start",
    "on_frame_begin",
    "on_pass_begin",
    "on_pass_end",
    "on_frame_end",
    "on_error",
    "on_engine_close",
})


def register(event: str, fn: HookCallable | None = None) -> HookCallable | Callable[[HookCallable], HookCallable]:
    """Register a hook. Functional or decorator form.

    >>> register("on_pass_end", lambda **kw: print(kw["pass_name"]))
    >>> @register("on_frame_begin")
    ... def my_hook(**kw): ...
    """
    if event not in _VALID_EVENTS:
        raise ValidationError(f"Unknown lifecycle event '{event}'. "
                              f"Valid: {sorted(_VALID_EVENTS)}")
    if fn is not None:
        _HOOKS[event].append(fn)
        return fn

    def _decorator(real: HookCallable) -> HookCallable:
        _HOOKS[event].append(real)
        return real

    return _decorator


def unregister(event: str, fn: HookCallable) -> None:
    if event in _HOOKS and fn in _HOOKS[event]:
        _HOOKS[event].remove(fn)


def clear(event: str | None = None) -> None:
    """Drop all hooks (one event, or every event)."""
    if event is None:
        _HOOKS.clear()
    else:
        _HOOKS.pop(event, None)


def fire(event: str, **kwargs: Any) -> None:
    """Invoke every subscriber for ``event``. Exceptions inside hooks are
    logged and swallowed so a buggy hook can't take the renderer down."""
    for hook in list(_HOOKS.get(event, ())):
        try:
            hook(**kwargs)
        except Exception as exc:               # noqa: BLE001 — defensive
            logger.warning(f"hook {hook!r} for {event!r} raised: {exc!r}")


def listeners(event: str) -> list[HookCallable]:
    return list(_HOOKS.get(event, ()))
