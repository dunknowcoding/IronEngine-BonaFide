"""Generic name-keyed registry — the only place "decorators register things"
should be encoded. Used by passes, backends, asset loaders.

Explicit registration via ``@registry.register("name")`` (or
``registry.register("name", cls)``) keeps discovery deterministic and
debuggable. Registries are plain dicts under the hood; no metaclass magic,
no entry-point auto-discovery.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Generic, TypeVar

from ironengine_bonafide.errors import ValidationError

T = TypeVar("T")


class Registry(Generic[T]):
    """Map of name -> entry. Names must be unique; pass ``replace=True``
    to override (with a logged warning at the call site)."""

    __slots__ = ("kind", "_entries")

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: dict[str, T] = {}

    # ----------------------------------------------------------- write
    def register(
        self,
        name: str,
        entry: T | None = None,
        *,
        replace: bool = False,
    ) -> Callable[[T], T] | T:
        """Two-mode usage:

          functional:  registry.register("name", entry_obj)
          decorator:   @registry.register("name")
        """
        if not name:
            raise ValidationError(f"{self.kind} registry: name must be non-empty")
        if entry is not None:
            self._set(name, entry, replace=replace)
            return entry

        def _decorator(real_entry: T) -> T:
            self._set(name, real_entry, replace=replace)
            return real_entry

        return _decorator

    def _set(self, name: str, entry: T, *, replace: bool) -> None:
        if name in self._entries and not replace:
            existing = self._entries[name]
            if existing is entry:        # idempotent re-import
                return
            raise ValidationError(
                f"{self.kind} registry already has '{name}' -> {existing!r}; "
                f"pass replace=True to override."
            )
        self._entries[name] = entry

    def unregister(self, name: str) -> None:
        self._entries.pop(name, None)

    # ----------------------------------------------------------- read
    def get(self, name: str) -> T | None:
        return self._entries.get(name)

    def require(self, name: str) -> T:
        if name not in self._entries:
            raise ValidationError(
                f"No '{name}' in {self.kind} registry. "
                f"Available: {sorted(self._entries)}"
            )
        return self._entries[name]

    def names(self) -> list[str]:
        return sorted(self._entries)

    def items(self) -> list[tuple[str, T]]:
        return list(self._entries.items())

    # ----------------------------------------------------------- protocol
    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"<Registry {self.kind!r} entries={self.names()}>"
