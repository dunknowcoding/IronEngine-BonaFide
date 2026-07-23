"""LRU asset cache with VRAM-budget awareness.

Handles are keyed by SHA1(path + modtime). On overflow, the oldest entries
are evicted until total reported bytes fit under `budget_mb`.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Any


def _digest(path: Path) -> str:
    s = path.stat()
    h = hashlib.sha1(f"{path.resolve()}|{s.st_mtime_ns}|{s.st_size}".encode())
    return h.hexdigest()


class AssetCache:
    def __init__(self, budget_mb: float = 4096.0) -> None:
        self.budget_mb = float(budget_mb)
        self._entries: OrderedDict[str, tuple[Any, int]] = OrderedDict()
        self._total_bytes: int = 0

    def get(self, path: Path) -> Any | None:
        key = _digest(path)
        if key not in self._entries:
            return None
        self._entries.move_to_end(key)
        return self._entries[key][0]

    def put(self, path: Path, payload: Any, size_bytes: int) -> None:
        key = _digest(path)
        if key in self._entries:
            old = self._entries.pop(key)
            self._total_bytes -= old[1]
        self._entries[key] = (payload, int(size_bytes))
        self._total_bytes += int(size_bytes)
        self._enforce()

    def _enforce(self) -> None:
        budget_bytes = int(self.budget_mb * 1024 * 1024)
        while self._total_bytes > budget_bytes and self._entries:
            _, (_, size) = self._entries.popitem(last=False)
            self._total_bytes -= size

    @property
    def used_mb(self) -> float:
        return self._total_bytes / (1024 * 1024)

    def __len__(self) -> int:
        return len(self._entries)
