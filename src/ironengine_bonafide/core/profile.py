"""`with engine.profile()` context.

Records per-pass wall time + (when CUDA is available) GPU memory delta.
Returns a `ProfileReport` you can `.summary()` (rich-formatted) or
`.to_dict()` for serialization.
"""
from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch


@dataclass(slots=True)
class PassTiming:
    name: str
    cpu_ms: float
    gpu_alloc_mb: float = 0.0


@dataclass(slots=True)
class ProfileReport:
    timings: list[PassTiming] = field(default_factory=list)
    total_cpu_ms: float = 0.0

    def add(self, t: PassTiming) -> None:
        self.timings.append(t)
        self.total_cpu_ms += t.cpu_ms

    def summary(self) -> str:
        try:
            from rich.console import Console
            from rich.table import Table
        except ImportError:
            return self._summary_plain()
        table = Table(title="BonaFide render profile")
        table.add_column("Pass")
        table.add_column("CPU ms", justify="right")
        table.add_column("GPU MB", justify="right")
        for t in self.timings:
            table.add_row(t.name, f"{t.cpu_ms:.2f}", f"{t.gpu_alloc_mb:+.1f}")
        table.add_row("[bold]TOTAL[/bold]", f"[bold]{self.total_cpu_ms:.2f}[/bold]", "")
        with Console(record=True) as console:
            console.print(table)
            return console.export_text()

    def _summary_plain(self) -> str:
        lines = ["pass                   cpu_ms   gpu_mb"]
        for t in self.timings:
            lines.append(f"{t.name:<22} {t.cpu_ms:>7.2f} {t.gpu_alloc_mb:>+8.1f}")
        lines.append(f"TOTAL                  {self.total_cpu_ms:.2f}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "timings": [{"name": t.name, "cpu_ms": t.cpu_ms, "gpu_alloc_mb": t.gpu_alloc_mb}
                        for t in self.timings],
            "total_cpu_ms": self.total_cpu_ms,
        }


@contextmanager
def stopwatch(report: ProfileReport, name: str) -> Iterator[None]:
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.synchronize()
        gpu_before = torch.cuda.memory_allocated() / (1024 * 1024)
    t0 = time.perf_counter()
    yield
    if cuda:
        torch.cuda.synchronize()
        gpu_after = torch.cuda.memory_allocated() / (1024 * 1024)
        gpu_delta = gpu_after - gpu_before
    else:
        gpu_delta = 0.0
    report.add(PassTiming(
        name=name,
        cpu_ms=(time.perf_counter() - t0) * 1000.0,
        gpu_alloc_mb=gpu_delta,
    ))
