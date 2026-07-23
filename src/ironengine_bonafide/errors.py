"""Typed exception hierarchy for BonaFide.

Public modules raise these instead of bare ``RuntimeError`` / ``ValueError``
so users can ``except ironengine_bonafide.errors.BackendUnavailable`` cleanly.

Hierarchy::

    BonaFideError
      ├── ConfigurationError
      ├── ValidationError
      ├── BackendError
      │     ├── BackendUnavailable
      │     └── CapabilityMissing
      ├── AssetError
      │     ├── AssetNotFound
      │     ├── AssetFormatError
      │     └── AssetDecodeError
      ├── PassError
      │     └── PassDependencyError
      └── IntegrationError
"""
from __future__ import annotations


class BonaFideError(Exception):
    """Base class — every public-facing exception derives from this."""


# --------------------------------------------------------------- config / validation
class ConfigurationError(BonaFideError):
    """Bad RenderConfig field combination (e.g. samples=0)."""


class ValidationError(BonaFideError):
    """Argument shape / dtype / range validation failed."""


# --------------------------------------------------------------- backend
class BackendError(BonaFideError):
    """Base class for backend issues."""


class BackendUnavailable(BackendError):  # noqa: N818 — public API name, not changing
    """The selected backend cannot be constructed (no GPU, missing libs)."""


class CapabilityMissing(BackendError):  # noqa: N818 — public API name, not changing
    """A pass required a capability the backend doesn't advertise."""

    def __init__(self, backend: str, capability: str) -> None:
        super().__init__(f"Backend '{backend}' does not provide '{capability}'.")
        self.backend = backend
        self.capability = capability


# --------------------------------------------------------------- assets
class AssetError(BonaFideError):
    """Base class for asset I/O issues."""


class AssetNotFound(AssetError):  # noqa: N818 — public API name, not changing
    """A referenced asset file does not exist."""


class AssetFormatError(AssetError):
    """The on-disk format isn't supported, or the header was malformed."""


class AssetDecodeError(AssetError):
    """Format was recognized but decoding failed."""


# --------------------------------------------------------------- passes
class PassError(BonaFideError):
    """Base class for render-pass execution issues."""


class PassDependencyError(PassError):
    """A pass requires data the upstream passes did not produce."""


# --------------------------------------------------------------- integrations
class IntegrationError(BonaFideError):
    """3DCreator / Sim shim could not bind."""


__all__ = [
    "AssetDecodeError",
    "AssetError",
    "AssetFormatError",
    "AssetNotFound",
    "BackendError",
    "BackendUnavailable",
    "BonaFideError",
    "CapabilityMissing",
    "ConfigurationError",
    "IntegrationError",
    "PassDependencyError",
    "PassError",
    "ValidationError",
]
