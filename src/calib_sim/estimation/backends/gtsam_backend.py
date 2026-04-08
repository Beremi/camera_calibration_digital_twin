"""GTSAM backend availability probe.

The current repo environment does not ship a working gtsam wheel, so this
module intentionally exposes only a probe and a descriptive fallback.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class GtsamBackendStatus:
    available: bool
    reason: str


def probe_gtsam_backend() -> GtsamBackendStatus:
    try:
        import gtsam  # type: ignore  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        return GtsamBackendStatus(available=False, reason=f"gtsam unavailable: {exc}")
    return GtsamBackendStatus(available=True, reason="gtsam import succeeded")

