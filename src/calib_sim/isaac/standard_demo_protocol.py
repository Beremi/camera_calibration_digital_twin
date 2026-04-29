"""Snapshot helpers for the tabletop-only standard Isaac demo."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from calib_sim.isaac.demo_profiles import get_demo_profile
from calib_sim.isaac.demo_protocol import empty_snapshot, initial_catalog
from calib_sim.isaac.standard_streaming import default_streaming_status, standard_streaming_backend_catalog


def empty_standard_snapshot(*, profile_key: str | None = None) -> dict[str, Any]:
    profile = get_demo_profile(profile_key)
    payload = deepcopy(empty_snapshot(profile_key=profile.key))
    payload["session"]["streaming_backend"] = "webrtc"
    payload["streaming"] = default_streaming_status("webrtc")
    return payload


def initial_standard_catalog() -> dict[str, Any]:
    payload = initial_catalog()
    payload["default_streaming_backend"] = "webrtc"
    payload["streaming_backends"] = standard_streaming_backend_catalog()
    return payload


def standard_snapshot_with_error(message: str, *, profile_key: str | None = None) -> dict[str, Any]:
    payload = empty_standard_snapshot(profile_key=profile_key)
    payload["session"]["state"] = "error"
    payload["session"]["last_error"] = str(message)
    payload["estimation"]["summary"] = str(message)
    payload["streaming"]["state"] = "failed"
    payload["streaming"]["error"] = str(message)
    return payload


__all__ = [
    "empty_standard_snapshot",
    "initial_standard_catalog",
    "standard_snapshot_with_error",
]
