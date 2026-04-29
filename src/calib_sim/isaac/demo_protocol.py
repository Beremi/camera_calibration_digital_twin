"""Command and snapshot helpers for the tabletop-only Isaac demo."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from calib_sim.isaac.demo_profiles import DEFAULT_PROFILE_KEY, get_demo_profile, list_demo_profiles


CONTROL_MODES = ("manual_joint", "manual_task", "auto_demo")
IMU_NOISE_MODES = ("ideal", "nominal_phone", "stress_phone")
VISION_NOISE_MODES = ("clean", "nominal")
ACTUATION_NOISE_MODES = ("none", "servo_nominal", "servo_stress")


def validate_control_mode(mode: str, *, allow_auto_path: bool) -> str:
    del allow_auto_path
    candidate = str(mode).strip().lower()
    if candidate not in CONTROL_MODES:
        raise ValueError(f"Unsupported control mode: {mode}")
    return candidate


def validate_noise_modes(
    *,
    imu_mode: str | None = None,
    vision_mode: str | None = None,
    actuation_mode: str | None = None,
    detector_mode: str | None = None,
) -> dict[str, str]:
    payload: dict[str, str] = {}
    if imu_mode is not None:
        candidate = str(imu_mode).strip().lower()
        if candidate not in IMU_NOISE_MODES:
            raise ValueError(f"Unsupported IMU noise mode: {imu_mode}")
        payload["imu_mode"] = candidate
    if vision_mode is not None:
        candidate = str(vision_mode).strip().lower()
        if candidate not in VISION_NOISE_MODES:
            raise ValueError(f"Unsupported vision noise mode: {vision_mode}")
        payload["vision_mode"] = candidate
    if actuation_mode is not None:
        candidate = str(actuation_mode).strip().lower()
        if candidate not in ACTUATION_NOISE_MODES:
            raise ValueError(f"Unsupported actuation noise mode: {actuation_mode}")
        payload["actuation_mode"] = candidate
    if detector_mode not in (None, "", "new_pupil"):
        raise ValueError("Only the new_pupil detector is supported in this branch.")
    if detector_mode is not None:
        payload["detector_mode"] = "new_pupil"
    return payload


def empty_snapshot(*, profile_key: str | None = None) -> dict[str, Any]:
    profile = get_demo_profile(profile_key)
    return {
        "session": {
            "state": "idle",
            "profile_key": profile.key,
            "profile_label": profile.label,
            "last_error": None,
        },
        "frames": {
            "primary": {"image_data_url": "", "frame_index": None, "timestamp_s": None},
            "observers": [],
        },
        "robot": {"arm_joints": []},
        "control": {
            "control_mode": profile.default_control_mode,
            "tracking_error_norm_m": None,
            "safety_reason": None,
            "desired_position_world_m": None,
        },
        "estimation": {
            "estimator_mode": profile.default_estimator_mode,
            "anchor_visible": False,
            "position_radius_95_m": None,
            "innovation_norm": None,
            "summary": "Waiting for Isaac worker startup.",
        },
        "detections": {
            "count": 0,
            "pattern_locations": [],
            "backend": "new_pupil",
            "backend_state": "active",
            "backend_error": None,
            "input_transport": None,
            "bridge_latency_ms": None,
            "timeouts": 0,
            "fallback_active": False,
        },
        "imu": {
            "measured": {"accel_mps2": [0.0, 0.0, 0.0], "gyro_rps": [0.0, 0.0, 0.0]},
            "truth": {"accel_mps2": [0.0, 0.0, 0.0], "gyro_rps": [0.0, 0.0, 0.0]},
        },
        "noise": {
            "imu_mode": "ideal",
            "vision_mode": "clean",
            "actuation_mode": "none",
            "detector_mode": "new_pupil",
        },
        "recording": {
            "enabled": False,
            "active_run_dir": None,
            "last_run_dir": None,
            "output_root": "output/isaac_demo_runs",
        },
        "analysis": {
            "state": "idle",
            "last_started_at_utc": None,
            "last_completed_at_utc": None,
            "last_run_dir": None,
            "summary": None,
            "report_path": None,
            "error": None,
        },
    }


def initial_catalog() -> dict[str, Any]:
    return {
        "profiles": list_demo_profiles(),
        "default_profile_key": DEFAULT_PROFILE_KEY,
    }


def snapshot_with_error(message: str, *, profile_key: str | None = None) -> dict[str, Any]:
    payload = deepcopy(empty_snapshot(profile_key=profile_key))
    payload["session"]["state"] = "error"
    payload["session"]["last_error"] = str(message)
    payload["estimation"]["summary"] = str(message)
    return payload


__all__ = [
    "ACTUATION_NOISE_MODES",
    "CONTROL_MODES",
    "IMU_NOISE_MODES",
    "VISION_NOISE_MODES",
    "empty_snapshot",
    "initial_catalog",
    "snapshot_with_error",
    "validate_control_mode",
    "validate_noise_modes",
]
