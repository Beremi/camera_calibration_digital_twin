"""Noise preset loading utilities for the batch estimator and simulator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (_repo_root() / candidate).resolve()


def _float_tuple(values: Any, *, length: int, label: str) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise ValueError(f"{label} must contain exactly {length} numeric values.")
    return tuple(float(value) for value in values)


@dataclass(slots=True)
class ImuNoisePreset:
    name: str
    accel_noise_std: tuple[float, float, float]
    gyro_noise_std: tuple[float, float, float]
    accel_bias_random_walk_std: tuple[float, float, float]
    gyro_bias_random_walk_std: tuple[float, float, float]
    accel_bias_mps2: tuple[float, float, float]
    gyro_bias_rps: tuple[float, float, float]
    sample_drop_probability: float = 0.0
    likelihood_scale: float = 1.0
    position_std_floor: float = 2.0e-2
    velocity_std_floor: float = 2.0e-1
    rotation_std_floor: float = 5.0e-2


@dataclass(slots=True)
class VisionNoisePreset:
    name: str
    corner_noise_std_px: float
    detection_drop_probability: float
    quality_scale: float = 1.0
    visibility_failure_probability: float = 0.0
    outlier_probability: float = 0.0


@dataclass(slots=True)
class TimingNoisePreset:
    name: str
    physics_rate_hz: float
    camera_rate_hz: float
    imu_rate_hz: float
    imu_jitter_std_s: float = 0.0
    camera_jitter_std_s: float = 0.0
    frame_drop_probability: float = 0.0


_IMU_PRESETS = {
    "imu_ideal": ImuNoisePreset(
        name="imu_ideal",
        accel_noise_std=(0.0, 0.0, 0.0),
        gyro_noise_std=(0.0, 0.0, 0.0),
        accel_bias_random_walk_std=(0.0, 0.0, 0.0),
        gyro_bias_random_walk_std=(0.0, 0.0, 0.0),
        accel_bias_mps2=(0.0, 0.0, 0.0),
        gyro_bias_rps=(0.0, 0.0, 0.0),
    ),
    "imu_nominal_phone": ImuNoisePreset(
        name="imu_nominal_phone",
        accel_noise_std=(0.0047856453, 0.0047856453, 0.0047856453),
        gyro_noise_std=(0.0012217305, 0.0012217305, 0.0012217305),
        accel_bias_random_walk_std=(0.00012, 0.00012, 0.00012),
        gyro_bias_random_walk_std=(2.5e-05, 2.5e-05, 2.5e-05),
        accel_bias_mps2=(0.0, 0.0, 0.0),
        gyro_bias_rps=(0.0, 0.0, 0.0),
    ),
    "imu_stress_phone": ImuNoisePreset(
        name="imu_stress_phone",
        accel_noise_std=(0.012, 0.012, 0.012),
        gyro_noise_std=(0.0032, 0.0032, 0.0032),
        accel_bias_random_walk_std=(0.00045, 0.00045, 0.00045),
        gyro_bias_random_walk_std=(0.00012, 0.00012, 0.00012),
        accel_bias_mps2=(0.0, 0.0, 0.0),
        gyro_bias_rps=(0.0, 0.0, 0.0),
        sample_drop_probability=0.01,
    ),
}

_VISION_PRESETS = {
    "vision_nominal": VisionNoisePreset(
        name="vision_nominal",
        corner_noise_std_px=0.75,
        detection_drop_probability=0.05,
        quality_scale=1.0,
        visibility_failure_probability=0.03,
        outlier_probability=0.01,
    ),
}

_TIMING_PRESETS = {
    "timing_nominal": TimingNoisePreset(
        name="timing_nominal",
        physics_rate_hz=240.0,
        camera_rate_hz=30.0,
        imu_rate_hz=58.236,
        imu_jitter_std_s=0.0,
        camera_jitter_std_s=0.0,
        frame_drop_probability=0.0,
    ),
}


def _load_mapping(path_or_name: str | Path, *, registry: dict[str, Any]) -> dict[str, Any]:
    candidate = Path(path_or_name)
    if candidate.exists():
        loaded = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"Noise preset file must contain a mapping: {candidate}")
        return loaded
    key = str(path_or_name)
    if key in registry:
        item = registry[key]
        return {
            field: getattr(item, field)
            for field in item.__dataclass_fields__.keys()  # type: ignore[attr-defined]
        }
    raise FileNotFoundError(f"Unknown noise preset: {path_or_name}")


def load_imu_noise_preset(path_or_name: str | Path) -> ImuNoisePreset:
    data = _load_mapping(path_or_name, registry=_IMU_PRESETS)
    return ImuNoisePreset(
        name=str(data.get("name", Path(path_or_name).stem)),
        accel_noise_std=_float_tuple(data.get("accel_noise_std", [0.0, 0.0, 0.0]), length=3, label="accel_noise_std"),
        gyro_noise_std=_float_tuple(data.get("gyro_noise_std", [0.0, 0.0, 0.0]), length=3, label="gyro_noise_std"),
        accel_bias_random_walk_std=_float_tuple(
            data.get("accel_bias_random_walk_std", [0.0, 0.0, 0.0]),
            length=3,
            label="accel_bias_random_walk_std",
        ),
        gyro_bias_random_walk_std=_float_tuple(
            data.get("gyro_bias_random_walk_std", [0.0, 0.0, 0.0]),
            length=3,
            label="gyro_bias_random_walk_std",
        ),
        accel_bias_mps2=_float_tuple(data.get("accel_bias_mps2", [0.0, 0.0, 0.0]), length=3, label="accel_bias_mps2"),
        gyro_bias_rps=_float_tuple(data.get("gyro_bias_rps", [0.0, 0.0, 0.0]), length=3, label="gyro_bias_rps"),
        sample_drop_probability=float(data.get("sample_drop_probability", 0.0)),
        likelihood_scale=float(data.get("likelihood_scale", 1.0)),
        position_std_floor=float(data.get("position_std_floor", 2.0e-2)),
        velocity_std_floor=float(data.get("velocity_std_floor", 2.0e-1)),
        rotation_std_floor=float(data.get("rotation_std_floor", 5.0e-2)),
    )


def load_vision_noise_preset(path_or_name: str | Path) -> VisionNoisePreset:
    data = _load_mapping(path_or_name, registry=_VISION_PRESETS)
    return VisionNoisePreset(
        name=str(data.get("name", Path(path_or_name).stem)),
        corner_noise_std_px=float(data.get("corner_noise_std_px", 0.0)),
        detection_drop_probability=float(data.get("detection_drop_probability", 0.0)),
        quality_scale=float(data.get("quality_scale", 1.0)),
        visibility_failure_probability=float(data.get("visibility_failure_probability", 0.0)),
        outlier_probability=float(data.get("outlier_probability", 0.0)),
    )


def load_timing_noise_preset(path_or_name: str | Path) -> TimingNoisePreset:
    data = _load_mapping(path_or_name, registry=_TIMING_PRESETS)
    return TimingNoisePreset(
        name=str(data.get("name", Path(path_or_name).stem)),
        physics_rate_hz=float(data.get("physics_rate_hz", 240.0)),
        camera_rate_hz=float(data.get("camera_rate_hz", 30.0)),
        imu_rate_hz=float(data.get("imu_rate_hz", 58.236)),
        imu_jitter_std_s=float(data.get("imu_jitter_std_s", 0.0)),
        camera_jitter_std_s=float(data.get("camera_jitter_std_s", 0.0)),
        frame_drop_probability=float(data.get("frame_drop_probability", 0.0)),
    )


def load_noise_preset(path_or_name: str | Path):
    """Dispatch to the category-specific preset loader based on the prefix."""
    key = str(path_or_name)
    stem = Path(key).stem
    if stem.startswith("imu_"):
        return load_imu_noise_preset(path_or_name)
    if stem.startswith("vision_"):
        return load_vision_noise_preset(path_or_name)
    if stem.startswith("timing_"):
        return load_timing_noise_preset(path_or_name)
    raise ValueError(f"Cannot infer noise preset category from: {path_or_name}")
