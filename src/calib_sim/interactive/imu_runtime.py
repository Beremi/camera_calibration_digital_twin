"""Reusable IMU timing and noise helpers for the interactive simulator.

This module is intentionally simulator-agnostic. It provides:
- noise preset loading from YAML
- fixed-rate IMU scheduling with a checkpoint compatibility mode
- additive white-noise and bias random-walk helpers
- a small runtime wrapper that turns truth samples into noisy measurements

The interactive simulator will wire these helpers in later without forcing the
estimation package to depend on the browser runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import yaml

from calib_sim.common.models import Vec3


REPO_ROOT = Path(__file__).resolve().parents[3]


def describe_imu_measurement_convention(*, compat_single_sample: bool, read_gravity: bool) -> dict[str, str]:
    """Return a machine-readable description of the simulator IMU convention."""

    return {
        "imu_measurement_convention": "body_specific_force_plus_bias" if read_gravity else "body_linear_acceleration_plus_bias",
        "imu_timestamp_semantics": "frame_locked_sim_time" if compat_single_sample else "independent_sim_time",
        "imu_gravity_handling": "gravity_removed_from_specific_force" if read_gravity else "gravity_retained_in_body_acceleration",
        "imu_sampling_mode": "end_sampled_interval" if compat_single_sample else "independent_fixed_rate",
        "accelerometer_measurement_equation": "tilde_f_k = R_IW_k (a_W_k - g_W) + b_a + n_a",
        "gyroscope_measurement_equation": "tilde_omega_k = omega_k + b_g + n_g",
    }


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def _to_vec3(values: Sequence[float] | np.ndarray, *, name: str) -> Vec3:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size != 3:
        raise ValueError(f"{name} must contain exactly 3 values.")
    return float(array[0]), float(array[1]), float(array[2])


def _to_array3(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    return np.asarray(_to_vec3(values, name=name), dtype=np.float64)


def _load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    resolved = _resolve_repo_path(path)
    data = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML file: {resolved}")
    return data


@dataclass(slots=True)
class ImuNoisePreset:
    """Noise and timing preset for the synthetic IMU."""

    name: str
    mode: str
    rate_hz: float
    read_gravity: bool
    accel_noise_std_mps2: np.ndarray
    gyro_noise_std_rps: np.ndarray
    accel_bias_initial_mps2: np.ndarray
    gyro_bias_initial_rps: np.ndarray
    accel_bias_walk_std_mps2_per_sqrt_s: np.ndarray
    gyro_bias_walk_std_rps_per_sqrt_s: np.ndarray
    gravity_mps2: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -9.81], dtype=np.float64))
    sample_jitter_std_s: float = 0.0
    drop_probability: float = 0.0

    @property
    def period_s(self) -> float:
        return 1.0 / max(float(self.rate_hz), 1e-12)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode,
            "rate_hz": float(self.rate_hz),
            "read_gravity": bool(self.read_gravity),
            "accel_noise_std_mps2": [float(value) for value in self.accel_noise_std_mps2.tolist()],
            "gyro_noise_std_rps": [float(value) for value in self.gyro_noise_std_rps.tolist()],
            "accel_bias_initial_mps2": [float(value) for value in self.accel_bias_initial_mps2.tolist()],
            "gyro_bias_initial_rps": [float(value) for value in self.gyro_bias_initial_rps.tolist()],
            "accel_bias_walk_std_mps2_per_sqrt_s": [
                float(value) for value in self.accel_bias_walk_std_mps2_per_sqrt_s.tolist()
            ],
            "gyro_bias_walk_std_rps_per_sqrt_s": [float(value) for value in self.gyro_bias_walk_std_rps_per_sqrt_s.tolist()],
            "gravity_mps2": [float(value) for value in self.gravity_mps2.tolist()],
            "sample_jitter_std_s": float(self.sample_jitter_std_s),
            "drop_probability": float(self.drop_probability),
        }


@dataclass(slots=True)
class VisionNoisePreset:
    """Measurement-layer visual noise preset."""

    name: str
    corner_noise_std_px: float
    dropout_probability: float
    visibility_failure_probability: float
    tag_quality_sigma_scale: float
    outlier_probability: float = 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "corner_noise_std_px": float(self.corner_noise_std_px),
            "dropout_probability": float(self.dropout_probability),
            "visibility_failure_probability": float(self.visibility_failure_probability),
            "tag_quality_sigma_scale": float(self.tag_quality_sigma_scale),
            "outlier_probability": float(self.outlier_probability),
        }


@dataclass(slots=True)
class TimingPreset:
    """Clocking preset for the interactive runtime."""

    name: str
    physics_rate_hz: float
    camera_rate_hz: float
    imu_rate_hz: float
    checkpoint_compat_rate_hz: float
    camera_jitter_std_s: float = 0.0
    imu_jitter_std_s: float = 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "physics_rate_hz": float(self.physics_rate_hz),
            "camera_rate_hz": float(self.camera_rate_hz),
            "imu_rate_hz": float(self.imu_rate_hz),
            "checkpoint_compat_rate_hz": float(self.checkpoint_compat_rate_hz),
            "camera_jitter_std_s": float(self.camera_jitter_std_s),
            "imu_jitter_std_s": float(self.imu_jitter_std_s),
        }


@dataclass(slots=True)
class ImuTruthSample:
    """Noise-free inertial truth at a point in time."""

    sim_time_s: float
    accel_body_mps2: np.ndarray
    gyro_body_rps: np.ndarray


@dataclass(slots=True)
class ScheduledImuTick:
    """One scheduled IMU emission time."""

    sample_index: int
    sim_time_s: float
    dt_s: float


@dataclass(slots=True)
class ImuMeasurement:
    """Noisy IMU packet emitted by the runtime helper."""

    tick_index: int
    sample_index: int
    sim_time_s: float
    dt_s: float
    accel_body_mps2: np.ndarray
    gyro_body_rps: np.ndarray
    accel_bias_mps2: np.ndarray
    gyro_bias_rps: np.ndarray
    is_checkpoint_compat: bool = False

    def as_row(self) -> dict[str, Any]:
        return {
            "tick_index": int(self.tick_index),
            "sample_index": int(self.sample_index),
            "sim_time_s": float(self.sim_time_s),
            "dt_s": float(self.dt_s),
            "accel_body_mps2": [float(value) for value in self.accel_body_mps2.tolist()],
            "gyro_body_rps": [float(value) for value in self.gyro_body_rps.tolist()],
            "accel_bias_mps2": [float(value) for value in self.accel_bias_mps2.tolist()],
            "gyro_bias_rps": [float(value) for value in self.gyro_bias_rps.tolist()],
            "is_checkpoint_compat": bool(self.is_checkpoint_compat),
        }


@dataclass(slots=True)
class ImuBiasState:
    """Additive bias state with a random-walk update."""

    accel_bias_mps2: np.ndarray
    gyro_bias_rps: np.ndarray

    @classmethod
    def from_preset(cls, preset: ImuNoisePreset) -> "ImuBiasState":
        return cls(
            accel_bias_mps2=np.asarray(preset.accel_bias_initial_mps2, dtype=np.float64).reshape(3).copy(),
            gyro_bias_rps=np.asarray(preset.gyro_bias_initial_rps, dtype=np.float64).reshape(3).copy(),
        )

    def copy(self) -> "ImuBiasState":
        return ImuBiasState(
            accel_bias_mps2=self.accel_bias_mps2.copy(),
            gyro_bias_rps=self.gyro_bias_rps.copy(),
        )

    def step(self, *, dt_s: float, rng: np.random.Generator, preset: ImuNoisePreset) -> "ImuBiasState":
        dt_s = float(max(dt_s, 0.0))
        if dt_s <= 0.0:
            return self
        root_dt = float(np.sqrt(dt_s))
        self.accel_bias_mps2 = self.accel_bias_mps2 + rng.normal(size=3) * preset.accel_bias_walk_std_mps2_per_sqrt_s * root_dt
        self.gyro_bias_rps = self.gyro_bias_rps + rng.normal(size=3) * preset.gyro_bias_walk_std_rps_per_sqrt_s * root_dt
        return self


def load_imu_noise_preset(path: str | Path) -> ImuNoisePreset:
    """Load an IMU noise preset from YAML."""

    data = _load_yaml_mapping(path)
    name = str(data.get("name") or Path(path).stem)
    mode = str(data.get("mode") or "async_realism")
    return ImuNoisePreset(
        name=name,
        mode=mode,
        rate_hz=float(data.get("rate_hz", 58.236)),
        read_gravity=bool(data.get("read_gravity", True)),
        accel_noise_std_mps2=_to_array3(data.get("accel_noise_std_mps2", data.get("accel_noise_std", (0.0, 0.0, 0.0))), name="accel_noise_std_mps2"),
        gyro_noise_std_rps=_to_array3(data.get("gyro_noise_std_rps", data.get("gyro_noise_std", (0.0, 0.0, 0.0))), name="gyro_noise_std_rps"),
        accel_bias_initial_mps2=_to_array3(
            data.get("accel_bias_initial_mps2", data.get("accel_bias", (0.0, 0.0, 0.0))),
            name="accel_bias_initial_mps2",
        ),
        gyro_bias_initial_rps=_to_array3(
            data.get("gyro_bias_initial_rps", data.get("gyro_bias", (0.0, 0.0, 0.0))),
            name="gyro_bias_initial_rps",
        ),
        accel_bias_walk_std_mps2_per_sqrt_s=_to_array3(
            data.get("accel_bias_walk_std_mps2_per_sqrt_s", (0.0, 0.0, 0.0)),
            name="accel_bias_walk_std_mps2_per_sqrt_s",
        ),
        gyro_bias_walk_std_rps_per_sqrt_s=_to_array3(
            data.get("gyro_bias_walk_std_rps_per_sqrt_s", (0.0, 0.0, 0.0)),
            name="gyro_bias_walk_std_rps_per_sqrt_s",
        ),
        gravity_mps2=_to_array3(data.get("gravity_mps2", (0.0, 0.0, -9.81)), name="gravity_mps2"),
        sample_jitter_std_s=float(data.get("sample_jitter_std_s", 0.0)),
        drop_probability=float(data.get("drop_probability", 0.0)),
    )


def load_vision_noise_preset(path: str | Path) -> VisionNoisePreset:
    """Load a visual measurement noise preset from YAML."""

    data = _load_yaml_mapping(path)
    return VisionNoisePreset(
        name=str(data.get("name") or Path(path).stem),
        corner_noise_std_px=float(data.get("corner_noise_std_px", 0.0)),
        dropout_probability=float(data.get("dropout_probability", 0.0)),
        visibility_failure_probability=float(data.get("visibility_failure_probability", 0.0)),
        tag_quality_sigma_scale=float(data.get("tag_quality_sigma_scale", 1.0)),
        outlier_probability=float(data.get("outlier_probability", 0.0)),
    )


def load_timing_preset(path: str | Path) -> TimingPreset:
    """Load a runtime timing preset from YAML."""

    data = _load_yaml_mapping(path)
    return TimingPreset(
        name=str(data.get("name") or Path(path).stem),
        physics_rate_hz=float(data.get("physics_rate_hz", 240.0)),
        camera_rate_hz=float(data.get("camera_rate_hz", 12.0)),
        imu_rate_hz=float(data.get("imu_rate_hz", 58.236)),
        checkpoint_compat_rate_hz=float(data.get("checkpoint_compat_rate_hz", 12.0)),
        camera_jitter_std_s=float(data.get("camera_jitter_std_s", 0.0)),
        imu_jitter_std_s=float(data.get("imu_jitter_std_s", 0.0)),
    )


def load_builtin_noise_presets() -> dict[str, ImuNoisePreset]:
    """Load the shipped IMU presets from `config/noise`."""

    preset_files = {
        "imu_ideal": REPO_ROOT / "config" / "noise" / "imu_ideal.yaml",
        "imu_nominal_phone": REPO_ROOT / "config" / "noise" / "imu_nominal_phone.yaml",
        "imu_stress_phone": REPO_ROOT / "config" / "noise" / "imu_stress_phone.yaml",
    }
    return {name: load_imu_noise_preset(path) for name, path in preset_files.items()}


@dataclass(slots=True)
class FixedRateImuScheduler:
    """Emit IMU sample timestamps at a fixed rate."""

    rate_hz: float
    start_time_s: float = 0.0
    compat_single_sample: bool = False
    _next_sample_index: int = field(init=False, default=0)
    _next_sample_time_s: float = field(init=False, default=0.0)
    _last_emit_time_s: float | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.reset(self.start_time_s)

    @property
    def period_s(self) -> float:
        return 1.0 / max(float(self.rate_hz), 1e-12)

    def reset(self, start_time_s: float = 0.0) -> None:
        self.start_time_s = float(start_time_s)
        self._next_sample_index = 0
        self._next_sample_time_s = self.start_time_s + self.period_s
        self._last_emit_time_s = None

    def advance_to(self, current_time_s: float, *, frame_dt_s: float | None = None) -> list[ScheduledImuTick]:
        current_time_s = float(current_time_s)
        if self.compat_single_sample:
            if self._last_emit_time_s is not None and current_time_s <= self._last_emit_time_s + 1e-12:
                return []
            dt_s = float(frame_dt_s if frame_dt_s is not None else (
                current_time_s - self._last_emit_time_s if self._last_emit_time_s is not None else current_time_s - self.start_time_s
            ))
            tick = ScheduledImuTick(
                sample_index=self._next_sample_index,
                sim_time_s=current_time_s,
                dt_s=max(dt_s, 0.0),
            )
            self._next_sample_index += 1
            self._last_emit_time_s = current_time_s
            return [tick]

        scheduled: list[ScheduledImuTick] = []
        while self._next_sample_time_s <= current_time_s + 1e-12:
            dt_s = self._next_sample_time_s - (
                self._last_emit_time_s if self._last_emit_time_s is not None else self.start_time_s
            )
            scheduled.append(
                ScheduledImuTick(
                    sample_index=self._next_sample_index,
                    sim_time_s=self._next_sample_time_s,
                    dt_s=max(float(dt_s), 0.0),
                )
            )
            self._last_emit_time_s = self._next_sample_time_s
            self._next_sample_index += 1
            self._next_sample_time_s += self.period_s
        return scheduled


TruthFn = Callable[[float, float], ImuTruthSample | tuple[Sequence[float], Sequence[float]]]


@dataclass(slots=True)
class ImuRuntime:
    """Combine a schedule, bias state, and preset into noisy IMU measurements."""

    preset: ImuNoisePreset
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    compat_single_sample: bool = False
    scheduler: FixedRateImuScheduler = field(init=False)
    bias_state: ImuBiasState = field(init=False)

    def __post_init__(self) -> None:
        self.scheduler = FixedRateImuScheduler(
            rate_hz=float(self.preset.rate_hz),
            start_time_s=0.0,
            compat_single_sample=bool(self.compat_single_sample or self.preset.mode == "checkpoint_01_compat"),
        )
        self.bias_state = ImuBiasState.from_preset(self.preset)

    def reset(self, *, start_time_s: float = 0.0) -> None:
        self.scheduler.reset(start_time_s)
        self.bias_state = ImuBiasState.from_preset(self.preset)

    def _coerce_truth(self, value: ImuTruthSample | tuple[Sequence[float], Sequence[float]], sim_time_s: float) -> ImuTruthSample:
        if isinstance(value, ImuTruthSample):
            return value
        accel, gyro = value
        return ImuTruthSample(
            sim_time_s=float(sim_time_s),
            accel_body_mps2=np.asarray(accel, dtype=np.float64).reshape(3),
            gyro_body_rps=np.asarray(gyro, dtype=np.float64).reshape(3),
        )

    def _sample_noise(self, std: np.ndarray) -> np.ndarray:
        std = np.asarray(std, dtype=np.float64).reshape(3)
        if np.allclose(std, 0.0):
            return np.zeros(3, dtype=np.float64)
        return self.rng.normal(size=3) * std

    def _apply_drop(self) -> bool:
        if self.preset.drop_probability <= 0.0:
            return False
        return bool(self.rng.random() < float(self.preset.drop_probability))

    def advance_to(
        self,
        current_time_s: float,
        truth_fn: TruthFn,
        *,
        frame_dt_s: float | None = None,
        frame_tick_index: int | None = None,
    ) -> list[ImuMeasurement]:
        """Advance the runtime and emit zero or more noisy IMU packets."""

        measurements: list[ImuMeasurement] = []
        scheduled_ticks = self.scheduler.advance_to(current_time_s, frame_dt_s=frame_dt_s)
        for scheduled_tick in scheduled_ticks:
            truth_sample = self._coerce_truth(truth_fn(scheduled_tick.sim_time_s, scheduled_tick.dt_s), scheduled_tick.sim_time_s)
            self.bias_state.step(dt_s=scheduled_tick.dt_s, rng=self.rng, preset=self.preset)
            if self._apply_drop():
                continue
            emitted_time_s = float(scheduled_tick.sim_time_s)
            if not self.scheduler.compat_single_sample and self.preset.sample_jitter_std_s > 0.0:
                emitted_time_s = max(
                    0.0,
                    emitted_time_s + float(self.rng.normal(scale=float(self.preset.sample_jitter_std_s))),
                )
            accel_noise = self._sample_noise(self.preset.accel_noise_std_mps2)
            gyro_noise = self._sample_noise(self.preset.gyro_noise_std_rps)
            accel = truth_sample.accel_body_mps2 + self.bias_state.accel_bias_mps2 + accel_noise
            gyro = truth_sample.gyro_body_rps + self.bias_state.gyro_bias_rps + gyro_noise
            measurements.append(
                ImuMeasurement(
                    tick_index=int(frame_tick_index if frame_tick_index is not None else scheduled_tick.sample_index),
                    sample_index=int(scheduled_tick.sample_index),
                    sim_time_s=emitted_time_s,
                    dt_s=float(scheduled_tick.dt_s),
                    accel_body_mps2=np.asarray(accel, dtype=np.float64).reshape(3),
                    gyro_body_rps=np.asarray(gyro, dtype=np.float64).reshape(3),
                    accel_bias_mps2=self.bias_state.accel_bias_mps2.copy(),
                    gyro_bias_rps=self.bias_state.gyro_bias_rps.copy(),
                    is_checkpoint_compat=bool(self.scheduler.compat_single_sample),
                )
            )
        return measurements


def checkpoint_01_compat_imu_runtime(preset: ImuNoisePreset, *, rng: np.random.Generator | None = None) -> ImuRuntime:
    """Return an IMU runtime that mirrors the current checkpoint behavior."""

    runtime = ImuRuntime(preset=preset, rng=rng if rng is not None else np.random.default_rng(), compat_single_sample=True)
    runtime.scheduler.compat_single_sample = True
    return runtime


def load_checkpoint_compat_preset(path: str | Path) -> ImuNoisePreset:
    """Load a preset and force checkpoint-compatible semantics."""

    preset = load_imu_noise_preset(path)
    return ImuNoisePreset(
        name=preset.name,
        mode="checkpoint_01_compat",
        rate_hz=preset.rate_hz,
        read_gravity=preset.read_gravity,
        accel_noise_std_mps2=preset.accel_noise_std_mps2.copy(),
        gyro_noise_std_rps=preset.gyro_noise_std_rps.copy(),
        accel_bias_initial_mps2=preset.accel_bias_initial_mps2.copy(),
        gyro_bias_initial_rps=preset.gyro_bias_initial_rps.copy(),
        accel_bias_walk_std_mps2_per_sqrt_s=np.zeros(3, dtype=np.float64),
        gyro_bias_walk_std_rps_per_sqrt_s=np.zeros(3, dtype=np.float64),
        gravity_mps2=preset.gravity_mps2.copy(),
        sample_jitter_std_s=0.0,
        drop_probability=0.0,
    )


def emit_checkpoint_01_compat_measurements(
    preset: ImuNoisePreset,
    *,
    current_time_s: float,
    truth_fn: TruthFn,
    frame_tick_index: int,
    frame_dt_s: float | None = None,
    rng: np.random.Generator | None = None,
) -> list[ImuMeasurement]:
    """Convenience helper for the current frame-locked checkpoint path."""

    runtime = checkpoint_01_compat_imu_runtime(preset, rng=rng)
    return runtime.advance_to(
        current_time_s,
        truth_fn,
        frame_dt_s=frame_dt_s,
        frame_tick_index=frame_tick_index,
    )


def build_nominal_imu_runtime(
    *,
    preset_path: str | Path,
    compat_single_sample: bool = False,
    rng: np.random.Generator | None = None,
) -> ImuRuntime:
    """Construct a runtime from a preset path."""

    return ImuRuntime(
        preset=load_imu_noise_preset(preset_path),
        rng=rng if rng is not None else np.random.default_rng(),
        compat_single_sample=compat_single_sample,
    )
