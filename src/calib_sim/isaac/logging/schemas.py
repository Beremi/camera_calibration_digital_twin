"""Typed schema helpers for Isaac raw, GT, and estimate logs."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any


def _float_list(values: tuple[float, ...] | list[float]) -> list[float]:
    return [float(value) for value in values]


@dataclass(slots=True)
class IsaacCameraFramePacket:
    frame_index: int
    timestamp_s: float
    sim_time_s: float
    sensor_time_s: float
    host_time_s: float
    rgb_path: str
    intrinsics_snapshot: dict[str, Any]
    extrinsics_snapshot: dict[str, Any]
    visible_gt_tag_ids: tuple[int, ...] = ()

    def as_json(self) -> dict[str, Any]:
        return {
            "frame_index": int(self.frame_index),
            "timestamp_s": float(self.timestamp_s),
            "sim_time_s": float(self.sim_time_s),
            "sensor_time_s": float(self.sensor_time_s),
            "host_time_s": float(self.host_time_s),
            "rgb_path": self.rgb_path,
            "intrinsics_snapshot": dict(self.intrinsics_snapshot),
            "extrinsics_snapshot": dict(self.extrinsics_snapshot),
            "visible_gt_tag_ids": [int(tag_id) for tag_id in self.visible_gt_tag_ids],
        }


@dataclass(slots=True)
class IsaacTagDetectionPacket:
    timestamp_s: float
    sim_time_s: float
    frame_index: int
    tag_id: int
    corners_xy: tuple[tuple[float, float], ...]
    corner_order: str
    score: float
    is_anchor: bool
    visibility_flags: dict[str, bool] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        return {
            "timestamp_s": float(self.timestamp_s),
            "sim_time_s": float(self.sim_time_s),
            "frame_index": int(self.frame_index),
            "tag_id": int(self.tag_id),
            "corners_xy": [[float(x), float(y)] for x, y in self.corners_xy],
            "corner_order": self.corner_order,
            "score": float(self.score),
            "is_anchor": bool(self.is_anchor),
            "visibility_flags": {str(key): bool(value) for key, value in self.visibility_flags.items()},
        }


@dataclass(slots=True)
class IsaacImuPacket:
    timestamp_s: float
    sim_time_s: float
    wx: float
    wy: float
    wz: float
    ax: float
    ay: float
    az: float
    imu_frame: str
    noise_preset: str
    sensor_time_s: float | None = None
    host_time_s: float | None = None

    @classmethod
    def csv_fieldnames(cls) -> list[str]:
        return [
            "timestamp_s",
            "sim_time_s",
            "wx",
            "wy",
            "wz",
            "ax",
            "ay",
            "az",
            "imu_frame",
            "noise_preset",
            "sensor_time_s",
            "host_time_s",
        ]

    def to_csv_row(self) -> dict[str, Any]:
        return {
            "timestamp_s": float(self.timestamp_s),
            "sim_time_s": float(self.sim_time_s),
            "wx": float(self.wx),
            "wy": float(self.wy),
            "wz": float(self.wz),
            "ax": float(self.ax),
            "ay": float(self.ay),
            "az": float(self.az),
            "imu_frame": self.imu_frame,
            "noise_preset": self.noise_preset,
            "sensor_time_s": None if self.sensor_time_s is None else float(self.sensor_time_s),
            "host_time_s": None if self.host_time_s is None else float(self.host_time_s),
        }


@dataclass(slots=True)
class IsaacJointCommandPacket:
    timestamp_s: float
    sim_time_s: float
    joint_id: str
    command_type: str
    command_value: float
    controller_mode: str

    @classmethod
    def csv_fieldnames(cls) -> list[str]:
        return [
            "timestamp_s",
            "sim_time_s",
            "joint_id",
            "command_type",
            "command_value",
            "controller_mode",
        ]

    def to_csv_row(self) -> dict[str, Any]:
        return {
            "timestamp_s": float(self.timestamp_s),
            "sim_time_s": float(self.sim_time_s),
            "joint_id": self.joint_id,
            "command_type": self.command_type,
            "command_value": float(self.command_value),
            "controller_mode": self.controller_mode,
        }


@dataclass(slots=True)
class IsaacRealizedJointPacket:
    timestamp_s: float
    sim_time_s: float
    joint_names: tuple[str, ...]
    positions: tuple[float, ...]
    velocities: tuple[float, ...]
    servo_internal_state: dict[str, Any] | None = None

    @classmethod
    def csv_fieldnames(cls) -> list[str]:
        return [
            "timestamp_s",
            "sim_time_s",
            "joint_names",
            "positions",
            "velocities",
            "servo_internal_state",
        ]

    def to_csv_row(self) -> dict[str, Any]:
        return {
            "timestamp_s": float(self.timestamp_s),
            "sim_time_s": float(self.sim_time_s),
            "joint_names": "|".join(self.joint_names),
            "positions": "|".join(str(float(value)) for value in self.positions),
            "velocities": "|".join(str(float(value)) for value in self.velocities),
            "servo_internal_state": "" if self.servo_internal_state is None else json.dumps(dict(self.servo_internal_state), sort_keys=True),
        }
