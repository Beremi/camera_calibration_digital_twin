"""Typed schema helpers for Isaac raw, GT, and estimate logs."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any


def _float_list(values: tuple[float, ...] | list[float]) -> list[float]:
    return [float(value) for value in values]


def _matrix_list(values: tuple[tuple[float, ...], ...] | list[list[float]]) -> list[list[float]]:
    return [[float(entry) for entry in row] for row in values]


def _pipe_join(values: tuple[str, ...] | tuple[float, ...] | list[str] | list[float]) -> str:
    return "|".join(str(value) for value in values)


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
    image_width_px: int = 0
    image_height_px: int = 0
    visible_gt_tag_ids: tuple[int, ...] = ()

    def as_json(self) -> dict[str, Any]:
        return {
            "frame_index": int(self.frame_index),
            "timestamp_s": float(self.timestamp_s),
            "sim_time_s": float(self.sim_time_s),
            "sensor_time_s": float(self.sensor_time_s),
            "host_time_s": float(self.host_time_s),
            "rgb_path": self.rgb_path,
            "image_width_px": int(self.image_width_px),
            "image_height_px": int(self.image_height_px),
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
    family: str = "apriltag36h11"
    tag_size_m: float = 0.0
    pnp_tag_size_m: float | None = None
    local_tag_points_m: tuple[tuple[float, float, float], ...] = ()
    pose_camera_rvec: tuple[float, float, float] | None = None
    pose_camera_tvec_m: tuple[float, float, float] | None = None
    detector_backend: str = ""
    visibility_flags: dict[str, bool] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        return {
            "timestamp_s": float(self.timestamp_s),
            "sim_time_s": float(self.sim_time_s),
            "frame_index": int(self.frame_index),
            "tag_id": int(self.tag_id),
            "family": self.family,
            "tag_size_m": float(self.tag_size_m),
            "pnp_tag_size_m": None if self.pnp_tag_size_m is None else float(self.pnp_tag_size_m),
            "corners_xy": [[float(x), float(y)] for x, y in self.corners_xy],
            "local_tag_points_m": [[float(x), float(y), float(z)] for x, y, z in self.local_tag_points_m],
            "corner_order": self.corner_order,
            "score": float(self.score),
            "is_anchor": bool(self.is_anchor),
            "pose_camera_rvec": None if self.pose_camera_rvec is None else _float_list(list(self.pose_camera_rvec)),
            "pose_camera_tvec_m": None if self.pose_camera_tvec_m is None else _float_list(list(self.pose_camera_tvec_m)),
            "detector_backend": self.detector_backend,
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
    packet_index: int = 0
    imu_semantics: str = "specific_force"
    dt_s: float | None = None
    orientation_wxyz: tuple[float, float, float, float] | None = None
    sensor_time_s: float | None = None
    host_time_s: float | None = None

    @classmethod
    def csv_fieldnames(cls) -> list[str]:
        return [
            "packet_index",
            "timestamp_s",
            "sim_time_s",
            "dt_s",
            "wx",
            "wy",
            "wz",
            "ax",
            "ay",
            "az",
            "imu_frame",
            "imu_semantics",
            "noise_preset",
            "orientation_wxyz",
            "sensor_time_s",
            "host_time_s",
        ]

    def to_csv_row(self) -> dict[str, Any]:
        return {
            "packet_index": int(self.packet_index),
            "timestamp_s": float(self.timestamp_s),
            "sim_time_s": float(self.sim_time_s),
            "dt_s": None if self.dt_s is None else float(self.dt_s),
            "wx": float(self.wx),
            "wy": float(self.wy),
            "wz": float(self.wz),
            "ax": float(self.ax),
            "ay": float(self.ay),
            "az": float(self.az),
            "imu_frame": self.imu_frame,
            "imu_semantics": self.imu_semantics,
            "noise_preset": self.noise_preset,
            "orientation_wxyz": "" if self.orientation_wxyz is None else _pipe_join(self.orientation_wxyz),
            "sensor_time_s": None if self.sensor_time_s is None else float(self.sensor_time_s),
            "host_time_s": None if self.host_time_s is None else float(self.host_time_s),
        }


@dataclass(slots=True)
class IsaacJointCommandPacket:
    timestamp_s: float
    sim_time_s: float
    joint_names: tuple[str, ...]
    desired_positions: tuple[float, ...]
    effective_positions: tuple[float, ...]
    estimator_mode: str
    controller_mode: str
    waypoint_index: int
    safety_reason: str
    tracking_error_world_m: tuple[float, float, float]
    dropped_command: bool = False

    @classmethod
    def csv_fieldnames(cls) -> list[str]:
        return [
            "timestamp_s",
            "sim_time_s",
            "joint_names",
            "desired_positions",
            "effective_positions",
            "estimator_mode",
            "controller_mode",
            "waypoint_index",
            "safety_reason",
            "tracking_error_world_m",
            "dropped_command",
        ]

    def to_csv_row(self) -> dict[str, Any]:
        return {
            "timestamp_s": float(self.timestamp_s),
            "sim_time_s": float(self.sim_time_s),
            "joint_names": _pipe_join(self.joint_names),
            "desired_positions": _pipe_join(self.desired_positions),
            "effective_positions": _pipe_join(self.effective_positions),
            "estimator_mode": self.estimator_mode,
            "controller_mode": self.controller_mode,
            "waypoint_index": int(self.waypoint_index),
            "safety_reason": self.safety_reason,
            "tracking_error_world_m": _pipe_join(self.tracking_error_world_m),
            "dropped_command": bool(self.dropped_command),
        }


@dataclass(slots=True)
class IsaacControllerDiagnosticPacket:
    timestamp_s: float
    sim_time_s: float
    estimator_mode: str
    controller_mode: str
    waypoint_index: int
    current_position_source: str
    tracking_error_norm_m: float
    command_delta_norm_m: float
    desired_position_world_m: tuple[float, float, float]
    safety_reason: str
    position_radius_95_m: float
    innovation_norm: float
    ik_success: bool
    ik_retry_alpha: float
    joint_target_delta_norm: float
    min_joint_limit_margin: float | None = None
    dropped_command: bool = False
    degraded_mode_active: bool = False
    orientation_policy: str = "fixed"

    @classmethod
    def csv_fieldnames(cls) -> list[str]:
        return [
            "timestamp_s",
            "sim_time_s",
            "estimator_mode",
            "controller_mode",
            "waypoint_index",
            "current_position_source",
            "tracking_error_norm_m",
            "command_delta_norm_m",
            "desired_position_world_m",
            "safety_reason",
            "position_radius_95_m",
            "innovation_norm",
            "ik_success",
            "ik_retry_alpha",
            "joint_target_delta_norm",
            "min_joint_limit_margin",
            "dropped_command",
            "degraded_mode_active",
            "orientation_policy",
        ]

    def to_csv_row(self) -> dict[str, Any]:
        return {
            "timestamp_s": float(self.timestamp_s),
            "sim_time_s": float(self.sim_time_s),
            "estimator_mode": self.estimator_mode,
            "controller_mode": self.controller_mode,
            "waypoint_index": int(self.waypoint_index),
            "current_position_source": self.current_position_source,
            "tracking_error_norm_m": float(self.tracking_error_norm_m),
            "command_delta_norm_m": float(self.command_delta_norm_m),
            "desired_position_world_m": _pipe_join(self.desired_position_world_m),
            "safety_reason": self.safety_reason,
            "position_radius_95_m": float(self.position_radius_95_m),
            "innovation_norm": float(self.innovation_norm),
            "ik_success": bool(self.ik_success),
            "ik_retry_alpha": float(self.ik_retry_alpha),
            "joint_target_delta_norm": float(self.joint_target_delta_norm),
            "min_joint_limit_margin": None if self.min_joint_limit_margin is None else float(self.min_joint_limit_margin),
            "dropped_command": bool(self.dropped_command),
            "degraded_mode_active": bool(self.degraded_mode_active),
            "orientation_policy": self.orientation_policy,
        }


@dataclass(slots=True)
class IsaacRealizedJointPacket:
    timestamp_s: float
    sim_time_s: float
    joint_names: tuple[str, ...]
    positions: tuple[float, ...]
    velocities: tuple[float, ...]
    end_effector_position_world_m: tuple[float, float, float] | None = None
    end_effector_orientation_wxyz: tuple[float, float, float, float] | None = None
    servo_internal_state: dict[str, Any] | None = None

    @classmethod
    def csv_fieldnames(cls) -> list[str]:
        return [
            "timestamp_s",
            "sim_time_s",
            "joint_names",
            "positions",
            "velocities",
            "end_effector_position_world_m",
            "end_effector_orientation_wxyz",
            "servo_internal_state",
        ]

    def to_csv_row(self) -> dict[str, Any]:
        return {
            "timestamp_s": float(self.timestamp_s),
            "sim_time_s": float(self.sim_time_s),
            "joint_names": _pipe_join(self.joint_names),
            "positions": _pipe_join(self.positions),
            "velocities": _pipe_join(self.velocities),
            "end_effector_position_world_m": ""
            if self.end_effector_position_world_m is None
            else _pipe_join(self.end_effector_position_world_m),
            "end_effector_orientation_wxyz": ""
            if self.end_effector_orientation_wxyz is None
            else _pipe_join(self.end_effector_orientation_wxyz),
            "servo_internal_state": ""
            if self.servo_internal_state is None
            else json.dumps(dict(self.servo_internal_state), sort_keys=True),
        }


__all__ = [
    "IsaacCameraFramePacket",
    "IsaacControllerDiagnosticPacket",
    "IsaacImuPacket",
    "IsaacJointCommandPacket",
    "IsaacRealizedJointPacket",
    "IsaacTagDetectionPacket",
]
