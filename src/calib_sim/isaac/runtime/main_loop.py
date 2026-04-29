"""Standalone Isaac runtime orchestration for the first live pass."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

from calib_sim.estimation.noise_models import VisionNoisePreset, load_vision_noise_preset
from calib_sim.isaac.actuation.servo_model import ServoCorruptionConfig, UncertainServoModel
from calib_sim.isaac.clocks import FixedRateClock, TimestampTriplet
from calib_sim.isaac.control.path_tracker import PathTracker
from calib_sim.isaac.control.safety_gates import SafetyGates
from calib_sim.isaac.control.waypoint_manager import Waypoint, WaypointManager
from calib_sim.isaac.estimation.fixed_lag_smoother import FixedLagSmoother
from calib_sim.isaac.estimation.online_filter import AnchoredOnlineFilter
from calib_sim.isaac.estimation.uncertainty import sanitize_covariance
from calib_sim.isaac.estimation.windowed_anchor_ba import WindowedAnchorBASmoother
from calib_sim.isaac.frontend.apriltag_frontend import IsaacAprilTagFrontend
from calib_sim.isaac.logging.schemas import (
    IsaacCameraFramePacket,
    IsaacControllerDiagnosticPacket,
    IsaacJointCommandPacket,
    IsaacRealizedJointPacket,
    IsaacTagDetectionPacket,
)
from calib_sim.isaac.logging.writer import IsaacRunWriter
from calib_sim.isaac.robot_builder import LiveRobotBinding, build_robot_binding
from calib_sim.isaac.sensors import IsaacCameraBinding, IsaacImuBinding, camera_spec_from_config, imu_spec_from_config
from calib_sim.isaac.standard_streaming import (
    build_webrtc_connection_info,
    default_streaming_status,
)
from calib_sim.isaac.stage_builder import StageBuildArtifacts, IsaacStageSpec, build_anchor_room_geometry, stage_spec_from_config
from calib_sim.isaac.tag_builder import TagPoseSpec
from calib_sim.sim.runtime import RuntimeConfig, SimulationRuntime


def _normalize(vector: np.ndarray, *, default: np.ndarray) -> np.ndarray:
    candidate = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(candidate))
    if norm < 1e-12:
        return np.asarray(default, dtype=np.float64).reshape(candidate.shape)
    return candidate / norm


def _rotation_matrix_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        return np.array(
            [
                0.25 * s,
                (matrix[2, 1] - matrix[1, 2]) / s,
                (matrix[0, 2] - matrix[2, 0]) / s,
                (matrix[1, 0] - matrix[0, 1]) / s,
            ],
            dtype=np.float64,
        )
    diagonal = np.diag(matrix)
    if diagonal[0] > diagonal[1] and diagonal[0] > diagonal[2]:
        s = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
        return np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / s,
                0.25 * s,
                (matrix[0, 1] + matrix[1, 0]) / s,
                (matrix[0, 2] + matrix[2, 0]) / s,
            ],
            dtype=np.float64,
        )
    if diagonal[1] > diagonal[2]:
        s = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
        return np.array(
            [
                (matrix[0, 2] - matrix[2, 0]) / s,
                (matrix[0, 1] + matrix[1, 0]) / s,
                0.25 * s,
                (matrix[1, 2] + matrix[2, 1]) / s,
            ],
            dtype=np.float64,
        )
    s = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
    return np.array(
        [
            (matrix[1, 0] - matrix[0, 1]) / s,
            (matrix[0, 2] + matrix[2, 0]) / s,
            (matrix[1, 2] + matrix[2, 1]) / s,
            0.25 * s,
        ],
        dtype=np.float64,
    )


def _quaternion_wxyz_to_rotation_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = quaternion / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _quaternion_angle_deg(lhs_wxyz: np.ndarray, rhs_wxyz: np.ndarray) -> float:
    lhs = _normalize(np.asarray(lhs_wxyz, dtype=np.float64).reshape(4), default=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
    rhs = _normalize(np.asarray(rhs_wxyz, dtype=np.float64).reshape(4), default=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
    alignment = float(np.clip(abs(np.dot(lhs, rhs)), -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(alignment)))


def _slerp_quaternion_wxyz(start_wxyz: np.ndarray, end_wxyz: np.ndarray, alpha: float) -> np.ndarray:
    start = _normalize(np.asarray(start_wxyz, dtype=np.float64).reshape(4), default=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
    end = _normalize(np.asarray(end_wxyz, dtype=np.float64).reshape(4), default=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
    dot = float(np.dot(start, end))
    if dot < 0.0:
        end = -end
        dot = -dot
    clipped_alpha = float(np.clip(alpha, 0.0, 1.0))
    if dot > 0.9995:
        return _normalize(start + clipped_alpha * (end - start), default=start)
    theta_0 = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    sin_theta_0 = float(np.sin(theta_0))
    if sin_theta_0 < 1e-12:
        return end.copy()
    theta = clipped_alpha * theta_0
    sin_theta = float(np.sin(theta))
    scale_start = float(np.sin(theta_0 - theta) / sin_theta_0)
    scale_end = float(sin_theta / sin_theta_0)
    return _normalize(scale_start * start + scale_end * end, default=end)


def _limit_quaternion_step(
    current_wxyz: np.ndarray,
    target_wxyz: np.ndarray,
    *,
    max_step_deg: float | None,
) -> np.ndarray:
    if max_step_deg in (None, ""):
        return _normalize(np.asarray(target_wxyz, dtype=np.float64).reshape(4), default=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
    max_step_deg = float(max_step_deg)
    if max_step_deg <= 0.0:
        return _normalize(np.asarray(current_wxyz, dtype=np.float64).reshape(4), default=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
    angle_deg = _quaternion_angle_deg(current_wxyz, target_wxyz)
    if angle_deg <= max_step_deg:
        return _normalize(np.asarray(target_wxyz, dtype=np.float64).reshape(4), default=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
    return _slerp_quaternion_wxyz(current_wxyz, target_wxyz, max_step_deg / max(angle_deg, 1e-9))


def _look_at_basis(
    eye_world_m: np.ndarray,
    target_world_m: np.ndarray,
    *,
    up_hint_world_m: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eye = np.asarray(eye_world_m, dtype=np.float64).reshape(3)
    target = np.asarray(target_world_m, dtype=np.float64).reshape(3)
    forward = _normalize(target - eye, default=np.array([0.0, 0.0, -1.0], dtype=np.float64))
    up_hint = np.asarray([0.0, 0.0, 1.0] if up_hint_world_m is None else up_hint_world_m, dtype=np.float64).reshape(3)
    up = _normalize(up_hint - np.dot(up_hint, forward) * forward, default=np.array([0.0, 1.0, 0.0], dtype=np.float64))
    left = _normalize(np.cross(up, forward), default=np.array([0.0, 1.0, 0.0], dtype=np.float64))
    up = _normalize(np.cross(forward, left), default=np.array([0.0, 0.0, 1.0], dtype=np.float64))
    return forward, left, up


def _look_at_orientation_wxyz(
    eye_world_m: np.ndarray,
    target_world_m: np.ndarray,
    *,
    up_hint_world_m: np.ndarray | None = None,
) -> np.ndarray:
    eye = np.asarray(eye_world_m, dtype=np.float64).reshape(3)
    target = np.asarray(target_world_m, dtype=np.float64).reshape(3)
    forward = _normalize(target - eye, default=np.array([0.0, 0.0, -1.0], dtype=np.float64))
    up_hint = np.asarray([0.0, 0.0, 1.0] if up_hint_world_m is None else up_hint_world_m, dtype=np.float64).reshape(3)
    if abs(float(np.dot(forward, _normalize(up_hint, default=np.array([0.0, 0.0, 1.0], dtype=np.float64))))) > 0.97:
        up_hint = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    right = _normalize(np.cross(forward, up_hint), default=np.array([1.0, 0.0, 0.0], dtype=np.float64))
    up = _normalize(np.cross(right, forward), default=np.array([0.0, 0.0, 1.0], dtype=np.float64))
    # Isaac/USD camera prims look down their local -Z axis with +Y up.
    return _rotation_matrix_to_quaternion_wxyz(np.column_stack((right, up, -forward)))


def _cubic_bezier_point(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    t: float,
) -> np.ndarray:
    t = float(np.clip(t, 0.0, 1.0))
    one_minus_t = 1.0 - t
    return (
        (one_minus_t ** 3) * np.asarray(p0, dtype=np.float64).reshape(3)
        + 3.0 * (one_minus_t ** 2) * t * np.asarray(p1, dtype=np.float64).reshape(3)
        + 3.0 * one_minus_t * (t ** 2) * np.asarray(p2, dtype=np.float64).reshape(3)
        + (t ** 3) * np.asarray(p3, dtype=np.float64).reshape(3)
    )


@dataclass(slots=True)
class IsaacRunSummary:
    run_id: str
    run_dir: str
    duration_s: float
    step_count: int
    counts: dict[str, int]
    warnings: list[str] = field(default_factory=list)
    complete: bool = False
    latest_any_promoted: bool = False
    latest_complete_promoted: bool = False

    def as_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "duration_s": float(self.duration_s),
            "step_count": int(self.step_count),
            "counts": {str(key): int(value) for key, value in self.counts.items()},
            "warnings": list(self.warnings),
            "complete": bool(self.complete),
            "latest_any_promoted": bool(self.latest_any_promoted),
            "latest_complete_promoted": bool(self.latest_complete_promoted),
        }


@dataclass(slots=True)
class IsaacRuntimeConfig:
    stage_path: str
    robot_preset: str
    anchor_tag_id: int
    run_id: str = "adhoc"
    run_dir: str = "output/isaac_runs/adhoc"
    headless: bool = True
    width: int = 1280
    height: int = 720
    streaming_backend: str = "webrtc"
    webrtc_signal_port: int = 49110
    webrtc_stream_port: int = 47990
    webrtc_target_fps: int = 60
    webrtc_allow_dynamic_resize: bool = True
    webrtc_public_ip: str = ""
    limit_cpu_threads: int | None = None
    disable_viewport_updates: bool | None = None
    seed: int = 7
    duration_s: float = 10.0
    max_steps: int | None = None
    estimator_mode: str = "fused"
    controller_mode: str = "closed-loop"
    bootstrap_control_policy: str = "hold_until_first_detection"
    mode: str | None = None
    promote_latest_complete: bool = False
    allow_gt_debug_control: bool = False
    config_paths: dict[str, str] = field(default_factory=dict)
    config_payloads: dict[str, dict[str, Any]] = field(default_factory=dict)

    def runtime_config(self) -> RuntimeConfig:
        streaming_backend = str(self.streaming_backend).strip().lower()
        disable_viewport_updates = (
            bool(self.headless and streaming_backend != "webrtc")
            if self.disable_viewport_updates is None
            else bool(self.disable_viewport_updates)
        )
        limit_cpu_threads = 32 if self.limit_cpu_threads is None else int(self.limit_cpu_threads)
        return RuntimeConfig(
            headless=self.headless,
            width=self.width,
            height=self.height,
            streaming_backend=self.streaming_backend,
            webrtc_signal_port=self.webrtc_signal_port,
            webrtc_stream_port=self.webrtc_stream_port,
            webrtc_target_fps=self.webrtc_target_fps,
            webrtc_allow_dynamic_resize=self.webrtc_allow_dynamic_resize,
            webrtc_public_ip=self.webrtc_public_ip,
            limit_cpu_threads=limit_cpu_threads,
            disable_viewport_updates=disable_viewport_updates,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "stage_path": self.stage_path,
            "robot_preset": self.robot_preset,
            "anchor_tag_id": int(self.anchor_tag_id),
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "headless": bool(self.headless),
            "width": int(self.width),
            "height": int(self.height),
            "streaming_backend": str(self.streaming_backend),
            "webrtc_signal_port": int(self.webrtc_signal_port),
            "webrtc_stream_port": int(self.webrtc_stream_port),
            "webrtc_target_fps": int(self.webrtc_target_fps),
            "webrtc_allow_dynamic_resize": bool(self.webrtc_allow_dynamic_resize),
            "webrtc_public_ip": str(self.webrtc_public_ip),
            "limit_cpu_threads": None if self.limit_cpu_threads is None else int(self.limit_cpu_threads),
            "disable_viewport_updates": None
            if self.disable_viewport_updates is None
            else bool(self.disable_viewport_updates),
            "seed": int(self.seed),
            "duration_s": float(self.duration_s),
            "max_steps": None if self.max_steps is None else int(self.max_steps),
            "estimator_mode": self.estimator_mode,
            "controller_mode": self.controller_mode,
            "bootstrap_control_policy": self.bootstrap_control_policy,
            "mode": self.mode,
            "promote_latest_complete": bool(self.promote_latest_complete),
            "allow_gt_debug_control": bool(self.allow_gt_debug_control),
            "config_paths": dict(self.config_paths),
        }


class IsaacStandaloneRuntime:
    """Real first-pass runtime orchestrator over Isaac Sim."""

    def __init__(self, config: IsaacRuntimeConfig) -> None:
        self.config = config
        self._runtime = SimulationRuntime(config.runtime_config())
        self._writer = IsaacRunWriter(config.run_dir)
        self._rng = np.random.default_rng(int(config.seed))
        self._warnings: list[str] = []
        self._counts = {
            "camera_frames": 0,
            "detections": 0,
            "imu_packets": 0,
            "commands": 0,
            "controller_diagnostics": 0,
            "realized_joints": 0,
            "filter_states": 0,
            "smoother_states": 0,
            "uncertainty_states": 0,
        }
        self._timeline = None
        self._world = None
        self._stage_artifacts: StageBuildArtifacts | None = None
        self._robot_binding: LiveRobotBinding | None = None
        self._camera_binding: IsaacCameraBinding | None = None
        self._observer_camera_bindings: list[IsaacCameraBinding] = []
        self._imu_binding: IsaacImuBinding | None = None
        self._frontend: IsaacAprilTagFrontend | None = None
        self._filter: AnchoredOnlineFilter | None = None
        self._smoother: FixedLagSmoother | WindowedAnchorBASmoother | None = None
        self._tracker: PathTracker | None = None
        self._filter_clock: FixedRateClock | None = None
        self._smoother_clock: FixedRateClock | None = None
        self._controller_clock: FixedRateClock | None = None
        self._servo_models: list[UncertainServoModel] = []
        self._tag_pose_map: dict[int, TagPoseSpec] = {}
        self._latest_filter_snapshot = None
        self._latest_uncertainty = None
        self._latest_auxiliary_update_summary = None
        self._last_anchor_visible = False
        self._last_anchor_visible_raw = False
        self._last_anchor_update_suppressed = False
        self._anchor_lock_acquired = False
        self._last_detections = ()
        self._physics_dt_s = 1.0 / float(self.config.config_payloads["scene"]["physics_rate_hz"])
        self._sim_time_s = 0.0
        self.started = False
        self.steps_executed = 0
        self._control_target_orientation_wxyz: np.ndarray | None = None
        self._last_open_loop_target_position: np.ndarray | None = None
        self._stage_from_file_loaded = False
        self._previous_joint_positions: np.ndarray | None = None
        self._ik_failure_count = 0
        self._controller_consecutive_ik_failures = 0
        self._finalized_summary: IsaacRunSummary | None = None
        self._tag_feedback_diagnostics: dict[int, dict[str, Any]] = {}
        self._last_camera_intrinsics_snapshot: dict[str, Any] | None = None
        self._smoother_interval_imu_packets: list[Any] = []
        self._camera_interval_imu_packets: list[Any] = []
        self._last_camera_timestamp_s: float | None = None
        self._disable_imu_prediction_while_anchor_suppressed = False
        self._suppression_imu_specific_force_gate_mps2: float | None = None
        self._suppression_propagation_mode = "full_imu"
        self._last_imu_packets_used_for_prediction = 0
        self._last_imu_packets_rejected_for_prediction = 0
        self._latest_camera_frame_packet: IsaacCameraFramePacket | None = None
        self._latest_camera_image_rgb: np.ndarray | None = None
        self._latest_display_camera_frame_packet: IsaacCameraFramePacket | None = None
        self._latest_display_camera_image_rgb: np.ndarray | None = None
        self._latest_observer_views: dict[str, dict[str, Any]] = {}
        self._latest_imu_packet = None
        self._latest_imu_truth = {
            "accel_mps2": np.zeros(3, dtype=np.float64),
            "gyro_rps": np.zeros(3, dtype=np.float64),
        }
        self._latest_realized_joint_packet: IsaacRealizedJointPacket | None = None
        self._latest_controller_diagnostic: IsaacControllerDiagnosticPacket | None = None
        self._latest_command_packet: IsaacJointCommandPacket | None = None
        self._latest_anchor_update_result: dict[str, Any] | None = None
        self._last_detected_frame_by_tag: dict[int, int] = {}
        self._interactive_control_mode: str | None = None
        self._manual_joint_target_positions: np.ndarray | None = None
        self._manual_task_target_world_m: np.ndarray | None = None
        self._home_joint_positions: np.ndarray | None = None
        self._camera_rotation_ec: np.ndarray | None = None
        self._camera_translation_ec_m: np.ndarray | None = None
        self._scene_motion_program_cache: dict[str, Any] | None = None
        self._auto_demo_visibility_miss_count = 0
        self._auto_demo_curve_ready = False
        self._auto_demo_curve_anchor_time_s = 0.0
        self._auto_demo_curve_phase = 0.0
        self._auto_demo_last_curve_tracking_good = False
        self._auto_demo_camera_orientation_correction: np.ndarray | None = None
        self._auto_demo_commanded_camera_position_world_m: np.ndarray | None = None
        self._auto_demo_commanded_camera_orientation_wxyz: np.ndarray | None = None
        self._live_vision_noise_mode = "clean"
        scene_payload = self.config.config_payloads.get("scene", {})
        tabletop_nominal_preset = scene_payload.get("live_vision_nominal_preset", "vision_nominal")
        self._vision_nominal_preset: VisionNoisePreset = load_vision_noise_preset(tabletop_nominal_preset)
        self._recording_count_baseline = {key: 0 for key in self._counts}
        self._recording_time_baseline_s = 0.0
        self._recording_step_baseline = 0
        self._standard_streaming_status = default_streaming_status(self.config.streaming_backend)
        self._tag_detection_mode = "new_pupil"

    @property
    def writer(self) -> IsaacRunWriter:
        return self._writer

    def set_writer(self, writer: Any, *, reset_recording_baseline: bool = False) -> None:
        self._writer = writer
        if reset_recording_baseline:
            self._recording_count_baseline = dict(self._counts)
            self._recording_time_baseline_s = float(self._sim_time_s)
            self._recording_step_baseline = int(self.steps_executed)

    def set_interactive_control_mode(self, mode: str | None) -> None:
        normalized = None if mode in (None, "") else str(mode).strip().lower()
        if normalized not in (None, "manual_joint", "manual_task", "auto_demo", "auto_path"):
            raise ValueError(f"Unsupported interactive control mode: {mode}")
        if normalized == "auto_demo" and self._interactive_control_mode != "auto_demo":
            self._scene_motion_program_cache = None
            self._auto_demo_curve_ready = False
            self._auto_demo_curve_anchor_time_s = float(self._sim_time_s)
            self._auto_demo_curve_phase = 0.0
            self._auto_demo_last_curve_tracking_good = False
            self._auto_demo_camera_orientation_correction = None
            self._auto_demo_commanded_camera_position_world_m = None
            self._auto_demo_commanded_camera_orientation_wxyz = None
        if normalized != "auto_demo":
            self._auto_demo_visibility_miss_count = 0
            self._auto_demo_curve_ready = False
            self._auto_demo_curve_phase = 0.0
            self._auto_demo_last_curve_tracking_good = False
            self._auto_demo_camera_orientation_correction = None
            self._auto_demo_commanded_camera_position_world_m = None
            self._auto_demo_commanded_camera_orientation_wxyz = None
        self._interactive_control_mode = normalized
        if normalized == "manual_joint" and self._robot_binding is not None:
            self._manual_joint_target_positions = self._robot_binding.get_joint_positions().copy()
        if normalized == "manual_task" and self._manual_task_target_world_m is None and self._robot_binding is not None:
            ee_position_world_m, _ = self._robot_binding.get_end_effector_pose()
            self._manual_task_target_world_m = self._current_control_position_world_m(
                ee_position_world_m=np.asarray(ee_position_world_m, dtype=np.float64).reshape(3)
            )

    def set_manual_joint_targets_rad(self, joint_targets_rad: np.ndarray) -> None:
        if self._robot_binding is None:
            return
        current = self._robot_binding.get_joint_positions()
        candidate = current.copy()
        provided = np.asarray(joint_targets_rad, dtype=np.float64).reshape(-1)
        arm_joint_count = max(min(len(self._robot_binding.robot.joint_names) - 2, len(provided)), 0)
        if arm_joint_count > 0:
            candidate[:arm_joint_count] = provided[:arm_joint_count]
        self.set_interactive_control_mode("manual_joint")
        self._manual_joint_target_positions = candidate

    def set_manual_joint_targets_deg(self, joint_targets_deg: np.ndarray) -> None:
        self.set_manual_joint_targets_rad(np.radians(np.asarray(joint_targets_deg, dtype=np.float64)))

    def set_manual_task_target_world_m(self, position_world_m: np.ndarray) -> None:
        if self._interactive_control_mode == "manual_joint":
            return
        self._manual_task_target_world_m = self._clamp_to_workspace(position_world_m)
        self._interactive_control_mode = "manual_task"

    def nudge_manual_task_target(self, delta_world_m: np.ndarray) -> None:
        if self._interactive_control_mode == "manual_joint":
            return
        delta = np.asarray(delta_world_m, dtype=np.float64).reshape(3)
        if self._manual_task_target_world_m is None:
            if self._robot_binding is None:
                return
            ee_position_world_m, _ = self._robot_binding.get_end_effector_pose()
            self._manual_task_target_world_m = self._current_control_position_world_m(
                ee_position_world_m=np.asarray(ee_position_world_m, dtype=np.float64).reshape(3)
            )
        self._manual_task_target_world_m = self._clamp_to_workspace(self._manual_task_target_world_m + delta)
        self._interactive_control_mode = "manual_task"

    def set_imu_noise_preset(self, preset_name: str) -> None:
        if self._imu_binding is None:
            self.config.config_payloads["imu"]["noise_preset"] = str(preset_name)
            return
        self._imu_binding.set_noise_preset(str(preset_name))
        self.config.config_payloads["imu"]["noise_preset"] = str(self._imu_binding.spec.noise_preset)

    def set_vision_noise_mode(self, mode: str) -> None:
        normalized = str(mode).strip().lower()
        if normalized not in {"clean", "nominal"}:
            raise ValueError(f"Unsupported vision noise mode: {mode}")
        self._live_vision_noise_mode = normalized

    def _scene_frontend_config(self) -> dict[str, Any]:
        scene_payload = self.config.config_payloads.setdefault("scene", {})
        frontend_config = scene_payload.get("frontend", {})
        if not isinstance(frontend_config, dict):
            frontend_config = {}
            scene_payload["frontend"] = frontend_config
        return frontend_config

    def _build_frontend_from_scene_config(self, frontend_config: dict[str, Any]) -> IsaacAprilTagFrontend:
        return IsaacAprilTagFrontend(
            anchor_tag_id=int(self.config.anchor_tag_id),
            detector_backend=str(frontend_config.get("detector_backend", "pupil_apriltags")),
            direct_detection_retry_scale=float(frontend_config.get("direct_detection_retry_scale", 1.75)),
            direct_detection_retry_scales=tuple(
                float(value) for value in frontend_config.get("direct_detection_retry_scales", [])
            ),
            pupil_nthreads=int(frontend_config.get("pupil_nthreads", 8)),
            pupil_quad_decimate=float(frontend_config.get("pupil_quad_decimate", 1.0)),
            pupil_quad_sigma=float(frontend_config.get("pupil_quad_sigma", 0.0)),
            pupil_refine_edges=bool(frontend_config.get("pupil_refine_edges", True)),
            pupil_decode_sharpening=float(frontend_config.get("pupil_decode_sharpening", 0.25)),
            solve_tag_pose=bool(frontend_config.get("solve_tag_pose", True)),
            roi_recovery_enabled=bool(frontend_config.get("roi_recovery_enabled", False)),
            roi_recovery_max_gap_frames=int(frontend_config.get("roi_recovery_max_gap_frames", 1)),
            roi_padding_px=int(frontend_config.get("roi_padding_px", 48)),
            roi_prediction_flow_win_size=int(frontend_config.get("roi_prediction_flow_win_size", 27)),
            roi_accept_max_corner_error_px=float(frontend_config.get("roi_accept_max_corner_error_px", 10.0)),
            roi_accept_max_edge_ratio=float(frontend_config.get("roi_accept_max_edge_ratio", 3.0)),
            roi_accept_max_area_delta_ratio=float(frontend_config.get("roi_accept_max_area_delta_ratio", 0.40)),
            temporal_tracking_enabled=bool(frontend_config.get("temporal_tracking_enabled", True)),
            temporal_max_gap_frames=int(frontend_config.get("temporal_max_gap_frames", 3)),
            temporal_flow_max_error_px=float(frontend_config.get("temporal_flow_max_error_px", 5.0)),
            anchor_temporal_clean_max_gap_frames=int(frontend_config.get("anchor_temporal_clean_max_gap_frames", 1)),
            anchor_temporal_flow_max_error_px=float(frontend_config.get("anchor_temporal_flow_max_error_px", 3.0)),
            anchor_temporal_max_edge_ratio=float(frontend_config.get("anchor_temporal_max_edge_ratio", 2.4)),
            anchor_temporal_max_area_delta_ratio=float(frontend_config.get("anchor_temporal_max_area_delta_ratio", 0.28)),
        )

    def _tag_detection_enabled(self) -> bool:
        return bool(self._scene_frontend_config().get("detection_enabled", True))

    def _tag_pose_estimation_enabled(self) -> bool:
        return bool(self._scene_frontend_config().get("solve_tag_pose", True))

    def _using_external_cuda_detector(self) -> bool:
        return False

    def set_tag_detection_mode(self, mode: str) -> None:
        normalized = str(mode).strip().lower()
        if normalized not in {"new_pupil", "off"}:
            raise ValueError(f"Unsupported tag detection mode: {mode}")
        self._tag_detection_mode = normalized
        frontend_config = self._scene_frontend_config()
        if normalized == "off":
            frontend_config["detector_backend"] = "pupil_apriltags"
            frontend_config["roi_recovery_enabled"] = True
            frontend_config["detection_enabled"] = False
        else:
            frontend_config["detector_backend"] = "pupil_apriltags"
            frontend_config["roi_recovery_enabled"] = True
            frontend_config["detection_enabled"] = True
            frontend_config.setdefault("pupil_nthreads", 8)
            frontend_config.setdefault("pupil_quad_decimate", 1.0)
            frontend_config.setdefault("pupil_quad_sigma", 0.0)
            frontend_config.setdefault("pupil_refine_edges", True)
            frontend_config.setdefault("pupil_decode_sharpening", 0.25)
            frontend_config.setdefault("roi_recovery_max_gap_frames", 1)
            frontend_config.setdefault("roi_padding_px", 48)
            frontend_config.setdefault("roi_prediction_flow_win_size", 27)
            frontend_config.setdefault("roi_accept_max_corner_error_px", 10.0)
            frontend_config.setdefault("roi_accept_max_edge_ratio", 3.0)
            frontend_config.setdefault("roi_accept_max_area_delta_ratio", 0.40)
        self._frontend = self._build_frontend_from_scene_config(frontend_config)

    def _auxiliary_feedback_diagnostics_for_filter(self) -> dict[int, dict[str, Any]]:
        smoother_backend = str(self.config.config_payloads["estimation"].get("smoother", {}).get("backend", "lightweight")).strip().lower()
        if smoother_backend == "lightweight":
            return {}
        return dict(self._tag_feedback_diagnostics)

    def _build_servo_config(self, actuation_config: dict[str, Any]) -> ServoCorruptionConfig:
        return ServoCorruptionConfig(
            delay_s=float(actuation_config.get("delay_s", 0.0)),
            gain_error=float(actuation_config.get("gain_error", 1.0)),
            bias=float(actuation_config.get("bias", 0.0)),
            lag_time_constant_s=float(actuation_config.get("lag_time_constant_s", 0.05)),
            rate_limit_per_s=float(actuation_config.get("rate_limit_per_s", 4.0)),
            deadband=float(actuation_config.get("deadband", 0.0)),
            backlash=float(actuation_config.get("backlash", 0.0)),
            process_noise_std=float(actuation_config.get("process_noise_std", 0.0)),
            drop_probability=float(actuation_config.get("drop_probability", 0.0)),
            jitter_std_s=float(actuation_config.get("jitter_std_s", 0.0)),
        )

    def _rebuild_servo_models(self, actuation_config: dict[str, Any]) -> None:
        if self._robot_binding is None:
            return
        joint_positions = self._robot_binding.get_joint_positions()
        servo_config = self._build_servo_config(actuation_config)
        self._servo_models = [
            UncertainServoModel(
                servo_config,
                initial_position=float(joint_positions[index]) if index < len(joint_positions) else 0.0,
                rng=np.random.default_rng(int(self.config.seed) + index + 1),
            )
            for index in range(len(self._robot_binding.robot.joint_names))
        ]

    def set_actuation_config_payload(self, actuation_config: dict[str, Any]) -> None:
        self.config.config_payloads["actuation"] = copy.deepcopy(actuation_config)
        self._rebuild_servo_models(self.config.config_payloads["actuation"])

    def home(self) -> None:
        if self._robot_binding is None:
            return
        if self._interactive_control_mode == "manual_joint":
            return
        if self._home_joint_positions is None:
            self._home_joint_positions = self._robot_binding.get_joint_positions().copy()
        self._robot_binding.articulation.set_joint_positions(self._home_joint_positions.copy())
        for index, servo_model in enumerate(self._servo_models):
            if index < len(self._home_joint_positions):
                servo_model.position = float(self._home_joint_positions[index])
                servo_model.velocity = 0.0
        self._manual_joint_target_positions = self._home_joint_positions.copy()
        ee_position_world_m, _ = self._robot_binding.get_end_effector_pose()
        self._manual_task_target_world_m = self._current_control_position_world_m(
            ee_position_world_m=np.asarray(ee_position_world_m, dtype=np.float64).reshape(3)
        )

    def boot_kit(self) -> None:
        self._runtime.start()
        import omni.timeline

        self._timeline = omni.timeline.get_timeline_interface()

    def open_or_build_stage(self) -> None:
        stage_path = Path(self.config.stage_path)
        if stage_path.exists():
            import omni.usd

            omni.usd.get_context().open_stage(str(stage_path.resolve()))
            for _ in range(10):
                self._runtime.step()
            self._stage_from_file_loaded = True
        else:
            self._warnings.append(
                f"Configured stage_path={self.config.stage_path!r} was not found; falling back to programmatic build."
            )
            self._stage_from_file_loaded = False

    def bind_world(self) -> None:
        from isaacsim.core.api import World

        scene_config = self.config.config_payloads["scene"]
        camera_rate_hz = float(scene_config.get("camera_rate_hz", self.config.config_payloads["camera"]["rate_hz"]))
        self._world = World(
            stage_units_in_meters=1.0,
            physics_dt=self._physics_dt_s,
            rendering_dt=1.0 / max(camera_rate_hz, 1.0),
        )
        self._world.scene.add_default_ground_plane()
        stage_spec: IsaacStageSpec = stage_spec_from_config(scene_config)
        if self._stage_from_file_loaded:
            self._stage_artifacts = StageBuildArtifacts(
                stage_spec=stage_spec,
                tag_pose_specs={int(tag.tag_id): tag for tag in stage_spec.tag_pose_specs},
                warnings=[],
            )
        else:
            self._stage_artifacts = build_anchor_room_geometry(
                scene_config=scene_config,
                generated_asset_dir=Path(self.config.run_dir) / "generated_assets" / "tags",
            )
        self._tag_pose_map = dict(self._stage_artifacts.tag_pose_specs)
        self._warnings.extend(self._stage_artifacts.warnings)

    def bind_robot(self) -> None:
        if self._world is None:
            raise RuntimeError("World must be bound before the robot.")
        self._robot_binding = build_robot_binding(world=self._world, robot_config=self.config.config_payloads["robot"])

    def bind_sensors(self) -> None:
        if self._robot_binding is None or self._stage_artifacts is None:
            raise RuntimeError("Robot must be bound before sensors.")
        camera_spec = camera_spec_from_config(
            self.config.config_payloads["camera"],
            default_prim_path=self._robot_binding.robot.camera_mount_prim_path,
        )
        imu_spec = imu_spec_from_config(
            self.config.config_payloads["imu"],
            default_prim_path=self._robot_binding.robot.imu_mount_prim_path,
            parent_prim_path=f"{self._robot_binding.robot.articulation_prim_path}/{self._robot_binding.robot.ee_frame}",
        )
        self._camera_binding = IsaacCameraBinding.create(camera_spec)
        self._imu_binding = IsaacImuBinding.create(imu_spec)
        self._observer_camera_bindings = []
        for observer_spec in self._stage_artifacts.stage_spec.observer_cameras:
            horizontal_aperture_mm = 20.955
            focal_length_mm = 0.5 * horizontal_aperture_mm / np.tan(np.radians(float(observer_spec.fov_deg)) / 2.0)
            observer_rate_hz = (
                float(observer_spec.rate_hz)
                if observer_spec.rate_hz not in (None, "")
                else float(self.config.config_payloads["scene"].get("camera_rate_hz", camera_spec.rate_hz))
            )
            self._observer_camera_bindings.append(
                IsaacCameraBinding.create(
                    camera_spec_from_config(
                        {
                            "name": str(observer_spec.name),
                            "prim_path": str(observer_spec.prim_path),
                            "width_px": int(observer_spec.width_px),
                            "height_px": int(observer_spec.height_px),
                            "rate_hz": float(observer_rate_hz),
                            "frame_id": f"observer:{observer_spec.name}",
                            "local_translation_m": [0.0, 0.0, 0.0],
                            "local_orientation_rpy_deg": [0.0, 0.0, 0.0],
                            "focal_length_mm": float(focal_length_mm),
                            "horizontal_aperture_mm": float(horizontal_aperture_mm),
                            "vertical_aperture_mm": float(horizontal_aperture_mm) * float(observer_spec.height_px) / max(float(observer_spec.width_px), 1.0),
                            "preserve_world_pose_on_initialize": True,
                        },
                        default_prim_path=str(observer_spec.prim_path),
                    )
                )
            )

    def bind_frontend_estimation_control(self) -> None:
        if self._robot_binding is None:
            raise RuntimeError("Robot must be bound before frontend/estimation/control.")
        estimation_config = self.config.config_payloads["estimation"]
        control_config = self.config.config_payloads["control"]
        actuation_config = self.config.config_payloads["actuation"]
        frontend_config = self._scene_frontend_config()
        self._frontend = self._build_frontend_from_scene_config(frontend_config)
        self._filter = AnchoredOnlineFilter.identity_initialized(estimator_mode=self.config.estimator_mode)
        filter_config = estimation_config.get("filter", {})
        smoother_config = estimation_config.get("smoother", {})
        self._filter.use_auxiliary_tag_updates = bool(filter_config.get("use_aux_tags_in_filter", True))
        self._filter.vision_covariance_scale = float(filter_config.get("vision_covariance_scale", 1.0))
        if filter_config.get("anchor_vision_covariance_scale") not in (None, ""):
            self._filter.anchor_vision_covariance_scale = float(filter_config.get("anchor_vision_covariance_scale"))
        if filter_config.get("aux_vision_covariance_scale") not in (None, ""):
            self._filter.aux_vision_covariance_scale = float(filter_config.get("aux_vision_covariance_scale"))
        self._filter.imu_process_covariance_scale = float(filter_config.get("imu_process_covariance_scale", 1.0))
        if filter_config.get("gyro_process_covariance_scale") not in (None, ""):
            self._filter.gyro_process_covariance_scale = float(filter_config.get("gyro_process_covariance_scale"))
        if filter_config.get("accel_process_covariance_scale") not in (None, ""):
            self._filter.accel_process_covariance_scale = float(filter_config.get("accel_process_covariance_scale"))
        self._filter.post_relocalization_covariance_scale = float(
            filter_config.get("post_relocalization_covariance_scale", 1.0)
        )
        self._filter.allow_anchor_reacquisition_after_first_lock = bool(
            filter_config.get("allow_anchor_reacquisition_after_first_lock", True)
        )
        if filter_config.get("dropout_post_reacquisition_covariance_scale") not in (None, ""):
            self._filter.dropout_post_reacquisition_covariance_scale = float(
                filter_config.get("dropout_post_reacquisition_covariance_scale")
            )
        if filter_config.get("dropout_max_reacquisition_position_correction_m") not in (None, ""):
            self._filter.dropout_max_reacquisition_position_correction_m = float(
                filter_config.get("dropout_max_reacquisition_position_correction_m")
            )
        if filter_config.get("dropout_max_reacquisition_rotation_correction_deg") not in (None, ""):
            self._filter.dropout_max_reacquisition_rotation_correction_deg = float(
                filter_config.get("dropout_max_reacquisition_rotation_correction_deg")
            )
        if filter_config.get("dropout_max_reacquisition_velocity_correction_mps") not in (None, ""):
            self._filter.dropout_max_reacquisition_velocity_correction_mps = float(
                filter_config.get("dropout_max_reacquisition_velocity_correction_mps")
            )
        if filter_config.get("anchor_reacquisition_max_innovation_norm") not in (None, ""):
            self._filter.anchor_reacquisition_max_innovation_norm = float(
                filter_config.get("anchor_reacquisition_max_innovation_norm")
            )
        if filter_config.get("anchor_reacquisition_max_nis") not in (None, ""):
            self._filter.anchor_reacquisition_max_nis = float(filter_config.get("anchor_reacquisition_max_nis"))
        self._disable_imu_prediction_while_anchor_suppressed = bool(
            filter_config.get("disable_imu_prediction_while_anchor_suppressed", False)
        )
        if filter_config.get("suppression_imu_specific_force_gate_mps2") not in (None, ""):
            self._suppression_imu_specific_force_gate_mps2 = float(
                filter_config.get("suppression_imu_specific_force_gate_mps2")
            )
        self._suppression_propagation_mode = str(filter_config.get("suppression_propagation_mode", "full_imu")).strip().lower()
        smoother_backend = str(smoother_config.get("backend", "lightweight")).strip().lower()
        if smoother_backend == "windowed_ba":
            self._smoother = WindowedAnchorBASmoother(
                lag_size=int(smoother_config.get("lag_size", 30)),
                use_auxiliary_tags=bool(smoother_config.get("use_aux_tags_in_smoother", True)),
            )
        else:
            self._smoother = FixedLagSmoother(
                lag_size=int(smoother_config.get("lag_size", 30)),
                use_auxiliary_tags=bool(smoother_config.get("use_aux_tags_in_smoother", True)),
            )
        waypoints = tuple(
            Waypoint(
                position_world_m=tuple(float(value) for value in waypoint["position_world_m"]),
                tolerance_m=float(waypoint.get("tolerance_m", 0.02)),
            )
            for waypoint in control_config.get("waypoints", [])
        )
        self._tracker = PathTracker(
            waypoint_manager=WaypointManager(waypoints=waypoints),
            safety_gates=SafetyGates(
                max_position_radius_95_m=float(control_config.get("gates", {}).get("max_position_radius_95_m", 0.10)),
                max_innovation_norm=float(control_config.get("gates", {}).get("max_innovation_norm", 0.25)),
                max_anchor_lost_steps=int(control_config.get("gates", {}).get("max_anchor_lost_steps", 5)),
            ),
            proportional_gain=float(control_config.get("proportional_gain", 1.0)),
            max_command_abs=float(control_config.get("max_command_abs", 0.25)),
        )
        self._filter_clock = FixedRateClock(rate_hz=float(filter_config.get("output_rate_hz", 60.0)))
        self._smoother_clock = FixedRateClock(rate_hz=float(smoother_config.get("solve_rate_hz", 10.0)))
        self._controller_clock = FixedRateClock(rate_hz=float(self.config.config_payloads["scene"].get("controller_rate_hz", 50.0)))
        servo_config = self._build_servo_config(actuation_config)
        joint_count = len(self._robot_binding.robot.joint_names)
        self._servo_models = [
            UncertainServoModel(
                servo_config,
                initial_position=0.0,
                rng=np.random.default_rng(int(self.config.seed) + index + 1),
            )
            for index in range(joint_count)
        ]

    def warmup(self) -> None:
        if self._world is None or self._robot_binding is None or self._camera_binding is None:
            raise RuntimeError("World, robot, and camera must be initialized before warmup.")
        self._timeline.play()
        self._timeline.commit()
        self._world.reset()
        self._robot_binding.set_default_joint_positions()
        self._robot_binding.end_effector_prim.initialize()
        self._camera_binding.initialize()
        for observer_binding in self._observer_camera_bindings:
            observer_binding.initialize()
            observer_binding.set_updates_enabled(False)
        self._orient_observer_cameras_once()
        current_positions = self._robot_binding.get_joint_positions()
        for index, model in enumerate(self._servo_models):
            model.position = float(current_positions[index])
        self._previous_joint_positions = current_positions.copy()
        self._home_joint_positions = current_positions.copy()
        for _ in range(30):
            self._world.step(render=True)
        ee_position_world_m, ee_orientation_wxyz = self._robot_binding.get_end_effector_pose()
        self._calibrate_camera_mount_once(ee_position_world_m)
        for _ in range(10):
            self._world.step(render=True)
        ee_position_world_m, ee_orientation_wxyz = self._robot_binding.get_end_effector_pose()
        self._capture_camera_mount_relation(
            ee_position_world_m=np.asarray(ee_position_world_m, dtype=np.float64).reshape(3),
            ee_orientation_wxyz=np.asarray(ee_orientation_wxyz, dtype=np.float64).reshape(4),
        )
        self._control_target_orientation_wxyz = ee_orientation_wxyz.copy()
        self._last_open_loop_target_position = self._current_control_position_world_m(
            ee_position_world_m=np.asarray(ee_position_world_m, dtype=np.float64).reshape(3)
        )
        self._write_tag_ground_truth()

    def _scene_motion_program_mode(self) -> str:
        scene_payload = self.config.config_payloads.get("scene", {})
        motion_program = scene_payload.get("motion_program", {})
        if not isinstance(motion_program, dict):
            return ""
        return str(motion_program.get("mode", "")).strip().lower()

    def _camera_is_robot_mounted(self) -> bool:
        if self._camera_binding is None or self._robot_binding is None:
            return False
        camera_prim_path = str(self._camera_binding.spec.prim_path).rstrip("/")
        camera_mount_path = str(self._robot_binding.robot.camera_mount_prim_path).rstrip("/")
        if not camera_prim_path or not camera_mount_path:
            return False
        return camera_prim_path == camera_mount_path or camera_prim_path.startswith(camera_mount_path + "/")

    def _should_preserve_robot_mounted_camera_pose(self) -> bool:
        return self._camera_is_robot_mounted() and self._scene_motion_program_mode() in {"bezier_task_space", "keyframes"}

    def _calibrate_camera_mount_once(self, ee_position_world_m: np.ndarray) -> None:
        if self._camera_binding is None:
            return
        if self._should_preserve_robot_mounted_camera_pose():
            return
        look_at_world_m = self._camera_binding.spec.look_at_world_m
        if look_at_world_m is None:
            return
        look_at_world_m = np.asarray(look_at_world_m, dtype=np.float64).reshape(3)
        configured_offset_world_m = np.asarray(self._camera_binding.spec.mount_offset_world_m, dtype=np.float64).reshape(3)
        if float(np.linalg.norm(configured_offset_world_m)) > 1e-9:
            camera_position_world_m = np.asarray(ee_position_world_m, dtype=np.float64).reshape(3) + configured_offset_world_m
        else:
            camera_position_world_m, _ = self._camera_binding.get_world_pose()
            camera_position_world_m = np.asarray(camera_position_world_m, dtype=np.float64).reshape(3)
            camera_forward_world = _normalize(
                look_at_world_m - camera_position_world_m,
                default=np.array([0.0, 0.0, -1.0], dtype=np.float64),
            )
            camera_position_world_m = camera_position_world_m + 0.06 * camera_forward_world
        orientation_wxyz = _look_at_orientation_wxyz(
            camera_position_world_m,
            look_at_world_m,
        )
        self._camera_binding.set_world_pose(
            position_world_m=camera_position_world_m,
            orientation_wxyz=orientation_wxyz,
        )

    def _camera_mount_relation_from_live_poses(
        self,
        *,
        ee_position_world_m: np.ndarray,
        ee_orientation_wxyz: np.ndarray,
        camera_position_world_m: np.ndarray | None = None,
        camera_orientation_wxyz: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self._camera_binding is None:
            return None, None
        if camera_position_world_m is None or camera_orientation_wxyz is None:
            try:
                camera_position_world_m, camera_orientation_wxyz = self._camera_binding.get_world_pose()
            except Exception:
                return self._camera_rotation_ec, self._camera_translation_ec_m
        ee_position = np.asarray(ee_position_world_m, dtype=np.float64).reshape(3)
        ee_rotation_we = _quaternion_wxyz_to_rotation_matrix(ee_orientation_wxyz)
        camera_position = np.asarray(camera_position_world_m, dtype=np.float64).reshape(3)
        camera_rotation_wc = _quaternion_wxyz_to_rotation_matrix(camera_orientation_wxyz)
        return (
            ee_rotation_we.T @ camera_rotation_wc,
            ee_rotation_we.T @ (camera_position - ee_position),
        )

    def _capture_camera_mount_relation(
        self,
        *,
        ee_position_world_m: np.ndarray,
        ee_orientation_wxyz: np.ndarray,
    ) -> None:
        if self._camera_binding is None:
            self._camera_rotation_ec = None
            self._camera_translation_ec_m = None
            return
        try:
            camera_position_world_m, camera_orientation_wxyz = self._camera_binding.get_world_pose()
        except Exception:
            self._camera_rotation_ec = None
            self._camera_translation_ec_m = None
            return
        self._camera_rotation_ec, self._camera_translation_ec_m = self._camera_mount_relation_from_live_poses(
            ee_position_world_m=ee_position_world_m,
            ee_orientation_wxyz=ee_orientation_wxyz,
            camera_position_world_m=np.asarray(camera_position_world_m, dtype=np.float64).reshape(3),
            camera_orientation_wxyz=np.asarray(camera_orientation_wxyz, dtype=np.float64).reshape(4),
        )

    def _orient_observer_cameras_once(self) -> None:
        for binding, observer_spec in zip(
            self._observer_camera_bindings,
            self._stage_artifacts.stage_spec.observer_cameras,
            strict=False,
        ):
            position_world_m = np.asarray(observer_spec.position_world_m, dtype=np.float64).reshape(3)
            look_at_world_m = np.asarray(observer_spec.look_at_world_m, dtype=np.float64).reshape(3)
            binding.set_world_pose(
                position_world_m=position_world_m,
                orientation_wxyz=_look_at_orientation_wxyz(position_world_m, look_at_world_m),
            )

    def start(self) -> None:
        self.boot_kit()
        self.open_or_build_stage()
        self.bind_world()
        self.bind_robot()
        self.bind_sensors()
        self.bind_frontend_estimation_control()
        self.warmup()
        self._configure_standard_streaming()
        self.started = True

    def standard_streaming_status(self) -> dict[str, Any]:
        return copy.deepcopy(self._standard_streaming_status)

    def _configure_standard_streaming(self) -> None:
        backend = str(self.config.streaming_backend).strip().lower()
        if backend != "webrtc":
            backend = "webrtc"
        self._standard_streaming_status = default_streaming_status("webrtc")
        self._standard_streaming_status.update(
            {
                "state": "active",
                "transport_active": True,
                "connection_info": build_webrtc_connection_info(
                    signal_port=int(self.config.webrtc_signal_port),
                    stream_port=int(self.config.webrtc_stream_port),
                    target_fps=int(self.config.webrtc_target_fps),
                    allow_dynamic_resize=bool(self.config.webrtc_allow_dynamic_resize),
                    public_ip=str(self.config.webrtc_public_ip),
                ),
                "notes": [
                    "Mounted-camera viewing is provided by Isaac's WebRTC livestream path.",
                    "Use the external Isaac Sim WebRTC viewer or compatible browser client.",
                ],
            }
        )

    def _ensure_internal_detector_transport(self) -> None:
        return None

    def _setup_ros2_bridge_publishers(self) -> None:
        return None

    def _timestamps(self) -> TimestampTriplet:
        return TimestampTriplet(sim_time_s=self._sim_time_s, sensor_time_s=self._sim_time_s, host_time_s=time.time())

    def _write_tag_ground_truth(self) -> None:
        tags_payload = []
        for tag_id, tag_pose in sorted(self._tag_pose_map.items()):
            tags_payload.append(
                {
                    "tag_id": int(tag_id),
                    "is_anchor": bool(tag_pose.is_anchor),
                    "size_m": float(tag_pose.size_m),
                    "position_world_m": [float(value) for value in tag_pose.position_world_m],
                    "rotation_wt": [[float(entry) for entry in row] for row in tag_pose.rotation_wt],
                }
            )
        self._writer.write_tag_gt({"anchor_tag_id": int(self.config.anchor_tag_id), "tags": tags_payload})

    def _visibility_config(self) -> dict[str, Any]:
        payload = self.config.config_payloads.get("visibility", {})
        return payload if isinstance(payload, dict) else {}

    def _anchor_dropout_intervals_s(self) -> tuple[tuple[float, float], ...]:
        payload = self._visibility_config()
        if not bool(payload.get("enabled", True)):
            return ()
        raw_intervals = payload.get("suppressed_intervals_s", [])
        if not isinstance(raw_intervals, list):
            return ()
        intervals: list[tuple[float, float]] = []
        for interval in raw_intervals:
            if not isinstance(interval, (list, tuple)) or len(interval) < 2:
                continue
            start_s = float(interval[0])
            end_s = float(interval[1])
            if end_s <= start_s:
                continue
            intervals.append((start_s, end_s))
        return tuple(intervals)

    def _anchor_updates_suppressed(self, *, timestamp_s: float) -> bool:
        for start_s, end_s in self._anchor_dropout_intervals_s():
            if start_s <= float(timestamp_s) <= end_s:
                return True
        return False

    def _write_dropout_debug_event(
        self,
        *,
        timestamp_s: float,
        frame_index: int,
        event_kind: str,
        attempted: bool,
        accepted: bool,
        reason: str,
        is_reacquisition: bool = False,
        pose_innovation_norm_m: float | None = None,
        anchor_nis: float | None = None,
        orientation_innovation_norm_deg: float | None = None,
        velocity_innovation_norm_mps: float | None = None,
        relocalization_correction_norm_m: float | None = None,
        post_update_covariance_trace: float | None = None,
    ) -> None:
        self._writer.write_dropout_debug_event(
            {
                "timestamp_s": float(timestamp_s),
                "frame_index": int(frame_index),
                "event_kind": str(event_kind),
                "anchor_update_attempted": bool(attempted),
                "accepted": bool(accepted),
                "reason": str(reason),
                "is_reacquisition": bool(is_reacquisition),
                "pose_innovation_norm_m": None if pose_innovation_norm_m is None else float(pose_innovation_norm_m),
                "anchor_nis": None if anchor_nis is None else float(anchor_nis),
                "orientation_innovation_norm_deg": None
                if orientation_innovation_norm_deg is None
                else float(orientation_innovation_norm_deg),
                "velocity_innovation_norm_mps": None
                if velocity_innovation_norm_mps is None
                else float(velocity_innovation_norm_mps),
                "relocalization_correction_norm_m": None
                if relocalization_correction_norm_m is None
                else float(relocalization_correction_norm_m),
                "post_update_covariance_trace": None
                if post_update_covariance_trace is None
                else float(post_update_covariance_trace),
            }
        )

    def _write_dropout_debug_frame(
        self,
        *,
        frame_index: int,
        timestamp_s: float,
        frame_packet: IsaacCameraFramePacket,
        suppression_active: bool,
        imu_prediction_disabled: bool,
    ) -> None:
        if self._filter is None:
            return
        gt_position_world_m = np.asarray(
            frame_packet.extrinsics_snapshot.get("position_world_m", [0.0, 0.0, 0.0]),
            dtype=np.float64,
        ).reshape(3)
        position_error_norm_m = float(
            np.linalg.norm(np.asarray(self._filter.state.position_world_m, dtype=np.float64).reshape(3) - gt_position_world_m)
        )
        covariance = sanitize_covariance(np.asarray(self._filter.covariance, dtype=np.float64))
        eigenvalues = np.linalg.eigvalsh(covariance)
        propagation_dt_s = 0.0 if self._last_camera_timestamp_s is None else float(timestamp_s - self._last_camera_timestamp_s)
        self._writer.write_dropout_debug_frame(
            {
                "timestamp_s": float(timestamp_s),
                "frame_index": int(frame_index),
                "anchor_visible_raw": bool(self._last_anchor_visible_raw),
                "anchor_visible_effective": bool(self._last_anchor_visible),
                "suppression_active": bool(suppression_active),
                "imu_prediction_disabled": bool(imu_prediction_disabled),
                "imu_packets_since_last_frame": int(len(self._camera_interval_imu_packets)),
                "imu_packets_used_for_prediction": int(self._last_imu_packets_used_for_prediction),
                "imu_packets_rejected_for_prediction": int(self._last_imu_packets_rejected_for_prediction),
                "propagation_dt_s": float(max(propagation_dt_s, 0.0)),
                "position_error_norm_m": position_error_norm_m,
                "velocity_norm_mps": float(np.linalg.norm(np.asarray(self._filter.state.velocity_world_mps, dtype=np.float64))),
                "gyro_bias_norm_rps": float(np.linalg.norm(np.asarray(self._filter.state.gyro_bias_rps, dtype=np.float64))),
                "accel_bias_norm_mps2": float(np.linalg.norm(np.asarray(self._filter.state.accel_bias_mps2, dtype=np.float64))),
                "covariance_trace": float(np.trace(covariance)),
                "covariance_min_eigenvalue": float(np.min(eigenvalues)),
                "covariance_max_eigenvalue": float(np.max(eigenvalues)),
            }
        )
        self._last_camera_timestamp_s = float(timestamp_s)
        self._camera_interval_imu_packets.clear()

    def _filter_imu_packets_for_prediction(
        self,
        imu_packets: tuple[IsaacImuPacket, ...],
        *,
        suppression_active: bool,
    ) -> tuple[IsaacImuPacket, ...]:
        if not imu_packets:
            self._last_imu_packets_used_for_prediction = 0
            self._last_imu_packets_rejected_for_prediction = 0
            return ()
        filtered_packets = imu_packets
        rejected_count = 0
        if suppression_active and self._suppression_imu_specific_force_gate_mps2 not in (None, ""):
            gate = max(float(self._suppression_imu_specific_force_gate_mps2), 0.0)
            filtered_packets = tuple(
                packet
                for packet in imu_packets
                if float(np.linalg.norm(np.array([packet.ax, packet.ay, packet.az], dtype=np.float64))) <= gate
            )
            rejected_count = len(imu_packets) - len(filtered_packets)
        self._last_imu_packets_used_for_prediction = len(filtered_packets)
        self._last_imu_packets_rejected_for_prediction = rejected_count
        return filtered_packets

    def _propagate_filter_from_imu(
        self,
        imu_packets: tuple[IsaacImuPacket, ...],
        *,
        suppression_active: bool,
    ) -> None:
        if self._filter is None:
            self._last_imu_packets_used_for_prediction = 0
            self._last_imu_packets_rejected_for_prediction = 0
            return
        if not imu_packets:
            self._last_imu_packets_used_for_prediction = 0
            self._last_imu_packets_rejected_for_prediction = 0
            return
        if suppression_active and self._disable_imu_prediction_while_anchor_suppressed:
            self._last_imu_packets_used_for_prediction = 0
            self._last_imu_packets_rejected_for_prediction = len(imu_packets)
            return
        packets_for_prediction = self._filter_imu_packets_for_prediction(
            imu_packets,
            suppression_active=bool(suppression_active),
        )
        if not packets_for_prediction:
            return
        if suppression_active:
            self._filter.predict_with_mode(
                packets_for_prediction,
                mode=self._suppression_propagation_mode,
            )
            return
        self._filter.predict(packets_for_prediction)

    def _log_realized_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._robot_binding is None:
            raise RuntimeError("Robot binding missing.")
        joint_positions = self._robot_binding.get_joint_positions()
        if self._previous_joint_positions is None or self._previous_joint_positions.shape != joint_positions.shape:
            joint_velocities = np.zeros_like(joint_positions)
        else:
            joint_velocities = (joint_positions - self._previous_joint_positions) / float(self._physics_dt_s)
        self._previous_joint_positions = joint_positions.copy()
        ee_position_world_m, ee_orientation_wxyz = self._robot_binding.get_end_effector_pose()
        packet = IsaacRealizedJointPacket(
            timestamp_s=float(self._sim_time_s),
            sim_time_s=float(self._sim_time_s),
            joint_names=self._robot_binding.robot.joint_names,
            positions=tuple(float(value) for value in joint_positions.tolist()),
            velocities=tuple(float(value) for value in joint_velocities.tolist()),
            end_effector_position_world_m=tuple(float(value) for value in ee_position_world_m.tolist()),
            end_effector_orientation_wxyz=tuple(float(value) for value in ee_orientation_wxyz.tolist()),
        )
        self._latest_realized_joint_packet = packet
        self._writer.write_realized_state(packet)
        self._counts["realized_joints"] += 1
        gt_joint_row = {
            "timestamp_s": float(self._sim_time_s),
            "sim_time_s": float(self._sim_time_s),
            "ee_px": float(ee_position_world_m[0]),
            "ee_py": float(ee_position_world_m[1]),
            "ee_pz": float(ee_position_world_m[2]),
        }
        for joint_name, position, velocity in zip(self._robot_binding.robot.joint_names, joint_positions, joint_velocities):
            gt_joint_row[f"{joint_name}_position"] = float(position)
            gt_joint_row[f"{joint_name}_velocity"] = float(velocity)
        self._writer.write_gt(namespace="joint", payload=gt_joint_row)
        return joint_positions, joint_velocities, ee_position_world_m, ee_orientation_wxyz

    def _process_imu(
        self,
        timestamps: TimestampTriplet,
        *,
        ee_position_world_m: np.ndarray,
        ee_orientation_wxyz: np.ndarray,
    ) -> tuple:
        if self._imu_binding is None:
            return ()
        position_world_m = np.asarray(ee_position_world_m, dtype=np.float64).reshape(3)
        orientation_wxyz = np.asarray(ee_orientation_wxyz, dtype=np.float64).reshape(4)
        if self._camera_binding is not None:
            try:
                camera_position_world_m, camera_orientation_wxyz = self._camera_binding.get_world_pose()
                position_world_m = np.asarray(camera_position_world_m, dtype=np.float64).reshape(3)
                orientation_wxyz = np.asarray(camera_orientation_wxyz, dtype=np.float64).reshape(4)
            except Exception:
                position_world_m = np.asarray(ee_position_world_m, dtype=np.float64).reshape(3)
                orientation_wxyz = np.asarray(ee_orientation_wxyz, dtype=np.float64).reshape(4)
        qw, qx, qy, qz = orientation_wxyz
        rotation_wi = np.array(
            [
                [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
                [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
                [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
            ],
            dtype=np.float64,
        )
        packets = []
        for tick in self._imu_binding.due_ticks(self._sim_time_s):
            packet = self._imu_binding.sample(
                tick=tick,
                timestamps=timestamps,
                position_world_m=position_world_m,
                rotation_wi=rotation_wi,
            )
            if packet is None:
                continue
            packets.append(packet)
            self._latest_imu_packet = packet
            if self._imu_binding is not None:
                self._latest_imu_truth = {
                    "accel_mps2": np.asarray(self._imu_binding.last_truth_specific_force_body_mps2, dtype=np.float64).reshape(3),
                    "gyro_rps": np.asarray(self._imu_binding.last_truth_gyro_body_rps, dtype=np.float64).reshape(3),
                    "dropped": bool(self._imu_binding.last_packet_dropped),
                }
            self._writer.write_imu_packet(packet)
            self._writer.write_gt(
                namespace="imu",
                payload={
                    "timestamp_s": float(packet.timestamp_s),
                    "sim_time_s": float(packet.sim_time_s),
                    "ax": float(packet.ax),
                    "ay": float(packet.ay),
                    "az": float(packet.az),
                    "wx": float(packet.wx),
                    "wy": float(packet.wy),
                    "wz": float(packet.wz),
                },
            )
            self._counts["imu_packets"] += 1
            self._smoother_interval_imu_packets.append(packet)
            self._camera_interval_imu_packets.append(packet)
        return tuple(packets)

    def external_detector_status(self) -> dict[str, Any]:
        return {
            "backend": "new_pupil",
            "backend_state": "active" if self._tag_detection_enabled() else "idle",
            "backend_error": None,
            "input_transport": None,
            "bridge_latency_ms": None,
            "timeouts": 0,
            "fallback_active": False,
        }

    def set_external_detector_plain_ros_transport_enabled(self, enabled: bool) -> None:
        return None

    def _cache_external_detector_frame(
        self,
        *,
        frame_packet: IsaacCameraFramePacket,
        image_rgb: np.ndarray,
    ) -> None:
        return None

    def _external_detection_to_model(self, detection: Any) -> Any:
        return detection

    def _apply_frame_measurement_pack(self, pack: FrameMeasurementPack) -> None:
        frame_packet = pack.camera_frame
        for detection in pack.detections:
            self._writer.write_detection(detection)
            self._counts["detections"] += 1
            self._last_detected_frame_by_tag[int(detection.tag_id)] = int(detection.frame_index)
        self._last_detections = pack.detections
        self._latest_display_camera_frame_packet = frame_packet
        self._latest_display_camera_image_rgb = self._latest_camera_image_rgb
        previous_anchor_visible = bool(self._last_anchor_visible)
        anchor_visible_raw = bool(pack.metadata.get("anchor_visible", False))
        anchor_update_suppressed = self._anchor_updates_suppressed(timestamp_s=float(frame_packet.timestamp_s))
        self._last_anchor_visible_raw = anchor_visible_raw
        self._last_anchor_update_suppressed = bool(anchor_update_suppressed)
        self._last_anchor_visible = bool(anchor_visible_raw and not anchor_update_suppressed)
        self._latest_anchor_update_result = None
        if not self._tag_pose_estimation_enabled():
            self._latest_auxiliary_update_summary = None
            auxiliary_detections = tuple(detection for detection in pack.detections if not detection.is_anchor)
            self._writer.write_estimator_input(
                {
                    "timestamp_s": float(frame_packet.timestamp_s),
                    "kind": "frame_pack",
                    "frame_index": int(frame_packet.frame_index),
                    "anchor_visible": bool(self._last_anchor_visible),
                    "anchor_visible_raw": bool(anchor_visible_raw),
                    "anchor_visible_effective": bool(self._last_anchor_visible),
                    "anchor_update_suppressed": bool(anchor_update_suppressed),
                    "detected_tag_ids": list(pack.metadata.get("detected_tag_ids", [])),
                    "auxiliary_visible_count": int(len(auxiliary_detections)),
                    "native_auxiliary_pose_ready_count": 0,
                    "accepted_auxiliary_update_count": 0,
                    "rejected_auxiliary_update_count": 0,
                    "accepted_auxiliary_tag_ids": [],
                    "rejected_auxiliary_tag_ids": [],
                    "auxiliary_rejection_reasons": {},
                    "auxiliary_update_decisions": [],
                    "tag_pose_estimation_enabled": False,
                    "corners_only": True,
                }
            )
            self._write_dropout_debug_frame(
                frame_index=int(frame_packet.frame_index),
                timestamp_s=float(frame_packet.timestamp_s),
                frame_packet=frame_packet,
                suppression_active=bool(anchor_update_suppressed),
                imu_prediction_disabled=False,
            )
            return
        anchor_detections = tuple(pack.anchor_detections)
        anchor_pose_detections = tuple(pack.anchor_pose_detections)
        anchor_measurement_std_m = float(
            self.config.config_payloads["estimation"].get("filter", {}).get("anchor_measurement_std_m", 0.01)
        )
        is_reacquisition = bool(self._anchor_lock_acquired and not previous_anchor_visible and self._last_anchor_visible)
        anchor_update_result = None
        if not anchor_visible_raw:
            self._write_dropout_debug_event(
                timestamp_s=float(frame_packet.timestamp_s),
                frame_index=int(frame_packet.frame_index),
                event_kind="anchor_visibility",
                attempted=False,
                accepted=False,
                reason="not_visible",
            )
        elif anchor_update_suppressed:
            self._write_dropout_debug_event(
                timestamp_s=float(frame_packet.timestamp_s),
                frame_index=int(frame_packet.frame_index),
                event_kind="anchor_visibility",
                attempted=False,
                accepted=False,
                reason="suppressed_window",
            )
        elif self._anchor_lock_acquired and not self._filter.allow_anchor_reacquisition_after_first_lock:
            self._write_dropout_debug_event(
                timestamp_s=float(frame_packet.timestamp_s),
                frame_index=int(frame_packet.frame_index),
                event_kind="anchor_update",
                attempted=False,
                accepted=False,
                reason="first_lock_only_mode",
                is_reacquisition=is_reacquisition,
            )
        elif anchor_pose_detections:
            anchor_detection = max(anchor_pose_detections, key=lambda detection: float(detection.score))
            anchor_update_result = self._filter.update_anchor(
                tag_detection=anchor_detection,
                tag_pose=self._tag_pose_map[int(anchor_detection.tag_id)],
                measurement_std_m=anchor_measurement_std_m,
                is_reacquisition=is_reacquisition,
            )
            if bool(anchor_update_result.accepted):
                self._anchor_lock_acquired = True
            self._write_dropout_debug_event(
                timestamp_s=float(frame_packet.timestamp_s),
                frame_index=int(frame_packet.frame_index),
                event_kind="anchor_update",
                attempted=True,
                accepted=bool(anchor_update_result.accepted),
                reason=str(anchor_update_result.reason),
                is_reacquisition=bool(anchor_update_result.is_reacquisition),
                pose_innovation_norm_m=float(anchor_update_result.innovation_norm_m),
                anchor_nis=None if anchor_update_result.anchor_nis is None else float(anchor_update_result.anchor_nis),
                orientation_innovation_norm_deg=float(anchor_update_result.orientation_innovation_norm_deg),
                velocity_innovation_norm_mps=float(anchor_update_result.velocity_innovation_norm_mps),
                relocalization_correction_norm_m=float(anchor_update_result.relocalization_correction_norm_m),
                post_update_covariance_trace=float(anchor_update_result.post_update_covariance_trace),
            )
            self._latest_anchor_update_result = {
                "accepted": bool(anchor_update_result.accepted),
                "reason": str(anchor_update_result.reason),
                "tag_id": int(anchor_detection.tag_id),
                "is_reacquisition": bool(anchor_update_result.is_reacquisition),
            }
        elif anchor_detections:
            self._filter.rejected_updates_count += len(anchor_detections)
            self._write_dropout_debug_event(
                timestamp_s=float(frame_packet.timestamp_s),
                frame_index=int(frame_packet.frame_index),
                event_kind="anchor_update",
                attempted=True,
                accepted=False,
                reason="no_pose_solution",
                is_reacquisition=is_reacquisition,
            )
            self._latest_anchor_update_result = {
                "accepted": False,
                "reason": "no_pose_solution",
                "tag_id": int(anchor_detections[0].tag_id),
                "is_reacquisition": bool(is_reacquisition),
            }
        auxiliary_detections = tuple(detection for detection in pack.detections if not detection.is_anchor)
        auxiliary_update_summary = self._filter.update_aux_tags(
            tag_detections=auxiliary_detections,
            mapped_tag_poses=self._tag_pose_map,
            trust=float(self.config.config_payloads["estimation"].get("filter", {}).get("auxiliary_update_trust", 0.15)),
            intrinsics_snapshot=frame_packet.intrinsics_snapshot,
            image_width_px=int(frame_packet.image_width_px),
            image_height_px=int(frame_packet.image_height_px),
            min_corner_margin_px=float(
                self.config.config_payloads["estimation"].get("filter", {}).get("auxiliary_update_min_corner_margin_px", 0.0)
            ),
            min_visibility_streak=int(
                self.config.config_payloads["estimation"].get("filter", {}).get("auxiliary_update_min_visibility_streak", 1)
            ),
            max_reprojection_error_px=self.config.config_payloads["estimation"]
            .get("filter", {})
            .get("auxiliary_update_max_reprojection_error_px"),
            max_innovation_norm=self.config.config_payloads["estimation"]
            .get("filter", {})
            .get("auxiliary_update_max_innovation_norm"),
            max_mahalanobis_score=self.config.config_payloads["estimation"]
            .get("filter", {})
            .get("auxiliary_update_max_mahalanobis_score"),
            max_feedback_correction_norm_m=self.config.config_payloads["estimation"]
            .get("filter", {})
            .get("auxiliary_update_max_feedback_correction_m"),
            tag_feedback_diagnostics=self._auxiliary_feedback_diagnostics_for_filter(),
        )
        self._latest_auxiliary_update_summary = auxiliary_update_summary
        auxiliary_summary_payload = (
            {} if auxiliary_update_summary is None else auxiliary_update_summary.as_json()
        )
        self._writer.write_estimator_input(
            {
                "timestamp_s": float(frame_packet.timestamp_s),
                "kind": "frame_pack",
                "frame_index": int(frame_packet.frame_index),
                "anchor_visible": bool(self._last_anchor_visible),
                "anchor_visible_raw": bool(anchor_visible_raw),
                "anchor_visible_effective": bool(self._last_anchor_visible),
                "anchor_update_suppressed": bool(anchor_update_suppressed),
                "detected_tag_ids": list(pack.metadata.get("detected_tag_ids", [])),
                "auxiliary_visible_count": int(auxiliary_summary_payload.get("visible_count", 0)),
                "native_auxiliary_pose_ready_count": int(auxiliary_summary_payload.get("native_pose_ready_count", 0)),
                "accepted_auxiliary_update_count": int(auxiliary_summary_payload.get("accepted_count", 0)),
                "rejected_auxiliary_update_count": int(auxiliary_summary_payload.get("rejected_count", 0)),
                "accepted_auxiliary_tag_ids": list(auxiliary_summary_payload.get("accepted_tag_ids", [])),
                "rejected_auxiliary_tag_ids": list(auxiliary_summary_payload.get("rejected_tag_ids", [])),
                "auxiliary_rejection_reasons": dict(auxiliary_summary_payload.get("rejection_reason_counts", {})),
                "auxiliary_update_decisions": list(auxiliary_summary_payload.get("decisions", [])),
            }
        )
        self._write_dropout_debug_frame(
            frame_index=int(frame_packet.frame_index),
            timestamp_s=float(frame_packet.timestamp_s),
            frame_packet=frame_packet,
            suppression_active=bool(anchor_update_suppressed),
            imu_prediction_disabled=bool(
                anchor_update_suppressed and self._disable_imu_prediction_while_anchor_suppressed
            ),
        )

    def _process_external_detector_result(self, result: Any) -> None:
        return None

    def ingest_external_detector_results(self, results: list[Any]) -> None:
        return None

    def expire_external_detector_frames(self) -> int:
        return 0

    def _process_camera(self, timestamps: TimestampTriplet) -> None:
        detection_enabled = self._tag_detection_enabled()
        if self._camera_binding is None or self._filter is None:
            return
        if detection_enabled and self._frontend is None:
            return
        tag_size_by_id = {int(tag_id): float(tag.size_m) for tag_id, tag in self._tag_pose_map.items()}
        for tick in self._camera_binding.due_ticks(self._sim_time_s):
            sample = self._camera_binding.sample(
                tick=tick,
                timestamps=timestamps,
            )
            if sample is None:
                continue
            frame_packet, image_rgb = sample
            rgb_rel_path = self._writer.save_rgb_image(frame_index=frame_packet.frame_index, image_rgb=image_rgb)
            frame_packet = IsaacCameraFramePacket(
                frame_index=frame_packet.frame_index,
                timestamp_s=frame_packet.timestamp_s,
                sim_time_s=frame_packet.sim_time_s,
                sensor_time_s=frame_packet.sensor_time_s,
                host_time_s=frame_packet.host_time_s,
                rgb_path=rgb_rel_path,
                intrinsics_snapshot={
                    **frame_packet.intrinsics_snapshot,
                    "distortion_coefficients": list(self._camera_binding.spec.distortion_coefficients),
                },
                extrinsics_snapshot=frame_packet.extrinsics_snapshot,
                image_width_px=frame_packet.image_width_px,
                image_height_px=frame_packet.image_height_px,
                visible_gt_tag_ids=frame_packet.visible_gt_tag_ids,
            )
            self._writer.write_camera_frame(frame_packet)
            self._counts["camera_frames"] += 1
            self._last_camera_intrinsics_snapshot = dict(frame_packet.intrinsics_snapshot)
            self._latest_camera_frame_packet = frame_packet
            self._latest_camera_image_rgb = np.asarray(image_rgb, dtype=np.uint8)
            self._writer.write_gt(
                namespace="camera",
                payload={
                    "timestamp_s": float(frame_packet.timestamp_s),
                    "sim_time_s": float(frame_packet.sim_time_s),
                    "px": float(frame_packet.extrinsics_snapshot["position_world_m"][0]),
                    "py": float(frame_packet.extrinsics_snapshot["position_world_m"][1]),
                    "pz": float(frame_packet.extrinsics_snapshot["position_world_m"][2]),
                    "qw": float(frame_packet.extrinsics_snapshot["orientation_wxyz"][0]),
                    "qx": float(frame_packet.extrinsics_snapshot["orientation_wxyz"][1]),
                    "qy": float(frame_packet.extrinsics_snapshot["orientation_wxyz"][2]),
                    "qz": float(frame_packet.extrinsics_snapshot["orientation_wxyz"][3]),
                },
            )
            if not detection_enabled:
                self._latest_display_camera_frame_packet = frame_packet
                self._latest_display_camera_image_rgb = np.asarray(image_rgb, dtype=np.uint8)
                self._last_detections = ()
                self._last_anchor_visible_raw = False
                self._last_anchor_visible = False
                self._last_anchor_update_suppressed = False
                self._latest_anchor_update_result = None
                self._latest_auxiliary_update_summary = None
                self._writer.write_estimator_input(
                    {
                        "timestamp_s": float(frame_packet.timestamp_s),
                        "kind": "frame_pack",
                        "frame_index": int(frame_packet.frame_index),
                        "anchor_visible": False,
                        "anchor_visible_raw": False,
                        "anchor_visible_effective": False,
                        "anchor_update_suppressed": False,
                        "detected_tag_ids": [],
                        "auxiliary_visible_count": 0,
                        "native_auxiliary_pose_ready_count": 0,
                        "accepted_auxiliary_update_count": 0,
                        "rejected_auxiliary_update_count": 0,
                        "accepted_auxiliary_tag_ids": [],
                        "rejected_auxiliary_tag_ids": [],
                        "auxiliary_rejection_reasons": {},
                        "auxiliary_update_decisions": [],
                        "detector_disabled": True,
                    }
                )
                self._write_dropout_debug_frame(
                    frame_index=int(frame_packet.frame_index),
                    timestamp_s=float(frame_packet.timestamp_s),
                    frame_packet=frame_packet,
                    suppression_active=False,
                    imu_prediction_disabled=False,
                )
                continue
            pack = self._frontend.process_bgr_frame(
                cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
                frame_packet=frame_packet,
                tag_size_by_id=tag_size_by_id,
            )
            pack = self._apply_live_vision_noise(pack)
            self._apply_frame_measurement_pack(pack)
        self._process_observer_cameras(timestamps)

    def _solve_detection_pose_from_corners(
        self,
        *,
        corners_xy: np.ndarray,
        local_tag_points_m: np.ndarray,
        intrinsics_snapshot: dict[str, Any],
    ) -> tuple[tuple[float, float, float] | None, tuple[float, float, float] | None]:
        if local_tag_points_m.size == 0:
            return None, None
        fx = float(intrinsics_snapshot.get("fx_px", 0.0))
        fy = float(intrinsics_snapshot.get("fy_px", 0.0))
        cx = float(intrinsics_snapshot.get("cx_px", 0.0))
        cy = float(intrinsics_snapshot.get("cy_px", 0.0))
        if fx <= 0.0 or fy <= 0.0:
            return None, None
        camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        dist_coeffs = np.asarray(
            intrinsics_snapshot.get("distortion_coefficients", [0.0, 0.0, 0.0, 0.0, 0.0]),
            dtype=np.float64,
        )
        success, rvec, tvec = cv2.solvePnP(
            objectPoints=np.asarray(local_tag_points_m, dtype=np.float64),
            imagePoints=np.asarray(corners_xy, dtype=np.float64),
            cameraMatrix=camera_matrix,
            distCoeffs=dist_coeffs,
            flags=getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE),
        )
        if not success:
            return None, None
        return (
            tuple(float(value) for value in np.asarray(rvec, dtype=np.float64).reshape(3)),
            tuple(float(value) for value in np.asarray(tvec, dtype=np.float64).reshape(3)),
        )

    def _apply_live_vision_noise(self, pack: Any) -> Any:
        if self._live_vision_noise_mode != "nominal":
            return pack
        preset = self._vision_nominal_preset
        pose_estimation_enabled = bool(self._tag_pose_estimation_enabled())
        mutated_detections = []
        for detection in pack.detections:
            try:
                if float(self._rng.random()) < float(preset.visibility_failure_probability):
                    continue
                if float(self._rng.random()) < float(preset.detection_drop_probability):
                    continue
                corners_xy = np.asarray(detection.corners_xy, dtype=np.float64).reshape(-1, 2).copy()
                if corners_xy.size == 0:
                    continue
                corners_xy = corners_xy + float(preset.corner_noise_std_px) * self._rng.normal(size=corners_xy.shape)
                if float(self._rng.random()) < float(preset.outlier_probability):
                    corners_xy = corners_xy + (4.0 * float(preset.corner_noise_std_px)) * self._rng.normal(size=corners_xy.shape)
                local_points = np.asarray(detection.local_tag_points_m, dtype=np.float64).reshape(-1, 3)
                if pose_estimation_enabled:
                    pose_camera_rvec, pose_camera_tvec_m = self._solve_detection_pose_from_corners(
                        corners_xy=corners_xy,
                        local_tag_points_m=local_points,
                        intrinsics_snapshot=pack.camera_frame.intrinsics_snapshot,
                    )
                else:
                    pose_camera_rvec, pose_camera_tvec_m = None, None
                mutated_detections.append(
                    IsaacTagDetectionPacket(
                        timestamp_s=float(detection.timestamp_s),
                        sim_time_s=float(detection.sim_time_s),
                        frame_index=int(detection.frame_index),
                        tag_id=int(detection.tag_id),
                        family=str(detection.family),
                        tag_size_m=float(detection.tag_size_m),
                        pnp_tag_size_m=None if detection.pnp_tag_size_m is None else float(detection.pnp_tag_size_m),
                        corners_xy=tuple((float(x), float(y)) for x, y in corners_xy.tolist()),
                        local_tag_points_m=tuple(tuple(float(value) for value in row) for row in local_points.tolist()),
                        corner_order=str(detection.corner_order),
                        score=float(detection.score) / max(float(preset.quality_scale), 1.0),
                        is_anchor=bool(detection.is_anchor),
                        pose_camera_rvec=pose_camera_rvec,
                        pose_camera_tvec_m=pose_camera_tvec_m,
                        detector_backend=str(detection.detector_backend),
                        measurement_source=str(getattr(detection, "measurement_source", "")),
                        temporal_gap_frames=int(detection.temporal_gap_frames),
                        visibility_flags={
                            **dict(detection.visibility_flags),
                            "noise_mode_nominal": True,
                            "pose_ready": pose_camera_tvec_m is not None,
                        },
                    )
                )
            except Exception:
                mutated_detections.append(
                    IsaacTagDetectionPacket(
                        timestamp_s=float(detection.timestamp_s),
                        sim_time_s=float(detection.sim_time_s),
                        frame_index=int(detection.frame_index),
                        tag_id=int(detection.tag_id),
                        family=str(detection.family),
                        tag_size_m=float(detection.tag_size_m),
                        pnp_tag_size_m=None if detection.pnp_tag_size_m is None else float(detection.pnp_tag_size_m),
                        corners_xy=tuple((float(x), float(y)) for x, y in detection.corners_xy),
                        local_tag_points_m=tuple(
                            tuple(float(value) for value in row) for row in detection.local_tag_points_m
                        ),
                        corner_order=str(detection.corner_order),
                        score=float(detection.score),
                        is_anchor=bool(detection.is_anchor),
                        pose_camera_rvec=None
                        if detection.pose_camera_rvec is None
                        else tuple(float(value) for value in detection.pose_camera_rvec),
                        pose_camera_tvec_m=None
                        if detection.pose_camera_tvec_m is None
                        else tuple(float(value) for value in detection.pose_camera_tvec_m),
                        detector_backend=str(detection.detector_backend),
                        measurement_source=str(getattr(detection, "measurement_source", "")),
                        temporal_gap_frames=int(detection.temporal_gap_frames),
                        visibility_flags={
                            **dict(detection.visibility_flags),
                            "noise_mode_nominal": True,
                            "noise_fallback": True,
                        },
                    )
                )
        pack.detections = tuple(mutated_detections)
        pack.metadata = {
            **dict(pack.metadata),
            "anchor_visible": any(bool(item.is_anchor) for item in mutated_detections),
            "detected_tag_ids": [int(item.tag_id) for item in mutated_detections],
            "detections_per_frame": len(mutated_detections),
            "vision_noise_mode": self._live_vision_noise_mode,
        }
        return pack

    def _process_observer_cameras(self, timestamps: TimestampTriplet) -> None:
        for binding in self._observer_camera_bindings:
            latest_view = self._latest_observer_views.get(binding.spec.name, {})
            for tick in binding.due_ticks(self._sim_time_s):
                sample = binding.sample(tick=tick, timestamps=timestamps)
                if sample is None:
                    continue
                frame_packet, image_rgb = sample
                self._writer.write_observer_camera_frame(
                    camera_name=str(binding.spec.name),
                    payload={
                        "camera_name": str(binding.spec.name),
                        "frame_index": int(frame_packet.frame_index),
                        "timestamp_s": float(frame_packet.timestamp_s),
                        "sim_time_s": float(frame_packet.sim_time_s),
                        "sensor_time_s": float(frame_packet.sensor_time_s),
                        "host_time_s": float(frame_packet.host_time_s),
                        "rgb_path": "",
                        "image_width_px": int(frame_packet.image_width_px),
                        "image_height_px": int(frame_packet.image_height_px),
                        "extrinsics_snapshot": dict(frame_packet.extrinsics_snapshot),
                    },
                    image_rgb=image_rgb,
                )
                latest_view = {
                    "name": str(binding.spec.name),
                    "frame_index": int(frame_packet.frame_index),
                    "timestamp_s": float(frame_packet.timestamp_s),
                    "image_rgb": np.asarray(image_rgb, dtype=np.uint8),
                }
            if latest_view:
                self._latest_observer_views[str(binding.spec.name)] = latest_view

    def _log_filter_and_uncertainty(self) -> None:
        if self._filter is None or self._filter_clock is None:
            return
        for _ in self._filter_clock.advance_to(self._sim_time_s):
            snapshot = self._filter.export_snapshot(sim_time_s=self._sim_time_s)
            snapshot.anchor_visible = bool(self._last_anchor_visible)
            snapshot.mode = self._filter.estimator_mode
            self._writer.write_filter_state(snapshot)
            self._counts["filter_states"] += 1
            self._latest_filter_snapshot = snapshot
            uncertainty = self._filter.current_uncertainty_summary(sim_time_s=self._sim_time_s)
            self._writer.write_uncertainty(uncertainty)
            self._counts["uncertainty_states"] += 1
            self._latest_uncertainty = uncertainty

    def _log_smoother(self) -> None:
        if self._smoother is None or self._smoother_clock is None or self._latest_filter_snapshot is None:
            return
        for _ in self._smoother_clock.advance_to(self._sim_time_s):
            self._smoother.push_snapshot(self._latest_filter_snapshot)
            self._smoother.observe_imu_interval(imu_packets=tuple(self._smoother_interval_imu_packets))
            self._smoother_interval_imu_packets.clear()
            self._smoother.observe_auxiliary_detections(
                detections=tuple(self._last_detections),
                current_state=self._latest_filter_snapshot,
                anchor_pose_map=self._tag_pose_map,
                intrinsics_snapshot=self._last_camera_intrinsics_snapshot,
            )
            smoother_snapshot = self._smoother.solve()
            smoother_diagnostics = dict(smoother_snapshot.diagnostics)
            feedback_observation_count = int(
                self.config.config_payloads["estimation"].get("smoother", {}).get("min_feedback_observation_count", 5)
            )
            max_feedback_correction_norm_m = float(
                self.config.config_payloads["estimation"].get("smoother", {}).get("max_feedback_correction_norm_m", 0.15)
            )
            anchored_window = bool(smoother_diagnostics.get("anchored_window", False))
            residual_before = smoother_diagnostics.get("residual_rmse_before_px")
            residual_after = smoother_diagnostics.get("residual_rmse_after_px")
            residual_improved = not (
                residual_before not in (None, "")
                and residual_after not in (None, "")
                and float(residual_after) > float(residual_before) + 1e-6
            )
            covariance_trace_before = smoother_diagnostics.get("covariance_trace_before")
            covariance_trace_after = smoother_diagnostics.get("covariance_trace_after")
            covariance_contracted = not (
                covariance_trace_before not in (None, "")
                and covariance_trace_after not in (None, "")
                and float(covariance_trace_after) > float(covariance_trace_before) + 1e-6
            )
            total_feedback_candidates = 0
            trusted_feedback_count = 0
            max_feedback_correction_norm_seen = 0.0
            active_estimates = self._smoother.active_tag_estimates(min_observation_count=feedback_observation_count)
            for tag_id, estimate in active_estimates.items():
                if int(tag_id) == int(self.config.anchor_tag_id):
                    continue
                total_feedback_candidates += 1
                previous_pose = self._tag_pose_map.get(int(tag_id))
                previous_position = (
                    np.asarray(previous_pose.position_world_m, dtype=np.float64).reshape(3)
                    if previous_pose is not None
                    else np.asarray(estimate.pose_wt[:3, 3], dtype=np.float64).reshape(3)
                )
                candidate_position = np.asarray(estimate.pose_wt[:3, 3], dtype=np.float64).reshape(3)
                correction_norm_m = float(np.linalg.norm(candidate_position - previous_position))
                trusted = (
                    anchored_window
                    and residual_improved
                    and covariance_contracted
                    and correction_norm_m <= max_feedback_correction_norm_m
                )
                max_feedback_correction_norm_seen = max(max_feedback_correction_norm_seen, correction_norm_m)
                self._tag_feedback_diagnostics[int(tag_id)] = {
                    "feedback_correction_norm_m": float(correction_norm_m),
                    "trusted": bool(trusted),
                    "observation_count": int(estimate.observation_count),
                    "anchored_window": bool(anchored_window),
                    "residual_improved": bool(residual_improved),
                    "covariance_contracted": bool(covariance_contracted),
                }
                if not trusted:
                    continue
                trusted_feedback_count += 1
                if not bool(self.config.config_payloads["control"].get("use_aux_map_for_control", True)):
                    continue
                self._tag_pose_map[int(tag_id)] = TagPoseSpec(
                    tag_id=int(tag_id),
                    size_m=float(estimate.size_m),
                    position_world_m=tuple(float(value) for value in candidate_position.tolist()),
                    rotation_wt=tuple(
                        tuple(float(entry) for entry in row)
                        for row in np.asarray(estimate.pose_wt[:3, :3], dtype=np.float64).tolist()
                    ),
                    is_anchor=bool(estimate.is_anchor),
                )
            self._smoother.record_feedback_diagnostics(
                feedback_correction_norm_m=float(max_feedback_correction_norm_seen),
                max_feedback_correction_norm_m=float(max_feedback_correction_norm_m),
                trusted_feedback_count=int(trusted_feedback_count),
                total_feedback_candidates=int(total_feedback_candidates),
            )
            smoother_snapshot.diagnostics.update(
                {
                    "feedback_correction_norm_m": float(max_feedback_correction_norm_seen),
                    "max_feedback_correction_norm_m": float(max_feedback_correction_norm_m),
                    "trusted_feedback_count": float(trusted_feedback_count),
                    "total_feedback_candidates": float(total_feedback_candidates),
                    "anchored_window": bool(anchored_window),
                    "residual_improved": bool(residual_improved),
                    "covariance_contracted": bool(covariance_contracted),
                }
            )
            self._writer.write_smoother_state(smoother_snapshot)
            self._counts["smoother_states"] += 1

    def _control_config(self) -> dict[str, Any]:
        return self.config.config_payloads["control"]

    def _workspace_bounds(self, key: str) -> tuple[np.ndarray, np.ndarray] | None:
        workspace = self._control_config().get(key)
        if not isinstance(workspace, dict):
            return None
        lower = workspace.get("min")
        upper = workspace.get("max")
        if not isinstance(lower, (list, tuple)) or not isinstance(upper, (list, tuple)):
            return None
        if len(lower) < 3 or len(upper) < 3:
            return None
        return (
            np.asarray([float(lower[0]), float(lower[1]), float(lower[2])], dtype=np.float64),
            np.asarray([float(upper[0]), float(upper[1]), float(upper[2])], dtype=np.float64),
        )

    def _control_workspace_bounds(self) -> tuple[np.ndarray, np.ndarray] | None:
        return self._workspace_bounds("workspace_aabb_world_m")

    def _ik_workspace_bounds(self) -> tuple[np.ndarray, np.ndarray] | None:
        return self._workspace_bounds("ik_workspace_aabb_world_m")

    def _clamp_to_workspace(self, position_world_m: np.ndarray) -> np.ndarray:
        candidate = np.asarray(position_world_m, dtype=np.float64).reshape(3)
        bounds = self._control_workspace_bounds()
        if bounds is None:
            return candidate
        lower, upper = bounds
        return np.clip(candidate, lower, upper)

    def _clamp_ik_target(self, position_world_m: np.ndarray) -> np.ndarray:
        candidate = np.asarray(position_world_m, dtype=np.float64).reshape(3)
        bounds = self._ik_workspace_bounds()
        if bounds is None:
            return candidate
        lower, upper = bounds
        return np.clip(candidate, lower, upper)

    def _limit_step_delta(self, delta_world_m: np.ndarray, *, max_step_m: float | None = None) -> np.ndarray:
        candidate = np.asarray(delta_world_m, dtype=np.float64).reshape(3)
        configured_max_step_m = float(self._control_config().get("max_position_step_m", 0.02)) if max_step_m is None else float(max_step_m)
        norm = float(np.linalg.norm(candidate))
        if norm <= configured_max_step_m or norm < 1e-12:
            return candidate
        return candidate * (configured_max_step_m / norm)

    def _limit_joint_target_step(
        self,
        current_joint_positions: np.ndarray,
        joint_targets: np.ndarray,
        *,
        max_step_deg: float | None = None,
    ) -> np.ndarray:
        if max_step_deg in (None, ""):
            return np.asarray(joint_targets, dtype=np.float64).reshape(-1)
        max_step_rad = np.radians(float(max_step_deg))
        if max_step_rad <= 0.0:
            return np.asarray(joint_targets, dtype=np.float64).reshape(-1)
        current = np.asarray(current_joint_positions, dtype=np.float64).reshape(-1)
        target = np.asarray(joint_targets, dtype=np.float64).reshape(-1)
        limited = current.copy()
        mapped_joint_count = min(len(current), len(target))
        if mapped_joint_count <= 0:
            return limited
        limited[:mapped_joint_count] = current[:mapped_joint_count] + np.clip(
            target[:mapped_joint_count] - current[:mapped_joint_count],
            -max_step_rad,
            max_step_rad,
        )
        if mapped_joint_count < len(target):
            limited = np.concatenate([limited[:mapped_joint_count], target[mapped_joint_count:]])
        return limited

    def _control_target_to_ik_target(
        self,
        *,
        current_control_position_world_m: np.ndarray,
        desired_control_position_world_m: np.ndarray,
        current_ee_position_world_m: np.ndarray,
    ) -> np.ndarray:
        control_position = np.asarray(current_control_position_world_m, dtype=np.float64).reshape(3)
        desired_control_position = np.asarray(desired_control_position_world_m, dtype=np.float64).reshape(3)
        current_ee_position = np.asarray(current_ee_position_world_m, dtype=np.float64).reshape(3)
        control_delta_world_m = desired_control_position - control_position
        return current_ee_position + control_delta_world_m

    def _current_control_position_world_m(self, *, ee_position_world_m: np.ndarray) -> np.ndarray:
        if self._camera_binding is not None:
            try:
                position_world_m, _ = self._camera_binding.get_world_pose()
                return np.asarray(position_world_m, dtype=np.float64).reshape(3)
            except Exception:
                pass
        return np.asarray(ee_position_world_m, dtype=np.float64).reshape(3)

    def _current_control_pose_world(
        self,
        *,
        ee_position_world_m: np.ndarray,
        ee_orientation_wxyz: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._camera_binding is not None:
            try:
                position_world_m, orientation_wxyz = self._camera_binding.get_world_pose()
                return (
                    np.asarray(position_world_m, dtype=np.float64).reshape(3),
                    np.asarray(orientation_wxyz, dtype=np.float64).reshape(4),
                )
            except Exception:
                pass
        return (
            np.asarray(ee_position_world_m, dtype=np.float64).reshape(3),
            np.asarray(ee_orientation_wxyz, dtype=np.float64).reshape(4),
        )

    def _auto_demo_camera_target_orientation_wxyz(
        self,
        *,
        target_camera_position_world_m: np.ndarray,
        look_at_world_m: np.ndarray,
        current_control_position_world_m: np.ndarray | None = None,
        current_control_orientation_wxyz: np.ndarray | None = None,
    ) -> np.ndarray:
        del current_control_position_world_m
        up_hint_world_m = None
        if current_control_orientation_wxyz is not None:
            current_rotation_wc = _quaternion_wxyz_to_rotation_matrix(
                np.asarray(current_control_orientation_wxyz, dtype=np.float64).reshape(4)
            )
            up_hint_world_m = np.asarray(current_rotation_wc[:, 1], dtype=np.float64).reshape(3)
        return _look_at_orientation_wxyz(
            target_camera_position_world_m,
            look_at_world_m,
            up_hint_world_m=up_hint_world_m,
        )

    def _predicted_camera_position_world_m(
        self,
        *,
        ee_position_world_m: np.ndarray,
        ee_orientation_wxyz: np.ndarray,
    ) -> np.ndarray:
        if self._camera_translation_ec_m is None:
            return self._current_control_position_world_m(ee_position_world_m=ee_position_world_m)
        ee_position = np.asarray(ee_position_world_m, dtype=np.float64).reshape(3)
        ee_rotation_we = _quaternion_wxyz_to_rotation_matrix(ee_orientation_wxyz)
        return ee_position + ee_rotation_we @ np.asarray(self._camera_translation_ec_m, dtype=np.float64).reshape(3)

    def _resolve_control_state(
        self,
        *,
        ee_position_world_m: np.ndarray,
    ) -> tuple[np.ndarray | None, str, bool]:
        current_control_position = self._current_control_position_world_m(ee_position_world_m=ee_position_world_m)
        if self.config.controller_mode == "open-loop":
            if self._last_open_loop_target_position is None:
                self._last_open_loop_target_position = current_control_position.copy()
            return self._last_open_loop_target_position.copy(), "open_loop_memory", False
        if self._counts["detections"] == 0:
            if self.config.allow_gt_debug_control or self.config.bootstrap_control_policy == "gt_until_first_detection":
                return current_control_position, "gt_debug", False
            return None, "bootstrap_hold", True
        if self.config.allow_gt_debug_control:
            return current_control_position, "gt_debug", False
        if self._latest_filter_snapshot is not None and self._filter is not None:
            return self._filter.state.position_world_m.copy(), "estimate", False
        return None, "bootstrap_hold", True

    def _solve_control_ik(
        self,
        *,
        current_position_world_m: np.ndarray,
        current_joint_positions: np.ndarray,
        desired_position_world_m: np.ndarray,
        current_orientation_wxyz: np.ndarray,
        desired_orientation_wxyz: np.ndarray | None = None,
        allow_orientation_relaxation: bool = True,
    ) -> tuple[np.ndarray, bool, float, str, np.ndarray]:
        if self._robot_binding is None:
            raise RuntimeError("Robot binding missing.")
        alphas = self._control_config().get("ik_retry_alphas", [1.0, 0.5, 0.25, 0.1, 0.0])
        fixed_orientation = (
            np.asarray(desired_orientation_wxyz, dtype=np.float64).reshape(4)
            if desired_orientation_wxyz is not None
            else (
                self._control_target_orientation_wxyz.copy()
                if self._control_target_orientation_wxyz is not None
                else np.asarray(current_orientation_wxyz, dtype=np.float64).reshape(4)
            )
        )
        orientation_candidates = [("fixed", fixed_orientation)]
        relaxed_orientation = np.asarray(current_orientation_wxyz, dtype=np.float64).reshape(4)
        if allow_orientation_relaxation and not np.allclose(relaxed_orientation, fixed_orientation):
            orientation_candidates.append(("relaxed_current", relaxed_orientation))
        desired_position = np.asarray(desired_position_world_m, dtype=np.float64).reshape(3)
        current_position = np.asarray(current_position_world_m, dtype=np.float64).reshape(3)
        delta = desired_position - current_position
        for orientation_policy, orientation in orientation_candidates:
            for alpha in alphas:
                candidate_position = self._clamp_ik_target(current_position + float(alpha) * delta)
                joint_targets, success = self._robot_binding.compute_joint_targets_from_pose(
                    target_position_world_m=candidate_position,
                    target_orientation_wxyz=orientation,
                )
                if success:
                    return joint_targets, True, float(alpha), orientation_policy, candidate_position
        return current_joint_positions.copy(), False, 0.0, orientation_candidates[-1][0], current_position.copy()

    def _table_surface_height_world_m(self) -> float:
        scene_payload = self.config.config_payloads.get("scene", {})
        for box in scene_payload.get("scene_boxes", []):
            if not isinstance(box, dict):
                continue
            if str(box.get("name", "")).strip().lower() != "table_top":
                continue
            center = np.asarray(box.get("center_world_m", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
            size = np.asarray(box.get("size_m", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
            return float(center[2] + 0.5 * size[2])
        center = np.asarray(scene_payload.get("table_center_world_m", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
        scale = np.asarray(scene_payload.get("table_scale_m", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
        return float(center[2] + 0.5 * scale[2])

    def _motion_program_bounds_world_m(
        self,
        motion_program: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        custom_bounds = motion_program.get("control_waypoint_bounds_world_m")
        if isinstance(custom_bounds, dict):
            lower = np.asarray(custom_bounds.get("min", [-0.2, 0.45, 0.88]), dtype=np.float64).reshape(3)
            upper = np.asarray(custom_bounds.get("max", [0.24, 0.82, 1.04]), dtype=np.float64).reshape(3)
        else:
            bounds = self._control_workspace_bounds()
            if bounds is None:
                lower = np.asarray([-0.2, 0.45, 0.88], dtype=np.float64)
                upper = np.asarray([0.24, 0.82, 1.04], dtype=np.float64)
            else:
                lower, upper = bounds
        table_clearance_m = float(motion_program.get("table_clearance_m", 0.14))
        lower = lower.copy()
        upper = upper.copy()
        lower[2] = max(lower[2], self._table_surface_height_world_m() + table_clearance_m)
        return lower, upper

    def _camera_half_fov_tangents(self) -> tuple[float, float]:
        if self._camera_binding is None:
            return float(np.tan(np.radians(36.0))), float(np.tan(np.radians(22.0)))
        spec = self._camera_binding.spec
        fx_px = float(spec.intrinsics.get("fx_px", 0.0))
        fy_px = float(spec.intrinsics.get("fy_px", 0.0))
        if fx_px <= 1e-9 or fy_px <= 1e-9:
            return float(np.tan(np.radians(36.0))), float(np.tan(np.radians(22.0)))
        return float(spec.width_px) / (2.0 * fx_px), float(spec.height_px) / (2.0 * fy_px)

    def _point_visible_from_look_at_pose(
        self,
        *,
        camera_position_world_m: np.ndarray,
        look_at_world_m: np.ndarray,
        point_world_m: np.ndarray,
        margin_scale: float,
        max_distance_m: float | None,
    ) -> bool:
        forward, left, up = _look_at_basis(camera_position_world_m, look_at_world_m)
        vector_world = np.asarray(point_world_m, dtype=np.float64).reshape(3) - np.asarray(camera_position_world_m, dtype=np.float64).reshape(3)
        depth = float(np.dot(vector_world, forward))
        if depth <= 1e-6:
            return False
        if max_distance_m is not None and depth > float(max_distance_m):
            return False
        tan_half_h, tan_half_v = self._camera_half_fov_tangents()
        horizontal = abs(float(np.dot(vector_world, left))) / depth
        vertical = abs(float(np.dot(vector_world, up))) / depth
        return horizontal <= margin_scale * tan_half_h and vertical <= margin_scale * tan_half_v

    def _sample_closed_bezier_camera_position(
        self,
        control_points_world_m: tuple[np.ndarray, ...],
        *,
        phase: float,
        tangent_scale: float,
    ) -> np.ndarray:
        if not control_points_world_m:
            raise ValueError("Bezier demo requires at least one control point.")
        if len(control_points_world_m) == 1:
            return np.asarray(control_points_world_m[0], dtype=np.float64).reshape(3)
        point_count = len(control_points_world_m)
        normalized_phase = float(phase) % 1.0
        scaled_phase = normalized_phase * point_count
        segment_index = int(np.floor(scaled_phase)) % point_count
        local_t = scaled_phase - np.floor(scaled_phase)
        p0 = np.asarray(control_points_world_m[(segment_index - 1) % point_count], dtype=np.float64).reshape(3)
        p1 = np.asarray(control_points_world_m[segment_index % point_count], dtype=np.float64).reshape(3)
        p2 = np.asarray(control_points_world_m[(segment_index + 1) % point_count], dtype=np.float64).reshape(3)
        p3 = np.asarray(control_points_world_m[(segment_index + 2) % point_count], dtype=np.float64).reshape(3)
        scale = float(max(tangent_scale, 0.0))
        b0 = p1
        b1 = p1 + scale * (p2 - p0)
        b2 = p2 - scale * (p3 - p1)
        b3 = p2
        return _cubic_bezier_point(b0, b1, b2, b3, local_t)

    def _build_bezier_task_space_motion_program_cache(self) -> dict[str, Any] | None:
        motion_program = self.config.config_payloads.get("scene", {}).get("motion_program", {})
        if not isinstance(motion_program, dict):
            return None
        if str(motion_program.get("mode", "")).strip().lower() != "bezier_task_space":
            return None
        explicit_control_points_world_m = motion_program.get("control_points_world_m")
        explicit_point_signature: tuple[tuple[float, float, float], ...] | None = None
        if isinstance(explicit_control_points_world_m, (list, tuple)):
            normalized_points: list[tuple[float, float, float]] = []
            for raw_point in explicit_control_points_world_m:
                try:
                    point = np.asarray(raw_point, dtype=np.float64).reshape(3)
                except Exception:
                    continue
                normalized_points.append((float(point[0]), float(point[1]), float(point[2])))
            if normalized_points:
                explicit_point_signature = tuple(normalized_points)
        if self._scene_motion_program_cache is not None:
            cached_signature = self._scene_motion_program_cache.get("signature")
            current_signature = (
                str(motion_program.get("mode", "")),
                float(motion_program.get("loop_duration_s", 20.0)),
                int(motion_program.get("seed", self.config.seed)),
                explicit_point_signature,
            )
            if cached_signature == current_signature:
                return self._scene_motion_program_cache
        lower, upper = self._motion_program_bounds_world_m(motion_program)
        entry_position_world_m = np.asarray(
            motion_program.get(
                "entry_control_position_world_m",
                [0.12, 0.76, max(self._table_surface_height_world_m() + 0.18, 0.94)],
            ),
            dtype=np.float64,
        ).reshape(3)
        entry_position_world_m = np.clip(entry_position_world_m, lower, upper)
        recovery_position_world_m = np.asarray(
            motion_program.get(
                "recovery_camera_position_world_m",
                entry_position_world_m.tolist(),
            ),
            dtype=np.float64,
        ).reshape(3)
        recovery_position_world_m = np.clip(recovery_position_world_m, lower, upper)
        look_at_world_m = np.asarray(
            motion_program.get("look_at_world_m", [0.10, 0.02, 1.02]),
            dtype=np.float64,
        ).reshape(3)
        recovery_look_at_world_m = np.asarray(
            motion_program.get("recovery_look_at_world_m", look_at_world_m.tolist()),
            dtype=np.float64,
        ).reshape(3)
        candidate_tag_ids = tuple(
            int(tag_id)
            for tag_id in motion_program.get("visibility_tag_ids", [19, 42, 0, 88])
        )
        visible_tag_positions_world_m = tuple(
            np.asarray(self._tag_pose_map[tag_id].position_world_m, dtype=np.float64).reshape(3)
            for tag_id in candidate_tag_ids
            if int(tag_id) in self._tag_pose_map
        )
        if not visible_tag_positions_world_m:
            visible_tag_positions_world_m = tuple(
                np.asarray(tag.position_world_m, dtype=np.float64).reshape(3)
                for tag in self._tag_pose_map.values()
            )
        tangent_scale = float(motion_program.get("tangent_scale", 0.12))
        margin_scale = float(motion_program.get("visibility_margin_scale", 0.82))
        max_visibility_distance_m = motion_program.get("max_visibility_distance_m")
        max_visibility_distance_m = None if max_visibility_distance_m in (None, "") else float(max_visibility_distance_m)
        random_point_count = max(int(motion_program.get("random_point_count", 4)), 2)
        min_pairwise_distance_m = float(motion_program.get("min_pairwise_distance_m", 0.12))
        sampling_attempts = max(int(motion_program.get("sampling_attempts", 48)), 1)
        point_sampling_draw_limit = max(
            int(motion_program.get("point_sampling_draw_limit", 192)),
            random_point_count + 1,
        )
        visibility_samples_per_loop = max(int(motion_program.get("visibility_samples_per_loop", 120)), 24)
        rng = np.random.default_rng(int(motion_program.get("seed", self.config.seed)))

        if isinstance(explicit_control_points_world_m, (list, tuple)) and len(explicit_control_points_world_m) >= 3:
            parsed_points: list[np.ndarray] = []
            for raw_point in explicit_control_points_world_m:
                try:
                    point = np.asarray(raw_point, dtype=np.float64).reshape(3)
                except Exception:
                    continue
                parsed_points.append(np.clip(point, lower, upper))
            if len(parsed_points) >= 3:
                chosen_points = tuple(parsed_points)
                self._scene_motion_program_cache = {
                    "signature": (
                        str(motion_program.get("mode", "")),
                        float(motion_program.get("loop_duration_s", 20.0)),
                        int(motion_program.get("seed", self.config.seed)),
                        explicit_point_signature,
                    ),
                    "control_points_world_m": chosen_points,
                    "entry_control_position_world_m": chosen_points[0].copy(),
                    "recovery_camera_position_world_m": recovery_position_world_m.copy(),
                    "look_at_world_m": look_at_world_m,
                    "recovery_look_at_world_m": recovery_look_at_world_m,
                    "tangent_scale": tangent_scale,
                    "loop_duration_s": max(float(motion_program.get("loop_duration_s", 24.0)), 1e-6),
                }
                return self._scene_motion_program_cache

        def _is_valid_curve(points: tuple[np.ndarray, ...]) -> bool:
            for sample_index in range(visibility_samples_per_loop):
                sample_phase = float(sample_index) / float(visibility_samples_per_loop)
                camera_position_world_m = self._sample_closed_bezier_camera_position(
                    points,
                    phase=sample_phase,
                    tangent_scale=tangent_scale,
                )
                if np.any(camera_position_world_m < lower - 1e-6) or np.any(camera_position_world_m > upper + 1e-6):
                    return False
                if camera_position_world_m[2] < self._table_surface_height_world_m() + float(motion_program.get("table_clearance_m", 0.14)) - 1e-6:
                    return False
                if not any(
                    self._point_visible_from_look_at_pose(
                        camera_position_world_m=camera_position_world_m,
                        look_at_world_m=look_at_world_m,
                        point_world_m=tag_position_world_m,
                        margin_scale=margin_scale,
                        max_distance_m=max_visibility_distance_m,
                    )
                    for tag_position_world_m in visible_tag_positions_world_m
                ):
                    return False
            return True

        chosen_points: tuple[np.ndarray, ...] | None = None
        for _ in range(sampling_attempts):
            points = [entry_position_world_m.copy()]
            for _draw_index in range(point_sampling_draw_limit):
                if len(points) >= random_point_count + 1:
                    break
                candidate = rng.uniform(lower, upper)
                if all(float(np.linalg.norm(candidate - existing)) >= min_pairwise_distance_m for existing in points):
                    points.append(candidate)
            if len(points) < random_point_count + 1:
                continue
            candidate_points = tuple(np.asarray(point, dtype=np.float64).reshape(3) for point in points)
            if _is_valid_curve(candidate_points):
                chosen_points = candidate_points
                break
        if chosen_points is None:
            fallback_points = (
                entry_position_world_m.copy(),
                np.asarray([0.24, 0.72, max(lower[2], 0.98)], dtype=np.float64),
                np.asarray([0.18, 0.58, max(lower[2], 0.90)], dtype=np.float64),
                np.asarray([0.00, 0.54, max(lower[2], 0.96)], dtype=np.float64),
                np.asarray([-0.04, 0.70, max(lower[2], 1.00)], dtype=np.float64),
            )
            chosen_points = tuple(np.clip(point, lower, upper) for point in fallback_points)
        self._scene_motion_program_cache = {
            "signature": (
                str(motion_program.get("mode", "")),
                float(motion_program.get("loop_duration_s", 20.0)),
                int(motion_program.get("seed", self.config.seed)),
                explicit_point_signature,
            ),
            "control_points_world_m": chosen_points,
            "entry_control_position_world_m": entry_position_world_m.copy(),
            "recovery_camera_position_world_m": recovery_position_world_m.copy(),
            "look_at_world_m": look_at_world_m,
            "recovery_look_at_world_m": recovery_look_at_world_m,
            "tangent_scale": tangent_scale,
            "loop_duration_s": max(float(motion_program.get("loop_duration_s", 24.0)), 1e-6),
        }
        return self._scene_motion_program_cache

    def _camera_target_to_end_effector_target(
        self,
        *,
        target_camera_position_world_m: np.ndarray,
        target_camera_orientation_wxyz: np.ndarray,
        current_ee_position_world_m: np.ndarray,
        current_ee_orientation_wxyz: np.ndarray | None = None,
        camera_rotation_ec: np.ndarray | None = None,
        camera_translation_ec_m: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        target_camera_position = np.asarray(target_camera_position_world_m, dtype=np.float64).reshape(3)
        target_camera_rotation_wc = _quaternion_wxyz_to_rotation_matrix(target_camera_orientation_wxyz)
        resolved_camera_rotation_ec = None if camera_rotation_ec is None else np.asarray(camera_rotation_ec, dtype=np.float64).reshape(3, 3)
        resolved_camera_translation_ec_m = None if camera_translation_ec_m is None else np.asarray(camera_translation_ec_m, dtype=np.float64).reshape(3)
        if current_ee_orientation_wxyz is not None:
            live_rotation_ec, live_translation_ec_m = self._camera_mount_relation_from_live_poses(
                ee_position_world_m=np.asarray(current_ee_position_world_m, dtype=np.float64).reshape(3),
                ee_orientation_wxyz=np.asarray(current_ee_orientation_wxyz, dtype=np.float64).reshape(4),
            )
            if live_rotation_ec is not None and live_translation_ec_m is not None:
                resolved_camera_rotation_ec = np.asarray(live_rotation_ec, dtype=np.float64).reshape(3, 3)
                resolved_camera_translation_ec_m = np.asarray(live_translation_ec_m, dtype=np.float64).reshape(3)
        if resolved_camera_rotation_ec is None:
            resolved_camera_rotation_ec = self._camera_rotation_ec
        if resolved_camera_translation_ec_m is None:
            resolved_camera_translation_ec_m = self._camera_translation_ec_m
        if resolved_camera_rotation_ec is None:
            target_ee_orientation_wxyz = (
                self._control_target_orientation_wxyz.copy()
                if self._control_target_orientation_wxyz is not None
                else np.asarray(target_camera_orientation_wxyz, dtype=np.float64).reshape(4)
            )
            if self._camera_binding is not None:
                current_camera_position_world_m, _ = self._camera_binding.get_world_pose()
                target_ee_position_world_m = (
                    np.asarray(current_ee_position_world_m, dtype=np.float64).reshape(3)
                    + target_camera_position
                    - np.asarray(current_camera_position_world_m, dtype=np.float64).reshape(3)
                )
            else:
                target_ee_position_world_m = np.asarray(current_ee_position_world_m, dtype=np.float64).reshape(3)
            return target_ee_position_world_m, target_ee_orientation_wxyz
        target_ee_rotation_we = target_camera_rotation_wc @ np.asarray(resolved_camera_rotation_ec, dtype=np.float64).reshape(3, 3).T
        target_ee_orientation_wxyz = _rotation_matrix_to_quaternion_wxyz(target_ee_rotation_we)
        if resolved_camera_translation_ec_m is None:
            target_ee_position_world_m = np.asarray(current_ee_position_world_m, dtype=np.float64).reshape(3)
        else:
            target_ee_position_world_m = target_camera_position - target_ee_rotation_we @ np.asarray(
                resolved_camera_translation_ec_m,
                dtype=np.float64,
            ).reshape(3)
        return target_ee_position_world_m, target_ee_orientation_wxyz

    def _scene_motion_program_command(
        self,
        *,
        current_joint_positions: np.ndarray,
        current_control_position_world_m: np.ndarray,
        current_ee_position_world_m: np.ndarray,
        current_ee_orientation_wxyz: np.ndarray,
    ) -> dict[str, Any] | None:
        motion_program = self.config.config_payloads.get("scene", {}).get("motion_program", {})
        if not isinstance(motion_program, dict):
            return None
        mode = str(motion_program.get("mode", "")).strip().lower()
        if mode == "keyframes":
            joint_targets = self._scene_motion_program_joint_targets(current_joint_positions)
            if joint_targets is None:
                return None
            return {"kind": "joint_targets", "joint_targets": joint_targets}
        if mode != "bezier_task_space":
            return None
        cache = self._build_bezier_task_space_motion_program_cache()
        if cache is None:
            return None
        loop_duration_s = float(cache["loop_duration_s"])
        controller_period_s = self._physics_dt_s if self._controller_clock is None else float(self._controller_clock.period_s)
        phase_step = 0.0 if loop_duration_s <= 1e-9 else controller_period_s / loop_duration_s
        look_at_world_m = np.asarray(cache["look_at_world_m"], dtype=np.float64).reshape(3)
        entry_control_position_world_m = np.asarray(cache["entry_control_position_world_m"], dtype=np.float64).reshape(3)
        entry_lock_tolerance_m = float(motion_program.get("entry_lock_tolerance_m", 0.03))
        entry_lock_orientation_tolerance_deg = float(motion_program.get("entry_lock_orientation_tolerance_deg", 6.0))
        tracking_error_gate_m = float(motion_program.get("phase_advance_tracking_error_threshold_m", 0.06))
        tracking_orientation_gate_deg = float(motion_program.get("phase_advance_orientation_error_threshold_deg", 8.0))
        current_control_position_world_m, current_control_orientation_wxyz = self._current_control_pose_world(
            ee_position_world_m=current_ee_position_world_m,
            ee_orientation_wxyz=current_ee_orientation_wxyz,
        )
        if not self._auto_demo_curve_ready:
            entry_orientation_wxyz = self._auto_demo_camera_target_orientation_wxyz(
                target_camera_position_world_m=entry_control_position_world_m,
                look_at_world_m=look_at_world_m,
                current_control_position_world_m=current_control_position_world_m,
                current_control_orientation_wxyz=current_control_orientation_wxyz,
            )
            entry_position_error_m = float(
                np.linalg.norm(entry_control_position_world_m - np.asarray(current_control_position_world_m, dtype=np.float64).reshape(3))
            )
            entry_orientation_error_deg = _quaternion_angle_deg(entry_orientation_wxyz, current_control_orientation_wxyz)
            if (
                entry_position_error_m <= entry_lock_tolerance_m
                and entry_orientation_error_deg <= entry_lock_orientation_tolerance_deg
            ):
                self._auto_demo_curve_ready = True
                self._auto_demo_curve_anchor_time_s = float(self._sim_time_s)
                self._auto_demo_curve_phase = 0.0
                self._auto_demo_last_curve_tracking_good = False
            else:
                target_camera_position_world_m = entry_control_position_world_m.copy()
                target_camera_orientation_wxyz = entry_orientation_wxyz
                target_ee_position_world_m, target_ee_orientation_wxyz = self._camera_target_to_end_effector_target(
                    target_camera_position_world_m=target_camera_position_world_m,
                    target_camera_orientation_wxyz=target_camera_orientation_wxyz,
                    current_ee_position_world_m=current_ee_position_world_m,
                    current_ee_orientation_wxyz=current_ee_orientation_wxyz,
                )
                return {
                    "kind": "camera_curve",
                    "target_camera_position_world_m": target_camera_position_world_m,
                    "target_camera_orientation_wxyz": target_camera_orientation_wxyz,
                    "target_ee_position_world_m": target_ee_position_world_m,
                    "target_ee_orientation_wxyz": target_ee_orientation_wxyz,
                    "look_at_world_m": look_at_world_m,
                    "curve_ready": False,
                    "recovery_active": False,
                    "tracking_error_gate_m": tracking_error_gate_m,
                    "tracking_orientation_gate_deg": tracking_orientation_gate_deg,
                }
        if len(self._last_detections) <= 0:
            self._auto_demo_visibility_miss_count += 1
        else:
            self._auto_demo_visibility_miss_count = 0
        recovery_enabled = bool(motion_program.get("visibility_recovery_enabled", True))
        recovery_after_missing_steps = max(int(motion_program.get("visibility_recovery_missing_steps", 3)), 1)
        recovery_active = recovery_enabled and self._auto_demo_visibility_miss_count >= recovery_after_missing_steps
        if not recovery_active and self._auto_demo_last_curve_tracking_good and phase_step > 0.0:
            self._auto_demo_curve_phase = (float(self._auto_demo_curve_phase) + float(phase_step)) % 1.0
        if recovery_active:
            target_camera_position_world_m = np.asarray(cache["recovery_camera_position_world_m"], dtype=np.float64).reshape(3)
            look_at_world_m = np.asarray(cache["recovery_look_at_world_m"], dtype=np.float64).reshape(3)
        else:
            target_camera_position_world_m = self._sample_closed_bezier_camera_position(
                cache["control_points_world_m"],
                phase=float(self._auto_demo_curve_phase),
                tangent_scale=float(cache["tangent_scale"]),
            )
        target_camera_orientation_wxyz = self._auto_demo_camera_target_orientation_wxyz(
            target_camera_position_world_m=target_camera_position_world_m,
            look_at_world_m=look_at_world_m,
            current_control_position_world_m=current_control_position_world_m,
            current_control_orientation_wxyz=current_control_orientation_wxyz,
        )
        target_ee_position_world_m, target_ee_orientation_wxyz = self._camera_target_to_end_effector_target(
            target_camera_position_world_m=target_camera_position_world_m,
            target_camera_orientation_wxyz=target_camera_orientation_wxyz,
            current_ee_position_world_m=current_ee_position_world_m,
            current_ee_orientation_wxyz=current_ee_orientation_wxyz,
        )
        return {
            "kind": "camera_curve",
            "target_camera_position_world_m": target_camera_position_world_m,
            "target_camera_orientation_wxyz": target_camera_orientation_wxyz,
            "target_ee_position_world_m": target_ee_position_world_m,
            "target_ee_orientation_wxyz": target_ee_orientation_wxyz,
            "look_at_world_m": look_at_world_m,
            "curve_ready": True,
            "recovery_active": recovery_active,
            "tracking_error_gate_m": tracking_error_gate_m,
            "tracking_orientation_gate_deg": tracking_orientation_gate_deg,
        }

    def _scene_motion_program_joint_targets(self, current_joint_positions: np.ndarray) -> np.ndarray | None:
        motion_program = self.config.config_payloads.get("scene", {}).get("motion_program", {})
        if not isinstance(motion_program, dict):
            return None
        if str(motion_program.get("mode", "")).strip().lower() != "keyframes":
            return None
        raw_keyframes = motion_program.get("keyframes", [])
        if not isinstance(raw_keyframes, list) or not raw_keyframes:
            return None
        keyframes: list[tuple[float, str, np.ndarray]] = []
        for entry in raw_keyframes:
            if not isinstance(entry, dict):
                continue
            time_s = float(entry.get("time_s", 0.0))
            if "joint_targets_deg" in entry or "joint_positions_deg" in entry:
                joint_targets_deg = np.asarray(
                    entry.get("joint_targets_deg", entry.get("joint_positions_deg", [])),
                    dtype=np.float64,
                ).reshape(-1)
                if joint_targets_deg.size > 0:
                    keyframes.append((time_s, "joint_targets_deg", joint_targets_deg))
                continue
            legacy_servos_deg = np.asarray(entry.get("servos_deg", []), dtype=np.float64).reshape(-1)
            if legacy_servos_deg.size > 0:
                keyframes.append((time_s, "legacy_servo_offsets_deg", legacy_servos_deg))
        if not keyframes:
            return None
        ordered_keyframes = sorted(keyframes, key=lambda item: item[0])
        loop_duration_s = max(
            float(motion_program.get("loop_duration_s", ordered_keyframes[-1][0])),
            float(ordered_keyframes[-1][0]),
            1e-6,
        )
        phase_s = float(self._sim_time_s) % loop_duration_s
        previous_time_s, previous_mode, previous_targets_deg = ordered_keyframes[-1]
        next_time_s, next_mode, next_targets_deg = ordered_keyframes[0]
        for candidate_time_s, candidate_mode, candidate_targets_deg in ordered_keyframes:
            if phase_s < candidate_time_s:
                next_time_s, next_mode, next_targets_deg = candidate_time_s, candidate_mode, candidate_targets_deg
                break
            previous_time_s, previous_mode, previous_targets_deg = candidate_time_s, candidate_mode, candidate_targets_deg
        else:
            next_time_s, next_mode, next_targets_deg = ordered_keyframes[0]
        if next_time_s <= previous_time_s:
            interval = max(loop_duration_s - previous_time_s + next_time_s, 1e-6)
            alpha = (phase_s - previous_time_s) / interval if phase_s >= previous_time_s else (loop_duration_s - previous_time_s + phase_s) / interval
        else:
            interval = max(next_time_s - previous_time_s, 1e-6)
            alpha = np.clip((phase_s - previous_time_s) / interval, 0.0, 1.0)
        if previous_mode == next_mode and previous_targets_deg.shape == next_targets_deg.shape:
            target_mode = previous_mode
            interpolated_targets_deg = previous_targets_deg + float(alpha) * (next_targets_deg - previous_targets_deg)
        elif float(alpha) < 0.5:
            target_mode = previous_mode
            interpolated_targets_deg = previous_targets_deg
        else:
            target_mode = next_mode
            interpolated_targets_deg = next_targets_deg
        if self._home_joint_positions is None:
            self._home_joint_positions = np.asarray(current_joint_positions, dtype=np.float64).reshape(-1).copy()
        joint_targets = np.asarray(self._home_joint_positions, dtype=np.float64).reshape(-1).copy()
        if target_mode == "joint_targets_deg":
            mapped_joint_count = int(min(len(joint_targets), len(interpolated_targets_deg)))
            if mapped_joint_count <= 0:
                return joint_targets
            joint_targets[:mapped_joint_count] = np.radians(interpolated_targets_deg[:mapped_joint_count])
            return joint_targets
        robot_config = self.config.config_payloads.get("robot", {})
        reference_initial_servos_deg = np.asarray(
            robot_config.get("replica_reference_initial_servos_deg", ordered_keyframes[0][2].tolist()),
            dtype=np.float64,
        ).reshape(-1)
        mapped_joint_count = int(min(3, len(joint_targets), len(interpolated_targets_deg), len(reference_initial_servos_deg)))
        if mapped_joint_count <= 0:
            return joint_targets
        joint_targets[:mapped_joint_count] = joint_targets[:mapped_joint_count] + np.radians(
            interpolated_targets_deg[:mapped_joint_count] - reference_initial_servos_deg[:mapped_joint_count]
        )
        return joint_targets

    def _apply_control(self, ee_position_world_m: np.ndarray, ee_orientation_wxyz: np.ndarray) -> None:
        if self._tracker is None or self._controller_clock is None or self._robot_binding is None:
            return
        from isaacsim.core.utils.types import ArticulationAction

        for _ in self._controller_clock.advance_to(self._sim_time_s):
            current_joint_positions = self._robot_binding.get_joint_positions()
            current_control_position, current_control_orientation_wxyz = self._current_control_pose_world(
                ee_position_world_m=np.asarray(ee_position_world_m, dtype=np.float64).reshape(3),
                ee_orientation_wxyz=np.asarray(ee_orientation_wxyz, dtype=np.float64).reshape(4),
            )
            current_position, current_position_source, hold_for_bootstrap = self._resolve_control_state(
                ee_position_world_m=ee_position_world_m
            )
            position_radius_95_m = 0.05 if self._latest_uncertainty is None else float(self._latest_uncertainty.position_radius_95_m)
            innovation_norm = 0.0 if self._filter is None else float(self._filter.last_innovation_norm)
            degraded_after = int(self._control_config().get("degrade_after_consecutive_ik_failures", 5))
            degraded_gain_scale = float(self._control_config().get("degraded_gain_scale", 0.5))
            degraded_mode_active = self._controller_consecutive_ik_failures >= degraded_after
            interactive_mode = self._interactive_control_mode
            path_command = None
            diagnostic_desired_position = current_control_position.copy()
            actual_control_position = current_control_position.copy()
            actual_ee_position = np.asarray(ee_position_world_m, dtype=np.float64).reshape(3).copy()
            target_ee_position_for_diagnostic = np.asarray(ee_position_world_m, dtype=np.float64).reshape(3).copy()
            camera_orientation_error_deg: float | None = None
            recovery_active = False
            if interactive_mode == "manual_joint" and self._manual_joint_target_positions is not None:
                tracked_target_position = current_control_position.copy()
                ik_target_position = np.asarray(ee_position_world_m, dtype=np.float64).reshape(3)
                target_ee_position_for_diagnostic = ik_target_position.copy()
                joint_targets = self._manual_joint_target_positions.copy()
                ik_success = True
                ik_retry_alpha = 1.0
                orientation_policy = "manual_joint"
                joint_target_delta_norm = float(np.linalg.norm(joint_targets - current_joint_positions))
                min_joint_limit_margin = self._robot_binding.joint_limit_margin(joint_targets)
                command_delta_norm_m = 0.0
                tracking_error_norm_m = 0.0
                current_position_source = "manual_joint"
                safety_reason = "manual_joint"
            elif interactive_mode == "manual_task":
                tracked_target_position = (
                    current_control_position.copy()
                    if self._manual_task_target_world_m is None
                    else self._manual_task_target_world_m.copy()
                )
                joint_targets, ik_success, ik_retry_alpha, orientation_policy, ik_target_position = self._solve_control_ik(
                    current_position_world_m=np.asarray(ee_position_world_m, dtype=np.float64).reshape(3),
                    current_joint_positions=current_joint_positions,
                    desired_position_world_m=self._control_target_to_ik_target(
                        current_control_position_world_m=current_control_position,
                        desired_control_position_world_m=tracked_target_position,
                        current_ee_position_world_m=ee_position_world_m,
                    ),
                    current_orientation_wxyz=ee_orientation_wxyz,
                )
                target_ee_position_for_diagnostic = np.asarray(ik_target_position, dtype=np.float64).reshape(3).copy()
                joint_target_delta_norm = float(np.linalg.norm(joint_targets - current_joint_positions))
                min_joint_limit_margin = self._robot_binding.joint_limit_margin(joint_targets)
                command_delta_norm_m = float(np.linalg.norm(tracked_target_position - current_control_position))
                tracking_error_norm_m = float(np.linalg.norm(tracked_target_position - current_control_position))
                current_position_source = "manual_task"
                safety_reason = "manual_task" if ik_success else "manual_task_ik_hold"
                if not ik_success:
                    joint_targets = current_joint_positions.copy()
                    ik_target_position = np.asarray(ee_position_world_m, dtype=np.float64).reshape(3)
            elif interactive_mode == "auto_demo":
                demo_command = self._scene_motion_program_command(
                    current_joint_positions=current_joint_positions,
                    current_control_position_world_m=current_control_position,
                    current_ee_position_world_m=np.asarray(ee_position_world_m, dtype=np.float64).reshape(3),
                    current_ee_orientation_wxyz=np.asarray(ee_orientation_wxyz, dtype=np.float64).reshape(4),
                )
                if demo_command is None:
                    interactive_mode = "auto_path"
                elif str(demo_command.get("kind")) == "joint_targets":
                    joint_targets = np.asarray(demo_command["joint_targets"], dtype=np.float64).reshape(-1)
                    motion_program = self.config.config_payloads.get("scene", {}).get("motion_program", {})
                    if not isinstance(motion_program, dict):
                        motion_program = {}
                    joint_targets = self._limit_joint_target_step(
                        current_joint_positions,
                        joint_targets,
                        max_step_deg=motion_program.get("max_joint_step_deg"),
                    )
                    tracked_target_position = current_control_position.copy()
                    ik_target_position = np.asarray(ee_position_world_m, dtype=np.float64).reshape(3)
                    target_ee_position_for_diagnostic = ik_target_position.copy()
                    ik_success = True
                    ik_retry_alpha = 1.0
                    orientation_policy = "motion_program"
                    joint_target_delta_norm = float(np.linalg.norm(joint_targets - current_joint_positions))
                    min_joint_limit_margin = self._robot_binding.joint_limit_margin(joint_targets)
                    command_delta_norm_m = joint_target_delta_norm
                    tracking_error_norm_m = 0.0
                    current_position_source = "motion_program"
                    safety_reason = "auto_demo"
                    self._auto_demo_last_curve_tracking_good = False
                else:
                    motion_program = self.config.config_payloads.get("scene", {}).get("motion_program", {})
                    if not isinstance(motion_program, dict):
                        motion_program = {}
                    raw_curve_target_position = np.asarray(
                        demo_command["target_camera_position_world_m"],
                        dtype=np.float64,
                    ).reshape(3)
                    diagnostic_desired_position = raw_curve_target_position.copy()
                    commanded_camera_position = (
                        np.asarray(self._auto_demo_commanded_camera_position_world_m, dtype=np.float64).reshape(3)
                        if self._auto_demo_commanded_camera_position_world_m is not None
                        else current_control_position.copy()
                    )
                    curve_delta = raw_curve_target_position - current_control_position
                    commanded_curve_delta = raw_curve_target_position - commanded_camera_position
                    tracked_target_position = self._clamp_to_workspace(
                        commanded_camera_position
                        + self._limit_step_delta(
                            commanded_curve_delta,
                            max_step_m=motion_program.get("max_tracking_step_m"),
                        )
                    )
                    raw_target_camera_orientation_wxyz = self._auto_demo_camera_target_orientation_wxyz(
                        target_camera_position_world_m=raw_curve_target_position,
                        look_at_world_m=np.asarray(demo_command.get("look_at_world_m", raw_curve_target_position), dtype=np.float64).reshape(3),
                        current_control_position_world_m=current_control_position,
                        current_control_orientation_wxyz=current_control_orientation_wxyz,
                    )
                    commanded_camera_orientation_wxyz = (
                        np.asarray(self._auto_demo_commanded_camera_orientation_wxyz, dtype=np.float64).reshape(4)
                        if self._auto_demo_commanded_camera_orientation_wxyz is not None
                        else current_control_orientation_wxyz.copy()
                    )
                    tracked_camera_orientation_wxyz = _limit_quaternion_step(
                        commanded_camera_orientation_wxyz,
                        raw_target_camera_orientation_wxyz,
                        max_step_deg=motion_program.get("max_orientation_step_deg"),
                    )
                    desired_camera_orientation_error_deg = _quaternion_angle_deg(
                        raw_target_camera_orientation_wxyz,
                        current_control_orientation_wxyz,
                    )
                    desired_ee_position_world_m, desired_ee_orientation_wxyz = self._camera_target_to_end_effector_target(
                        target_camera_position_world_m=tracked_target_position,
                        target_camera_orientation_wxyz=tracked_camera_orientation_wxyz,
                        current_ee_position_world_m=np.asarray(ee_position_world_m, dtype=np.float64).reshape(3),
                        current_ee_orientation_wxyz=np.asarray(ee_orientation_wxyz, dtype=np.float64).reshape(4),
                    )
                    desired_ee_position_world_m = np.asarray(desired_ee_position_world_m, dtype=np.float64).reshape(3)
                    desired_ee_orientation_wxyz = np.asarray(desired_ee_orientation_wxyz, dtype=np.float64).reshape(4)
                    target_ee_position_for_diagnostic = desired_ee_position_world_m.copy()
                    joint_targets, ik_success, ik_retry_alpha, orientation_policy, ik_target_position = self._solve_control_ik(
                        current_position_world_m=np.asarray(ee_position_world_m, dtype=np.float64).reshape(3),
                        current_joint_positions=current_joint_positions,
                        desired_position_world_m=desired_ee_position_world_m,
                        current_orientation_wxyz=ee_orientation_wxyz,
                        desired_orientation_wxyz=desired_ee_orientation_wxyz,
                        allow_orientation_relaxation=bool(motion_program.get("allow_orientation_relaxation", False)),
                    )
                    joint_targets = self._limit_joint_target_step(
                        current_joint_positions,
                        joint_targets,
                        max_step_deg=motion_program.get("max_joint_step_deg"),
                    )
                    joint_target_delta_norm = float(np.linalg.norm(joint_targets - current_joint_positions))
                    min_joint_limit_margin = self._robot_binding.joint_limit_margin(joint_targets)
                    command_delta_norm_m = float(np.linalg.norm(tracked_target_position - current_control_position))
                    tracking_error_norm_m = float(np.linalg.norm(raw_curve_target_position - current_control_position))
                    recovery_active = bool(demo_command.get("recovery_active", False))
                    camera_orientation_error_deg = _quaternion_angle_deg(
                        raw_target_camera_orientation_wxyz,
                        current_control_orientation_wxyz,
                    )
                    current_position_source = "motion_program_bezier_recovery" if recovery_active else "motion_program_bezier"
                    safety_reason = "auto_demo_bezier_recovery" if recovery_active else "auto_demo_bezier"
                    if not ik_success:
                        safety_reason = f"{safety_reason}_ik_hold"
                    if not ik_success:
                        joint_targets = current_joint_positions.copy()
                        ik_target_position = np.asarray(ee_position_world_m, dtype=np.float64).reshape(3)
                    else:
                        self._auto_demo_commanded_camera_position_world_m = tracked_target_position.copy()
                        self._auto_demo_commanded_camera_orientation_wxyz = tracked_camera_orientation_wxyz.copy()
                    self._auto_demo_last_curve_tracking_good = bool(
                        demo_command.get("curve_ready", False)
                        and not recovery_active
                        and ik_success
                        and tracking_error_norm_m <= float(demo_command.get("tracking_error_gate_m", 0.06))
                        and desired_camera_orientation_error_deg
                        <= float(demo_command.get("tracking_orientation_gate_deg", 8.0))
                    )
            if interactive_mode in (None, "auto_path"):
                self._auto_demo_last_curve_tracking_good = False
                if hold_for_bootstrap or current_position is None:
                    tracked_target_position = current_control_position.copy()
                    ik_target_position = tracked_target_position.copy()
                    target_ee_position_for_diagnostic = ik_target_position.copy()
                    joint_targets = current_joint_positions.copy()
                    ik_success = True
                    ik_retry_alpha = 0.0
                    orientation_policy = "hold"
                    joint_target_delta_norm = 0.0
                    min_joint_limit_margin = self._robot_binding.joint_limit_margin(joint_targets)
                    command_delta_norm_m = 0.0
                    tracking_error_norm_m = 0.0
                    safety_reason = "bootstrap_hold"
                else:
                    current_position = np.asarray(current_position, dtype=np.float64).reshape(3)
                    path_command = self._tracker.compute_control(
                        current_position_world_m=current_position,
                        position_radius_95_m=position_radius_95_m,
                        innovation_norm=innovation_norm,
                        anchor_visible=bool(self._last_anchor_visible or self.config.controller_mode == "open-loop"),
                    )
                    command_delta = np.asarray(path_command.command_delta_world_m, dtype=np.float64)
                    if degraded_mode_active:
                        command_delta = command_delta * degraded_gain_scale
                    command_delta = self._limit_step_delta(command_delta)
                    tracked_target_position = self._clamp_to_workspace(current_position + command_delta)
                    ik_target_position = self._control_target_to_ik_target(
                        current_control_position_world_m=current_position,
                        desired_control_position_world_m=tracked_target_position,
                        current_ee_position_world_m=ee_position_world_m,
                    )
                    joint_targets, ik_success, ik_retry_alpha, orientation_policy, ik_target_position = self._solve_control_ik(
                        current_position_world_m=np.asarray(ee_position_world_m, dtype=np.float64).reshape(3),
                        current_joint_positions=current_joint_positions,
                        desired_position_world_m=ik_target_position,
                        current_orientation_wxyz=ee_orientation_wxyz,
                    )
                    target_ee_position_for_diagnostic = np.asarray(ik_target_position, dtype=np.float64).reshape(3).copy()
                    if not ik_success:
                        self._ik_failure_count += 1
                        self._controller_consecutive_ik_failures += 1
                        joint_targets = current_joint_positions.copy()
                        tracked_target_position = current_position.copy()
                        ik_target_position = np.asarray(ee_position_world_m, dtype=np.float64).reshape(3)
                        target_ee_position_for_diagnostic = ik_target_position.copy()
                        safety_reason = "ik_hold"
                    else:
                        self._controller_consecutive_ik_failures = 0
                        safety_reason = path_command.safety_reason if not degraded_mode_active else "ik_degraded"
                    joint_target_delta_norm = float(np.linalg.norm(joint_targets - current_joint_positions))
                    min_joint_limit_margin = self._robot_binding.joint_limit_margin(joint_targets)
                    command_delta_norm_m = float(np.linalg.norm(command_delta))
                    tracking_error_norm_m = float(np.linalg.norm(np.asarray(path_command.tracking_error_world_m, dtype=np.float64)))
            elif interactive_mode not in ("manual_joint", "manual_task", "auto_demo"):
                self._auto_demo_last_curve_tracking_good = False
                tracked_target_position = current_control_position.copy()
                ik_target_position = np.asarray(ee_position_world_m, dtype=np.float64).reshape(3)
                target_ee_position_for_diagnostic = ik_target_position.copy()
                joint_targets = current_joint_positions.copy()
                ik_success = True
                ik_retry_alpha = 0.0
                orientation_policy = "hold"
                joint_target_delta_norm = 0.0
                min_joint_limit_margin = self._robot_binding.joint_limit_margin(joint_targets)
                command_delta_norm_m = 0.0
                tracking_error_norm_m = 0.0
                safety_reason = "hold"
            effective_positions: list[float] = []
            servo_state: dict[str, Any] = {}
            dropped_command = False
            dt_control_s = float(self._controller_clock.period_s)
            if interactive_mode == "manual_joint":
                for joint_name, desired_joint_position, servo_model in zip(
                    self._robot_binding.robot.joint_names,
                    joint_targets[: len(self._robot_binding.robot.joint_names)],
                    self._servo_models,
                ):
                    desired_value = float(desired_joint_position)
                    servo_model.position = desired_value
                    servo_model.velocity = 0.0
                    effective_positions.append(desired_value)
                    servo_state[joint_name] = {
                        "effective_command": desired_value,
                        "target": desired_value,
                        "dropped": False,
                        "realized_velocity": 0.0,
                    }
            else:
                for joint_name, desired_joint_position, servo_model in zip(
                    self._robot_binding.robot.joint_names,
                    joint_targets[: len(self._robot_binding.robot.joint_names)],
                    self._servo_models,
                ):
                    step = servo_model.step(command=float(desired_joint_position), dt_s=dt_control_s)
                    effective_positions.append(float(step.realized_position))
                    dropped_command = dropped_command or bool(step.dropped)
                    servo_state[joint_name] = {
                        "effective_command": float(step.effective_command),
                        "target": float(step.target),
                        "dropped": bool(step.dropped),
                        "realized_velocity": float(step.realized_velocity),
                    }
            action = ArticulationAction(joint_positions=np.asarray(effective_positions, dtype=np.float64))
            self._robot_binding.articulation.apply_action(action)
            command_packet = IsaacJointCommandPacket(
                timestamp_s=float(self._sim_time_s),
                sim_time_s=float(self._sim_time_s),
                joint_names=self._robot_binding.robot.joint_names,
                desired_positions=tuple(float(value) for value in joint_targets[: len(self._robot_binding.robot.joint_names)].tolist()),
                effective_positions=tuple(float(value) for value in effective_positions),
                estimator_mode=self.config.estimator_mode,
                controller_mode=self.config.controller_mode,
                waypoint_index=int(self._tracker.waypoint_manager.current_index),
                safety_reason=safety_reason,
                tracking_error_world_m=(0.0, 0.0, 0.0)
                if path_command is None
                else tuple(float(value) for value in path_command.tracking_error_world_m),
                dropped_command=bool(dropped_command),
            )
            self._latest_command_packet = command_packet
            self._writer.write_command(command_packet)
            diagnostic_packet = IsaacControllerDiagnosticPacket(
                timestamp_s=float(self._sim_time_s),
                sim_time_s=float(self._sim_time_s),
                estimator_mode=self.config.estimator_mode,
                controller_mode=self.config.controller_mode,
                waypoint_index=int(self._tracker.waypoint_manager.current_index),
                current_position_source=current_position_source,
                tracking_error_norm_m=float(tracking_error_norm_m),
                command_delta_norm_m=float(command_delta_norm_m),
                desired_position_world_m=tuple(float(value) for value in diagnostic_desired_position.tolist()),
                safety_reason=safety_reason,
                position_radius_95_m=float(position_radius_95_m),
                innovation_norm=float(innovation_norm),
                ik_success=bool(ik_success),
                ik_retry_alpha=float(ik_retry_alpha),
                joint_target_delta_norm=float(joint_target_delta_norm),
                min_joint_limit_margin=min_joint_limit_margin,
                dropped_command=bool(dropped_command),
                degraded_mode_active=bool(degraded_mode_active),
                orientation_policy=orientation_policy,
                actual_control_position_world_m=tuple(float(value) for value in actual_control_position.tolist()),
                actual_ee_position_world_m=tuple(float(value) for value in actual_ee_position.tolist()),
                ik_target_position_world_m=tuple(float(value) for value in np.asarray(ik_target_position, dtype=np.float64).reshape(3).tolist()),
                target_ee_position_world_m=tuple(float(value) for value in target_ee_position_for_diagnostic.tolist()),
                camera_orientation_error_deg=camera_orientation_error_deg,
                recovery_active=bool(recovery_active),
            )
            self._latest_controller_diagnostic = diagnostic_packet
            self._writer.write_controller_diagnostic(
                diagnostic_packet
            )
            self._counts["commands"] += 1
            self._counts["controller_diagnostics"] += 1
            self._last_open_loop_target_position = tracked_target_position.copy()

    def _step_once(self) -> None:
        if self._world is None:
            raise RuntimeError("Runtime world is not initialized.")
        next_sim_time_s = float(self.steps_executed + 1) * self._physics_dt_s
        camera_due = bool(self._camera_binding is not None and self._camera_binding.is_due(next_sim_time_s))
        due_observer_bindings = [binding for binding in self._observer_camera_bindings if binding.is_due(next_sim_time_s)]
        for binding in self._observer_camera_bindings:
            binding.set_updates_enabled(binding in due_observer_bindings)
        should_render = bool(camera_due or due_observer_bindings)
        self._world.step(render=bool(should_render))
        self.steps_executed += 1
        self._sim_time_s = float(self.steps_executed) * self._physics_dt_s
        timestamps = self._timestamps()
        _, _, ee_position_world_m, ee_orientation_wxyz = self._log_realized_state()
        imu_packets = self._process_imu(
            timestamps,
            ee_position_world_m=ee_position_world_m,
            ee_orientation_wxyz=ee_orientation_wxyz,
        )
        if self._filter is not None and imu_packets:
            suppression_active = self._anchor_updates_suppressed(timestamp_s=float(self._sim_time_s))
            self._propagate_filter_from_imu(
                imu_packets,
                suppression_active=bool(suppression_active),
            )
        else:
            self._last_imu_packets_used_for_prediction = 0
            self._last_imu_packets_rejected_for_prediction = 0
        self._process_camera(timestamps)
        self._log_filter_and_uncertainty()
        self._log_smoother()
        self._apply_control(ee_position_world_m, ee_orientation_wxyz)

    def step(self, steps: int = 1) -> None:
        if not self.started:
            raise RuntimeError("Isaac runtime has not been started.")
        for _ in range(max(int(steps), 0)):
            self._step_once()

    def run_until_done(self, max_steps: int | None = None) -> IsaacRunSummary:
        if not self.started:
            raise RuntimeError("Isaac runtime has not been started.")
        step_limit = self.config.max_steps if max_steps is None else int(max_steps)
        while self._sim_time_s < float(self.config.duration_s):
            if step_limit is not None and self.steps_executed >= int(step_limit):
                break
            self._step_once()
        summary = self.summary()
        self.persist_summary(summary)
        return summary

    def persist_summary(self, summary: IsaacRunSummary) -> IsaacRunSummary:
        self._finalized_summary = summary
        self._writer.finalize_summary(summary.as_json())
        return summary

    def summary(self) -> IsaacRunSummary:
        summary_counts = {
            key: max(int(value) - int(self._recording_count_baseline.get(key, 0)), 0)
            for key, value in self._counts.items()
        }
        summary_duration_s = max(float(self._sim_time_s) - float(self._recording_time_baseline_s), 0.0)
        summary_step_count = max(int(self.steps_executed) - int(self._recording_step_baseline), 0)
        complete = all(
            summary_counts[key] > 0
            for key in (
                "camera_frames",
                "imu_packets",
                "commands",
                "controller_diagnostics",
                "filter_states",
                "smoother_states",
                "uncertainty_states",
            )
        )
        warnings = list(dict.fromkeys(self._warnings))
        if self._ik_failure_count > 0:
            warnings.append(
                f"IK failed {self._ik_failure_count} controller ticks; the runtime held the current joint configuration on those updates."
            )
        return IsaacRunSummary(
            run_id=self.config.run_id,
            run_dir=str(Path(self.config.run_dir).resolve()),
            duration_s=float(summary_duration_s),
            step_count=int(summary_step_count),
            counts=summary_counts,
            warnings=warnings,
            complete=bool(complete),
        )

    def shutdown(self) -> None:
        if self._timeline is not None:
            self._timeline.stop()
        if self.started:
            summary = self._finalized_summary if self._finalized_summary is not None else self.summary()
            self._writer.finalize_summary(summary.as_json())
        self._runtime.shutdown()
        self.started = False

    def validate_stage_path(self) -> Path:
        if not self.config.stage_path:
            return Path(self.config.stage_path)
        return Path(self.config.stage_path)
