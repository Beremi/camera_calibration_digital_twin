"""Standalone Isaac runtime orchestration for the first live pass."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

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
)
from calib_sim.isaac.logging.writer import IsaacRunWriter
from calib_sim.isaac.robot_builder import LiveRobotBinding, build_robot_binding
from calib_sim.isaac.sensors import IsaacCameraBinding, IsaacImuBinding, camera_spec_from_config, imu_spec_from_config
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
    up = _normalize(up_hint - np.dot(up_hint, forward) * forward, default=np.array([0.0, 1.0, 0.0], dtype=np.float64))
    left = _normalize(np.cross(up, forward), default=np.array([0.0, 1.0, 0.0], dtype=np.float64))
    up = _normalize(np.cross(forward, left), default=np.array([0.0, 0.0, 1.0], dtype=np.float64))
    return _rotation_matrix_to_quaternion_wxyz(np.column_stack((forward, left, up)))


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
    use_ros2_bridge: bool = False
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
        return RuntimeConfig(headless=self.headless, width=self.width, height=self.height)

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
            "use_ros2_bridge": bool(self.use_ros2_bridge),
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
        self._last_imu_packets_used_for_prediction = 0
        self._last_imu_packets_rejected_for_prediction = 0

    @property
    def writer(self) -> IsaacRunWriter:
        return self._writer

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
            rendering_dt=1.0 / max(camera_rate_hz, 30.0),
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
        if self._robot_binding is None:
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

    def bind_frontend_estimation_control(self) -> None:
        if self._robot_binding is None:
            raise RuntimeError("Robot must be bound before frontend/estimation/control.")
        estimation_config = self.config.config_payloads["estimation"]
        control_config = self.config.config_payloads["control"]
        actuation_config = self.config.config_payloads["actuation"]
        self._frontend = IsaacAprilTagFrontend(anchor_tag_id=int(self.config.anchor_tag_id))
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
        servo_config = ServoCorruptionConfig(
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
        current_positions = self._robot_binding.get_joint_positions()
        for index, model in enumerate(self._servo_models):
            model.position = float(current_positions[index])
        self._previous_joint_positions = current_positions.copy()
        for _ in range(30):
            self._world.step(render=True)
        ee_position_world_m, ee_orientation_wxyz = self._robot_binding.get_end_effector_pose()
        self._calibrate_camera_mount_once(ee_position_world_m)
        for _ in range(10):
            self._world.step(render=True)
        self._control_target_orientation_wxyz = ee_orientation_wxyz.copy()
        self._last_open_loop_target_position = self._current_control_position_world_m(
            ee_position_world_m=np.asarray(ee_position_world_m, dtype=np.float64).reshape(3)
        )
        self._write_tag_ground_truth()

    def _calibrate_camera_mount_once(self, ee_position_world_m: np.ndarray) -> None:
        if self._camera_binding is None:
            return
        look_at_world_m = self._camera_binding.spec.look_at_world_m
        if look_at_world_m is None:
            return
        camera_position_world_m = np.asarray(ee_position_world_m, dtype=np.float64).reshape(3) + np.asarray(
            self._camera_binding.spec.mount_offset_world_m,
            dtype=np.float64,
        ).reshape(3)
        orientation_wxyz = _look_at_orientation_wxyz(
            camera_position_world_m,
            np.asarray(look_at_world_m, dtype=np.float64).reshape(3),
        )
        self._camera_binding.set_world_pose(
            position_world_m=camera_position_world_m,
            orientation_wxyz=orientation_wxyz,
        )

    def start(self) -> None:
        self.boot_kit()
        self.open_or_build_stage()
        self.bind_world()
        self.bind_robot()
        self.bind_sensors()
        self.bind_frontend_estimation_control()
        self.warmup()
        self.started = True

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

    def _process_camera(self, timestamps: TimestampTriplet) -> None:
        if self._camera_binding is None or self._frontend is None or self._filter is None:
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
            pack = self._frontend.process_bgr_frame(
                cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
                frame_packet=frame_packet,
                tag_size_by_id=tag_size_by_id,
            )
            for detection in pack.detections:
                self._writer.write_detection(detection)
                self._counts["detections"] += 1
            self._last_detections = pack.detections
            previous_anchor_visible = bool(self._last_anchor_visible)
            anchor_visible_raw = bool(pack.metadata.get("anchor_visible", False))
            anchor_update_suppressed = self._anchor_updates_suppressed(timestamp_s=float(frame_packet.timestamp_s))
            self._last_anchor_visible_raw = anchor_visible_raw
            self._last_anchor_update_suppressed = bool(anchor_update_suppressed)
            self._last_anchor_visible = bool(anchor_visible_raw and not anchor_update_suppressed)
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
                    orientation_innovation_norm_deg=float(anchor_update_result.orientation_innovation_norm_deg),
                    velocity_innovation_norm_mps=float(anchor_update_result.velocity_innovation_norm_mps),
                    relocalization_correction_norm_m=float(anchor_update_result.relocalization_correction_norm_m),
                    post_update_covariance_trace=float(anchor_update_result.post_update_covariance_trace),
                )
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
                tag_feedback_diagnostics=self._tag_feedback_diagnostics,
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

    def _limit_step_delta(self, delta_world_m: np.ndarray) -> np.ndarray:
        candidate = np.asarray(delta_world_m, dtype=np.float64).reshape(3)
        max_step_m = float(self._control_config().get("max_position_step_m", 0.02))
        norm = float(np.linalg.norm(candidate))
        if norm <= max_step_m or norm < 1e-12:
            return candidate
        return candidate * (max_step_m / norm)

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
    ) -> tuple[np.ndarray, bool, float, str, np.ndarray]:
        if self._robot_binding is None:
            raise RuntimeError("Robot binding missing.")
        alphas = self._control_config().get("ik_retry_alphas", [1.0, 0.5, 0.25, 0.1, 0.0])
        fixed_orientation = (
            self._control_target_orientation_wxyz.copy()
            if self._control_target_orientation_wxyz is not None
            else np.asarray(current_orientation_wxyz, dtype=np.float64).reshape(4)
        )
        orientation_candidates = [("fixed", fixed_orientation)]
        relaxed_orientation = np.asarray(current_orientation_wxyz, dtype=np.float64).reshape(4)
        if not np.allclose(relaxed_orientation, fixed_orientation):
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

    def _apply_control(self, ee_position_world_m: np.ndarray, ee_orientation_wxyz: np.ndarray) -> None:
        if self._tracker is None or self._controller_clock is None or self._robot_binding is None:
            return
        from isaacsim.core.utils.types import ArticulationAction

        for _ in self._controller_clock.advance_to(self._sim_time_s):
            current_position, current_position_source, hold_for_bootstrap = self._resolve_control_state(
                ee_position_world_m=ee_position_world_m
            )
            current_joint_positions = self._robot_binding.get_joint_positions()
            position_radius_95_m = 0.05 if self._latest_uncertainty is None else float(self._latest_uncertainty.position_radius_95_m)
            innovation_norm = 0.0 if self._filter is None else float(self._filter.last_innovation_norm)
            degraded_after = int(self._control_config().get("degrade_after_consecutive_ik_failures", 5))
            degraded_gain_scale = float(self._control_config().get("degraded_gain_scale", 0.5))
            degraded_mode_active = self._controller_consecutive_ik_failures >= degraded_after
            if hold_for_bootstrap or current_position is None:
                tracked_target_position = self._current_control_position_world_m(
                    ee_position_world_m=np.asarray(ee_position_world_m, dtype=np.float64).reshape(3)
                )
                ik_target_position = tracked_target_position.copy()
                path_command = None
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
                if not ik_success:
                    self._ik_failure_count += 1
                    self._controller_consecutive_ik_failures += 1
                    joint_targets = current_joint_positions.copy()
                    tracked_target_position = current_position.copy()
                    ik_target_position = np.asarray(ee_position_world_m, dtype=np.float64).reshape(3)
                    safety_reason = "ik_hold"
                else:
                    self._controller_consecutive_ik_failures = 0
                    safety_reason = path_command.safety_reason if not degraded_mode_active else "ik_degraded"
                joint_target_delta_norm = float(np.linalg.norm(joint_targets - current_joint_positions))
                min_joint_limit_margin = self._robot_binding.joint_limit_margin(joint_targets)
                command_delta_norm_m = float(np.linalg.norm(command_delta))
                tracking_error_norm_m = float(np.linalg.norm(np.asarray(path_command.tracking_error_world_m, dtype=np.float64)))
            effective_positions: list[float] = []
            servo_state: dict[str, Any] = {}
            dropped_command = False
            dt_control_s = float(self._controller_clock.period_s)
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
            self._writer.write_command(
                IsaacJointCommandPacket(
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
            )
            self._writer.write_controller_diagnostic(
                IsaacControllerDiagnosticPacket(
                    timestamp_s=float(self._sim_time_s),
                    sim_time_s=float(self._sim_time_s),
                    estimator_mode=self.config.estimator_mode,
                    controller_mode=self.config.controller_mode,
                    waypoint_index=int(self._tracker.waypoint_manager.current_index),
                    current_position_source=current_position_source,
                    tracking_error_norm_m=float(tracking_error_norm_m),
                    command_delta_norm_m=float(command_delta_norm_m),
                    desired_position_world_m=tuple(float(value) for value in tracked_target_position.tolist()),
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
                )
            )
            self._counts["commands"] += 1
            self._counts["controller_diagnostics"] += 1
            self._last_open_loop_target_position = tracked_target_position.copy()

    def _step_once(self) -> None:
        if self._world is None:
            raise RuntimeError("Runtime world is not initialized.")
        self._world.step(render=True)
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
            if suppression_active and self._disable_imu_prediction_while_anchor_suppressed:
                self._last_imu_packets_used_for_prediction = 0
                self._last_imu_packets_rejected_for_prediction = len(imu_packets)
                pass
            else:
                packets_for_prediction = self._filter_imu_packets_for_prediction(
                    imu_packets,
                    suppression_active=bool(suppression_active),
                )
                if packets_for_prediction:
                    self._filter.predict(packets_for_prediction)
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
        complete = all(
            self._counts[key] > 0
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
            duration_s=float(self._sim_time_s),
            step_count=int(self.steps_executed),
            counts=dict(self._counts),
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
