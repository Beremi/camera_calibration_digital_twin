"""Typed containers for batch calibration datasets and evaluation data."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _summary_float_tuple(values: tuple[float, ...]) -> list[float]:
    return [float(value) for value in values]


@dataclass(slots=True)
class CameraPoseEstimationBridgeSnapshot:
    pattern_half_extent_m: float
    measurement_unit: str
    measurement_order: tuple[str, ...]
    y_axis: str
    undistort_before_export: bool

    def summary(self) -> dict[str, Any]:
        return {
            "pattern_half_extent_m": float(self.pattern_half_extent_m),
            "measurement_unit": self.measurement_unit,
            "measurement_order": list(self.measurement_order),
            "y_axis": self.y_axis,
            "undistort_before_export": bool(self.undistort_before_export),
        }


@dataclass(slots=True)
class CameraModelSnapshot:
    name: str
    source_toml_path: str
    projection_model: str
    apply_lens_distortion_in_render: bool
    output_width_px: int
    output_height_px: int
    focal_length_mm: float
    focus_distance_diopters: float
    target_frames_per_second: float
    timestamp_source: str
    sensor_orientation_deg: int
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    skew_px: float
    crop_x_px: float
    crop_y_px: float
    crop_width_px: float
    crop_height_px: float
    distortion_model: str
    distortion_coefficients: tuple[float, float, float, float, float]
    pose_bridge: CameraPoseEstimationBridgeSnapshot

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_toml_path": self.source_toml_path,
            "projection_model": self.projection_model,
            "apply_lens_distortion_in_render": bool(self.apply_lens_distortion_in_render),
            "output_resolution_px": [int(self.output_width_px), int(self.output_height_px)],
            "focal_length_mm": float(self.focal_length_mm),
            "focus_distance_diopters": float(self.focus_distance_diopters),
            "target_frames_per_second": float(self.target_frames_per_second),
            "timestamp_source": self.timestamp_source,
            "sensor_orientation_deg": int(self.sensor_orientation_deg),
            "effective_intrinsics_px": {
                "fx": float(self.fx_px),
                "fy": float(self.fy_px),
                "cx": float(self.cx_px),
                "cy": float(self.cy_px),
                "skew": float(self.skew_px),
            },
            "crop_rect_active_array_px": {
                "x": float(self.crop_x_px),
                "y": float(self.crop_y_px),
                "width": float(self.crop_width_px),
                "height": float(self.crop_height_px),
            },
            "distortion": {
                "model": self.distortion_model,
                "coefficients": [float(value) for value in self.distortion_coefficients],
            },
            "camera_pose_estimation": self.pose_bridge.summary(),
        }


@dataclass(slots=True)
class DeviceCameraSnapshot:
    width: int
    height: int
    fps: float
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: str
    distortion_coefficients: tuple[float, float, float, float, float]

    def summary(self) -> dict[str, Any]:
        return {
            "width": int(self.width),
            "height": int(self.height),
            "fps": float(self.fps),
            "fx": float(self.fx),
            "fy": float(self.fy),
            "cx": float(self.cx),
            "cy": float(self.cy),
            "distortion_model": self.distortion_model,
            "distortion_coefficients": [float(value) for value in self.distortion_coefficients],
        }


@dataclass(slots=True)
class DeviceImuSnapshot:
    rate_hz: float
    read_gravity: bool
    accel_noise_std: tuple[float, float, float]
    gyro_noise_std: tuple[float, float, float]
    accel_bias: tuple[float, float, float]
    gyro_bias: tuple[float, float, float]

    def summary(self) -> dict[str, Any]:
        return {
            "rate_hz": float(self.rate_hz),
            "read_gravity": bool(self.read_gravity),
            "accel_noise_std": _summary_float_tuple(self.accel_noise_std),
            "gyro_noise_std": _summary_float_tuple(self.gyro_noise_std),
            "accel_bias": _summary_float_tuple(self.accel_bias),
            "gyro_bias": _summary_float_tuple(self.gyro_bias),
        }


@dataclass(slots=True)
class ImuConventionSnapshot:
    imu_measurement_convention: str
    imu_timestamp_semantics: str
    imu_gravity_handling: str
    imu_sampling_mode: str
    accelerometer_measurement_equation: str
    gyroscope_measurement_equation: str

    def summary(self) -> dict[str, Any]:
        return {
            "imu_measurement_convention": self.imu_measurement_convention,
            "imu_timestamp_semantics": self.imu_timestamp_semantics,
            "imu_gravity_handling": self.imu_gravity_handling,
            "imu_sampling_mode": self.imu_sampling_mode,
            "accelerometer_measurement_equation": self.accelerometer_measurement_equation,
            "gyroscope_measurement_equation": self.gyroscope_measurement_equation,
        }


@dataclass(slots=True)
class DeviceMountSnapshot:
    camera_frame: str
    imu_frame: str
    imu_translation_m: tuple[float, float, float]
    imu_rpy_deg: tuple[float, float, float]

    def summary(self) -> dict[str, Any]:
        return {
            "camera_frame": self.camera_frame,
            "imu_frame": self.imu_frame,
            "imu_translation_m": _summary_float_tuple(self.imu_translation_m),
            "imu_rpy_deg": _summary_float_tuple(self.imu_rpy_deg),
        }


@dataclass(slots=True)
class DeviceConfigSnapshot:
    name: str
    namespace: str
    camera_model_toml: str
    camera: DeviceCameraSnapshot
    imu: DeviceImuSnapshot
    mount: DeviceMountSnapshot

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "namespace": self.namespace,
            "camera_model_toml": self.camera_model_toml,
            "camera_rgb": self.camera.summary(),
            "imu": self.imu.summary(),
            "mount": self.mount.summary(),
        }


@dataclass(slots=True)
class TagSpec:
    tag_id: int
    family: str
    size_m: float
    position_m: tuple[float, float, float]
    mount: str
    orientation_rpy_deg: tuple[float, float, float] | None = None

    def corner_points_local_m(self) -> np.ndarray:
        half_extent = float(self.size_m) * 0.5
        return np.array(
            [
                [-half_extent, -half_extent, 0.0],
                [half_extent, -half_extent, 0.0],
                [half_extent, half_extent, 0.0],
                [-half_extent, half_extent, 0.0],
            ],
            dtype=np.float64,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "tag_id": int(self.tag_id),
            "family": self.family,
            "size_m": float(self.size_m),
            "position_m": _summary_float_tuple(self.position_m),
            "mount": self.mount,
            "orientation_rpy_deg": None if self.orientation_rpy_deg is None else _summary_float_tuple(self.orientation_rpy_deg),
        }


@dataclass(slots=True)
class CameraFrame:
    frame_index: int
    recording_frame_index: int
    tick_index: int
    timestamp_s: float
    servo_positions_deg: tuple[float, float, float]
    servo_targets_deg: tuple[float, float, float]
    auto_demo_enabled: bool
    camera_model_name: str


@dataclass(slots=True)
class TagDetectionObservation:
    frame_index: int
    recording_frame_index: int
    tick_index: int
    timestamp_s: float
    tag_id: int
    family: str
    corners_xy_clockwise_px: tuple[tuple[float, float], ...]
    center_xy_px: tuple[float, float]
    points5_xy_px: tuple[tuple[float, float], ...]
    physical_edge_length_m: float
    pose_camera_rvec_rad: tuple[float, float, float] | None = None
    pose_camera_tvec_m: tuple[float, float, float] | None = None
    quality: dict[str, Any] = field(default_factory=dict)
    camera_model_name: str = ""


@dataclass(slots=True)
class ImuPacket:
    tick_index: int
    timestamp_s: float
    accel_body_mps2: tuple[float, float, float]
    gyro_body_rps: tuple[float, float, float]


@dataclass(slots=True)
class CameraTruthSample:
    frame_index: int
    recording_frame_index: int
    tick_index: int
    timestamp_s: float
    position_world_m: tuple[float, float, float]
    velocity_world_mps: tuple[float, float, float]
    rotation_cw: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    rotation_xyz_deg: tuple[float, float, float]
    servo_positions_deg: tuple[float, float, float]


@dataclass(slots=True)
class TagTruthSample:
    frame_index: int
    recording_frame_index: int
    tick_index: int
    timestamp_s: float
    tag_id: int
    family: str
    size_m: float
    visible: bool
    tag_center_world_m: tuple[float, float, float]
    camera_position_tag_m: tuple[float, float, float]
    camera_rotation_tc: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    rendered_image_points_px: tuple[tuple[float, float], ...]
    ideal_image_points_px: tuple[tuple[float, float], ...] | None = None
    image_plane_points_m: tuple[tuple[float, float], ...] | None = None


@dataclass(slots=True)
class BatchCalibrationDataset:
    run_dir: Path
    run_name: str
    schema_version: int
    config_name: str
    recording_frame_rate_hz: float
    camera_model: CameraModelSnapshot
    device_config: DeviceConfigSnapshot
    imu_convention: ImuConventionSnapshot
    tag_catalog: dict[int, TagSpec]
    camera_frames: tuple[CameraFrame, ...]
    tag_detections: tuple[TagDetectionObservation, ...]
    imu_packets: tuple[ImuPacket, ...]


@dataclass(slots=True)
class BatchCalibrationEvaluationData:
    run_dir: Path
    camera_truth: tuple[CameraTruthSample, ...]
    tag_truth_by_frame: dict[int, tuple[TagTruthSample, ...]]
    global_gyro_bias_rps_truth: tuple[float, float, float] | None = None
    global_accel_bias_mps2_truth: tuple[float, float, float] | None = None


@dataclass(slots=True)
class InitialGuess:
    stage: str
    anchor_frame_index: int
    camera_pose_vectors: dict[int, np.ndarray]
    tag_pose_vectors: dict[int, np.ndarray]
    velocity_world_mps_by_frame: dict[int, np.ndarray] = field(default_factory=dict)
    gyro_bias_rps_by_frame: dict[int, np.ndarray] = field(default_factory=dict)
    accel_bias_mps2_by_frame: dict[int, np.ndarray] = field(default_factory=dict)
    global_gyro_bias_rps: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    global_accel_bias_mps2: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    notes: tuple[str, ...] = ()


@dataclass(slots=True)
class PoseUncertaintySummary:
    index: int
    position_std_m: tuple[float, float, float]
    rotation_std_deg: tuple[float, float, float]
    position_radius_95_m: float
    rotation_radius_95_deg: float


@dataclass(slots=True)
class VectorUncertaintySummary:
    index: int
    std: tuple[float, float, float]
    radius_95: float


@dataclass(slots=True)
class GlobalVectorUncertaintySummary:
    name: str
    std: tuple[float, float, float]
    radius_95: float


@dataclass(slots=True)
class UncertaintySummary:
    camera_pose_uncertainty: tuple[PoseUncertaintySummary, ...]
    tag_pose_uncertainty: tuple[PoseUncertaintySummary, ...]
    covariance_condition_number: float | None = None
    velocity_uncertainty: tuple[VectorUncertaintySummary, ...] = ()
    gyro_bias_uncertainty: GlobalVectorUncertaintySummary | None = None
    accel_bias_uncertainty: GlobalVectorUncertaintySummary | None = None
    coverage_by_level: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BatchSolveResult:
    stage: str
    variant: str
    success: bool
    solver_name: str
    iterations: int
    final_cost: float
    initial_cost: float
    damping: float
    camera_pose_vectors: dict[int, np.ndarray]
    tag_pose_vectors: dict[int, np.ndarray]
    velocity_world_mps_by_frame: dict[int, np.ndarray] = field(default_factory=dict)
    gyro_bias_rps_by_frame: dict[int, np.ndarray] = field(default_factory=dict)
    accel_bias_mps2_by_frame: dict[int, np.ndarray] = field(default_factory=dict)
    global_gyro_bias_rps: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    global_accel_bias_mps2: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    hessian: np.ndarray | None = None
    gradient: np.ndarray | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvalSummary:
    variant: str
    frames_analyzed: int
    estimated_tags: int
    mean_position_error_m: float | None
    median_position_error_m: float | None
    p95_position_error_m: float | None
    max_position_error_m: float | None
    mean_rotation_error_deg: float | None
    mean_reprojection_rmse_px: float | None
    joint_world_mean_position_error_m: float | None
    joint_world_mean_rotation_error_deg: float | None
    solve_rate: float
    runtime_ms_per_iteration: float | None
    per_frame_records: tuple[dict[str, Any], ...] = ()
    per_tag_records: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
