"""Sensor specifications and live Isaac wrappers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tomllib
from typing import Any

import numpy as np

from calib_sim.common.inertial import accelerometer_specific_force_body, gyroscope_measurement_body
from calib_sim.isaac.clocks import FixedRateClock, ScheduledSensorTick, TimestampTriplet
from calib_sim.isaac.logging.schemas import IsaacCameraFramePacket, IsaacImuPacket


def _reading_attr(reading: Any, primary: str, secondary: str | None = None, default: float = 0.0) -> float:
    if hasattr(reading, primary):
        return float(getattr(reading, primary))
    if secondary is not None and hasattr(reading, secondary):
        return float(getattr(reading, secondary))
    return float(default)


def _rotation_matrix_to_quaternion_wxyz(rotation: np.ndarray) -> tuple[float, float, float, float]:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        return (
            float(0.25 * s),
            float((matrix[2, 1] - matrix[1, 2]) / s),
            float((matrix[0, 2] - matrix[2, 0]) / s),
            float((matrix[1, 0] - matrix[0, 1]) / s),
        )
    diagonal = np.diag(matrix)
    if diagonal[0] > diagonal[1] and diagonal[0] > diagonal[2]:
        s = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
        return (
            float((matrix[2, 1] - matrix[1, 2]) / s),
            float(0.25 * s),
            float((matrix[0, 1] + matrix[1, 0]) / s),
            float((matrix[0, 2] + matrix[2, 0]) / s),
        )
    if diagonal[1] > diagonal[2]:
        s = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
        return (
            float((matrix[0, 2] - matrix[2, 0]) / s),
            float((matrix[0, 1] + matrix[1, 0]) / s),
            float(0.25 * s),
            float((matrix[1, 2] + matrix[2, 1]) / s),
        )
    s = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
    return (
        float((matrix[1, 0] - matrix[0, 1]) / s),
        float((matrix[0, 2] + matrix[2, 0]) / s),
        float((matrix[1, 2] + matrix[2, 1]) / s),
        float(0.25 * s),
    )


@dataclass(slots=True)
class CameraSensorSpec:
    name: str
    width_px: int
    height_px: int
    rate_hz: float
    frame_id: str
    prim_path: str
    source_toml_path: str | None = None
    local_translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    local_orientation_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mount_offset_world_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    look_at_world_m: tuple[float, float, float] | None = None
    focal_length_mm: float | None = None
    horizontal_aperture_mm: float | None = None
    vertical_aperture_mm: float | None = None
    distortion_model: str = "opencv_pinhole"
    distortion_coefficients: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)
    intrinsics: dict[str, float] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "width_px": int(self.width_px),
            "height_px": int(self.height_px),
            "rate_hz": float(self.rate_hz),
            "frame_id": self.frame_id,
            "prim_path": self.prim_path,
            "source_toml_path": self.source_toml_path,
            "local_translation_m": [float(value) for value in self.local_translation_m],
            "local_orientation_rpy_deg": [float(value) for value in self.local_orientation_rpy_deg],
            "mount_offset_world_m": [float(value) for value in self.mount_offset_world_m],
            "look_at_world_m": None
            if self.look_at_world_m is None
            else [float(value) for value in self.look_at_world_m],
            "focal_length_mm": None if self.focal_length_mm is None else float(self.focal_length_mm),
            "horizontal_aperture_mm": None
            if self.horizontal_aperture_mm is None
            else float(self.horizontal_aperture_mm),
            "vertical_aperture_mm": None
            if self.vertical_aperture_mm is None
            else float(self.vertical_aperture_mm),
            "distortion_model": self.distortion_model,
            "distortion_coefficients": [float(value) for value in self.distortion_coefficients],
            "intrinsics": {str(key): float(value) for key, value in self.intrinsics.items()},
        }


@dataclass(slots=True)
class ImuSensorSpec:
    name: str
    rate_hz: float
    frame_id: str
    prim_path: str
    parent_prim_path: str
    local_translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    local_orientation_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    read_gravity: bool = False
    gravity_world_mps2: tuple[float, float, float] = (0.0, 0.0, -9.81)
    imu_semantics: str = "specific_force"
    noise_preset: str = "imu_nominal"

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rate_hz": float(self.rate_hz),
            "frame_id": self.frame_id,
            "prim_path": self.prim_path,
            "parent_prim_path": self.parent_prim_path,
            "local_translation_m": [float(value) for value in self.local_translation_m],
            "local_orientation_rpy_deg": [float(value) for value in self.local_orientation_rpy_deg],
            "read_gravity": bool(self.read_gravity),
            "gravity_world_mps2": [float(value) for value in self.gravity_world_mps2],
            "imu_semantics": self.imu_semantics,
            "noise_preset": self.noise_preset,
        }


def camera_spec_from_config(camera_config: dict[str, Any], *, default_prim_path: str) -> CameraSensorSpec:
    source_toml_path = camera_config.get("source_toml_path")
    intrinsics = {str(key): float(value) for key, value in dict(camera_config.get("intrinsics", {})).items()}
    focal_length_mm = camera_config.get("focal_length_mm")
    if focal_length_mm is None and source_toml_path:
        resolved_toml = Path(str(source_toml_path)).resolve()
        if resolved_toml.exists():
            toml_payload = tomllib.loads(resolved_toml.read_text(encoding="utf-8"))
            if isinstance(toml_payload, dict):
                optics = toml_payload.get("optics", {})
                if isinstance(optics, dict):
                    focal_length_mm = optics.get("focal_length_mm")
    horizontal_aperture_mm = camera_config.get("horizontal_aperture_mm")
    vertical_aperture_mm = camera_config.get("vertical_aperture_mm")
    fx_px = float(intrinsics.get("fx_px", 0.0))
    fy_px = float(intrinsics.get("fy_px", 0.0))
    if focal_length_mm is not None:
        focal_length_mm = float(focal_length_mm)
        if horizontal_aperture_mm is None and fx_px > 0.0:
            horizontal_aperture_mm = float(focal_length_mm) * float(camera_config["width_px"]) / fx_px
        if vertical_aperture_mm is None and fy_px > 0.0:
            vertical_aperture_mm = float(focal_length_mm) * float(camera_config["height_px"]) / fy_px
    return CameraSensorSpec(
        name=str(camera_config.get("name", "camera")),
        width_px=int(camera_config["width_px"]),
        height_px=int(camera_config["height_px"]),
        rate_hz=float(camera_config["rate_hz"]),
        frame_id=str(camera_config.get("frame_id", "C")),
        prim_path=str(camera_config.get("prim_path", default_prim_path)),
        source_toml_path=None if source_toml_path is None else str(source_toml_path),
        local_translation_m=tuple(float(value) for value in camera_config.get("local_translation_m", (0.0, 0.0, 0.0))),
        local_orientation_rpy_deg=tuple(float(value) for value in camera_config.get("local_orientation_rpy_deg", (0.0, 0.0, 0.0))),
        mount_offset_world_m=tuple(float(value) for value in camera_config.get("mount_offset_world_m", (0.0, 0.0, 0.0))),
        look_at_world_m=None
        if camera_config.get("look_at_world_m") is None
        else tuple(float(value) for value in camera_config.get("look_at_world_m", (0.0, 0.0, 0.0))),
        focal_length_mm=None if focal_length_mm is None else float(focal_length_mm),
        horizontal_aperture_mm=None if horizontal_aperture_mm is None else float(horizontal_aperture_mm),
        vertical_aperture_mm=None if vertical_aperture_mm is None else float(vertical_aperture_mm),
        distortion_model=str(camera_config.get("distortion_model", "opencv_pinhole")),
        distortion_coefficients=tuple(float(value) for value in camera_config.get("distortion_coefficients", (0.0, 0.0, 0.0, 0.0, 0.0))),
        intrinsics=intrinsics,
    )


def imu_spec_from_config(imu_config: dict[str, Any], *, default_prim_path: str, parent_prim_path: str) -> ImuSensorSpec:
    return ImuSensorSpec(
        name=str(imu_config.get("name", "imu")),
        rate_hz=float(imu_config["rate_hz"]),
        frame_id=str(imu_config.get("frame_id", "I")),
        prim_path=str(imu_config.get("prim_path", default_prim_path)),
        parent_prim_path=str(imu_config.get("parent_prim_path", parent_prim_path)),
        local_translation_m=tuple(float(value) for value in imu_config.get("local_translation_m", (0.0, 0.0, 0.0))),
        local_orientation_rpy_deg=tuple(float(value) for value in imu_config.get("local_orientation_rpy_deg", (0.0, 0.0, 0.0))),
        read_gravity=bool(imu_config.get("read_gravity", False)),
        gravity_world_mps2=tuple(float(value) for value in imu_config.get("gravity_world_mps2", (0.0, 0.0, -9.81))),
        imu_semantics=str(imu_config.get("imu_semantics", "specific_force")),
        noise_preset=str(imu_config.get("noise_preset", "imu_nominal")),
    )


@dataclass(slots=True)
class IsaacCameraBinding:
    spec: CameraSensorSpec
    camera: Any
    clock: FixedRateClock
    initialized: bool = False

    @classmethod
    def create(cls, spec: CameraSensorSpec) -> "IsaacCameraBinding":
        from isaacsim.sensors.camera import Camera

        camera = Camera(
            prim_path=spec.prim_path,
            name=spec.name,
            frequency=float(spec.rate_hz),
            resolution=(int(spec.width_px), int(spec.height_px)),
        )
        return cls(spec=spec, camera=camera, clock=FixedRateClock(rate_hz=spec.rate_hz))

    def initialize(self) -> None:
        from isaacsim.core.utils.rotations import euler_angles_to_quat

        self.camera.initialize()
        self.camera.set_local_pose(
            translation=np.asarray(self.spec.local_translation_m, dtype=np.float64),
            orientation=euler_angles_to_quat(
                np.asarray(self.spec.local_orientation_rpy_deg, dtype=np.float64),
                degrees=True,
            ),
            camera_axes="world",
        )
        if self.spec.focal_length_mm is not None:
            self.camera.set_focal_length(float(self.spec.focal_length_mm))
        if self.spec.horizontal_aperture_mm is not None:
            self.camera.set_horizontal_aperture(float(self.spec.horizontal_aperture_mm))
        if self.spec.vertical_aperture_mm is not None:
            self.camera.set_vertical_aperture(float(self.spec.vertical_aperture_mm), maintain_square_pixels=False)
        self.camera.set_lens_aperture(0.0)
        self.camera.set_focus_distance(1.0)
        self.camera.set_clipping_range(0.01, 10.0)
        self.initialized = True

    def spec_summary(self) -> dict[str, Any]:
        return self.spec.summary()

    def set_world_pose(self, *, position_world_m: np.ndarray, orientation_wxyz: np.ndarray) -> None:
        self.camera.set_world_pose(
            position=np.asarray(position_world_m, dtype=np.float64).reshape(3),
            orientation=np.asarray(orientation_wxyz, dtype=np.float64).reshape(4),
            camera_axes="world",
        )

    def is_due(self, sim_time_s: float) -> bool:
        return self.clock.next_time_s <= float(sim_time_s) + 1e-12

    def due_ticks(self, sim_time_s: float) -> list[ScheduledSensorTick]:
        return self.clock.advance_to(float(sim_time_s))

    def sample(
        self,
        *,
        tick: ScheduledSensorTick,
        timestamps: TimestampTriplet,
        visible_gt_tag_ids: tuple[int, ...] = (),
    ) -> tuple[IsaacCameraFramePacket, np.ndarray] | None:
        if not self.initialized:
            raise RuntimeError("Camera binding was not initialized before sampling.")
        rgb = self.camera.get_rgb()
        if rgb is None:
            return None
        frame_data = self.camera.get_current_frame()
        position_world, orientation_wxyz = self.camera.get_world_pose()
        packet = IsaacCameraFramePacket(
            frame_index=int(tick.sample_index),
            timestamp_s=float(tick.sim_time_s),
            sim_time_s=float(timestamps.sim_time_s),
            sensor_time_s=float(frame_data.get("rendering_time", timestamps.sensor_time_s)),
            host_time_s=float(timestamps.host_time_s),
            rgb_path="",
            intrinsics_snapshot=dict(self.spec.intrinsics),
            extrinsics_snapshot={
                "position_world_m": [float(value) for value in np.asarray(position_world, dtype=np.float64).reshape(3)],
                "orientation_wxyz": [float(value) for value in np.asarray(orientation_wxyz, dtype=np.float64).reshape(4)],
                "frame_id": self.spec.frame_id,
            },
            image_width_px=int(self.spec.width_px),
            image_height_px=int(self.spec.height_px),
            visible_gt_tag_ids=tuple(int(tag_id) for tag_id in visible_gt_tag_ids),
        )
        return packet, np.asarray(rgb, dtype=np.uint8)


@dataclass(slots=True)
class IsaacImuBinding:
    spec: ImuSensorSpec
    clock: FixedRateClock
    packet_index: int = 0
    previous_position_world_m: np.ndarray | None = None
    previous_velocity_world_mps: np.ndarray | None = None
    previous_rotation_wi: np.ndarray | None = None

    @classmethod
    def create(cls, spec: ImuSensorSpec) -> "IsaacImuBinding":
        return cls(spec=spec, clock=FixedRateClock(rate_hz=spec.rate_hz))

    def spec_summary(self) -> dict[str, Any]:
        return self.spec.summary()

    def is_due(self, sim_time_s: float) -> bool:
        return self.clock.next_time_s <= float(sim_time_s) + 1e-12

    def due_ticks(self, sim_time_s: float) -> list[ScheduledSensorTick]:
        return self.clock.advance_to(float(sim_time_s))

    def sample(
        self,
        *,
        tick: ScheduledSensorTick,
        timestamps: TimestampTriplet,
        position_world_m: np.ndarray,
        rotation_wi: np.ndarray,
    ) -> IsaacImuPacket | None:
        position_world_m = np.asarray(position_world_m, dtype=np.float64).reshape(3)
        rotation_wi = np.asarray(rotation_wi, dtype=np.float64).reshape(3, 3)
        if self.previous_position_world_m is None or self.previous_velocity_world_mps is None or self.previous_rotation_wi is None:
            world_velocity_mps = np.zeros(3, dtype=np.float64)
            world_acceleration_mps2 = np.zeros(3, dtype=np.float64)
            gyro_body_rps = np.zeros(3, dtype=np.float64)
        else:
            dt_s = max(float(tick.dt_s), 1e-6)
            world_velocity_mps = (position_world_m - self.previous_position_world_m) / dt_s
            world_acceleration_mps2 = (world_velocity_mps - self.previous_velocity_world_mps) / dt_s
            gyro_body_rps = gyroscope_measurement_body(
                start_rotation_wi=self.previous_rotation_wi,
                end_rotation_wi=rotation_wi,
                delta_time_s=dt_s,
            )
        specific_force_body_mps2 = accelerometer_specific_force_body(
            rotation_wi=rotation_wi,
            acceleration_world_mps2=world_acceleration_mps2,
            gravity_world_mps2=self.spec.gravity_world_mps2,
        )
        orientation_wxyz = _rotation_matrix_to_quaternion_wxyz(rotation_wi)
        packet = IsaacImuPacket(
            packet_index=int(self.packet_index),
            timestamp_s=float(tick.sim_time_s),
            sim_time_s=float(timestamps.sim_time_s),
            dt_s=float(tick.dt_s),
            wx=float(gyro_body_rps[0]),
            wy=float(gyro_body_rps[1]),
            wz=float(gyro_body_rps[2]),
            ax=float(specific_force_body_mps2[0]),
            ay=float(specific_force_body_mps2[1]),
            az=float(specific_force_body_mps2[2]),
            imu_frame=self.spec.frame_id,
            imu_semantics=self.spec.imu_semantics,
            noise_preset=self.spec.noise_preset,
            orientation_wxyz=orientation_wxyz,
            sensor_time_s=float(timestamps.sensor_time_s),
            host_time_s=float(timestamps.host_time_s),
        )
        self.packet_index += 1
        self.previous_position_world_m = position_world_m.copy()
        self.previous_velocity_world_mps = world_velocity_mps.copy()
        self.previous_rotation_wi = rotation_wi.copy()
        return packet


__all__ = [
    "CameraSensorSpec",
    "ImuSensorSpec",
    "IsaacCameraBinding",
    "IsaacImuBinding",
    "camera_spec_from_config",
    "imu_spec_from_config",
]
