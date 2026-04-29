"""Shared data models used across the project.

These models are intentionally lightweight. They define the public shapes that
multiple modules agree on: poses, intrinsics, robot configs, and tag detection
results. Keeping them centralized avoids silent schema drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


Vec3 = Tuple[float, float, float]
Vec2 = Tuple[float, float]


@dataclass(slots=True)
class Pose:
    """Simple pose container.

    The scaffold uses translation in meters and roll/pitch/yaw in degrees for
    human readability. A production system may prefer quaternions internally.
    """

    xyz_m: Vec3
    rpy_deg: Vec3


@dataclass(slots=True)
class CameraIntrinsics:
    """Minimal intrinsics model used by both the simulator and detector service."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: str = "opencv_pinhole"
    distortion_coefficients: Sequence[float] = field(default_factory=lambda: [0, 0, 0, 0, 0])


@dataclass(slots=True)
class ImuModel:
    """Noise/bias metadata for the synthetic IMU."""

    rate_hz: float
    read_gravity: bool = True
    accel_noise_std: Vec3 = (0.02, 0.02, 0.02)
    gyro_noise_std: Vec3 = (0.002, 0.002, 0.002)
    accel_bias: Vec3 = (0.0, 0.0, 0.0)
    gyro_bias: Vec3 = (0.0, 0.0, 0.0)


@dataclass(slots=True)
class PhoneRigConfig:
    """Logical phone device attached to the robot."""

    namespace: str
    camera: CameraIntrinsics
    imu: ImuModel
    camera_frame: str = "phone_camera"
    imu_frame: str = "phone_imu"


@dataclass(slots=True)
class RobotConfig:
    """Normalized robot description.

    This is the key to making the robot swapable by configuration rather than by
    code edits.
    """

    name: str
    asset_mode: str
    asset_path: str
    base_frame: str
    ee_frame: str
    mount_frame: str
    joint_order: List[str]
    control_mode: str = "position"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MotionSample:
    """One sample of a time-parameterized robot target pose."""

    t_s: float
    xyz_m: Vec3
    rpy_deg: Vec3


@dataclass(slots=True)
class MotionPreset:
    """Named motion preset emitted by the motion library."""

    preset_id: str
    duration_s: float
    sample_rate_hz: float
    samples: List[MotionSample]


@dataclass(slots=True)
class TagDetection:
    """Single tag detection result.

    `points5_xy` is explicitly included because the user asked for the 5 points:
    4 corners + center.
    """

    family: str
    tag_id: int
    corners_xy_clockwise: List[Vec2]
    center_xy: Vec2
    points5_xy: List[Vec2]
    pose_camera_rvec: Optional[Vec3] = None
    pose_camera_tvec: Optional[Vec3] = None
    quality: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FrameDetections:
    """Detections associated with a single image frame."""

    frame_index: int
    timestamp_s: Optional[float]
    detections: List[TagDetection]
