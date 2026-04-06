"""Synthetic browser-friendly simulator for calibration workflows.

This module deliberately avoids any Isaac Sim dependency so the repo can offer
an interactive, game-like demo today. The generated views are synthetic camera
feeds backed by simple 3D geometry, AprilTag rendering, and the real detector.
"""

from __future__ import annotations

import base64
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import yaml

from calib_sim.common.video import ManagedMp4Writer
from calib_sim.interactive.analysis import analyze_recording_run
from calib_sim.interactive.camera_model import PhoneCameraModel, load_phone_camera_model
from calib_sim.tag_service.detector import AprilTag36h11Detector

REPO_ROOT = Path(__file__).resolve().parents[3]
WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def _display_path(path: str | Path) -> str:
    resolved = _repo_path(path)
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _to_vec3(values: Sequence[float]) -> tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))


def _clip_angle_delta_deg(delta_deg: np.ndarray) -> np.ndarray:
    return (delta_deg + 180.0) % 360.0 - 180.0


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return vector.copy()
    return vector / norm


def _look_at_rotation(camera_position: np.ndarray, target_position: np.ndarray) -> np.ndarray:
    forward = _normalize(target_position - camera_position)
    # Build a right-handed camera frame: x=right, y=up, z=forward.
    right = _normalize(np.cross(WORLD_UP, forward))
    if float(np.linalg.norm(right)) < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    up = _normalize(np.cross(forward, right))
    return np.column_stack((right, up, forward))


def _camera_intrinsics(width: int, height: int, fov_deg: float) -> tuple[float, float, float, float]:
    fx = 0.5 * float(width) / math.tan(math.radians(fov_deg) * 0.5)
    fy = fx
    cx = float(width) * 0.5
    cy = float(height) * 0.5
    return fx, fy, cx, cy


def _encode_jpeg_data_url(image_bgr: np.ndarray, *, quality: int = 82) -> str:
    ok, buffer = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("Could not encode stream frame.")
    payload = base64.b64encode(buffer).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _rotation_matrix_xyz_deg(angles_deg: Sequence[float]) -> np.ndarray:
    rx, ry, rz = [math.radians(float(value)) for value in angles_deg]
    cx, cy, cz = math.cos(rx), math.cos(ry), math.cos(rz)
    sx, sy, sz = math.sin(rx), math.sin(ry), math.sin(rz)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rot_z @ rot_y @ rot_x


@dataclass(slots=True)
class SceneTagConfig:
    tag_id: int
    family: str
    size_m: float
    position_m: tuple[float, float, float]
    mount: str = "wall"
    orientation_rpy_deg: tuple[float, float, float] | None = None


@dataclass(slots=True)
class ObserverCameraConfig:
    name: str
    position_m: tuple[float, float, float]
    look_at_m: tuple[float, float, float]
    width: int
    height: int
    fov_deg: float


@dataclass(slots=True)
class PrimaryCameraConfig:
    width: int
    height: int
    fov_deg: float


@dataclass(slots=True)
class SceneBoxConfig:
    name: str
    center_m: tuple[float, float, float]
    size_m: tuple[float, float, float]
    color_bgr: tuple[int, int, int]
    edge_color_bgr: tuple[int, int, int]


@dataclass(slots=True)
class AutoDemoChannelConfig:
    offset_deg: float
    amplitude_deg: float
    frequency_hz: float
    phase_deg: float


@dataclass(slots=True)
class AutoDemoKeyframeConfig:
    time_s: float
    servos_deg: tuple[float, float, float]


@dataclass(slots=True)
class AutoDemoConfig:
    enabled_by_default: bool
    mode: str
    loop_duration_s: float
    channels: tuple[AutoDemoChannelConfig, AutoDemoChannelConfig, AutoDemoChannelConfig]
    keyframes: tuple[AutoDemoKeyframeConfig, ...]

    def summary(self) -> dict[str, Any]:
        return {
            "enabled_by_default": self.enabled_by_default,
            "mode": self.mode,
            "loop_duration_s": self.loop_duration_s,
            "channels": [
                {
                    "offset_deg": channel.offset_deg,
                    "amplitude_deg": channel.amplitude_deg,
                    "frequency_hz": channel.frequency_hz,
                    "phase_deg": channel.phase_deg,
                }
                for channel in self.channels
            ],
            "keyframes": [
                {
                    "time_s": frame.time_s,
                    "servos_deg": list(frame.servos_deg),
                }
                for frame in self.keyframes
            ],
        }


@dataclass(slots=True)
class RobotArmPresetConfig:
    preset_id: str
    name: str
    label: str
    base_position_m: tuple[float, float, float]
    shoulder_height_m: float
    link_lengths_m: tuple[float, float, float]
    initial_servos_deg: tuple[float, float, float]
    servo_limits_deg: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    servo_speed_deg_s: tuple[float, float, float]

    def summary(self) -> dict[str, Any]:
        return {
            "preset_id": self.preset_id,
            "name": self.name,
            "label": self.label,
            "base_position_m": list(self.base_position_m),
            "shoulder_height_m": self.shoulder_height_m,
            "link_lengths_m": list(self.link_lengths_m),
            "initial_servos_deg": list(self.initial_servos_deg),
            "servo_limits_deg": [list(item) for item in self.servo_limits_deg],
            "servo_speed_deg_s": list(self.servo_speed_deg_s),
        }


@dataclass(slots=True)
class InteractiveSimConfig:
    config_path: str
    name: str
    device_config_path: str
    camera_model_toml_path: str | None
    output_dir: str
    stream_fps: float
    primary_camera: PrimaryCameraConfig
    wall_y_m: float
    wall_width_m: float
    wall_height_m: float
    scene_look_at_m: tuple[float, float, float]
    scene_boxes: list[SceneBoxConfig]
    robot_arm: RobotArmPresetConfig
    auto_demo: AutoDemoConfig
    auto_analyze_after_recording: bool
    tags: list[SceneTagConfig]
    observer_cameras: list[ObserverCameraConfig]

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "config_path": self.config_path,
            "device_config_path": self.device_config_path,
            "camera_model_toml_path": self.camera_model_toml_path,
            "output_dir": self.output_dir,
            "stream_fps": self.stream_fps,
            "primary_camera": {
                "width": self.primary_camera.width,
                "height": self.primary_camera.height,
                "fov_deg": self.primary_camera.fov_deg,
            },
            "scene": {
                "wall_y_m": self.wall_y_m,
                "wall_width_m": self.wall_width_m,
                "wall_height_m": self.wall_height_m,
                "look_at_m": list(self.scene_look_at_m),
                "boxes": [
                    {
                        "name": box.name,
                        "center_m": list(box.center_m),
                        "size_m": list(box.size_m),
                    }
                    for box in self.scene_boxes
                ],
            },
            "robot_arm": self.robot_arm.summary(),
            "auto_demo": self.auto_demo.summary(),
            "auto_analyze_after_recording": self.auto_analyze_after_recording,
            "observer_cameras": [
                {
                    "name": camera.name,
                    "width": camera.width,
                    "height": camera.height,
                    "fov_deg": camera.fov_deg,
                }
                for camera in self.observer_cameras
            ],
            "tags": [
                {
                    "tag_id": tag.tag_id,
                    "family": tag.family,
                    "size_m": tag.size_m,
                    "position_m": list(tag.position_m),
                    "mount": tag.mount,
                    "orientation_rpy_deg": list(tag.orientation_rpy_deg) if tag.orientation_rpy_deg is not None else None,
                }
                for tag in self.tags
            ],
        }


@dataclass(slots=True)
class CameraPose:
    position_m: np.ndarray
    rotation_cw: np.ndarray


@dataclass(slots=True)
class SceneTagRuntime:
    config: SceneTagConfig
    texture_bgr: np.ndarray
    texture_render_scale: float = 1.0

    def rotation_wt(self) -> np.ndarray:
        if self.config.orientation_rpy_deg is not None:
            return _rotation_matrix_xyz_deg(self.config.orientation_rpy_deg)
        if self.config.mount == "top":
            return np.eye(3, dtype=np.float64)
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        )

    def _corners_world(self, half_extent_m: float) -> np.ndarray:
        center = np.asarray(self.config.position_m, dtype=np.float64)
        local_corners = np.array(
            [
                [-half_extent_m, -half_extent_m, 0.0],
                [half_extent_m, -half_extent_m, 0.0],
                [half_extent_m, half_extent_m, 0.0],
                [-half_extent_m, half_extent_m, 0.0],
            ],
            dtype=np.float64,
        )
        rotation_wt = self.rotation_wt()
        return np.array([center + rotation_wt @ point for point in local_corners], dtype=np.float64)

    def detection_corners_world(self) -> np.ndarray:
        return self._corners_world(self.config.size_m * 0.5)

    def render_corners_world(self) -> np.ndarray:
        return self._corners_world(self.config.size_m * 0.5 * float(self.texture_render_scale))


def _to_color_bgr(values: Sequence[int] | None, default: tuple[int, int, int]) -> tuple[int, int, int]:
    if not values or len(values) < 3:
        return default
    return (int(values[0]), int(values[1]), int(values[2]))


def _load_robot_arm_preset(preset_path: str | Path) -> RobotArmPresetConfig:
    resolved_path = _repo_path(preset_path)
    raw = yaml.safe_load(resolved_path.read_text()) or {}
    limits_raw = raw.get("servo_limits_deg", [[-45.0, 45.0], [-10.0, 78.0], [-65.0, 65.0]])
    return RobotArmPresetConfig(
        preset_id=str(raw.get("preset_id", resolved_path.stem)),
        name=str(raw.get("name", resolved_path.stem)),
        label=str(raw.get("label", raw.get("name", resolved_path.stem))),
        base_position_m=_to_vec3(raw.get("base_position_m", [0.0, 1.15, 0.16])),
        shoulder_height_m=float(raw.get("shoulder_height_m", 0.18)),
        link_lengths_m=_to_vec3(raw.get("link_lengths_m", [0.30, 0.25, 0.18])),
        initial_servos_deg=_to_vec3(raw.get("initial_servos_deg", [0.0, 25.0, -15.0])),
        servo_limits_deg=tuple((float(item[0]), float(item[1])) for item in limits_raw),  # type: ignore[arg-type]
        servo_speed_deg_s=_to_vec3(raw.get("servo_speed_deg_s", [90.0, 110.0, 130.0])),
    )


def available_scene_presets() -> list[dict[str, str]]:
    presets = []
    for path in sorted((REPO_ROOT / "config" / "interactive").glob("*.yaml")):
        try:
            raw = yaml.safe_load(path.read_text()) or {}
        except Exception:
            raw = {}
        presets.append(
            {
                "config_path": _display_path(path),
                "name": str(raw.get("name", path.stem)),
                "label": str(raw.get("label", raw.get("name", path.stem))),
            }
        )
    return presets


def available_robot_arm_presets() -> list[dict[str, str]]:
    presets = []
    for path in sorted((REPO_ROOT / "config" / "interactive" / "robot_arms").glob("*.yaml")):
        preset = _load_robot_arm_preset(path)
        presets.append(
            {
                "preset_id": preset.preset_id,
                "name": preset.name,
                "label": preset.label,
                "preset_path": _display_path(path),
            }
        )
    return presets


def _default_robot_arm_preset_path_for_scene(config_path: str | Path) -> str:
    resolved_path = _repo_path(config_path)
    raw = yaml.safe_load(resolved_path.read_text()) or {}
    robot_raw = raw.get("robot", {}) or {}
    return _display_path(robot_raw.get("preset_path", "config/interactive/robot_arms/compact_bench.yaml"))


def load_interactive_config(config_path: str | Path, *, robot_arm_preset_path: str | Path | None = None) -> InteractiveSimConfig:
    resolved_path = _repo_path(config_path)
    raw = yaml.safe_load(resolved_path.read_text()) or {}

    device_config_path = str(raw.get("device_config_path", "config/device/phone_default.yaml"))
    device_raw = yaml.safe_load(_repo_path(device_config_path).read_text()) or {}
    camera_rgb = device_raw.get("camera_rgb", {})
    camera_model_toml_path = raw.get("camera_model_toml_path", device_raw.get("camera_model_toml"))

    primary_raw = raw.get("primary_camera", {})
    primary_camera = PrimaryCameraConfig(
        width=int(primary_raw.get("width", camera_rgb.get("width", 960))),
        height=int(primary_raw.get("height", camera_rgb.get("height", 540))),
        fov_deg=float(primary_raw.get("fov_deg", 72.0)),
    )

    scene_raw = raw.get("scene", {})
    observer_raw = list(raw.get("observer_cameras") or [])
    observer_cameras = [
        ObserverCameraConfig(
            name=str(entry["name"]),
            position_m=_to_vec3(entry.get("position_m", entry.get("translation_m", [1.5, 2.5, 1.5]))),
            look_at_m=_to_vec3(entry.get("look_at_m", scene_raw.get("look_at_m", [0.0, 0.9, 1.0]))),
            width=int(entry.get("width", 640)),
            height=int(entry.get("height", 360)),
            fov_deg=float(entry.get("fov_deg", 55.0)),
        )
        for entry in observer_raw
    ]

    tags = [
        SceneTagConfig(
            tag_id=int(entry["tag_id"]),
            family=str(entry.get("family", "36h11")),
            size_m=float(entry.get("size_m", entry.get("edge_length_m", 0.18))),
            position_m=_to_vec3(entry.get("position_m", [0.0, scene_raw.get("wall_y_m", 0.0), 1.0])),
            mount=str(entry.get("mount", "wall")),
            orientation_rpy_deg=(
                _to_vec3(entry["orientation_rpy_deg"])
                if entry.get("orientation_rpy_deg") is not None
                else None
            ),
        )
        for entry in raw.get("tags", [])
    ]

    controls_raw = raw.get("controls", {})
    robot_raw = raw.get("robot", {})
    robot_preset_path = str(robot_arm_preset_path or robot_raw.get("preset_path", "config/interactive/robot_arms/compact_bench.yaml"))
    robot_preset = _load_robot_arm_preset(robot_preset_path)
    limits_raw = controls_raw.get("servo_limits_deg", robot_preset.servo_limits_deg)
    speed_raw = controls_raw.get("servo_speed_deg_s", robot_preset.servo_speed_deg_s)
    auto_demo_raw = raw.get("auto_demo", {})
    auto_demo_channels_raw = auto_demo_raw.get("channels") or []
    default_channels = [
        {"offset_deg": 0.0, "amplitude_deg": 22.0, "frequency_hz": 0.07, "phase_deg": 0.0},
        {"offset_deg": 28.0, "amplitude_deg": 18.0, "frequency_hz": 0.05, "phase_deg": 85.0},
        {"offset_deg": -18.0, "amplitude_deg": 26.0, "frequency_hz": 0.09, "phase_deg": 165.0},
    ]
    channels = []
    for index, defaults in enumerate(default_channels):
        channel_raw = auto_demo_channels_raw[index] if index < len(auto_demo_channels_raw) else {}
        channels.append(
            AutoDemoChannelConfig(
                offset_deg=float(channel_raw.get("offset_deg", defaults["offset_deg"])),
                amplitude_deg=float(channel_raw.get("amplitude_deg", defaults["amplitude_deg"])),
                frequency_hz=float(channel_raw.get("frequency_hz", defaults["frequency_hz"])),
                phase_deg=float(channel_raw.get("phase_deg", defaults["phase_deg"])),
            )
        )

    auto_demo_keyframes = tuple(
        AutoDemoKeyframeConfig(
            time_s=float(entry["time_s"]),
            servos_deg=_to_vec3(entry.get("servos_deg", robot_preset.initial_servos_deg)),
        )
        for entry in (auto_demo_raw.get("keyframes") or [])
    )
    scene_boxes = [
        SceneBoxConfig(
            name=str(entry.get("name", f"box_{index}")),
            center_m=_to_vec3(entry.get("center_m", [0.0, 0.7, 0.4])),
            size_m=_to_vec3(entry.get("size_m", [0.2, 0.2, 0.2])),
            color_bgr=_to_color_bgr(entry.get("color_bgr"), (104, 128, 160)),
            edge_color_bgr=_to_color_bgr(entry.get("edge_color_bgr"), (220, 228, 238)),
        )
        for index, entry in enumerate(scene_raw.get("boxes", []) or [])
    ]

    return InteractiveSimConfig(
        config_path=_display_path(resolved_path),
        name=str(raw.get("name", "browser_game_demo")),
        device_config_path=device_config_path,
        camera_model_toml_path=str(camera_model_toml_path) if camera_model_toml_path else None,
        output_dir=str(raw.get("output_dir", "output/interactive_runs")),
        stream_fps=float(raw.get("stream_fps", 8.0)),
        primary_camera=primary_camera,
        wall_y_m=float(scene_raw.get("wall_y_m", 0.0)),
        wall_width_m=float(scene_raw.get("wall_width_m", 2.6)),
        wall_height_m=float(scene_raw.get("wall_height_m", 1.8)),
        scene_look_at_m=_to_vec3(scene_raw.get("look_at_m", [0.0, 0.0, 1.0])),
        scene_boxes=scene_boxes,
        robot_arm=RobotArmPresetConfig(
            preset_id=robot_preset.preset_id,
            name=robot_preset.name,
            label=robot_preset.label,
            base_position_m=_to_vec3(robot_raw.get("base_position_m", robot_preset.base_position_m)),
            shoulder_height_m=float(robot_raw.get("shoulder_height_m", robot_preset.shoulder_height_m)),
            link_lengths_m=_to_vec3(robot_raw.get("link_lengths_m", robot_preset.link_lengths_m)),
            initial_servos_deg=_to_vec3(controls_raw.get("initial_servos_deg", robot_preset.initial_servos_deg)),
            servo_limits_deg=tuple((float(item[0]), float(item[1])) for item in limits_raw),  # type: ignore[arg-type]
            servo_speed_deg_s=_to_vec3(speed_raw),
        ),
        auto_demo=AutoDemoConfig(
            enabled_by_default=bool(auto_demo_raw.get("enabled_by_default", True)),
            mode=str(auto_demo_raw.get("mode", "channels")),
            loop_duration_s=float(auto_demo_raw.get("loop_duration_s", 16.0)),
            channels=tuple(channels),  # type: ignore[arg-type]
            keyframes=auto_demo_keyframes,
        ),
        auto_analyze_after_recording=bool((raw.get("analysis", {}) or {}).get("auto_run_on_stop", True)),
        tags=tags,
        observer_cameras=observer_cameras,
    )


class InteractiveCalibrationSim:
    """Lightweight interactive calibration sim with rendered views and IMU."""

    def __init__(self, config_path: str | Path) -> None:
        self.detector = AprilTag36h11Detector()
        self.selected_scene_config_path = _display_path(config_path)
        self.selected_robot_arm_preset_path = _default_robot_arm_preset_path_for_scene(config_path)
        self.config = load_interactive_config(
            self.selected_scene_config_path,
            robot_arm_preset_path=self.selected_robot_arm_preset_path,
        )
        self.primary_camera_model = self._build_primary_camera_model()
        self.current_servos_deg = np.array(self.config.robot_arm.initial_servos_deg, dtype=np.float64)
        self.target_servos_deg = np.array(self.config.robot_arm.initial_servos_deg, dtype=np.float64)
        self.time_s = 0.0
        self.tick_index = 0
        self.auto_demo_enabled = self.config.auto_demo.enabled_by_default
        self.recording_enabled = False
        self.active_run_dir: Path | None = None
        self.last_run_dir: Path | None = None
        self._recording_frame_index = 0
        self._recording_writer: ManagedMp4Writer | None = None
        self._imu_csv_handle: Any | None = None
        self._camera_csv_handle: Any | None = None
        self._prev_tool_position: np.ndarray | None = None
        self._prev_tool_velocity = np.zeros(3, dtype=np.float64)
        self._prev_rotation_cw: np.ndarray | None = None
        self._imu_accel_body_mps2 = np.zeros(3, dtype=np.float64)
        self._imu_gyro_body_rps = np.zeros(3, dtype=np.float64)
        self.analysis_status = {
            "state": "idle",
            "last_started_at_utc": None,
            "last_completed_at_utc": None,
            "last_run_dir": None,
            "summary": None,
            "report_path": None,
            "error": None,
        }
        self._load_runtime_assets()

    def catalog_summary(self) -> dict[str, Any]:
        return {
            "scene_presets": available_scene_presets(),
            "robot_arm_presets": available_robot_arm_presets(),
            "selected_scene_config_path": self.selected_scene_config_path,
            "selected_robot_arm_preset_path": self.selected_robot_arm_preset_path,
        }

    def _load_runtime_assets(self) -> None:
        self._tags_runtime = []
        for tag in self.config.tags:
            texture_bgr, texture_render_scale = self._generate_tag_texture(tag.tag_id)
            self._tags_runtime.append(
                SceneTagRuntime(
                    config=tag,
                    texture_bgr=texture_bgr,
                    texture_render_scale=texture_render_scale,
                )
            )

    def reload_config(self, config_path: str | Path) -> dict[str, Any]:
        self._close_recording_artifacts()
        self.selected_scene_config_path = _display_path(config_path)
        self.config = load_interactive_config(
            self.selected_scene_config_path,
            robot_arm_preset_path=self.selected_robot_arm_preset_path,
        )
        self.primary_camera_model = self._build_primary_camera_model()
        self.current_servos_deg = np.array(self.config.robot_arm.initial_servos_deg, dtype=np.float64)
        self.target_servos_deg = np.array(self.config.robot_arm.initial_servos_deg, dtype=np.float64)
        self.time_s = 0.0
        self.tick_index = 0
        self.auto_demo_enabled = self.config.auto_demo.enabled_by_default
        self._prev_tool_position = None
        self._prev_tool_velocity = np.zeros(3, dtype=np.float64)
        self._prev_rotation_cw = None
        self._imu_accel_body_mps2 = np.zeros(3, dtype=np.float64)
        self._imu_gyro_body_rps = np.zeros(3, dtype=np.float64)
        self.recording_enabled = False
        self.active_run_dir = None
        self._recording_frame_index = 0
        self._load_runtime_assets()
        return self.config.summary()

    def set_robot_arm_preset(self, preset_path: str | Path) -> dict[str, Any]:
        self._close_recording_artifacts()
        self.selected_robot_arm_preset_path = _display_path(preset_path)
        self.config = load_interactive_config(
            self.selected_scene_config_path,
            robot_arm_preset_path=self.selected_robot_arm_preset_path,
        )
        self.primary_camera_model = self._build_primary_camera_model()
        self.current_servos_deg = np.array(self.config.robot_arm.initial_servos_deg, dtype=np.float64)
        self.target_servos_deg = np.array(self.config.robot_arm.initial_servos_deg, dtype=np.float64)
        self.time_s = 0.0
        self.tick_index = 0
        self.auto_demo_enabled = self.config.auto_demo.enabled_by_default
        self._prev_tool_position = None
        self._prev_tool_velocity = np.zeros(3, dtype=np.float64)
        self._prev_rotation_cw = None
        self._imu_accel_body_mps2 = np.zeros(3, dtype=np.float64)
        self._imu_gyro_body_rps = np.zeros(3, dtype=np.float64)
        self.recording_enabled = False
        self.active_run_dir = None
        self._recording_frame_index = 0
        self._load_runtime_assets()
        return self.config.summary()

    def _build_primary_camera_model(self) -> PhoneCameraModel | None:
        if not self.config.camera_model_toml_path:
            return None
        return load_phone_camera_model(
            self.config.camera_model_toml_path,
            output_width_px=self.config.primary_camera.width,
            output_height_px=self.config.primary_camera.height,
        )

    def set_recording(self, enabled: bool) -> None:
        if enabled and not self.recording_enabled:
            self.active_run_dir = self._create_run_dir()
            self.last_run_dir = self.active_run_dir
            metadata = {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "config": self.config.summary(),
                "camera_model": self.primary_camera_model.summary() if self.primary_camera_model else None,
                "recording": {
                    "video_file": "phone_raw.mp4",
                    "samples_file": "samples.jsonl",
                    "imu_file": "imu.csv",
                    "camera_file": "camera_gt.csv",
                    "raw_frames_are_distorted": bool(
                        self.primary_camera_model and self.primary_camera_model.apply_lens_distortion_in_render
                    ),
                    "frame_rate_hz": float(self.config.stream_fps),
                },
            }
            (self.active_run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
            self._open_recording_artifacts()
        if not enabled and self.recording_enabled:
            last_run_dir = self.active_run_dir
            self._close_recording_artifacts()
            self.active_run_dir = None
            if last_run_dir is not None and self.config.auto_analyze_after_recording:
                self.run_analysis_on_run_dir(last_run_dir)
        self.recording_enabled = enabled

    def _open_recording_artifacts(self) -> None:
        if self.active_run_dir is None:
            return
        width = self.config.primary_camera.width
        height = self.config.primary_camera.height
        self._recording_writer = ManagedMp4Writer(
            self.active_run_dir / "phone_raw.mp4",
            fps=float(max(self.config.stream_fps, 1.0)),
            frame_size=(width, height),
        )
        self._imu_csv_handle = (self.active_run_dir / "imu.csv").open("w", encoding="utf-8")
        self._imu_csv_handle.write("tick_index,sim_time_s,ax_mps2,ay_mps2,az_mps2,gx_rps,gy_rps,gz_rps\n")
        self._camera_csv_handle = (self.active_run_dir / "camera_gt.csv").open("w", encoding="utf-8")
        self._camera_csv_handle.write(
            "tick_index,sim_time_s,cx_world_m,cy_world_m,cz_world_m,vx_world_mps,vy_world_mps,vz_world_mps,"
            "r00,r01,r02,r10,r11,r12,r20,r21,r22,servo0_deg,servo1_deg,servo2_deg\n"
        )
        self._recording_frame_index = 0

    def _close_recording_artifacts(self) -> None:
        if self._recording_writer is not None:
            self._recording_writer.close()
            self._recording_writer = None
        if self._imu_csv_handle is not None:
            self._imu_csv_handle.close()
            self._imu_csv_handle = None
        if self._camera_csv_handle is not None:
            self._camera_csv_handle.close()
            self._camera_csv_handle = None

    def set_auto_demo(self, enabled: bool) -> None:
        self.auto_demo_enabled = bool(enabled)

    def run_analysis_on_last_run(self) -> dict[str, Any]:
        if self.last_run_dir is None:
            raise FileNotFoundError("No recorded run is available yet.")
        return self.run_analysis_on_run_dir(self.last_run_dir)

    def run_analysis_on_run_dir(self, run_dir: str | Path) -> dict[str, Any]:
        resolved_run_dir = _repo_path(run_dir)
        started_at = datetime.now(timezone.utc).isoformat()
        self.analysis_status = {
            "state": "running",
            "last_started_at_utc": started_at,
            "last_completed_at_utc": None,
            "last_run_dir": self._relative_path(resolved_run_dir),
            "summary": None,
            "report_path": None,
            "error": None,
        }
        try:
            summary = analyze_recording_run(resolved_run_dir)
        except Exception as exc:
            self.analysis_status = {
                "state": "failed",
                "last_started_at_utc": started_at,
                "last_completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "last_run_dir": self._relative_path(resolved_run_dir),
                "summary": None,
                "report_path": None,
                "error": str(exc),
            }
            raise

        self.analysis_status = {
            "state": "completed",
            "last_started_at_utc": started_at,
            "last_completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "last_run_dir": self._relative_path(resolved_run_dir),
            "summary": summary,
            "report_path": self._relative_path(resolved_run_dir / "analysis" / "report.md"),
            "error": None,
        }
        return summary

    def set_servo_targets(self, targets_deg: Sequence[float]) -> None:
        clipped = []
        for index, limit in enumerate(self.config.robot_arm.servo_limits_deg):
            low, high = limit
            clipped.append(float(np.clip(float(targets_deg[index]), low, high)))
        self.target_servos_deg = np.array(clipped, dtype=np.float64)
        self.auto_demo_enabled = False

    def nudge_servo_targets(self, delta_deg: Sequence[float]) -> None:
        self.set_servo_targets(self.target_servos_deg + np.asarray(delta_deg, dtype=np.float64))

    def step(self, dt_s: float) -> None:
        dt_s = float(np.clip(dt_s, 1.0 / 240.0, 0.2))
        if self.auto_demo_enabled:
            self.target_servos_deg = self._demo_servo_targets(self.time_s + dt_s)
        max_step = np.asarray(self.config.robot_arm.servo_speed_deg_s, dtype=np.float64) * dt_s
        error = self.target_servos_deg - self.current_servos_deg
        self.current_servos_deg += np.clip(error, -max_step, max_step)
        self.time_s += dt_s
        self.tick_index += 1

        tool_pose = self._phone_camera_pose()
        tool_position = tool_pose.position_m

        if self._prev_tool_position is not None:
            velocity = (tool_position - self._prev_tool_position) / dt_s
            acceleration_world = (velocity - self._prev_tool_velocity) / dt_s
        else:
            velocity = np.zeros(3, dtype=np.float64)
            acceleration_world = np.zeros(3, dtype=np.float64)

        gravity_world = np.array([0.0, 0.0, -9.81], dtype=np.float64)
        self._imu_accel_body_mps2 = tool_pose.rotation_cw.T @ (acceleration_world - gravity_world)

        if self._prev_rotation_cw is not None:
            relative_rotation = self._prev_rotation_cw.T @ tool_pose.rotation_cw
            rvec, _ = cv2.Rodrigues(relative_rotation)
            self._imu_gyro_body_rps = rvec.reshape(3) / dt_s
        else:
            self._imu_gyro_body_rps = np.zeros(3, dtype=np.float64)

        self._prev_tool_position = tool_position.copy()
        self._prev_tool_velocity = velocity
        self._prev_rotation_cw = tool_pose.rotation_cw.copy()

    def _demo_servo_targets(self, time_s: float) -> np.ndarray:
        if self.config.auto_demo.mode == "keyframes" and self.config.auto_demo.keyframes:
            keyframes = self.config.auto_demo.keyframes
            loop_duration_s = max(float(self.config.auto_demo.loop_duration_s), float(keyframes[-1].time_s), 1e-6)
            wrapped_time_s = float(time_s % loop_duration_s)
            ordered_keyframes = sorted(keyframes, key=lambda item: item.time_s)
            previous = ordered_keyframes[-1]
            next_frame = ordered_keyframes[0]
            for candidate in ordered_keyframes:
                if candidate.time_s <= wrapped_time_s:
                    previous = candidate
                if candidate.time_s > wrapped_time_s:
                    next_frame = candidate
                    break
            previous_time = float(previous.time_s)
            next_time = float(next_frame.time_s)
            if next_time <= previous_time:
                next_time += loop_duration_s
            t = wrapped_time_s
            if t < previous_time:
                t += loop_duration_s
            alpha = (t - previous_time) / max(next_time - previous_time, 1e-9)
            eased_alpha = 0.5 - 0.5 * math.cos(math.pi * float(np.clip(alpha, 0.0, 1.0)))
            start = np.asarray(previous.servos_deg, dtype=np.float64)
            end = np.asarray(next_frame.servos_deg, dtype=np.float64)
            blended = start + (end - start) * eased_alpha
            clipped = []
            for index, limit in enumerate(self.config.robot_arm.servo_limits_deg):
                low, high = limit
                clipped.append(float(np.clip(blended[index], low, high)))
            return np.asarray(clipped, dtype=np.float64)
        targets = []
        for limits, channel in zip(self.config.robot_arm.servo_limits_deg, self.config.auto_demo.channels):
            value = channel.offset_deg + channel.amplitude_deg * math.sin(
                2.0 * math.pi * channel.frequency_hz * time_s + math.radians(channel.phase_deg)
            )
            low, high = limits
            targets.append(float(np.clip(value, low, high)))
        return np.asarray(targets, dtype=np.float64)

    def _update_detection_pose_estimates(
        self,
        detections: Any,
        *,
        intrinsics: dict[str, float],
    ) -> None:
        tag_size_by_id = {int(tag.tag_id): float(tag.size_m) for tag in self.config.tags}
        dist_coeffs = (
            self.primary_camera_model.distortion_coefficients_for_rendered_output()
            if self.primary_camera_model is not None
            else None
        )
        camera_matrix = np.array(
            [
                [float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])],
                [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        distortion = np.zeros((5, 1), dtype=np.float64) if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64)
        for item in detections.detections:
            tag_size_m = tag_size_by_id.get(int(item.tag_id))
            if tag_size_m is None:
                continue
            half = tag_size_m * 0.5
            object_points = np.array(
                [
                    [-half, -half, 0.0],
                    [half, -half, 0.0],
                    [half, half, 0.0],
                    [-half, half, 0.0],
                ],
                dtype=np.float64,
            )
            image_points = np.asarray(item.corners_xy_clockwise, dtype=np.float64).reshape(4, 2)
            success, rvec, tvec = cv2.solvePnP(
                objectPoints=object_points,
                imagePoints=image_points,
                cameraMatrix=camera_matrix,
                distCoeffs=distortion,
                flags=getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE),
            )
            if success:
                item.pose_camera_rvec = tuple(float(value) for value in rvec.reshape(3))
                item.pose_camera_tvec = tuple(float(value) for value in tvec.reshape(3))

    def render_snapshot(self) -> dict[str, Any]:
        phone_pose = self._phone_camera_pose()
        joint_chain = self._joint_chain_world()
        phone_raw = self._render_primary_camera(phone_pose)
        primary_intrinsics = self._primary_intrinsics()
        tag_size_m = float(self.config.tags[0].size_m) if self.config.tags else 0.18
        detections = self.detector.detect_image(
            phone_raw,
            frame_index=self.tick_index,
            timestamp_s=self.time_s,
            fx=primary_intrinsics["fx"],
            fy=primary_intrinsics["fy"],
            cx=primary_intrinsics["cx"],
            cy=primary_intrinsics["cy"],
            dist_coeffs=(
                self.primary_camera_model.distortion_coefficients_for_rendered_output()
                if self.primary_camera_model
                else None
            ),
            tag_size_m=tag_size_m,
        )
        self._update_detection_pose_estimates(detections, intrinsics=primary_intrinsics)
        phone_ground_truth = self._ground_truth_payload(phone_pose)
        phone_annotated = self._annotate_phone_frame(phone_raw.copy(), detections)
        phone_annotated = self._annotate_phone_status(phone_annotated)
        detected_ids = {item.tag_id for item in detections.detections}
        observer_frames = [
            {
                "name": camera.name,
                "image_data_url": _encode_jpeg_data_url(
                    self._render_observer_camera(camera, joint_chain, phone_pose, detected_ids)
                ),
            }
            for camera in self.config.observer_cameras
        ]

        phone_payload = self.detector.to_jsonable(detections)
        phone_payload["image_data_url"] = _encode_jpeg_data_url(phone_annotated)
        phone_payload["raw_image_data_url"] = _encode_jpeg_data_url(phone_raw)
        phone_payload["intrinsics"] = primary_intrinsics
        phone_payload["camera_model"] = (
            self.primary_camera_model.summary()
            if self.primary_camera_model is not None
            else {
                "name": "synthetic_pinhole_fov",
                "source_toml_path": None,
                "output_resolution_px": [self.config.primary_camera.width, self.config.primary_camera.height],
                "effective_intrinsics_px": primary_intrinsics,
                "distortion": {"model": "none", "coefficients": [0.0, 0.0, 0.0, 0.0, 0.0]},
            }
        )
        phone_payload["camera_pose_estimation"] = (
            [self.primary_camera_model.camera_pose_estimation_measurement(item) for item in detections.detections]
            if self.primary_camera_model is not None
            else []
        )
        phone_payload["ground_truth"] = phone_ground_truth

        snapshot = {
            "sim_time_s": round(self.time_s, 3),
            "tick_index": self.tick_index,
            "config": self.config.summary(),
            "catalog": self.catalog_summary(),
            "servo_positions_deg": [round(float(value), 2) for value in self.current_servos_deg.tolist()],
            "servo_targets_deg": [round(float(value), 2) for value in self.target_servos_deg.tolist()],
            "automation": {
                "auto_demo_enabled": self.auto_demo_enabled,
            },
            "imu": {
                "accel_mps2": [round(float(value), 3) for value in self._imu_accel_body_mps2.tolist()],
                "gyro_rps": [round(float(value), 4) for value in self._imu_gyro_body_rps.tolist()],
            },
            "phone_view": phone_payload,
            "observer_views": observer_frames,
            "recording": {
                "enabled": self.recording_enabled,
                "active_run_dir": self._relative_path(self.active_run_dir),
                "last_run_dir": self._relative_path(self.last_run_dir),
                "output_dir": self.config.output_dir,
            },
            "analysis": self.analysis_status,
        }

        if self.recording_enabled and self.active_run_dir is not None:
            self._write_recording_snapshot(snapshot, phone_raw, observer_frames)

        return snapshot

    def _relative_path(self, path: Path | None) -> str | None:
        if path is None:
            return None
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            return str(path)

    def _create_run_dir(self) -> Path:
        base_dir = _repo_path(self.config.output_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_dir = base_dir / timestamp
        suffix = 1
        while run_dir.exists():
            suffix += 1
            run_dir = base_dir / f"{timestamp}_{suffix:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _write_recording_snapshot(
        self,
        snapshot: dict[str, Any],
        phone_image_bgr: np.ndarray,
        observer_frames: list[dict[str, Any]],
    ) -> None:
        if self.active_run_dir is None:
            return

        self._recording_frame_index += 1
        if self._recording_writer is not None:
            self._recording_writer.write(phone_image_bgr)

        files = {
            "phone_raw_video": "phone_raw.mp4",
            "analysis_report": "analysis/report.md",
        }
        if observer_frames:
            files["observer_views"] = [observer["name"] for observer in observer_frames]

        if self._imu_csv_handle is not None:
            accel = self._imu_accel_body_mps2
            gyro = self._imu_gyro_body_rps
            self._imu_csv_handle.write(
                f"{self.tick_index},{self.time_s:.9f},"
                f"{accel[0]:.9f},{accel[1]:.9f},{accel[2]:.9f},"
                f"{gyro[0]:.9f},{gyro[1]:.9f},{gyro[2]:.9f}\n"
            )

        if self._camera_csv_handle is not None:
            camera_world_pose = snapshot["phone_view"]["ground_truth"]["camera_world_pose"]
            rotation = camera_world_pose["rotation_cw"]
            velocity_world_mps = self._prev_tool_velocity if self._prev_tool_velocity is not None else np.zeros(3, dtype=np.float64)
            self._camera_csv_handle.write(
                f"{self.tick_index},{self.time_s:.9f},"
                f"{camera_world_pose['position_m'][0]:.9f},{camera_world_pose['position_m'][1]:.9f},{camera_world_pose['position_m'][2]:.9f},"
                f"{velocity_world_mps[0]:.9f},{velocity_world_mps[1]:.9f},{velocity_world_mps[2]:.9f},"
                f"{rotation[0]:.9f},{rotation[1]:.9f},{rotation[2]:.9f},"
                f"{rotation[3]:.9f},{rotation[4]:.9f},{rotation[5]:.9f},"
                f"{rotation[6]:.9f},{rotation[7]:.9f},{rotation[8]:.9f},"
                f"{snapshot['servo_positions_deg'][0]:.9f},{snapshot['servo_positions_deg'][1]:.9f},{snapshot['servo_positions_deg'][2]:.9f}\n"
            )

        log_entry = {
            "sim_time_s": self.time_s,
            "tick_index": self.tick_index,
            "recording_frame_index": self._recording_frame_index,
            "servo_positions_deg": snapshot["servo_positions_deg"],
            "servo_targets_deg": snapshot["servo_targets_deg"],
            "automation": snapshot["automation"],
            "imu": {
                "accel_mps2": [float(value) for value in self._imu_accel_body_mps2.tolist()],
                "gyro_rps": [float(value) for value in self._imu_gyro_body_rps.tolist()],
            },
            "detections": snapshot["phone_view"]["detections"],
            "camera_model": snapshot["phone_view"].get("camera_model"),
            "camera_pose_estimation": snapshot["phone_view"].get("camera_pose_estimation", []),
            "ground_truth": snapshot["phone_view"].get("ground_truth"),
            "files": files,
        }
        with (self.active_run_dir / "samples.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log_entry) + "\n")

    def _ground_truth_payload(self, phone_pose: CameraPose) -> dict[str, Any]:
        camera_world_pose = {
            "position_m": [float(value) for value in phone_pose.position_m.tolist()],
            "rotation_cw": [float(value) for value in phone_pose.rotation_cw.reshape(-1).tolist()],
        }
        return {
            "camera_world_pose": camera_world_pose,
            "tags": [self._tag_ground_truth(tag, phone_pose) for tag in self._tags_runtime],
        }

    def _tag_ground_truth(self, tag: SceneTagRuntime, phone_pose: CameraPose) -> dict[str, Any]:
        tag_center_world = np.asarray(tag.config.position_m, dtype=np.float64)
        tag_rotation_wt = tag.rotation_wt()
        tag_rotation_tw = tag_rotation_wt.T
        half_extent = tag.config.size_m * 0.5
        object_points_tag = np.array(
            [
                [0.0, 0.0, 0.0],
                [half_extent, -half_extent, 0.0],
                [half_extent, half_extent, 0.0],
                [-half_extent, half_extent, 0.0],
                [-half_extent, -half_extent, 0.0],
            ],
            dtype=np.float64,
        )
        object_points_world = np.array([tag_center_world + tag_rotation_wt @ point for point in object_points_tag], dtype=np.float64)
        points_camera = (phone_pose.rotation_cw.T @ (object_points_world - phone_pose.position_m).T).T
        visible = bool(np.all(points_camera[:, 2] > 0.05))

        if self.primary_camera_model is not None:
            ideal_pixels, _ = self.primary_camera_model.project_camera_points(points_camera, apply_distortion=False)
            rendered_pixels, _ = self.primary_camera_model.project_camera_points(
                points_camera,
                apply_distortion=self.primary_camera_model.apply_lens_distortion_in_render,
            )
            distorted_pixels, _ = self.primary_camera_model.project_camera_points(points_camera, apply_distortion=True)
            image_plane_points_m = self.primary_camera_model.pixels_to_image_plane_mm(ideal_pixels, undistort=False) / 1000.0
        else:
            ideal_pixels, _ = self._project_points(
                phone_pose,
                self.config.primary_camera.width,
                self.config.primary_camera.height,
                object_points_world,
                intrinsics=self._primary_intrinsics(),
            )
            rendered_pixels = ideal_pixels.copy()
            distorted_pixels = ideal_pixels.copy()
            fx = float(self._primary_intrinsics()["fx"])
            fy = float(self._primary_intrinsics()["fy"])
            cx = float(self._primary_intrinsics()["cx"])
            cy = float(self._primary_intrinsics()["cy"])
            image_plane_points_m = np.column_stack(
                (
                    ((ideal_pixels[:, 0] - cx) / fx) * 0.008,
                    -((ideal_pixels[:, 1] - cy) / fy) * 0.008,
                )
            )

        camera_position_tag = tag_rotation_tw @ (phone_pose.position_m - tag_center_world)
        camera_rotation_tc = tag_rotation_tw @ phone_pose.rotation_cw
        camera_rvec_tag, _ = cv2.Rodrigues(camera_rotation_tc)
        return {
            "tag_id": tag.config.tag_id,
            "family": tag.config.family,
            "size_m": tag.config.size_m,
            "visible": visible,
            "tag_center_world_m": [float(value) for value in tag_center_world.tolist()],
            "camera_position_tag_m": [float(value) for value in camera_position_tag.tolist()],
            "camera_rotation_tc": [float(value) for value in camera_rotation_tc.reshape(-1).tolist()],
            "camera_rvec_tag_rad": [float(value) for value in camera_rvec_tag.reshape(3).tolist()],
            "object_points_tag_m": [[float(x), float(y), float(z)] for x, y, z in object_points_tag.tolist()],
            "points_camera_m": [[float(x), float(y), float(z)] for x, y, z in points_camera.tolist()],
            "ideal_image_points_px": [[float(x), float(y)] for x, y in ideal_pixels.tolist()],
            "rendered_image_points_px": [[float(x), float(y)] for x, y in rendered_pixels.tolist()],
            "distorted_image_points_px": [[float(x), float(y)] for x, y in distorted_pixels.tolist()],
            "image_plane_points_m": [[float(x), float(y)] for x, y in image_plane_points_m.tolist()],
        }

    def _generate_tag_texture(self, tag_id: int) -> tuple[np.ndarray, float]:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        marker_size_px = 512
        quiet_zone_px = 56
        marker = cv2.aruco.generateImageMarker(dictionary, int(tag_id), marker_size_px)
        marker = cv2.copyMakeBorder(
            marker,
            quiet_zone_px,
            quiet_zone_px,
            quiet_zone_px,
            quiet_zone_px,
            cv2.BORDER_CONSTANT,
            value=255,
        )
        texture_render_scale = (marker_size_px + 2.0 * quiet_zone_px) / float(marker_size_px)
        return cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR), float(texture_render_scale)

    def _joint_chain_world(self) -> list[np.ndarray]:
        q1, q2, q3 = np.radians(self.current_servos_deg)
        l1, l2, l3 = self.config.robot_arm.link_lengths_m
        base = np.asarray(self.config.robot_arm.base_position_m, dtype=np.float64)
        shoulder = base + np.array([0.0, 0.0, self.config.robot_arm.shoulder_height_m], dtype=np.float64)

        first_reach = l1 * math.cos(q2)
        first_rise = l1 * math.sin(q2)
        elbow = shoulder + np.array([first_reach * math.cos(q1), first_reach * math.sin(q1), first_rise])

        second_reach = l2 * math.cos(q2 + q3)
        second_rise = l2 * math.sin(q2 + q3)
        wrist = elbow + np.array([second_reach * math.cos(q1), second_reach * math.sin(q1), second_rise])

        tool_reach = l3 * math.cos(q2 + q3)
        tool_rise = l3 * math.sin(q2 + q3)
        tool = wrist + np.array([tool_reach * math.cos(q1), tool_reach * math.sin(q1), tool_rise])
        return [base, shoulder, elbow, wrist, tool]

    def _phone_camera_pose(self) -> CameraPose:
        tool_position = self._joint_chain_world()[-1]
        look_at_point = np.asarray(self.config.scene_look_at_m, dtype=np.float64)
        look_at_point = look_at_point + np.array(
            [
                0.20 * math.sin(math.radians(float(self.current_servos_deg[0]))),
                0.0,
                0.10 * math.sin(math.radians(float(self.current_servos_deg[2]))),
            ],
            dtype=np.float64,
        )
        rotation_cw = _look_at_rotation(tool_position, look_at_point)
        return CameraPose(position_m=tool_position, rotation_cw=rotation_cw)

    def _primary_intrinsics(self) -> dict[str, float]:
        if self.primary_camera_model is not None:
            return {
                "fx": self.primary_camera_model.fx_px,
                "fy": self.primary_camera_model.fy_px,
                "cx": self.primary_camera_model.cx_px,
                "cy": self.primary_camera_model.cy_px,
            }
        fx, fy, cx, cy = _camera_intrinsics(
            self.config.primary_camera.width,
            self.config.primary_camera.height,
            self.config.primary_camera.fov_deg,
        )
        return {"fx": fx, "fy": fy, "cx": cx, "cy": cy}

    def _project_points(
        self,
        camera_pose: CameraPose,
        width: int,
        height: int,
        points_world: Sequence[Sequence[float]] | np.ndarray,
        *,
        fov_deg: float | None = None,
        intrinsics: dict[str, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points_world, dtype=np.float64)
        camera_points = (camera_pose.rotation_cw.T @ (points - camera_pose.position_m).T).T
        if intrinsics is None:
            if fov_deg is None:
                raise ValueError("Either fov_deg or intrinsics must be provided.")
            fx, fy, cx, cy = _camera_intrinsics(width, height, fov_deg)
        else:
            fx = float(intrinsics["fx"])
            fy = float(intrinsics["fy"])
            cx = float(intrinsics["cx"])
            cy = float(intrinsics["cy"])
        with np.errstate(divide="ignore", invalid="ignore"):
            pixels_x = fx * (camera_points[:, 0] / camera_points[:, 2]) + cx
            pixels_y = cy - fy * (camera_points[:, 1] / camera_points[:, 2])
        return np.column_stack((pixels_x, pixels_y)), camera_points[:, 2] > 0.05

    def _draw_world_grid(
        self,
        canvas: np.ndarray,
        camera_pose: CameraPose,
        width: int,
        height: int,
        *,
        fov_deg: float | None = None,
        intrinsics: dict[str, float] | None = None,
    ) -> None:
        wall_y = self.config.wall_y_m
        wall_w = self.config.wall_width_m
        wall_h = self.config.wall_height_m
        wall_corners = np.array(
            [
                [-wall_w * 0.5, wall_y, 0.0],
                [wall_w * 0.5, wall_y, 0.0],
                [wall_w * 0.5, wall_y, wall_h],
                [-wall_w * 0.5, wall_y, wall_h],
            ],
            dtype=np.float64,
        )
        pixels, visible = self._project_points(
            camera_pose,
            width,
            height,
            wall_corners,
            fov_deg=fov_deg,
            intrinsics=intrinsics,
        )
        if bool(np.all(visible)):
            wall_poly = np.round(pixels).astype(np.int32)
            cv2.fillConvexPoly(canvas, wall_poly, (234, 233, 224))

        for x_coord in np.linspace(-wall_w * 0.5, wall_w * 0.5, 9):
            self._draw_segment(
                canvas,
                camera_pose,
                width,
                height,
                [x_coord, wall_y, 0.0],
                [x_coord, wall_y, wall_h],
                color=(190, 194, 198),
                thickness=1,
                fov_deg=fov_deg,
                intrinsics=intrinsics,
            )

        for z_coord in np.linspace(0.0, wall_h, 7):
            self._draw_segment(
                canvas,
                camera_pose,
                width,
                height,
                [-wall_w * 0.5, wall_y, z_coord],
                [wall_w * 0.5, wall_y, z_coord],
                color=(190, 194, 198),
                thickness=1,
                fov_deg=fov_deg,
                intrinsics=intrinsics,
            )

    def _draw_segment(
        self,
        canvas: np.ndarray,
        camera_pose: CameraPose,
        width: int,
        height: int,
        start_world: Sequence[float],
        end_world: Sequence[float],
        *,
        color: tuple[int, int, int],
        thickness: int,
        fov_deg: float | None = None,
        intrinsics: dict[str, float] | None = None,
    ) -> None:
        pixels, visible = self._project_points(
            camera_pose,
            width,
            height,
            [start_world, end_world],
            fov_deg=fov_deg,
            intrinsics=intrinsics,
        )
        if not bool(np.all(visible)):
            return
        start_px = tuple(int(round(value)) for value in pixels[0])
        end_px = tuple(int(round(value)) for value in pixels[1])
        cv2.line(canvas, start_px, end_px, color, thickness, cv2.LINE_AA)

    def _draw_scene_boxes(
        self,
        canvas: np.ndarray,
        camera_pose: CameraPose,
        width: int,
        height: int,
        *,
        fov_deg: float | None = None,
        intrinsics: dict[str, float] | None = None,
    ) -> None:
        for box in self.config.scene_boxes:
            center = np.asarray(box.center_m, dtype=np.float64)
            size = np.asarray(box.size_m, dtype=np.float64)
            half = size * 0.5
            corners = np.array(
                [
                    center + [-half[0], -half[1], -half[2]],
                    center + [half[0], -half[1], -half[2]],
                    center + [half[0], half[1], -half[2]],
                    center + [-half[0], half[1], -half[2]],
                    center + [-half[0], -half[1], half[2]],
                    center + [half[0], -half[1], half[2]],
                    center + [half[0], half[1], half[2]],
                    center + [-half[0], half[1], half[2]],
                ],
                dtype=np.float64,
            )
            pixels, visible = self._project_points(
                camera_pose,
                width,
                height,
                corners,
                fov_deg=fov_deg,
                intrinsics=intrinsics,
            )
            if not bool(np.all(visible)):
                continue
            faces = [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [0, 1, 5, 4],
                [1, 2, 6, 5],
                [2, 3, 7, 6],
                [3, 0, 4, 7],
            ]
            face_depths = []
            camera_space = (camera_pose.rotation_cw.T @ (corners - camera_pose.position_m).T).T
            for face in faces:
                face_depths.append((float(np.mean(camera_space[face, 2])), face))
            for _depth, face in sorted(face_depths, reverse=True):
                polygon = np.round(pixels[face]).astype(np.int32)
                cv2.fillConvexPoly(canvas, polygon, box.color_bgr)
                cv2.polylines(canvas, [polygon], True, box.edge_color_bgr, 1, cv2.LINE_AA)

    def _overlay_tag(
        self,
        canvas: np.ndarray,
        camera_pose: CameraPose,
        width: int,
        height: int,
        tag: SceneTagRuntime,
        *,
        fov_deg: float | None = None,
        intrinsics: dict[str, float] | None = None,
    ) -> bool:
        pixels, visible = self._project_points(
            camera_pose,
            width,
            height,
            tag.render_corners_world(),
            fov_deg=fov_deg,
            intrinsics=intrinsics,
        )
        if not bool(np.all(visible)):
            return False

        min_x, min_y = np.min(pixels[:, 0]), np.min(pixels[:, 1])
        max_x, max_y = np.max(pixels[:, 0]), np.max(pixels[:, 1])
        if max_x < 0 or max_y < 0 or min_x > width or min_y > height:
            return False
        if (max_x - min_x) < 12 or (max_y - min_y) < 12:
            return False

        texture = tag.texture_bgr
        src = np.array(
            [
                [0.0, 0.0],
                [texture.shape[1] - 1.0, 0.0],
                [texture.shape[1] - 1.0, texture.shape[0] - 1.0],
                [0.0, texture.shape[0] - 1.0],
            ],
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(src, pixels.astype(np.float32))
        warped = cv2.warpPerspective(
            texture,
            transform,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        mask = cv2.warpPerspective(
            np.full(texture.shape[:2], 255, dtype=np.uint8),
            transform,
            (width, height),
            flags=cv2.INTER_NEAREST,
        )
        canvas[mask > 0] = warped[mask > 0]
        return True

    def _render_primary_camera(self, camera_pose: CameraPose) -> np.ndarray:
        width = self.config.primary_camera.width
        height = self.config.primary_camera.height
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        gradient = np.linspace(18, 58, height, dtype=np.uint8).reshape(height, 1)
        canvas[:, :, 0] = gradient
        canvas[:, :, 1] = np.clip(gradient + 8, 0, 255)
        canvas[:, :, 2] = np.clip(gradient + 18, 0, 255)

        primary_intrinsics = self._primary_intrinsics()
        self._draw_world_grid(canvas, camera_pose, width, height, intrinsics=primary_intrinsics)
        self._draw_scene_boxes(canvas, camera_pose, width, height, intrinsics=primary_intrinsics)
        for tag in self._tags_runtime:
            self._overlay_tag(canvas, camera_pose, width, height, tag, intrinsics=primary_intrinsics)

        if self.primary_camera_model is not None:
            canvas = self.primary_camera_model.apply_lens_distortion(canvas)
        return canvas

    def _annotate_phone_frame(self, frame_bgr: np.ndarray, detections: Any) -> np.ndarray:
        for item in detections.detections:
            corners = np.asarray(item.corners_xy_clockwise, dtype=np.float64).reshape(-1, 2)
            polygon = np.round(corners).astype(np.int32)
            cv2.polylines(frame_bgr, [polygon], True, (16, 235, 122), 2, cv2.LINE_AA)

            for index, point in enumerate(item.points5_xy):
                center = (int(round(point[0])), int(round(point[1])))
                cv2.circle(frame_bgr, center, 5 if index == 4 else 4, (38, 147, 255), -1, cv2.LINE_AA)

            label = f"id {item.tag_id}"
            if item.pose_camera_tvec is not None:
                label += f"  z={item.pose_camera_tvec[2]:.2f}m"
            anchor = (int(round(item.center_xy[0])) + 10, int(round(item.center_xy[1])) - 10)
            cv2.putText(frame_bgr, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 12, 18), 4, cv2.LINE_AA)
            cv2.putText(frame_bgr, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 248, 250), 2, cv2.LINE_AA)
        return frame_bgr

    def _render_observer_camera(
        self,
        camera: ObserverCameraConfig,
        joint_chain: Sequence[np.ndarray],
        phone_pose: CameraPose,
        detected_ids: Iterable[int],
    ) -> np.ndarray:
        pose = CameraPose(
            position_m=np.asarray(camera.position_m, dtype=np.float64),
            rotation_cw=_look_at_rotation(np.asarray(camera.position_m, dtype=np.float64), np.asarray(camera.look_at_m, dtype=np.float64)),
        )
        canvas = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
        gradient = np.linspace(12, 44, camera.height, dtype=np.uint8).reshape(camera.height, 1)
        canvas[:, :, 0] = np.clip(gradient + 14, 0, 255)
        canvas[:, :, 1] = np.clip(gradient + 26, 0, 255)
        canvas[:, :, 2] = np.clip(gradient + 8, 0, 255)

        self._draw_world_grid(canvas, pose, camera.width, camera.height, fov_deg=camera.fov_deg)
        self._draw_scene_boxes(canvas, pose, camera.width, camera.height, fov_deg=camera.fov_deg)
        detected_ids = set(detected_ids)
        for tag in self._tags_runtime:
            rendered = self._overlay_tag(canvas, pose, camera.width, camera.height, tag, fov_deg=camera.fov_deg)
            if not rendered:
                continue
            pixels, visible = self._project_points(
                pose,
                camera.width,
                camera.height,
                tag.detection_corners_world(),
                fov_deg=camera.fov_deg,
            )
            if bool(np.all(visible)) and tag.config.tag_id in detected_ids:
                polygon = np.round(pixels).astype(np.int32)
                cv2.polylines(canvas, [polygon], True, (16, 235, 122), 2, cv2.LINE_AA)
                label_point = tuple(int(round(value)) for value in pixels.mean(axis=0))
                cv2.putText(
                    canvas,
                    f"detected {tag.config.tag_id}",
                    (label_point[0] - 42, label_point[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (245, 248, 250),
                    1,
                    cv2.LINE_AA,
                )

        projected_chain, visible = self._project_points(
            pose,
            camera.width,
            camera.height,
            joint_chain,
            fov_deg=camera.fov_deg,
        )
        for index in range(len(projected_chain) - 1):
            if not (visible[index] and visible[index + 1]):
                continue
            start = tuple(int(round(value)) for value in projected_chain[index])
            end = tuple(int(round(value)) for value in projected_chain[index + 1])
            cv2.line(canvas, start, end, (255, 182, 49), 4, cv2.LINE_AA)
            cv2.circle(canvas, start, 6, (255, 239, 184), -1, cv2.LINE_AA)
        if len(projected_chain) > 0 and bool(visible[-1]):
            tool_px = tuple(int(round(value)) for value in projected_chain[-1])
            cv2.circle(canvas, tool_px, 7, (72, 214, 255), -1, cv2.LINE_AA)

        self._draw_phone_frustum(canvas, pose, camera, phone_pose)
        cv2.putText(canvas, camera.name.upper(), (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245, 248, 250), 2, cv2.LINE_AA)
        return canvas

    def _draw_phone_frustum(
        self,
        canvas: np.ndarray,
        observer_pose: CameraPose,
        observer_camera: ObserverCameraConfig,
        phone_pose: CameraPose,
    ) -> None:
        depth = 0.22
        if self.primary_camera_model is not None:
            half_width = depth * math.tan(math.radians(self.primary_camera_model.horizontal_fov_deg()) * 0.5)
            half_height = depth * math.tan(math.radians(self.primary_camera_model.vertical_fov_deg()) * 0.5)
        else:
            aspect = self.config.primary_camera.width / max(1.0, float(self.config.primary_camera.height))
            half_width = depth * math.tan(math.radians(self.config.primary_camera.fov_deg) * 0.5)
            half_height = half_width / aspect
        frustum_camera = np.array(
            [
                [0.0, 0.0, 0.0],
                [-half_width, half_height, depth],
                [half_width, half_height, depth],
                [half_width, -half_height, depth],
                [-half_width, -half_height, depth],
            ],
            dtype=np.float64,
        )
        frustum_world = np.array([phone_pose.position_m + phone_pose.rotation_cw @ point for point in frustum_camera])
        pixels, visible = self._project_points(
            observer_pose,
            observer_camera.width,
            observer_camera.height,
            frustum_world,
            fov_deg=observer_camera.fov_deg,
        )
        if not bool(np.all(visible)):
            return
        origin = tuple(int(round(value)) for value in pixels[0])
        for corner_index in range(1, 5):
            corner = tuple(int(round(value)) for value in pixels[corner_index])
            cv2.line(canvas, origin, corner, (72, 214, 255), 1, cv2.LINE_AA)
        polygon = np.round(pixels[1:]).astype(np.int32)
        cv2.polylines(canvas, [polygon], True, (72, 214, 255), 1, cv2.LINE_AA)

    def _annotate_phone_status(self, frame_bgr: np.ndarray) -> np.ndarray:
        status = "AUTO DEMO" if self.auto_demo_enabled else "MANUAL"
        cv2.putText(frame_bgr, "PHONE CAM", (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 247, 250), 2, cv2.LINE_AA)
        cv2.putText(
            frame_bgr,
            f"t={self.time_s:05.2f}s  {status}",
            (24, frame_bgr.shape[0] - 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (240, 244, 247),
            2,
            cv2.LINE_AA,
        )
        return frame_bgr
