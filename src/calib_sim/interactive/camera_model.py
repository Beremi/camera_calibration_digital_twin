"""Camera model bridge for the interactive simulator.

The interactive sim renders a synthetic scene, but this module lets us drive
the primary phone camera from real device metadata and export measurements in
the same ordering and units expected by the CameraPoseEstimation repo.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from calib_sim.common.models import TagDetection


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (Path(__file__).resolve().parents[3] / candidate).resolve()


def _display_repo_path(path: str | Path) -> str:
    resolved = _resolve_repo_path(path)
    repo_root = Path(__file__).resolve().parents[3]
    try:
        return str(resolved.relative_to(repo_root))
    except ValueError:
        return str(resolved)


def _center_crop(
    active_width_px: int,
    active_height_px: int,
    output_width_px: int,
    output_height_px: int,
) -> tuple[float, float, float, float]:
    active_aspect = float(active_width_px) / float(active_height_px)
    output_aspect = float(output_width_px) / float(output_height_px)

    if active_aspect >= output_aspect:
        crop_height_px = float(active_height_px)
        crop_width_px = crop_height_px * output_aspect
    else:
        crop_width_px = float(active_width_px)
        crop_height_px = crop_width_px / output_aspect

    crop_x_px = (float(active_width_px) - crop_width_px) * 0.5
    crop_y_px = (float(active_height_px) - crop_height_px) * 0.5
    return crop_x_px, crop_y_px, crop_width_px, crop_height_px


@dataclass(slots=True)
class CameraPoseEstimationBridge:
    pattern_half_extent_m: float
    measurement_unit: str
    measurement_order: tuple[str, ...]
    y_axis: str
    undistort_before_export: bool

    def summary(self) -> dict[str, Any]:
        return {
            "pattern_half_extent_m": self.pattern_half_extent_m,
            "measurement_unit": self.measurement_unit,
            "measurement_order": list(self.measurement_order),
            "y_axis": self.y_axis,
            "undistort_before_export": self.undistort_before_export,
        }


@dataclass(slots=True)
class PhoneCameraModel:
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
    pose_bridge: CameraPoseEstimationBridge
    _remap_x: np.ndarray
    _remap_y: np.ndarray

    @property
    def camera_matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.fx_px, self.skew_px, self.cx_px],
                [0.0, self.fy_px, self.cy_px],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @property
    def distortion_array(self) -> np.ndarray:
        return np.asarray(self.distortion_coefficients, dtype=np.float64)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_toml_path": self.source_toml_path,
            "projection_model": self.projection_model,
            "apply_lens_distortion_in_render": self.apply_lens_distortion_in_render,
            "output_resolution_px": [self.output_width_px, self.output_height_px],
            "focal_length_mm": self.focal_length_mm,
            "focus_distance_diopters": self.focus_distance_diopters,
            "target_frames_per_second": self.target_frames_per_second,
            "timestamp_source": self.timestamp_source,
            "sensor_orientation_deg": self.sensor_orientation_deg,
            "effective_intrinsics_px": {
                "fx": self.fx_px,
                "fy": self.fy_px,
                "cx": self.cx_px,
                "cy": self.cy_px,
                "skew": self.skew_px,
            },
            "crop_rect_active_array_px": {
                "x": self.crop_x_px,
                "y": self.crop_y_px,
                "width": self.crop_width_px,
                "height": self.crop_height_px,
            },
            "distortion": {
                "model": self.distortion_model,
                "coefficients": list(self.distortion_coefficients),
            },
            "camera_pose_estimation": self.pose_bridge.summary(),
        }

    def horizontal_fov_deg(self) -> float:
        return float(np.degrees(2.0 * np.arctan(self.output_width_px / (2.0 * self.fx_px))))

    def vertical_fov_deg(self) -> float:
        return float(np.degrees(2.0 * np.arctan(self.output_height_px / (2.0 * self.fy_px))))

    def project_camera_points_ideal(self, points_camera_xyz: Sequence[Sequence[float]] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points_camera_xyz, dtype=np.float64)
        z = points[:, 2]
        visible = z > 0.05
        with np.errstate(divide="ignore", invalid="ignore"):
            x_norm = points[:, 0] / z
            y_norm = -points[:, 1] / z
            u = self.fx_px * x_norm + self.cx_px
            v = self.fy_px * y_norm + self.cy_px
        return np.column_stack((u, v)), visible

    def project_camera_points(
        self,
        points_camera_xyz: Sequence[Sequence[float]] | np.ndarray,
        *,
        apply_distortion: bool | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points_camera_xyz, dtype=np.float64).reshape(-1, 3)
        visible = points[:, 2] > 0.05
        if apply_distortion is None:
            apply_distortion = self.apply_lens_distortion_in_render
        if not apply_distortion:
            return self.project_camera_points_ideal(points)

        projected, _ = cv2.projectPoints(
            points,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            self.camera_matrix,
            self.distortion_array,
        )
        return projected.reshape(-1, 2).astype(np.float64), visible

    def apply_lens_distortion(self, ideal_image_bgr: np.ndarray) -> np.ndarray:
        if not self.apply_lens_distortion_in_render:
            return ideal_image_bgr
        return cv2.remap(
            ideal_image_bgr,
            self._remap_x,
            self._remap_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    def undistort_image(self, image_bgr: np.ndarray) -> np.ndarray:
        if not any(abs(value) > 1e-12 for value in self.distortion_coefficients):
            return image_bgr.copy()
        return cv2.undistort(image_bgr, self.camera_matrix, self.distortion_array)

    def analysis_image(self, image_bgr: np.ndarray) -> np.ndarray:
        if not self.apply_lens_distortion_in_render:
            return image_bgr.copy()
        return self.undistort_image(image_bgr)

    def distortion_coefficients_for_rendered_output(self) -> tuple[float, float, float, float, float] | None:
        if not self.apply_lens_distortion_in_render:
            return None
        return self.distortion_coefficients

    def undistort_pixel_points(self, points_px: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        points = np.asarray(points_px, dtype=np.float32).reshape(-1, 1, 2)
        undistorted = cv2.undistortPoints(points, self.camera_matrix, self.distortion_array, P=self.camera_matrix)
        return undistorted.reshape(-1, 2).astype(np.float64)

    def normalized_camera_coordinates(self, points_px: Sequence[Sequence[float]] | np.ndarray, *, undistort: bool = True) -> np.ndarray:
        points = np.asarray(points_px, dtype=np.float32).reshape(-1, 1, 2)
        if undistort:
            normalized = cv2.undistortPoints(points, self.camera_matrix, self.distortion_array)
            return normalized.reshape(-1, 2).astype(np.float64)

        raw = points.reshape(-1, 2).astype(np.float64)
        x = (raw[:, 0] - self.cx_px) / self.fx_px
        y = (raw[:, 1] - self.cy_px) / self.fy_px
        return np.column_stack((x, y))

    def pixels_to_image_plane_mm(self, points_px: Sequence[Sequence[float]] | np.ndarray, *, undistort: bool = True) -> np.ndarray:
        normalized = self.normalized_camera_coordinates(points_px, undistort=undistort)
        x_mm = normalized[:, 0] * self.focal_length_mm
        y_mm = -normalized[:, 1] * self.focal_length_mm
        return np.column_stack((x_mm, y_mm))

    def camera_pose_estimation_measurement(self, detection: TagDetection) -> dict[str, Any]:
        ordered_pixels = np.asarray(
            [
                detection.center_xy,
                detection.corners_xy_clockwise[1],
                detection.corners_xy_clockwise[2],
                detection.corners_xy_clockwise[3],
                detection.corners_xy_clockwise[0],
            ],
            dtype=np.float64,
        )
        image_plane_mm = self.pixels_to_image_plane_mm(
            ordered_pixels,
            undistort=self.pose_bridge.undistort_before_export,
        )
        measurement_vector_mm = [float(value) for value in image_plane_mm.reshape(-1).tolist()]
        return {
            "tag_id": detection.tag_id,
            "order": list(self.pose_bridge.measurement_order),
            "image_plane_points_mm": [[float(x), float(y)] for x, y in image_plane_mm.tolist()],
            "measurement_vector_mm": measurement_vector_mm,
            "measurement_vector_m": [value / 1000.0 for value in measurement_vector_mm],
            "camera_model_name": self.name,
            "pattern_half_extent_m": self.pose_bridge.pattern_half_extent_m,
        }


def load_phone_camera_model(
    toml_path: str | Path,
    *,
    output_width_px: int | None = None,
    output_height_px: int | None = None,
) -> PhoneCameraModel:
    resolved_path = _resolve_repo_path(toml_path)
    raw = tomllib.loads(resolved_path.read_text(encoding="utf-8"))

    sensor = raw["sensor"]
    active_array_width_px = int(sensor["active_array_width_px"])
    active_array_height_px = int(sensor["active_array_height_px"])

    video_mode = raw["video_mode"]
    final_width_px = int(output_width_px or video_mode["width_px"])
    final_height_px = int(output_height_px or video_mode["height_px"])

    crop_x_px, crop_y_px, crop_width_px, crop_height_px = _center_crop(
        active_array_width_px,
        active_array_height_px,
        final_width_px,
        final_height_px,
    )

    active_intrinsics = raw["intrinsics"]["active_array"]
    scale_x = float(final_width_px) / crop_width_px
    scale_y = float(final_height_px) / crop_height_px

    fx_px = float(active_intrinsics["fx_px"]) * scale_x
    fy_px = float(active_intrinsics["fy_px"]) * scale_y
    cx_px = (float(active_intrinsics["cx_px"]) - crop_x_px) * scale_x
    cy_px = (float(active_intrinsics["cy_px"]) - crop_y_px) * scale_y
    skew_px = float(active_intrinsics.get("skew_px", 0.0)) * scale_x

    distortion = raw["distortion"]
    distortion_coefficients = tuple(float(value) for value in distortion.get("coefficients", [0, 0, 0, 0, 0]))
    rendering = raw.get("rendering", {})

    pose_raw = raw.get("camera_pose_estimation", {})
    pose_bridge = CameraPoseEstimationBridge(
        pattern_half_extent_m=float(pose_raw.get("pattern_half_extent_m", 0.05)),
        measurement_unit=str(pose_raw.get("measurement_unit", "mm")),
        measurement_order=tuple(str(value) for value in pose_raw.get("measurement_order", ["center", "top_right", "bottom_right", "bottom_left", "top_left"])),
        y_axis=str(pose_raw.get("y_axis", "up")),
        undistort_before_export=bool(pose_raw.get("undistort_before_export", True)),
    )

    camera_matrix = np.array([[fx_px, skew_px, cx_px], [0.0, fy_px, cy_px], [0.0, 0.0, 1.0]], dtype=np.float64)
    distortion_array = np.asarray(distortion_coefficients, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(np.arange(final_width_px, dtype=np.float32), np.arange(final_height_px, dtype=np.float32))
    distorted_pixels = np.stack((grid_x, grid_y), axis=-1).reshape(-1, 1, 2)
    undistorted_pixels = cv2.undistortPoints(distorted_pixels, camera_matrix, distortion_array, P=camera_matrix)
    remap = undistorted_pixels.reshape(final_height_px, final_width_px, 2)

    optics = raw["optics"]
    return PhoneCameraModel(
        name=str(raw.get("name", "phone_camera_model")),
        source_toml_path=_display_repo_path(resolved_path),
        projection_model=str(rendering.get("projection_model", "pixel_metadata_distorted")),
        apply_lens_distortion_in_render=bool(rendering.get("apply_lens_distortion_in_render", True)),
        output_width_px=final_width_px,
        output_height_px=final_height_px,
        focal_length_mm=float(optics["focal_length_mm"]),
        focus_distance_diopters=float(optics.get("focus_distance_diopters", 0.0)),
        target_frames_per_second=float(optics.get("target_frames_per_second", video_mode.get("target_frames_per_second", 30.0))),
        timestamp_source=str(optics.get("timestamp_source", "REALTIME")),
        sensor_orientation_deg=int(sensor.get("orientation_deg", 0)),
        fx_px=fx_px,
        fy_px=fy_px,
        cx_px=cx_px,
        cy_px=cy_px,
        skew_px=skew_px,
        crop_x_px=crop_x_px,
        crop_y_px=crop_y_px,
        crop_width_px=crop_width_px,
        crop_height_px=crop_height_px,
        distortion_model=str(distortion.get("model", "opencv_brown_conrady")),
        distortion_coefficients=distortion_coefficients,
        pose_bridge=pose_bridge,
        _remap_x=remap[:, :, 0].astype(np.float32),
        _remap_y=remap[:, :, 1].astype(np.float32),
    )
