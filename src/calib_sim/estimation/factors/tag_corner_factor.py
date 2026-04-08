"""Pixel-space tag-corner factor helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np

from calib_sim.interactive.camera_model import PhoneCameraModel

TAG_CORNER_ORDER = ("top_right", "bottom_right", "bottom_left", "top_left")
DETECTOR_CLOCKWISE_ORDER = ("top_left", "top_right", "bottom_right", "bottom_left")


def canonical_tag_corner_order() -> tuple[str, str, str, str]:
    return TAG_CORNER_ORDER


def canonical_tag_object_points_m(tag_size_m: float) -> np.ndarray:
    half = float(tag_size_m) * 0.5
    return np.array(
        [
            [half, -half, 0.0],
            [half, half, 0.0],
            [-half, half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def canonical_tag_corner_pixels(detection: object) -> np.ndarray:
    corners = np.asarray(getattr(detection, "corners_xy_clockwise"), dtype=np.float64).reshape(4, 2)
    return np.asarray([corners[1], corners[2], corners[3], corners[0]], dtype=np.float64)


def _rotation_matrix_from_rvec(rvec: Sequence[float]) -> np.ndarray:
    rotation_matrix, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return rotation_matrix.astype(np.float64)


def _project_points_camera(
    points_camera_xyz: np.ndarray,
    *,
    camera_model: PhoneCameraModel,
    apply_distortion: bool,
) -> np.ndarray:
    projected, _ = camera_model.project_camera_points(points_camera_xyz, apply_distortion=apply_distortion)
    return np.asarray(projected, dtype=np.float64)


def project_tag_corners_pixels(
    *,
    camera_model: PhoneCameraModel,
    camera_position_world_m: Sequence[float],
    camera_rotation_cw: Sequence[Sequence[float]],
    tag_center_world_m: Sequence[float],
    tag_rotation_wt: Sequence[Sequence[float]],
    tag_size_m: float,
    apply_distortion: bool | None = None,
) -> np.ndarray:
    camera_position_world = np.asarray(camera_position_world_m, dtype=np.float64).reshape(3)
    camera_rotation_cw = np.asarray(camera_rotation_cw, dtype=np.float64).reshape(3, 3)
    tag_center_world = np.asarray(tag_center_world_m, dtype=np.float64).reshape(3)
    tag_rotation_wt = np.asarray(tag_rotation_wt, dtype=np.float64).reshape(3, 3)
    tag_corner_points = canonical_tag_object_points_m(tag_size_m)
    tag_corner_points_world = tag_center_world + (tag_rotation_wt @ tag_corner_points.T).T
    points_camera = (camera_rotation_cw.T @ (tag_corner_points_world - camera_position_world).T).T
    if apply_distortion is None:
        apply_distortion = camera_model.apply_lens_distortion_in_render
    return _project_points_camera(points_camera, camera_model=camera_model, apply_distortion=apply_distortion)


def tag_corner_pixel_residual(
    *,
    camera_model: PhoneCameraModel,
    observed_corner_pixels_px: Sequence[Sequence[float]],
    camera_position_world_m: Sequence[float],
    camera_rotation_cw: Sequence[Sequence[float]],
    tag_center_world_m: Sequence[float],
    tag_rotation_wt: Sequence[Sequence[float]],
    tag_size_m: float,
    apply_distortion: bool | None = None,
) -> np.ndarray:
    observed = np.asarray(observed_corner_pixels_px, dtype=np.float64).reshape(4, 2)
    predicted = project_tag_corners_pixels(
        camera_model=camera_model,
        camera_position_world_m=camera_position_world_m,
        camera_rotation_cw=camera_rotation_cw,
        tag_center_world_m=tag_center_world_m,
        tag_rotation_wt=tag_rotation_wt,
        tag_size_m=tag_size_m,
        apply_distortion=apply_distortion,
    )
    return (predicted - observed).reshape(-1)
