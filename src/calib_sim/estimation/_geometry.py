"""Shared geometry helpers for the batch estimation pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from calib_sim.interactive.camera_model import PhoneCameraModel


WALL_TAG_ROTATION_WT = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


def rotation_matrix_from_rpy_deg(angles_deg: Iterable[float]) -> np.ndarray:
    rx_deg, ry_deg, rz_deg = [float(value) for value in angles_deg]
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)
    cx, cy, cz = math.cos(rx), math.cos(ry), math.cos(rz)
    sx, sy, sz = math.sin(rx), math.sin(ry), math.sin(rz)
    rotation_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    rotation_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rotation_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rotation_z @ rotation_y @ rotation_x


def rotation_matrix_from_rvec(rvec: np.ndarray) -> np.ndarray:
    """Convert a Rodrigues vector into a 3x3 rotation matrix."""
    rotation_matrix, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return rotation_matrix.astype(np.float64)


def rvec_from_rotation_matrix(rotation_matrix: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix into a Rodrigues vector."""
    rvec, _ = cv2.Rodrigues(np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3))
    return rvec.reshape(3).astype(np.float64)


def normalize_rvec(rvec: np.ndarray) -> np.ndarray:
    """Wrap a Rodrigues vector into the principal branch."""
    vector = np.asarray(rvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(vector))
    if theta <= math.pi:
        return vector
    if theta < 1e-12:
        return np.zeros(3, dtype=np.float64)
    axis = vector / theta
    wrapped_theta = ((theta + math.pi) % (2.0 * math.pi)) - math.pi
    return axis * wrapped_theta


def compose_rvec(base_rvec: np.ndarray, delta_rvec: np.ndarray) -> np.ndarray:
    """Compose a small left-multiplied delta onto a Rodrigues vector pose."""
    base_rotation = rotation_matrix_from_rvec(base_rvec)
    delta_rotation = rotation_matrix_from_rvec(delta_rvec)
    return normalize_rvec(rvec_from_rotation_matrix(delta_rotation @ base_rotation))


def se3_pose_to_world_matrix(position_world_m: np.ndarray, rotation_wc: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(rotation_wc, dtype=np.float64).reshape(3, 3)
    matrix[:3, 3] = np.asarray(position_world_m, dtype=np.float64).reshape(3)
    return matrix


def transform_points(rotation_ab: np.ndarray, translation_a_b: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    """Transform points in frame b into frame a."""
    return (np.asarray(rotation_ab, dtype=np.float64).reshape(3, 3) @ np.asarray(points_b, dtype=np.float64).T).T + np.asarray(
        translation_a_b,
        dtype=np.float64,
    ).reshape(1, 3)


def normalized_image_points_y_up(camera_model: PhoneCameraModel, pixels_xy: np.ndarray) -> np.ndarray:
    """Convert image pixels into normalized y-up camera coordinates."""
    points = np.asarray(pixels_xy, dtype=np.float64)
    if hasattr(camera_model, "normalized_camera_coordinates"):
        normalized = camera_model.normalized_camera_coordinates(
            points,
            undistort=bool(getattr(camera_model, "apply_lens_distortion_in_render", False)),
        )
    else:
        fx_px = float(camera_model.fx_px)
        fy_px = float(camera_model.fy_px)
        cx_px = float(camera_model.cx_px)
        cy_px = float(camera_model.cy_px)
        if bool(getattr(camera_model, "apply_lens_distortion_in_render", False)):
            camera_matrix = np.array([[fx_px, float(getattr(camera_model, "skew_px", 0.0)), cx_px], [0.0, fy_px, cy_px], [0.0, 0.0, 1.0]], dtype=np.float64)
            distortion = np.asarray(getattr(camera_model, "distortion_coefficients", (0.0, 0.0, 0.0, 0.0, 0.0)), dtype=np.float64).reshape(-1)
            normalized = cv2.undistortPoints(points.astype(np.float32).reshape(-1, 1, 2), camera_matrix, distortion).reshape(-1, 2).astype(np.float64)
        else:
            normalized = np.column_stack(((points[:, 0] - cx_px) / fx_px, (points[:, 1] - cy_px) / fy_px)).astype(np.float64)
    return np.column_stack((normalized[:, 0], -normalized[:, 1])).astype(np.float64)


def tag_local_corners_clockwise(size_m: float) -> np.ndarray:
    half = float(size_m) * 0.5
    return np.array(
        [
            [-half, -half, 0.0],
            [half, -half, 0.0],
            [half, half, 0.0],
            [-half, half, 0.0],
        ],
        dtype=np.float64,
    )


def tag_local_corners_clockwise_2d(size_m: float) -> np.ndarray:
    return tag_local_corners_clockwise(size_m)[:, :2]


def project_world_points_to_pixels(
    *,
    camera_model: PhoneCameraModel,
    camera_position_world_m: np.ndarray,
    camera_rotation_wc: np.ndarray,
    world_points_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points into image pixels using the interactive camera convention."""
    points_camera = (
        np.asarray(camera_rotation_wc, dtype=np.float64).reshape(3, 3).T
        @ (np.asarray(world_points_m, dtype=np.float64) - np.asarray(camera_position_world_m, dtype=np.float64).reshape(1, 3)).T
    ).T
    if hasattr(camera_model, "project_camera_points"):
        projected_pixels, visible = camera_model.project_camera_points(
            points_camera,
            apply_distortion=bool(getattr(camera_model, "apply_lens_distortion_in_render", False)),
        )
        return projected_pixels.astype(np.float64), np.asarray(visible, dtype=bool)

    visible = np.asarray(points_camera[:, 2] > 0.05, dtype=bool)
    if bool(getattr(camera_model, "apply_lens_distortion_in_render", False)):
        camera_matrix = np.array(
            [
                [float(camera_model.fx_px), float(getattr(camera_model, "skew_px", 0.0)), float(camera_model.cx_px)],
                [0.0, float(camera_model.fy_px), float(camera_model.cy_px)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        distortion = np.asarray(getattr(camera_model, "distortion_coefficients", (0.0, 0.0, 0.0, 0.0, 0.0)), dtype=np.float64).reshape(-1)
        projected_pixels, _ = cv2.projectPoints(
            points_camera.reshape(-1, 3),
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            camera_matrix,
            distortion,
        )
        return projected_pixels.reshape(-1, 2).astype(np.float64), visible

    z = np.maximum(points_camera[:, 2], 1e-6)
    u = float(camera_model.fx_px) * (points_camera[:, 0] / z) + float(camera_model.cx_px)
    v = float(camera_model.fy_px) * (-(points_camera[:, 1] / z)) + float(camera_model.cy_px)
    return np.column_stack((u, v)).astype(np.float64), visible


def tag_pose_from_camera_pixels(
    *,
    camera_model: PhoneCameraModel,
    tag_size_m: float,
    ordered_corners_xy_px: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate tag-to-camera pose in the repo's y-up camera frame from four pixel corners."""
    object_points_xy = tag_local_corners_clockwise_2d(tag_size_m)
    normalized_pixels = normalized_image_points_y_up(camera_model, np.asarray(ordered_corners_xy_px, dtype=np.float64))
    homography, _ = cv2.findHomography(object_points_xy.astype(np.float64), normalized_pixels.astype(np.float64), method=0)
    if homography is None:
        raise RuntimeError("Homography initialization failed for tag observation.")

    h1 = homography[:, 0]
    h2 = homography[:, 1]
    h3 = homography[:, 2]
    lambda_scale = 1.0 / max((np.linalg.norm(h1) + np.linalg.norm(h2)) * 0.5, 1e-12)
    r1 = lambda_scale * h1
    r2 = lambda_scale * h2
    t_ct = lambda_scale * h3
    r3 = np.cross(r1, r2)
    rotation_raw = np.column_stack((r1, r2, r3))
    u_matrix, _, v_t = np.linalg.svd(rotation_raw)
    rotation_ct = u_matrix @ v_t
    if np.linalg.det(rotation_ct) < 0.0:
        rotation_ct[:, 2] *= -1.0
    return rotation_ct.astype(np.float64), np.asarray(t_ct, dtype=np.float64).reshape(3)


def camera_pose_from_tag_observation(
    *,
    camera_model: PhoneCameraModel,
    tag_size_m: float,
    ordered_corners_xy_px: np.ndarray,
    tag_position_world_m: np.ndarray,
    tag_rotation_wt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover a camera world pose from a single observed tag with known world pose."""
    rotation_ct, translation_ct = tag_pose_from_camera_pixels(
        camera_model=camera_model,
        tag_size_m=tag_size_m,
        ordered_corners_xy_px=ordered_corners_xy_px,
    )
    rotation_wc = np.asarray(tag_rotation_wt, dtype=np.float64).reshape(3, 3) @ rotation_ct.T
    camera_position_world_m = np.asarray(tag_position_world_m, dtype=np.float64).reshape(3) - rotation_wc @ np.asarray(
        translation_ct,
        dtype=np.float64,
    ).reshape(3)
    return camera_position_world_m.astype(np.float64), rotation_wc.astype(np.float64)


def pose_vector_from_components(position_world_m: np.ndarray, rotation_wc: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(position_world_m, dtype=np.float64).reshape(3),
            normalize_rvec(rvec_from_rotation_matrix(rotation_wc)),
        ]
    ).astype(np.float64)


def pose_components_from_vector(pose_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vector = np.asarray(pose_vector, dtype=np.float64).reshape(6)
    return vector[:3].copy(), rotation_matrix_from_rvec(vector[3:])


def apply_world_alignment_to_pose_vector(alignment_rotation: np.ndarray, alignment_translation: np.ndarray, pose_vector: np.ndarray) -> np.ndarray:
    position_world_m, rotation_wc = pose_components_from_vector(pose_vector)
    aligned_position = np.asarray(alignment_rotation, dtype=np.float64) @ position_world_m + np.asarray(
        alignment_translation,
        dtype=np.float64,
    ).reshape(3)
    aligned_rotation = np.asarray(alignment_rotation, dtype=np.float64) @ rotation_wc
    return pose_vector_from_components(aligned_position, aligned_rotation)


def world_alignment_from_reference_pose(estimated_pose_vector: np.ndarray, reference_pose_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    est_position, est_rotation = pose_components_from_vector(estimated_pose_vector)
    ref_position, ref_rotation = pose_components_from_vector(reference_pose_vector)
    alignment_rotation = ref_rotation @ est_rotation.T
    alignment_translation = ref_position - alignment_rotation @ est_position
    return alignment_rotation.astype(np.float64), alignment_translation.astype(np.float64)


def relative_rotation_error_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    relative = np.asarray(rotation_a, dtype=np.float64) @ np.asarray(rotation_b, dtype=np.float64).T
    cosine = float((np.trace(relative) - 1.0) * 0.5)
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def rotation_xyz_deg(rotation_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    sy = float(np.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2))
    singular = sy < 1e-8
    if not singular:
        x_angle = math.atan2(matrix[2, 1], matrix[2, 2])
        y_angle = math.atan2(-matrix[2, 0], sy)
        z_angle = math.atan2(matrix[1, 0], matrix[0, 0])
    else:
        x_angle = math.atan2(-matrix[1, 2], matrix[1, 1])
        y_angle = math.atan2(-matrix[2, 0], sy)
        z_angle = 0.0
    return np.degrees(np.array([x_angle, y_angle, z_angle], dtype=np.float64))


@dataclass(slots=True)
class TagWorldPose:
    tag_id: int
    position_world_m: np.ndarray
    rotation_wt: np.ndarray
    size_m: float


def tag_rotation_from_mount(mount: str, orientation_rpy_deg: Iterable[float] | None) -> np.ndarray:
    if orientation_rpy_deg is not None:
        return rotation_matrix_from_rpy_deg(orientation_rpy_deg)
    if str(mount) == "top":
        return np.eye(3, dtype=np.float64)
    return WALL_TAG_ROTATION_WT.copy()
