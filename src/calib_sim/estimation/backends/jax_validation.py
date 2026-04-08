"""JAX validation helpers for tag projection and factor consistency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from calib_sim.estimation.factors.tag_corner_factor import canonical_tag_corner_pixels, canonical_tag_object_points_m
from calib_sim.interactive.camera_model import PhoneCameraModel


def _import_jax() -> tuple[object, object]:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    return jax, jnp


def _skew(jnp: object, vector: object) -> object:
    vx, vy, vz = vector
    return jnp.array([[0.0, -vz, vy], [vz, 0.0, -vx], [-vy, vx, 0.0]], dtype=jnp.float64)


def _rotation_matrix_rvec_jax(jnp: object, rvec_rad: object) -> object:
    theta2 = jnp.dot(rvec_rad, rvec_rad)
    theta = jnp.sqrt(theta2 + 1e-12)
    k = _skew(jnp, rvec_rad)
    identity = jnp.eye(3, dtype=jnp.float64)
    return identity + (jnp.sin(theta) / theta) * k + ((1.0 - jnp.cos(theta)) / (theta * theta)) * (k @ k)


def project_camera_points_jax(
    *,
    camera_model: PhoneCameraModel,
    points_camera_xyz: Sequence[Sequence[float]] | np.ndarray,
    apply_distortion: bool = False,
) -> np.ndarray:
    _, jnp = _import_jax()
    points = jnp.asarray(np.asarray(points_camera_xyz, dtype=np.float64).reshape(-1, 3), dtype=jnp.float64)
    if apply_distortion:
        raise NotImplementedError("JAX validation currently covers the undistorted projection path only.")
    z = points[:, 2]
    uv = jnp.stack(
        (
            camera_model.fx_px * (points[:, 0] / z) + camera_model.cx_px,
            camera_model.fy_px * (-(points[:, 1] / z)) + camera_model.cy_px,
        ),
        axis=1,
    )
    return np.asarray(uv, dtype=np.float64)


def project_tag_corners_jax(
    *,
    camera_model: PhoneCameraModel,
    camera_position_world_m: Sequence[float],
    camera_rotation_cw: Sequence[Sequence[float]],
    tag_center_world_m: Sequence[float],
    tag_rotation_wt: Sequence[Sequence[float]],
    tag_size_m: float,
) -> np.ndarray:
    _, jnp = _import_jax()
    camera_position_world = jnp.asarray(np.asarray(camera_position_world_m, dtype=np.float64).reshape(3), dtype=jnp.float64)
    camera_rotation_cw = jnp.asarray(np.asarray(camera_rotation_cw, dtype=np.float64).reshape(3, 3), dtype=jnp.float64)
    tag_center_world = jnp.asarray(np.asarray(tag_center_world_m, dtype=np.float64).reshape(3), dtype=jnp.float64)
    tag_rotation_wt = jnp.asarray(np.asarray(tag_rotation_wt, dtype=np.float64).reshape(3, 3), dtype=jnp.float64)
    tag_corner_points = jnp.asarray(canonical_tag_object_points_m(tag_size_m), dtype=jnp.float64)
    tag_corner_points_world = tag_center_world + (tag_rotation_wt @ tag_corner_points.T).T
    points_camera = (camera_rotation_cw.T @ (tag_corner_points_world - camera_position_world).T).T
    return project_camera_points_jax(camera_model=camera_model, points_camera_xyz=points_camera, apply_distortion=False)


def validate_tag_corner_projection(
    *,
    camera_model: PhoneCameraModel,
    camera_position_world_m: Sequence[float],
    camera_rotation_cw: Sequence[Sequence[float]],
    tag_center_world_m: Sequence[float],
    tag_rotation_wt: Sequence[Sequence[float]],
    tag_size_m: float,
    atol_px: float = 1e-9,
) -> dict[str, float]:
    predicted_jax = project_tag_corners_jax(
        camera_model=camera_model,
        camera_position_world_m=camera_position_world_m,
        camera_rotation_cw=camera_rotation_cw,
        tag_center_world_m=tag_center_world_m,
        tag_rotation_wt=tag_rotation_wt,
        tag_size_m=tag_size_m,
    )
    camera_rotation_cw_arr = np.asarray(camera_rotation_cw, dtype=np.float64).reshape(3, 3)
    camera_position_world_arr = np.asarray(camera_position_world_m, dtype=np.float64).reshape(3)
    tag_center_world_arr = np.asarray(tag_center_world_m, dtype=np.float64).reshape(3)
    tag_rotation_wt_arr = np.asarray(tag_rotation_wt, dtype=np.float64).reshape(3, 3)
    corners_world = tag_center_world_arr + (tag_rotation_wt_arr @ canonical_tag_object_points_m(tag_size_m).T).T
    points_camera = (camera_rotation_cw_arr.T @ (corners_world - camera_position_world_arr).T).T
    predicted_cv, _ = camera_model.project_camera_points(points_camera, apply_distortion=False)
    predicted_cv = np.asarray(predicted_cv, dtype=np.float64)
    max_abs_error_px = float(np.max(np.abs(predicted_jax - predicted_cv)))
    if max_abs_error_px > float(atol_px):
        raise AssertionError(f"JAX projection mismatch: max abs error {max_abs_error_px:.3e} px")
    return {"max_abs_error_px": max_abs_error_px}
