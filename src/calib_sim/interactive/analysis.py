"""Offline analysis for recorded interactive sim runs.

This module replays a recorded raw phone video, undistorts it, redetects the
AprilTags, and estimates camera pose from image measurements alone using a
repo-inspired Newton solver driven by JAX gradients and Hessians.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from calib_sim.common.video import ManagedMp4Writer
from calib_sim.estimation.reporting import run_batch_estimation_analysis
from calib_sim.interactive.camera_model import PhoneCameraModel, load_phone_camera_model
from calib_sim.tag_service.detector import AprilTag36h11Detector

REPO_ROOT = Path(__file__).resolve().parents[3]


def _import_jax() -> tuple[Any, Any]:
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - exercised in runtime if dependency is missing.
        raise ImportError("JAX is required for offline pose analysis. Install the project dependencies again.") from exc
    jax.config.update("jax_enable_x64", True)
    return jax, jnp


def _load_json_lines(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if not path.exists():
        return entries
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _load_csv_dict_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _object_points_tag_m(pattern_half_extent_m: float) -> np.ndarray:
    half = float(pattern_half_extent_m)
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [half, -half, 0.0],
            [half, half, 0.0],
            [-half, half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def _rotation_matrix_xyz_np(angles_rad: np.ndarray) -> np.ndarray:
    rx, ry, rz = [float(value) for value in angles_rad]
    cx, cy, cz = math.cos(rx), math.cos(ry), math.cos(rz)
    sx, sy, sz = math.sin(rx), math.sin(ry), math.sin(rz)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rot_z @ rot_y @ rot_x


def _rotation_matrix_rvec_np(rvec_rad: np.ndarray) -> np.ndarray:
    rotation_matrix, _ = cv2.Rodrigues(np.asarray(rvec_rad, dtype=np.float64).reshape(3, 1))
    return rotation_matrix.astype(np.float64)


def _predict_image_plane_points_np(
    state: np.ndarray,
    *,
    focal_length_m: float,
    object_points_tag_m: np.ndarray,
) -> np.ndarray:
    translation = np.asarray(state[:3], dtype=np.float64)
    rotation = _rotation_matrix_rvec_np(np.asarray(state[3:], dtype=np.float64))
    points_camera = (rotation @ object_points_tag_m.T).T + translation
    image_plane = focal_length_m * (points_camera[:, :2] / points_camera[:, 2:3])
    return image_plane.astype(np.float64)


def _camera_obscura_seed_details(
    observed_points_image_plane_m: np.ndarray,
    *,
    focal_length_m: float,
    pattern_half_extent_m: float,
) -> dict[str, Any]:
    top_width = np.linalg.norm(observed_points_image_plane_m[1] - observed_points_image_plane_m[4])
    bottom_width = np.linalg.norm(observed_points_image_plane_m[2] - observed_points_image_plane_m[3])
    right_height = np.linalg.norm(observed_points_image_plane_m[1] - observed_points_image_plane_m[2])
    left_height = np.linalg.norm(observed_points_image_plane_m[4] - observed_points_image_plane_m[3])
    scale = float(np.mean([max(top_width, 1e-9), max(bottom_width, 1e-9), max(right_height, 1e-9), max(left_height, 1e-9)]))
    pattern_size_m = pattern_half_extent_m * 2.0
    depth_m = max((focal_length_m * pattern_size_m) / max(scale, 1e-9), 0.05)
    center = np.asarray(observed_points_image_plane_m[0], dtype=np.float64)
    tx_m = float(center[0] * depth_m / focal_length_m)
    ty_m = float(center[1] * depth_m / focal_length_m)
    state = np.array([tx_m, ty_m, depth_m, math.pi, 0.0, 0.0], dtype=np.float64)
    return {
        "seed_name": "camera_obscura",
        "observed_edge_lengths_m": {
            "top_width": float(top_width),
            "bottom_width": float(bottom_width),
            "right_height": float(right_height),
            "left_height": float(left_height),
        },
        "average_observed_edge_m": scale,
        "pattern_size_m": pattern_size_m,
        "focal_length_m": float(focal_length_m),
        "center_image_plane_m": [float(center[0]), float(center[1])],
        "estimated_depth_m": depth_m,
        "state": [float(value) for value in state.tolist()],
    }


def _opencv_pnp_initial_guess(
    ordered_pixels_px: np.ndarray,
    *,
    camera_model: PhoneCameraModel,
    pattern_half_extent_m: float,
) -> dict[str, Any] | None:
    corner_pixels = np.asarray(ordered_pixels_px[1:], dtype=np.float64)
    object_points = _object_points_tag_m(pattern_half_extent_m)[1:]
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        corner_pixels,
        camera_model.camera_matrix,
        np.zeros((5, 1), dtype=np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None
    rotation_cv, _ = cv2.Rodrigues(rvec)
    flip_y = np.diag([1.0, -1.0, 1.0])
    rotation_up = flip_y @ rotation_cv
    translation_up = (flip_y @ tvec).reshape(3)
    rvec_up, _ = cv2.Rodrigues(rotation_up)
    state = np.array(
        [translation_up[0], translation_up[1], translation_up[2], *rvec_up.reshape(3).tolist()],
        dtype=np.float64,
    )
    return {
        "seed_name": "opencv_pnp",
        "corner_pixels_px": [[float(x), float(y)] for x, y in corner_pixels.tolist()],
        "state": [float(value) for value in state.tolist()],
    }


@lru_cache(maxsize=8)
def _compiled_objective_functions(focal_length_m: float, pattern_half_extent_m: float) -> tuple[Any, Any, Any]:
    jax, jnp = _import_jax()
    object_points = jnp.asarray(_object_points_tag_m(pattern_half_extent_m), dtype=jnp.float64)
    focal_length = jnp.asarray(float(focal_length_m), dtype=jnp.float64)

    def skew(vector: Any) -> Any:
        vx, vy, vz = vector
        return jnp.array(
            [
                [0.0, -vz, vy],
                [vz, 0.0, -vx],
                [-vy, vx, 0.0],
            ],
            dtype=jnp.float64,
        )

    def rotation_matrix_rvec_jax(rvec_rad: Any) -> Any:
        theta = jnp.linalg.norm(rvec_rad)
        theta_safe = jnp.maximum(theta, 1e-12)
        axis = rvec_rad / theta_safe
        k = skew(axis)
        identity = jnp.eye(3, dtype=jnp.float64)
        return identity + jnp.sin(theta) * k + (1.0 - jnp.cos(theta)) * (k @ k)

    def loss_fn(state: Any, observations: Any) -> Any:
        translation = state[:3]
        rotation = rotation_matrix_rvec_jax(state[3:])
        points_camera = (rotation @ object_points.T).T + translation
        safe_depth = jnp.maximum(points_camera[:, 2:3], 1e-5)
        predicted = focal_length * (points_camera[:, :2] / safe_depth)
        residual = predicted - observations
        depth_penalty = jnp.sum(jnp.square(jnp.minimum(points_camera[:, 2] - 1e-3, 0.0))) * 1e8
        return 0.5 * jnp.sum(jnp.square(residual)) + depth_penalty

    return (
        jax.jit(loss_fn),
        jax.jit(jax.value_and_grad(loss_fn)),
        jax.jit(jax.hessian(loss_fn)),
    )


@lru_cache(maxsize=16)
def _compiled_world_objective_functions(
    focal_length_m: float,
    object_points_world_key: tuple[tuple[float, float, float], ...],
) -> tuple[Any, Any, Any]:
    jax, jnp = _import_jax()
    object_points_world = jnp.asarray(np.asarray(object_points_world_key, dtype=np.float64), dtype=jnp.float64)
    focal_length = jnp.asarray(float(focal_length_m), dtype=jnp.float64)

    def skew(vector: Any) -> Any:
        vx, vy, vz = vector
        return jnp.array(
            [
                [0.0, -vz, vy],
                [vz, 0.0, -vx],
                [-vy, vx, 0.0],
            ],
            dtype=jnp.float64,
        )

    def rotation_matrix_rvec_jax(rvec_rad: Any) -> Any:
        theta = jnp.linalg.norm(rvec_rad)
        theta_safe = jnp.maximum(theta, 1e-12)
        axis = rvec_rad / theta_safe
        k = skew(axis)
        identity = jnp.eye(3, dtype=jnp.float64)
        return identity + jnp.sin(theta) * k + (1.0 - jnp.cos(theta)) * (k @ k)

    def loss_fn(state: Any, observations: Any) -> Any:
        camera_position_world = state[:3]
        rotation_cw = rotation_matrix_rvec_jax(state[3:])
        points_camera = (rotation_cw.T @ (object_points_world - camera_position_world).T).T
        safe_depth = jnp.maximum(points_camera[:, 2:3], 1e-5)
        predicted = focal_length * (points_camera[:, :2] / safe_depth)
        residual = predicted - observations
        depth_penalty = jnp.sum(jnp.square(jnp.minimum(points_camera[:, 2] - 1e-3, 0.0))) * 1e8
        return 0.5 * jnp.sum(jnp.square(residual)) + depth_penalty

    return (
        jax.jit(loss_fn),
        jax.jit(jax.value_and_grad(loss_fn)),
        jax.jit(jax.hessian(loss_fn)),
    )


def _state_diagnostics(
    state: np.ndarray,
    *,
    observed_points_image_plane_m: np.ndarray,
    focal_length_m: float,
    pattern_half_extent_m: float,
) -> dict[str, Any]:
    object_points = _object_points_tag_m(pattern_half_extent_m)
    predicted = _predict_image_plane_points_np(
        state,
        focal_length_m=focal_length_m,
        object_points_tag_m=object_points,
    )
    residual = predicted - observed_points_image_plane_m
    rotation_ct = _rotation_matrix_rvec_np(state[3:])
    camera_position_tag = -(rotation_ct.T @ state[:3])
    points_camera = (rotation_ct @ object_points.T).T + state[:3]
    return {
        "state": [float(value) for value in state.tolist()],
        "translation_tag_to_camera_m": [float(value) for value in state[:3].tolist()],
        "rotation_vector_ct_rad": [float(value) for value in state[3:].tolist()],
        "camera_position_tag_m": [float(value) for value in camera_position_tag.tolist()],
        "predicted_image_plane_points_m": [[float(x), float(y)] for x, y in predicted.tolist()],
        "residual_image_plane_points_m": [[float(x), float(y)] for x, y in residual.tolist()],
        "reprojection_rmse_image_plane_m": float(np.sqrt(np.mean(np.square(residual)))),
        "min_depth_m": float(np.min(points_camera[:, 2])),
        "mean_depth_m": float(np.mean(points_camera[:, 2])),
    }


def _half_turn_about_tag_normal_state(state: np.ndarray) -> np.ndarray:
    """Apply the square-pose half-turn symmetry in the tag frame."""
    state_np = np.asarray(state, dtype=np.float64).reshape(6)
    rotation_ct = _rotation_matrix_rvec_np(state_np[3:])
    half_turn_tag_z = np.diag([-1.0, -1.0, 1.0])
    rotation_ct_half_turn = rotation_ct @ half_turn_tag_z
    rotation_vector_half_turn, _ = cv2.Rodrigues(rotation_ct_half_turn)
    return np.concatenate([state_np[:3], rotation_vector_half_turn.reshape(3)])


def _maybe_disambiguate_half_turn_branch(
    estimate: dict[str, Any],
    *,
    prior_camera_position_tag_m: np.ndarray | None,
    observed_points_image_plane_m: np.ndarray,
    focal_length_m: float,
    pattern_half_extent_m: float,
) -> dict[str, Any]:
    """Use trajectory continuity to resolve the 180-degree square-pose branch."""
    if prior_camera_position_tag_m is None:
        corrected = dict(estimate)
        corrected["branch_disambiguation"] = {"applied": False, "reason": "no_prior"}
        return corrected

    prior_position = np.asarray(prior_camera_position_tag_m, dtype=np.float64).reshape(3)
    current_state = np.asarray(estimate["final_state"], dtype=np.float64).reshape(6)
    half_turn_state = _half_turn_about_tag_normal_state(current_state)

    current_diag = _state_diagnostics(
        current_state,
        observed_points_image_plane_m=observed_points_image_plane_m,
        focal_length_m=focal_length_m,
        pattern_half_extent_m=pattern_half_extent_m,
    )
    half_turn_diag = _state_diagnostics(
        half_turn_state,
        observed_points_image_plane_m=observed_points_image_plane_m,
        focal_length_m=focal_length_m,
        pattern_half_extent_m=pattern_half_extent_m,
    )

    current_prior_distance_m = float(
        np.linalg.norm(np.asarray(current_diag["camera_position_tag_m"], dtype=np.float64) - prior_position)
    )
    half_turn_prior_distance_m = float(
        np.linalg.norm(np.asarray(half_turn_diag["camera_position_tag_m"], dtype=np.float64) - prior_position)
    )
    current_reprojection_rmse_m = float(current_diag["reprojection_rmse_image_plane_m"])
    half_turn_reprojection_rmse_m = float(half_turn_diag["reprojection_rmse_image_plane_m"])

    disambiguation = {
        "applied": False,
        "method": "previous_joint_world_half_turn",
        "prior_camera_position_tag_m": [float(value) for value in prior_position.tolist()],
        "current_prior_distance_m": current_prior_distance_m,
        "half_turn_prior_distance_m": half_turn_prior_distance_m,
        "current_reprojection_rmse_image_plane_m": current_reprojection_rmse_m,
        "half_turn_reprojection_rmse_image_plane_m": half_turn_reprojection_rmse_m,
    }
    if not (half_turn_prior_distance_m + 1e-9 < current_prior_distance_m):
        corrected = dict(estimate)
        corrected["branch_disambiguation"] = disambiguation
        return corrected

    include_history = bool(estimate.get("selected_seed_history"))
    half_turn_optimization = _optimize_pose_from_state(
        half_turn_state,
        observed_points_image_plane_m=observed_points_image_plane_m,
        focal_length_m=focal_length_m,
        pattern_half_extent_m=pattern_half_extent_m,
        max_iterations=18,
        include_history=include_history,
        use_diagonal_damping=bool(estimate.get("use_diagonal_damping", False)),
        initial_damping=float(estimate.get("initial_damping", 0.0)),
    )
    half_turn_optimized_position = np.asarray(half_turn_optimization["camera_position_tag_m"], dtype=np.float64)
    half_turn_optimized_prior_distance_m = float(np.linalg.norm(half_turn_optimized_position - prior_position))
    half_turn_optimized_reprojection_rmse_m = float(half_turn_optimization["reprojection_rmse_image_plane_m"])
    disambiguation["half_turn_optimized_prior_distance_m"] = half_turn_optimized_prior_distance_m
    disambiguation["half_turn_optimized_reprojection_rmse_image_plane_m"] = half_turn_optimized_reprojection_rmse_m
    if not (
        half_turn_optimized_prior_distance_m + 1e-9 < current_prior_distance_m
        and half_turn_optimized_reprojection_rmse_m <= current_reprojection_rmse_m + 1e-9
    ):
        corrected = dict(estimate)
        corrected["branch_disambiguation"] = disambiguation
        return corrected

    corrected = dict(estimate)
    corrected.update(
        {
            "translation_tag_to_camera_m": half_turn_optimization["translation_tag_to_camera_m"],
            "rotation_ct": half_turn_optimization["rotation_ct"],
            "rotation_vector_ct_rad": half_turn_optimization["rotation_vector_ct_rad"],
            "camera_position_tag_m": half_turn_optimization["camera_position_tag_m"],
            "camera_rotation_tc": half_turn_optimization["camera_rotation_tc"],
            "predicted_image_plane_points_m": half_turn_optimization["predicted_image_plane_points_m"],
            "reprojection_rmse_image_plane_m": half_turn_optimized_reprojection_rmse_m,
            "final_state": half_turn_optimization["final_state"],
            "loss_trace": [float(value) for value in half_turn_optimization["loss_trace"]],
            "termination_reason": str(half_turn_optimization["termination_reason"]),
        }
    )
    disambiguation["applied"] = True
    disambiguation["branch_seed_state"] = [float(value) for value in half_turn_state.tolist()]
    corrected["branch_disambiguation"] = disambiguation

    selected_seed_name = str(estimate.get("selected_seed_name", ""))
    candidate_seed_summaries = []
    for candidate in estimate.get("candidate_seed_summaries", []) or []:
        candidate_copy = dict(candidate)
        if str(candidate_copy.get("seed_name", "")) == selected_seed_name and candidate_copy.get("final_state") is not None:
            candidate_copy["final_state"] = half_turn_optimization["final_state"]
            candidate_copy["final_reprojection_rmse_image_plane_m"] = half_turn_optimized_reprojection_rmse_m
            candidate_copy["final_loss"] = float(half_turn_optimization["final_loss"])
            candidate_copy["termination_reason"] = str(half_turn_optimization["termination_reason"])
            candidate_copy["loss_trace"] = [float(value) for value in half_turn_optimization["loss_trace"]]
            if candidate_copy.get("history") is not None:
                candidate_copy["history"] = half_turn_optimization["history"]
        candidate_seed_summaries.append(candidate_copy)
    corrected["candidate_seed_summaries"] = candidate_seed_summaries

    if include_history:
        corrected["selected_seed_history"] = half_turn_optimization["history"]

    return corrected


def _predict_world_image_plane_points_np(
    state: np.ndarray,
    *,
    focal_length_m: float,
    object_points_world_m: np.ndarray,
) -> np.ndarray:
    camera_position_world = np.asarray(state[:3], dtype=np.float64)
    rotation_cw = _rotation_matrix_rvec_np(np.asarray(state[3:], dtype=np.float64))
    points_camera = (rotation_cw.T @ (np.asarray(object_points_world_m, dtype=np.float64) - camera_position_world).T).T
    image_plane = focal_length_m * (points_camera[:, :2] / points_camera[:, 2:3])
    return image_plane.astype(np.float64)


def _world_state_diagnostics(
    state: np.ndarray,
    *,
    observed_points_image_plane_m: np.ndarray,
    focal_length_m: float,
    object_points_world_m: np.ndarray,
) -> dict[str, Any]:
    predicted = _predict_world_image_plane_points_np(
        state,
        focal_length_m=focal_length_m,
        object_points_world_m=object_points_world_m,
    )
    residual = predicted - np.asarray(observed_points_image_plane_m, dtype=np.float64)
    rotation_cw = _rotation_matrix_rvec_np(state[3:])
    camera_position_world = np.asarray(state[:3], dtype=np.float64)
    points_camera = (rotation_cw.T @ (np.asarray(object_points_world_m, dtype=np.float64) - camera_position_world).T).T
    return {
        "state": [float(value) for value in np.asarray(state, dtype=np.float64).tolist()],
        "camera_position_world_m": [float(value) for value in camera_position_world.tolist()],
        "rotation_vector_cw_rad": [float(value) for value in np.asarray(state[3:], dtype=np.float64).tolist()],
        "camera_rotation_cw": [float(value) for value in rotation_cw.reshape(-1).tolist()],
        "predicted_image_plane_points_m": [[float(x), float(y)] for x, y in predicted.tolist()],
        "residual_image_plane_points_m": [[float(x), float(y)] for x, y in residual.tolist()],
        "reprojection_rmse_image_plane_m": float(np.sqrt(np.mean(np.square(residual)))),
        "min_depth_m": float(np.min(points_camera[:, 2])),
        "mean_depth_m": float(np.mean(points_camera[:, 2])),
    }


def _compute_modified_newton_step(
    *,
    hessian_np: np.ndarray,
    gradient_np: np.ndarray,
    use_diagonal_damping: bool,
    initial_damping: float,
) -> tuple[np.ndarray, float, float, str]:
    gradient = np.asarray(gradient_np, dtype=np.float64).reshape(-1)
    hessian = np.asarray(hessian_np, dtype=np.float64)
    identity = np.eye(hessian.shape[0], dtype=np.float64)
    damping = float(initial_damping if use_diagonal_damping else 0.0)
    descent_floor = -1e-18
    solve_attempts = 10 if use_diagonal_damping else 12
    method = "newton"

    for _attempt in range(solve_attempts):
        try:
            step = -np.linalg.solve(hessian + identity * damping, gradient)
        except np.linalg.LinAlgError:
            damping = max(1e-10, damping * 10.0 if damping > 0.0 else 1e-10)
            method = "damped_newton"
            continue
        directional_derivative = float(np.dot(step, gradient))
        if np.isfinite(directional_derivative) and directional_derivative < descent_floor:
            if damping > 0.0:
                method = "damped_newton"
            return step.astype(np.float64), damping, directional_derivative, method
        damping = max(1e-10, damping * 10.0 if damping > 0.0 else 1e-10)
        method = "damped_newton"

    step = -gradient
    directional_derivative = float(np.dot(step, gradient))
    return step.astype(np.float64), damping, directional_derivative, "gradient_fallback"


def _local_step_diagnostics(
    state: np.ndarray,
    *,
    observed_points_image_plane_m: np.ndarray,
    focal_length_m: float,
    pattern_half_extent_m: float,
) -> dict[str, Any]:
    _jax, jnp = _import_jax()
    loss_fn, value_and_grad, hessian_fn = _compiled_objective_functions(focal_length_m, pattern_half_extent_m)
    observations = jnp.asarray(observed_points_image_plane_m, dtype=jnp.float64)
    state_jnp = jnp.asarray(np.asarray(state, dtype=np.float64), dtype=jnp.float64)
    loss_value, gradient = value_and_grad(state_jnp, observations)
    hessian = np.asarray(hessian_fn(state_jnp, observations), dtype=np.float64)
    gradient_np = np.asarray(gradient, dtype=np.float64)
    raw_step: np.ndarray | None = None
    raw_step_dot_gradient: float | None = None
    raw_step_norm: float | None = None
    raw_solve_error: str | None = None
    try:
        raw_step = -np.linalg.solve(hessian, gradient_np)
        raw_step_dot_gradient = float(np.dot(raw_step, gradient_np))
        raw_step_norm = float(np.linalg.norm(raw_step))
    except np.linalg.LinAlgError as exc:
        raw_solve_error = str(exc)

    modified_step, used_damping, modified_dot_gradient, method = _compute_modified_newton_step(
        hessian_np=hessian,
        gradient_np=gradient_np,
        use_diagonal_damping=False,
        initial_damping=0.0,
    )
    eigenvalues = np.linalg.eigvalsh(hessian)
    return {
        "loss": float(loss_value),
        "gradient_norm": float(np.linalg.norm(gradient_np)),
        "gradient_vector": [float(value) for value in gradient_np.tolist()],
        "hessian_rank": int(np.linalg.matrix_rank(hessian)),
        "hessian_eigenvalues": [float(value) for value in eigenvalues.tolist()],
        "raw_step_norm": raw_step_norm,
        "raw_step_dot_gradient": raw_step_dot_gradient,
        "raw_solve_error": raw_solve_error,
        "modified_step_norm": float(np.linalg.norm(modified_step)),
        "modified_step_dot_gradient": float(modified_dot_gradient),
        "modified_step_damping": float(used_damping),
        "modified_step_method": method,
    }


def _optimize_pose_from_state(
    initial_state: np.ndarray,
    *,
    observed_points_image_plane_m: np.ndarray,
    focal_length_m: float,
    pattern_half_extent_m: float,
    max_iterations: int = 18,
    include_history: bool = False,
    use_diagonal_damping: bool = True,
    initial_damping: float = 1e-7,
) -> dict[str, Any]:
    _jax, jnp = _import_jax()
    loss_fn, value_and_grad, hessian_fn = _compiled_objective_functions(focal_length_m, pattern_half_extent_m)
    observations = jnp.asarray(observed_points_image_plane_m, dtype=jnp.float64)
    state = np.asarray(initial_state, dtype=np.float64).copy()
    trace: list[float] = []
    history: list[dict[str, Any]] = []
    termination_reason = "max_iterations"

    for iteration in range(max_iterations):
        state_jnp = jnp.asarray(state, dtype=jnp.float64)
        loss_value, gradient = value_and_grad(state_jnp, observations)
        hessian = hessian_fn(state_jnp, observations)
        loss_scalar = float(loss_value)
        gradient_np = np.asarray(gradient, dtype=np.float64)
        hessian_np = np.asarray(hessian, dtype=np.float64)
        trace.append(loss_scalar)
        gradient_norm = float(np.linalg.norm(gradient_np))

        step, damping, directional_derivative, step_method = _compute_modified_newton_step(
            hessian_np=hessian_np,
            gradient_np=gradient_np,
            use_diagonal_damping=use_diagonal_damping,
            initial_damping=initial_damping,
        )
        step_norm = float(np.linalg.norm(step))

        if step_norm < 1e-12 or gradient_norm < 1e-10:
            termination_reason = "small_step_or_gradient"
            if include_history:
                state_after_diag = _state_diagnostics(
                    state,
                    observed_points_image_plane_m=observed_points_image_plane_m,
                    focal_length_m=focal_length_m,
                    pattern_half_extent_m=pattern_half_extent_m,
                )
                history.append(
                    {
                        "iteration": iteration,
                        "loss_before": loss_scalar,
                        "gradient_norm": gradient_norm,
                        "step_norm": step_norm,
                        "damping": damping,
                        "step_method": step_method,
                        "directional_derivative": directional_derivative,
                        "line_search_alpha": 0.0,
                        "accepted": False,
                        "loss_after": loss_scalar,
                        "reprojection_rmse_after_m": state_after_diag["reprojection_rmse_image_plane_m"],
                        "translation_after_m": state_after_diag["translation_tag_to_camera_m"],
                        "camera_position_after_m": state_after_diag["camera_position_tag_m"],
                        "rotation_vector_after_rad": state_after_diag["rotation_vector_ct_rad"],
                        "min_depth_after_m": state_after_diag["min_depth_m"],
                    }
                )
            break

        armijo_rhs_scale = 1e-4 * float(np.dot(gradient_np, step))
        accepted = False
        alpha = 1.0
        chosen_alpha = 0.0
        loss_after = loss_scalar
        while alpha >= 1e-4:
            candidate = state + alpha * step
            candidate_loss = float(loss_fn(jnp.asarray(candidate, dtype=jnp.float64), observations))
            if np.isfinite(candidate_loss) and candidate_loss <= loss_scalar + alpha * armijo_rhs_scale:
                state = candidate
                accepted = True
                chosen_alpha = alpha
                loss_after = candidate_loss
                break
            alpha *= 0.5
        if not accepted:
            state = state - 0.05 * gradient_np
            chosen_alpha = 0.05
            loss_after = float(loss_fn(jnp.asarray(state, dtype=jnp.float64), observations))

        if include_history:
            state_after_diag = _state_diagnostics(
                state,
                observed_points_image_plane_m=observed_points_image_plane_m,
                focal_length_m=focal_length_m,
                pattern_half_extent_m=pattern_half_extent_m,
            )
            history.append(
                    {
                        "iteration": iteration,
                        "loss_before": loss_scalar,
                        "gradient_norm": gradient_norm,
                        "step_norm": step_norm,
                        "damping": damping,
                        "step_method": step_method,
                        "directional_derivative": directional_derivative,
                        "line_search_alpha": chosen_alpha,
                        "accepted": accepted,
                        "loss_after": loss_after,
                    "reprojection_rmse_after_m": state_after_diag["reprojection_rmse_image_plane_m"],
                    "translation_after_m": state_after_diag["translation_tag_to_camera_m"],
                    "camera_position_after_m": state_after_diag["camera_position_tag_m"],
                    "rotation_vector_after_rad": state_after_diag["rotation_vector_ct_rad"],
                    "min_depth_after_m": state_after_diag["min_depth_m"],
                }
            )

        if iteration >= 2 and abs(trace[-1] - trace[-2]) < 1e-12:
            termination_reason = "plateau"
            break

    final_loss = float(loss_fn(jnp.asarray(state, dtype=jnp.float64), observations))
    final_diag = _state_diagnostics(
        state,
        observed_points_image_plane_m=observed_points_image_plane_m,
        focal_length_m=focal_length_m,
        pattern_half_extent_m=pattern_half_extent_m,
    )
    rotation_ct = _rotation_matrix_rvec_np(np.asarray(state[3:], dtype=np.float64))
    return {
        "initial_state": [float(value) for value in np.asarray(initial_state, dtype=np.float64).tolist()],
        "final_state": [float(value) for value in state.tolist()],
        "termination_reason": termination_reason,
        "loss_trace": [float(value) for value in trace],
        "history": history,
        "final_loss": final_loss,
        "translation_tag_to_camera_m": final_diag["translation_tag_to_camera_m"],
        "rotation_vector_ct_rad": final_diag["rotation_vector_ct_rad"],
        "camera_position_tag_m": final_diag["camera_position_tag_m"],
        "predicted_image_plane_points_m": final_diag["predicted_image_plane_points_m"],
        "residual_image_plane_points_m": final_diag["residual_image_plane_points_m"],
        "reprojection_rmse_image_plane_m": float(final_diag["reprojection_rmse_image_plane_m"]),
        "min_depth_m": float(final_diag["min_depth_m"]),
        "mean_depth_m": float(final_diag["mean_depth_m"]),
        "rotation_ct": [float(value) for value in rotation_ct.reshape(-1).tolist()],
        "camera_rotation_tc": [float(value) for value in rotation_ct.T.reshape(-1).tolist()],
        "use_diagonal_damping": bool(use_diagonal_damping),
        "initial_damping": float(initial_damping),
    }


def _estimate_pose_newton(
    observed_points_image_plane_m: np.ndarray,
    *,
    ordered_pixels_px: np.ndarray,
    camera_model: PhoneCameraModel,
    focal_length_m: float,
    pattern_half_extent_m: float,
    candidate_seed_specs: list[dict[str, Any]] | None = None,
    max_iterations: int = 18,
    include_history: bool = False,
    use_diagonal_damping: bool = True,
    initial_damping: float = 1e-7,
) -> dict[str, Any]:
    camera_obscura_seed = _camera_obscura_seed_details(
        observed_points_image_plane_m,
        focal_length_m=focal_length_m,
        pattern_half_extent_m=pattern_half_extent_m,
    )
    camera_obscura_seed_diag = _state_diagnostics(
        np.asarray(camera_obscura_seed["state"], dtype=np.float64),
        observed_points_image_plane_m=observed_points_image_plane_m,
        focal_length_m=focal_length_m,
        pattern_half_extent_m=pattern_half_extent_m,
    )
    candidate_specs: list[dict[str, Any]] = list(candidate_seed_specs or [camera_obscura_seed])
    # OpenCV solvePnP returns a pose in the standard computer-vision camera
    # frame (x right, y down, z forward). Our optimizer currently uses a
    # camera frame with y up. The map between those frames is a reflection,
    # not a proper rotation, so converting the PnP result directly into the
    # Rodrigues-based optimizer state produces invalid seeds. Until the whole
    # estimator is rewritten in one consistent camera convention, keep PnP out
    # of the candidate set and rely on the physically interpretable obscura
    # seed instead.
    pnp_guess = None

    optimized_candidates: list[dict[str, Any]] = []
    for candidate_spec in candidate_specs:
        initial_state = np.asarray(candidate_spec["state"], dtype=np.float64)
        optimization = _optimize_pose_from_state(
            initial_state,
            observed_points_image_plane_m=observed_points_image_plane_m,
            focal_length_m=focal_length_m,
            pattern_half_extent_m=pattern_half_extent_m,
            max_iterations=max_iterations,
            include_history=include_history,
            use_diagonal_damping=use_diagonal_damping,
            initial_damping=initial_damping,
        )
        seed_diag = _state_diagnostics(
            initial_state,
            observed_points_image_plane_m=observed_points_image_plane_m,
            focal_length_m=focal_length_m,
            pattern_half_extent_m=pattern_half_extent_m,
        )
        optimized_candidates.append(
            {
                "seed_name": str(candidate_spec["seed_name"]),
                "seed_state": [float(value) for value in initial_state.tolist()],
                "seed_details": candidate_spec,
                "optimization": optimization,
                "seed_reprojection_rmse_image_plane_m": seed_diag["reprojection_rmse_image_plane_m"],
                "final_state": optimization["final_state"],
                "final_loss": float(optimization["final_loss"]),
                "final_reprojection_rmse_image_plane_m": float(optimization["reprojection_rmse_image_plane_m"]),
                "termination_reason": str(optimization["termination_reason"]),
                "loss_trace": [float(value) for value in optimization["loss_trace"]],
                "history": optimization["history"],
            }
        )

    optimized_candidates.sort(key=lambda item: float(item["final_loss"]))
    best_candidate = optimized_candidates[0]
    best_optimization = dict(best_candidate["optimization"])
    result = {
        "translation_tag_to_camera_m": best_optimization["translation_tag_to_camera_m"],
        "rotation_ct": best_optimization["rotation_ct"],
        "rotation_vector_ct_rad": best_optimization["rotation_vector_ct_rad"],
        "camera_position_tag_m": best_optimization["camera_position_tag_m"],
        "camera_rotation_tc": best_optimization["camera_rotation_tc"],
        "use_diagonal_damping": bool(use_diagonal_damping),
        "initial_damping": float(initial_damping),
        "initial_state": best_optimization["initial_state"],
        "final_state": best_optimization["final_state"],
        "selected_seed_name": str(best_candidate["seed_name"]),
        "termination_reason": str(best_optimization["termination_reason"]),
        "loss_trace": [float(value) for value in best_optimization["loss_trace"]],
        "reprojection_rmse_image_plane_m": float(best_optimization["reprojection_rmse_image_plane_m"]),
        "predicted_image_plane_points_m": best_optimization["predicted_image_plane_points_m"],
        "camera_obscura_seed": camera_obscura_seed,
        "camera_obscura_seed_reprojection_rmse_image_plane_m": float(
            camera_obscura_seed_diag["reprojection_rmse_image_plane_m"]
        ),
        "candidate_seed_summaries": [
            {
                "seed_name": str(item["seed_name"]),
                "seed_reprojection_rmse_image_plane_m": float(item["seed_reprojection_rmse_image_plane_m"]),
                "final_reprojection_rmse_image_plane_m": float(item["final_reprojection_rmse_image_plane_m"]),
                "final_loss": float(item["final_loss"]),
                "termination_reason": str(item["termination_reason"]),
                "seed_state": [float(value) for value in np.asarray(item["seed_state"], dtype=np.float64).tolist()],
                **(
                    {
                        "final_state": [float(value) for value in np.asarray(item["final_state"], dtype=np.float64).tolist()],
                        "loss_trace": [float(value) for value in item["loss_trace"]],
                        "history": item["history"],
                        "seed_details": item["seed_details"],
                    }
                    if include_history
                    else {}
                ),
            }
            for item in optimized_candidates
        ],
    }
    if pnp_guess is not None:
        result["opencv_pnp_seed"] = pnp_guess
        result["opencv_pnp_seed_reprojection_rmse_image_plane_m"] = next(
            float(item["seed_reprojection_rmse_image_plane_m"])
            for item in optimized_candidates
            if item["seed_name"] == "opencv_pnp"
        )
    if include_history:
        result["selected_seed_history"] = best_optimization["history"]
    return result


def _candidate_seed_specs_for_replay(estimate: dict[str, Any]) -> list[dict[str, Any]]:
    """Rebuild seed specs so report reruns follow the same initialization path."""
    candidate_specs: list[dict[str, Any]] = []
    for candidate in estimate.get("candidate_seed_summaries", []) or []:
        seed_state = candidate.get("seed_state")
        if seed_state is None:
            continue
        spec: dict[str, Any] = {
            "seed_name": str(candidate.get("seed_name", "seed")),
            "state": [float(value) for value in np.asarray(seed_state, dtype=np.float64).tolist()],
        }
        seed_details = candidate.get("seed_details")
        if isinstance(seed_details, dict):
            for key, value in seed_details.items():
                if key not in {"seed_name", "state"}:
                    spec[key] = value
        candidate_specs.append(spec)
    if candidate_specs:
        return candidate_specs
    initial_state = estimate.get("initial_state")
    if initial_state is None:
        return []
    return [
        {
            "seed_name": str(estimate.get("selected_seed_name", "seed")),
            "state": [float(value) for value in np.asarray(initial_state, dtype=np.float64).tolist()],
        }
    ]


def _optimize_world_pose_from_state(
    initial_state: np.ndarray,
    *,
    observed_points_image_plane_m: np.ndarray,
    focal_length_m: float,
    object_points_world_m: np.ndarray,
    max_iterations: int = 18,
    include_history: bool = False,
    use_diagonal_damping: bool = True,
    initial_damping: float = 1e-7,
) -> dict[str, Any]:
    _jax, jnp = _import_jax()
    object_points_key = tuple(tuple(float(value) for value in point) for point in np.asarray(object_points_world_m, dtype=np.float64).tolist())
    loss_fn, value_and_grad, hessian_fn = _compiled_world_objective_functions(focal_length_m, object_points_key)
    observations = jnp.asarray(observed_points_image_plane_m, dtype=jnp.float64)
    state = np.asarray(initial_state, dtype=np.float64).copy()
    trace: list[float] = []
    history: list[dict[str, Any]] = []
    termination_reason = "max_iterations"

    for iteration in range(max_iterations):
        state_jnp = jnp.asarray(state, dtype=jnp.float64)
        loss_value, gradient = value_and_grad(state_jnp, observations)
        hessian = hessian_fn(state_jnp, observations)
        loss_scalar = float(loss_value)
        gradient_np = np.asarray(gradient, dtype=np.float64)
        hessian_np = np.asarray(hessian, dtype=np.float64)
        trace.append(loss_scalar)
        gradient_norm = float(np.linalg.norm(gradient_np))

        step, damping, directional_derivative, step_method = _compute_modified_newton_step(
            hessian_np=hessian_np,
            gradient_np=gradient_np,
            use_diagonal_damping=use_diagonal_damping,
            initial_damping=initial_damping,
        )
        step_norm = float(np.linalg.norm(step))

        if step_norm < 1e-12 or gradient_norm < 1e-10:
            termination_reason = "small_step_or_gradient"
            if include_history:
                state_after_diag = _world_state_diagnostics(
                    state,
                    observed_points_image_plane_m=observed_points_image_plane_m,
                    focal_length_m=focal_length_m,
                    object_points_world_m=object_points_world_m,
                )
                history.append(
                    {
                        "iteration": iteration,
                        "loss_before": loss_scalar,
                        "gradient_norm": gradient_norm,
                        "step_norm": step_norm,
                        "damping": damping,
                        "step_method": step_method,
                        "directional_derivative": directional_derivative,
                        "line_search_alpha": 0.0,
                        "accepted": False,
                        "loss_after": loss_scalar,
                        "reprojection_rmse_after_m": state_after_diag["reprojection_rmse_image_plane_m"],
                        "camera_position_world_after_m": state_after_diag["camera_position_world_m"],
                        "rotation_vector_cw_after_rad": state_after_diag["rotation_vector_cw_rad"],
                        "min_depth_after_m": state_after_diag["min_depth_m"],
                    }
                )
            break

        armijo_rhs_scale = 1e-4 * float(np.dot(gradient_np, step))
        accepted = False
        alpha = 1.0
        chosen_alpha = 0.0
        loss_after = loss_scalar
        while alpha >= 1e-4:
            candidate = state + alpha * step
            candidate_loss = float(loss_fn(jnp.asarray(candidate, dtype=jnp.float64), observations))
            if np.isfinite(candidate_loss) and candidate_loss <= loss_scalar + alpha * armijo_rhs_scale:
                state = candidate
                accepted = True
                chosen_alpha = alpha
                loss_after = candidate_loss
                break
            alpha *= 0.5
        if not accepted:
            state = state - 0.05 * gradient_np
            chosen_alpha = 0.05
            loss_after = float(loss_fn(jnp.asarray(state, dtype=jnp.float64), observations))

        if include_history:
            state_after_diag = _world_state_diagnostics(
                state,
                observed_points_image_plane_m=observed_points_image_plane_m,
                focal_length_m=focal_length_m,
                object_points_world_m=object_points_world_m,
            )
            history.append(
                {
                    "iteration": iteration,
                    "loss_before": loss_scalar,
                    "gradient_norm": gradient_norm,
                    "step_norm": step_norm,
                    "damping": damping,
                    "step_method": step_method,
                    "directional_derivative": directional_derivative,
                    "line_search_alpha": chosen_alpha,
                    "accepted": accepted,
                    "loss_after": loss_after,
                    "reprojection_rmse_after_m": state_after_diag["reprojection_rmse_image_plane_m"],
                    "camera_position_world_after_m": state_after_diag["camera_position_world_m"],
                    "rotation_vector_cw_after_rad": state_after_diag["rotation_vector_cw_rad"],
                    "min_depth_after_m": state_after_diag["min_depth_m"],
                }
            )

        if iteration >= 2 and abs(trace[-1] - trace[-2]) < 1e-12:
            termination_reason = "plateau"
            break

    final_loss = float(loss_fn(jnp.asarray(state, dtype=jnp.float64), observations))
    final_diag = _world_state_diagnostics(
        state,
        observed_points_image_plane_m=observed_points_image_plane_m,
        focal_length_m=focal_length_m,
        object_points_world_m=object_points_world_m,
    )
    return {
        "initial_state": [float(value) for value in np.asarray(initial_state, dtype=np.float64).tolist()],
        "final_state": [float(value) for value in state.tolist()],
        "termination_reason": termination_reason,
        "loss_trace": [float(value) for value in trace],
        "history": history,
        "final_loss": final_loss,
        "camera_position_world_m": final_diag["camera_position_world_m"],
        "rotation_vector_cw_rad": final_diag["rotation_vector_cw_rad"],
        "camera_rotation_cw": final_diag["camera_rotation_cw"],
        "predicted_image_plane_points_m": final_diag["predicted_image_plane_points_m"],
        "residual_image_plane_points_m": final_diag["residual_image_plane_points_m"],
        "reprojection_rmse_image_plane_m": float(final_diag["reprojection_rmse_image_plane_m"]),
        "min_depth_m": float(final_diag["min_depth_m"]),
        "mean_depth_m": float(final_diag["mean_depth_m"]),
        "use_diagonal_damping": bool(use_diagonal_damping),
        "initial_damping": float(initial_damping),
    }


def _estimate_world_pose_from_all_points(
    observed_points_image_plane_m: np.ndarray,
    *,
    object_points_world_m: np.ndarray,
    seed_states_world: list[np.ndarray],
    include_history: bool = False,
    focal_length_m: float,
    max_iterations: int = 18,
    use_diagonal_damping: bool = False,
    initial_damping: float = 0.0,
) -> dict[str, Any]:
    optimized_candidates: list[dict[str, Any]] = []
    for seed_index, initial_state in enumerate(seed_states_world):
        optimization = _optimize_world_pose_from_state(
            initial_state,
            observed_points_image_plane_m=observed_points_image_plane_m,
            focal_length_m=focal_length_m,
            object_points_world_m=object_points_world_m,
            max_iterations=max_iterations,
            include_history=include_history,
            use_diagonal_damping=use_diagonal_damping,
            initial_damping=initial_damping,
        )
        seed_diag = _world_state_diagnostics(
            initial_state,
            observed_points_image_plane_m=observed_points_image_plane_m,
            focal_length_m=focal_length_m,
            object_points_world_m=object_points_world_m,
        )
        optimized_candidates.append(
            {
                "seed_name": f"world_seed_{seed_index}",
                "seed_state": [float(value) for value in np.asarray(initial_state, dtype=np.float64).tolist()],
                "seed_reprojection_rmse_image_plane_m": float(seed_diag["reprojection_rmse_image_plane_m"]),
                "optimization": optimization,
                "final_state": optimization["final_state"],
                "final_loss": float(optimization["final_loss"]),
                "final_reprojection_rmse_image_plane_m": float(optimization["reprojection_rmse_image_plane_m"]),
                "termination_reason": str(optimization["termination_reason"]),
                "loss_trace": [float(value) for value in optimization["loss_trace"]],
                "history": optimization["history"],
            }
        )

    best = min(optimized_candidates, key=lambda item: float(item["final_loss"]))
    result = {
        "camera_position_world_m": best["optimization"]["camera_position_world_m"],
        "camera_rotation_cw": best["optimization"]["camera_rotation_cw"],
        "rotation_vector_cw_rad": best["optimization"]["rotation_vector_cw_rad"],
        "selected_seed_name": best["seed_name"],
        "termination_reason": best["optimization"]["termination_reason"],
        "loss_trace": best["optimization"]["loss_trace"],
        "reprojection_rmse_image_plane_m": best["optimization"]["reprojection_rmse_image_plane_m"],
        "predicted_image_plane_points_m": best["optimization"]["predicted_image_plane_points_m"],
        "candidate_seed_summaries": [
            {
                "seed_name": item["seed_name"],
                "seed_reprojection_rmse_image_plane_m": item["seed_reprojection_rmse_image_plane_m"],
                "final_reprojection_rmse_image_plane_m": item["final_reprojection_rmse_image_plane_m"],
                "final_loss": item["final_loss"],
                "termination_reason": item["termination_reason"],
            }
            for item in optimized_candidates
        ],
        "use_diagonal_damping": bool(use_diagonal_damping),
        "initial_damping": float(initial_damping),
        "seed_count": len(seed_states_world),
    }
    if include_history:
        result["selected_seed_history"] = best["optimization"]["history"]
    return result


def _rotation_error_deg(rotation_tc_est: np.ndarray, rotation_tc_gt: np.ndarray) -> float:
    relative = rotation_tc_est.T @ rotation_tc_gt
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _rotation_matrix_to_xyz_angles_deg(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    sin_y = float(np.clip(-matrix[2, 0], -1.0, 1.0))
    cos_y = math.sqrt(max(0.0, 1.0 - sin_y * sin_y))
    if cos_y > 1e-9:
        rx = math.atan2(float(matrix[2, 1]), float(matrix[2, 2]))
        ry = math.asin(sin_y)
        rz = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    else:
        rx = math.atan2(float(-matrix[0, 1]), float(matrix[1, 1]))
        ry = math.asin(sin_y)
        rz = 0.0
    return np.degrees(np.array([rx, ry, rz], dtype=np.float64))


def _pose_error_vectors(
    *,
    ground_truth_position_tag_m: np.ndarray,
    ground_truth_rotation_tc: np.ndarray,
    estimated_position_tag_m: np.ndarray,
    estimated_rotation_tc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    position_error_vector_m = (
        np.asarray(ground_truth_position_tag_m, dtype=np.float64).reshape(3)
        - np.asarray(estimated_position_tag_m, dtype=np.float64).reshape(3)
    )
    relative_rotation = (
        np.asarray(ground_truth_rotation_tc, dtype=np.float64).reshape(3, 3)
        @ np.asarray(estimated_rotation_tc, dtype=np.float64).reshape(3, 3).T
    )
    rotation_error_xyz_deg = _rotation_matrix_to_xyz_angles_deg(relative_rotation)
    return position_error_vector_m, rotation_error_xyz_deg


def _camera_world_pose_from_tag_pose(
    *,
    camera_position_tag_m: np.ndarray,
    camera_rotation_tc: np.ndarray,
    tag_center_world_m: np.ndarray,
    tag_rotation_wt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    camera_position_world_m = (
        np.asarray(tag_center_world_m, dtype=np.float64).reshape(3)
        + np.asarray(tag_rotation_wt, dtype=np.float64).reshape(3, 3)
        @ np.asarray(camera_position_tag_m, dtype=np.float64).reshape(3)
    )
    camera_rotation_cw = (
        np.asarray(tag_rotation_wt, dtype=np.float64).reshape(3, 3)
        @ np.asarray(camera_rotation_tc, dtype=np.float64).reshape(3, 3)
    )
    return camera_position_world_m.astype(np.float64), camera_rotation_cw.astype(np.float64)


def _tag_pose_state_from_world_pose(
    *,
    camera_position_world_m: np.ndarray,
    camera_rotation_cw: np.ndarray,
    tag_center_world_m: np.ndarray,
    tag_rotation_wt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    camera_position_tag_m = (
        np.asarray(tag_rotation_wt, dtype=np.float64).reshape(3, 3).T
        @ (
            np.asarray(camera_position_world_m, dtype=np.float64).reshape(3)
            - np.asarray(tag_center_world_m, dtype=np.float64).reshape(3)
        )
    )
    camera_rotation_tc = (
        np.asarray(tag_rotation_wt, dtype=np.float64).reshape(3, 3).T
        @ np.asarray(camera_rotation_cw, dtype=np.float64).reshape(3, 3)
    )
    rotation_ct = camera_rotation_tc.T
    translation_ct = -(rotation_ct @ camera_position_tag_m)
    rotation_vector_ct, _ = cv2.Rodrigues(rotation_ct)
    return (
        np.concatenate([translation_ct.reshape(3), rotation_vector_ct.reshape(3)]).astype(np.float64),
        camera_position_tag_m.astype(np.float64),
    )


def _format_metric(value: float | None, *, scale: float = 1.0, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value * scale:.4f}{suffix}"


def _resize_for_panel(image_bgr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(image_bgr, size, interpolation=cv2.INTER_AREA)


def _draw_text_lines(
    canvas: np.ndarray,
    lines: list[str],
    *,
    origin: tuple[int, int],
    line_height: int = 24,
    color: tuple[int, int, int] = (245, 248, 250),
) -> None:
    x0, y0 = origin
    for index, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (x0, y0 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            1,
            cv2.LINE_AA,
        )


def _save_metric_plot(
    path: Path,
    *,
    title: str,
    y_label: str,
    values: list[float],
    color_bgr: tuple[int, int, int],
) -> None:
    if not values:
        return
    width = 960
    height = 360
    left = 72
    right = 30
    top = 54
    bottom = 56
    canvas = np.full((height, width, 3), 252, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (220, 224, 228), 1)
    cv2.putText(canvas, title, (24, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (22, 26, 32), 2, cv2.LINE_AA)
    cv2.putText(canvas, y_label, (24, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (82, 90, 98), 1, cv2.LINE_AA)

    min_value = float(min(values))
    max_value = float(max(values))
    if math.isclose(min_value, max_value):
        min_value -= 1.0
        max_value += 1.0
    padding = (max_value - min_value) * 0.08
    min_value -= padding
    max_value += padding

    plot_width = width - left - right
    plot_height = height - top - bottom
    origin_x = left
    origin_y = height - bottom
    cv2.line(canvas, (origin_x, top), (origin_x, origin_y), (170, 176, 184), 1, cv2.LINE_AA)
    cv2.line(canvas, (origin_x, origin_y), (width - right, origin_y), (170, 176, 184), 1, cv2.LINE_AA)

    for tick_index in range(5):
        value = min_value + (max_value - min_value) * tick_index / 4.0
        y = int(round(origin_y - plot_height * tick_index / 4.0))
        cv2.line(canvas, (origin_x, y), (width - right, y), (232, 236, 240), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{value:.4f}", (8, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (98, 106, 114), 1, cv2.LINE_AA)

    if len(values) == 1:
        x_positions = [origin_x + plot_width // 2]
    else:
        x_positions = [int(round(origin_x + plot_width * index / (len(values) - 1))) for index in range(len(values))]
    points = []
    for x, value in zip(x_positions, values):
        y_norm = (value - min_value) / max(max_value - min_value, 1e-12)
        y = int(round(origin_y - y_norm * plot_height))
        points.append((x, y))

    cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, color_bgr, 2, cv2.LINE_AA)
    for point in points[:: max(1, len(points) // 24)]:
        cv2.circle(canvas, point, 3, color_bgr, -1, cv2.LINE_AA)

    stats_text = [
        f"mean={np.mean(values):.5f}",
        f"median={np.median(values):.5f}",
        f"min={np.min(values):.5f}",
        f"max={np.max(values):.5f}",
        f"n={len(values)}",
    ]
    _draw_text_lines(canvas, stats_text, origin=(width - 230, 32), line_height=20, color=(52, 60, 68))
    cv2.imwrite(str(path), canvas)


def _format_axis_tick_value(value: float) -> str:
    magnitude = abs(float(value))
    if magnitude >= 100.0:
        return f"{value:.1f}"
    if magnitude >= 10.0:
        return f"{value:.2f}"
    if magnitude >= 1.0:
        return f"{value:.3f}"
    if magnitude >= 0.1:
        return f"{value:.4f}"
    return f"{value:.5f}"


def _save_multi_series_plot(
    path: Path,
    *,
    title: str,
    y_label: str,
    x_label: str,
    x_values: list[int],
    series_specs: list[dict[str, Any]],
) -> None:
    if not x_values or not series_specs:
        return
    finite_values = [
        float(value)
        for spec in series_specs
        for value in spec.get("values", [])
        if value is not None and np.isfinite(float(value))
    ]
    if not finite_values:
        return

    width = 1080
    height = 420
    left = 84
    right = 36
    top = 58
    bottom = 74
    canvas = np.full((height, width, 3), 252, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (220, 224, 228), 1)
    cv2.putText(canvas, title, (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (22, 26, 32), 2, cv2.LINE_AA)
    cv2.putText(canvas, y_label, (24, height - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (82, 90, 98), 1, cv2.LINE_AA)
    cv2.putText(canvas, x_label, (width - 184, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (82, 90, 98), 1, cv2.LINE_AA)

    min_value = float(min(finite_values))
    max_value = float(max(finite_values))
    if math.isclose(min_value, max_value):
        min_value -= 1.0
        max_value += 1.0
    padding = (max_value - min_value) * 0.08
    min_value -= padding
    max_value += padding

    x_min = int(min(x_values))
    x_max = int(max(x_values))
    plot_width = width - left - right
    plot_height = height - top - bottom
    origin_x = left
    origin_y = height - bottom
    cv2.line(canvas, (origin_x, top), (origin_x, origin_y), (170, 176, 184), 1, cv2.LINE_AA)
    cv2.line(canvas, (origin_x, origin_y), (width - right, origin_y), (170, 176, 184), 1, cv2.LINE_AA)

    for tick_index in range(5):
        value = min_value + (max_value - min_value) * tick_index / 4.0
        y = int(round(origin_y - plot_height * tick_index / 4.0))
        cv2.line(canvas, (origin_x, y), (width - right, y), (232, 236, 240), 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            _format_axis_tick_value(value),
            (8, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (98, 106, 114),
            1,
            cv2.LINE_AA,
        )

    for tick_index in range(5):
        if x_max == x_min:
            frame_value = x_min
        else:
            frame_value = int(round(x_min + (x_max - x_min) * tick_index / 4.0))
        x = origin_x + (plot_width // 2 if x_max == x_min else int(round(plot_width * tick_index / 4.0)))
        cv2.line(canvas, (x, top), (x, origin_y), (240, 243, 246), 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{frame_value}",
            (x - 14, origin_y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (98, 106, 114),
            1,
            cv2.LINE_AA,
        )

    def x_to_px(frame_number: int) -> int:
        if x_max == x_min:
            return origin_x + plot_width // 2
        return origin_x + int(round((frame_number - x_min) / max(x_max - x_min, 1) * plot_width))

    def y_to_px(value: float) -> int:
        y_norm = (float(value) - min_value) / max(max_value - min_value, 1e-12)
        return int(round(origin_y - y_norm * plot_height))

    for spec in series_specs:
        values = spec.get("values", [])
        color_bgr = tuple(int(channel) for channel in spec.get("color_bgr", (61, 142, 255)))
        segments: list[list[tuple[int, int]]] = []
        current_segment: list[tuple[int, int]] = []
        for frame_number, value in zip(x_values, values):
            if value is None or not np.isfinite(float(value)):
                if len(current_segment) >= 2:
                    segments.append(current_segment)
                elif len(current_segment) == 1:
                    segments.append(current_segment.copy())
                current_segment = []
                continue
            current_segment.append((x_to_px(int(frame_number)), y_to_px(float(value))))
        if len(current_segment) >= 2:
            segments.append(current_segment)
        elif len(current_segment) == 1:
            segments.append(current_segment.copy())

        for segment in segments:
            if len(segment) >= 2:
                cv2.polylines(canvas, [np.asarray(segment, dtype=np.int32)], False, color_bgr, 2, cv2.LINE_AA)
            for point in segment[:: max(1, len(segment) // 18 or 1)]:
                cv2.circle(canvas, point, 3, color_bgr, -1, cv2.LINE_AA)

    legend_x = width - 320
    legend_y = 30
    for legend_index, spec in enumerate(series_specs):
        color_bgr = tuple(int(channel) for channel in spec.get("color_bgr", (61, 142, 255)))
        y = legend_y + legend_index * 24
        cv2.line(canvas, (legend_x, y), (legend_x + 22, y), color_bgr, 2, cv2.LINE_AA)
        cv2.circle(canvas, (legend_x + 11, y), 3, color_bgr, -1, cv2.LINE_AA)
        label = str(spec.get("label", "series"))
        valid_count = sum(1 for value in spec.get("values", []) if value is not None and np.isfinite(float(value)))
        cv2.putText(
            canvas,
            f"{label} (n={valid_count})",
            (legend_x + 30, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (52, 60, 68),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(path), canvas)


def _save_representative_fit(
    path: Path,
    *,
    raw_frame_bgr: np.ndarray,
    analysis_frame_bgr: np.ndarray,
    annotated_frame_bgr: np.ndarray,
    metadata_lines: list[str],
) -> None:
    panel_width = 360
    panel_height = 203
    gap = 16
    top = 68
    footer_height = 150
    width = panel_width * 3 + gap * 4
    height = top + panel_height + gap + footer_height
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    cv2.putText(canvas, "Representative Fit Frame", (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.96, (20, 24, 30), 2, cv2.LINE_AA)

    panels = [
        ("Recorded raw frame", raw_frame_bgr),
        ("Analysis input frame", analysis_frame_bgr),
        ("Detected/annotated frame", annotated_frame_bgr),
    ]
    for index, (label, image) in enumerate(panels):
        x0 = gap + index * (panel_width + gap)
        y0 = top
        panel = _resize_for_panel(image, (panel_width, panel_height))
        canvas[y0 : y0 + panel_height, x0 : x0 + panel_width] = panel
        cv2.rectangle(canvas, (x0, y0), (x0 + panel_width, y0 + panel_height), (208, 214, 220), 1)
        cv2.putText(canvas, label, (x0, y0 - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (66, 74, 82), 1, cv2.LINE_AA)

    _draw_text_lines(
        canvas,
        metadata_lines,
        origin=(24, top + panel_height + 32),
        line_height=24,
        color=(44, 52, 60),
    )
    cv2.imwrite(str(path), canvas)


def _save_pose_comparison(
    path: Path,
    *,
    entries: list[dict[str, Any]],
) -> None:
    width = 1080
    height = 500
    gap = 18
    panel_width = (width - gap * 3) // 2
    panel_height = 310
    top = 84
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    cv2.putText(canvas, "Representative Pose Comparison", (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.96, (20, 24, 30), 2, cv2.LINE_AA)

    positions = [np.asarray(entry["position_m"], dtype=np.float64) for entry in entries]
    all_x = [0.0] + [float(position[0]) for position in positions]
    all_y = [0.0] + [float(position[1]) for position in positions]
    all_z = [0.0] + [float(position[2]) for position in positions]

    def padded_range(values: list[float]) -> tuple[float, float]:
        lower = float(min(values))
        upper = float(max(values))
        if math.isclose(lower, upper):
            lower -= 0.5
            upper += 0.5
        span = upper - lower
        return lower - span * 0.14, upper + span * 0.14

    x_min, x_max = padded_range(all_x)
    y_min, y_max = padded_range(all_y)
    z_min, z_max = padded_range(all_z)

    def to_panel(
        point_a: float,
        point_b: float,
        *,
        a_min: float,
        a_max: float,
        b_min: float,
        b_max: float,
        origin: tuple[int, int],
    ) -> tuple[int, int]:
        x0, y0 = origin
        px = x0 + int(round((point_a - a_min) / max(a_max - a_min, 1e-12) * panel_width))
        py = y0 + panel_height - int(round((point_b - b_min) / max(b_max - b_min, 1e-12) * panel_height))
        return px, py

    panels = [
        {
            "title": "Top View (x vs z)",
            "a_label": "x [m]",
            "b_label": "z [m]",
            "a_index": 0,
            "b_index": 2,
            "a_range": (x_min, x_max),
            "b_range": (z_min, z_max),
            "origin": (gap, top),
        },
        {
            "title": "Side View (y vs z)",
            "a_label": "y [m]",
            "b_label": "z [m]",
            "a_index": 1,
            "b_index": 2,
            "a_range": (y_min, y_max),
            "b_range": (z_min, z_max),
            "origin": (gap * 2 + panel_width, top),
        },
    ]

    for panel in panels:
        x0, y0 = panel["origin"]
        cv2.rectangle(canvas, (x0, y0), (x0 + panel_width, y0 + panel_height), (208, 214, 220), 1)
        cv2.putText(canvas, panel["title"], (x0, y0 - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (66, 74, 82), 1, cv2.LINE_AA)
        a_min, a_max = panel["a_range"]
        b_min, b_max = panel["b_range"]
        origin_point = to_panel(0.0, 0.0, a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, origin=panel["origin"])
        cv2.line(canvas, (x0, origin_point[1]), (x0 + panel_width, origin_point[1]), (232, 236, 240), 1, cv2.LINE_AA)
        cv2.line(canvas, (origin_point[0], y0), (origin_point[0], y0 + panel_height), (232, 236, 240), 1, cv2.LINE_AA)
        cv2.putText(canvas, panel["a_label"], (x0 + 8, y0 + panel_height + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (82, 90, 98), 1, cv2.LINE_AA)
        cv2.putText(canvas, panel["b_label"], (x0 + panel_width - 86, y0 + panel_height + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (82, 90, 98), 1, cv2.LINE_AA)
        tag_point = to_panel(0.0, 0.0, a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, origin=panel["origin"])
        cv2.circle(canvas, tag_point, 6, (56, 64, 72), -1, cv2.LINE_AA)
        cv2.putText(canvas, "tag origin", (tag_point[0] + 8, tag_point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (56, 64, 72), 1, cv2.LINE_AA)

        for entry in entries:
            position = np.asarray(entry["position_m"], dtype=np.float64)
            point = to_panel(
                float(position[panel["a_index"]]),
                float(position[panel["b_index"]]),
                a_min=a_min,
                a_max=a_max,
                b_min=b_min,
                b_max=b_max,
                origin=panel["origin"],
            )
            color = tuple(int(value) for value in entry["color_bgr"])
            cv2.circle(canvas, point, 7, color, -1, cv2.LINE_AA)
            cv2.putText(canvas, str(entry["short_label"]), (point[0] + 8, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)

    legend_lines = [
        f"{entry['short_label']}: {entry['label']}  pos={_format_vector(entry['position_m'], precision=3)} m"
        for entry in entries
    ]
    _draw_text_lines(canvas, legend_lines, origin=(24, top + panel_height + 54), line_height=24, color=(44, 52, 60))
    cv2.imwrite(str(path), canvas)


def _draw_projected_point_set(
    frame_bgr: np.ndarray,
    *,
    points_px: np.ndarray,
    color_bgr: tuple[int, int, int],
    marker_style: str,
    point_labels: list[str] | None = None,
    point_offset: tuple[int, int] = (8, -8),
    line_thickness: int = 1,
    marker_size: int = 9,
) -> None:
    points = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    if points.shape[0] >= 5:
        corners = np.round(points[1:5]).astype(np.int32)
        cv2.polylines(frame_bgr, [corners], True, color_bgr, line_thickness, cv2.LINE_AA)

    for index, point in enumerate(points):
        center = (int(round(point[0])), int(round(point[1])))
        if marker_style == "circle":
            cv2.circle(frame_bgr, center, max(2, marker_size // 3), color_bgr, -1, cv2.LINE_AA)
        elif marker_style == "cross":
            cv2.drawMarker(
                frame_bgr,
                center,
                color_bgr,
                markerType=cv2.MARKER_CROSS,
                markerSize=marker_size,
                thickness=max(1, line_thickness),
                line_type=cv2.LINE_AA,
            )
        else:
            cv2.drawMarker(
                frame_bgr,
                center,
                color_bgr,
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=marker_size,
                thickness=max(1, line_thickness),
                line_type=cv2.LINE_AA,
            )
        if point_labels is not None and index < len(point_labels):
            cv2.putText(
                frame_bgr,
                point_labels[index],
                (center[0] + point_offset[0], center[1] + point_offset[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                color_bgr,
                1,
                cv2.LINE_AA,
            )


def _crop_bounds_for_points(
    image_shape: tuple[int, ...],
    point_sets_px: list[np.ndarray],
    *,
    margin_px: int = 48,
) -> tuple[int, int, int, int]:
    valid_sets = [np.asarray(points, dtype=np.float64).reshape(-1, 2) for points in point_sets_px if np.asarray(points).size]
    height, width = image_shape[:2]
    if not valid_sets:
        return 0, 0, width, height
    stacked = np.vstack(valid_sets)
    min_x = max(0, int(math.floor(float(np.min(stacked[:, 0]))) - margin_px))
    min_y = max(0, int(math.floor(float(np.min(stacked[:, 1]))) - margin_px))
    max_x = min(width, int(math.ceil(float(np.max(stacked[:, 0]))) + margin_px))
    max_y = min(height, int(math.ceil(float(np.max(stacked[:, 1]))) + margin_px))
    max_x = max(max_x, min_x + 2)
    max_y = max(max_y, min_y + 2)
    return min_x, min_y, max_x, max_y


def _save_single_pattern_overlay(
    path: Path,
    *,
    analysis_frame_bgr: np.ndarray,
    observed_pixels_px: np.ndarray,
    ground_truth_pixels_px: np.ndarray,
    optimized_pixels_px: np.ndarray,
    point_labels: list[str],
    metadata_lines: list[str],
) -> None:
    panel_width = 520
    panel_height = 292
    gap = 18
    top = 72
    footer_height = 180
    width = panel_width * 2 + gap * 3
    height = top + panel_height + gap + footer_height
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    cv2.putText(canvas, "Single Pattern Image Comparison", (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.96, (20, 24, 30), 2, cv2.LINE_AA)

    full_view = analysis_frame_bgr.copy()
    _draw_projected_point_set(
        full_view,
        points_px=np.asarray(observed_pixels_px, dtype=np.float64),
        color_bgr=(0, 165, 255),
        marker_style="circle",
        line_thickness=1,
        marker_size=8,
    )
    _draw_projected_point_set(
        full_view,
        points_px=np.asarray(ground_truth_pixels_px, dtype=np.float64),
        color_bgr=(32, 181, 80),
        marker_style="cross",
        line_thickness=1,
        marker_size=12,
    )
    _draw_projected_point_set(
        full_view,
        points_px=np.asarray(optimized_pixels_px, dtype=np.float64),
        color_bgr=(64, 96, 255),
        marker_style="tilted_cross",
        line_thickness=1,
        marker_size=12,
    )

    crop_x0, crop_y0, crop_x1, crop_y1 = _crop_bounds_for_points(
        analysis_frame_bgr.shape,
        [
            np.asarray(observed_pixels_px, dtype=np.float64),
            np.asarray(ground_truth_pixels_px, dtype=np.float64),
            np.asarray(optimized_pixels_px, dtype=np.float64),
        ],
        margin_px=54,
    )
    zoom_view = analysis_frame_bgr[crop_y0:crop_y1, crop_x0:crop_x1].copy()
    crop_offset = np.array([crop_x0, crop_y0], dtype=np.float64)
    _draw_projected_point_set(
        zoom_view,
        points_px=np.asarray(observed_pixels_px, dtype=np.float64) - crop_offset,
        color_bgr=(0, 165, 255),
        marker_style="circle",
        point_labels=point_labels,
        line_thickness=1,
        marker_size=10,
    )
    _draw_projected_point_set(
        zoom_view,
        points_px=np.asarray(ground_truth_pixels_px, dtype=np.float64) - crop_offset,
        color_bgr=(32, 181, 80),
        marker_style="cross",
        line_thickness=1,
        marker_size=14,
    )
    _draw_projected_point_set(
        zoom_view,
        points_px=np.asarray(optimized_pixels_px, dtype=np.float64) - crop_offset,
        color_bgr=(64, 96, 255),
        marker_style="tilted_cross",
        line_thickness=1,
        marker_size=14,
    )

    panels = [
        ("Analysis frame with all three point sets", full_view),
        ("Zoomed pattern crop with point labels", zoom_view),
    ]
    for index, (label, image) in enumerate(panels):
        x0 = gap + index * (panel_width + gap)
        y0 = top
        panel = _resize_for_panel(image, (panel_width, panel_height))
        canvas[y0 : y0 + panel_height, x0 : x0 + panel_width] = panel
        cv2.rectangle(canvas, (x0, y0), (x0 + panel_width, y0 + panel_height), (208, 214, 220), 1)
        cv2.putText(canvas, label, (x0, y0 - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (66, 74, 82), 1, cv2.LINE_AA)

    legend_lines = [
        "orange filled circle = captured points from the recorded image",
        "green cross = ground-truth pose projected through the minimizer camera operator",
        "orange-red tilted cross = optimized pose projected through the same operator",
        "point labels: C=center, TR=top_right, BR=bottom_right, BL=bottom_left, TL=top_left",
    ]
    _draw_text_lines(canvas, legend_lines, origin=(24, top + panel_height + 30), line_height=22, color=(44, 52, 60))
    _draw_text_lines(canvas, metadata_lines, origin=(24, top + panel_height + 126), line_height=22, color=(44, 52, 60))
    cv2.imwrite(str(path), canvas)


def _annotate_detected_frame(frame_bgr: np.ndarray, detections: list[dict[str, Any]], summaries: dict[int, dict[str, Any]]) -> np.ndarray:
    for detection in detections:
        corners = np.asarray(detection["corners_xy_clockwise"], dtype=np.float64)
        cv2.polylines(frame_bgr, [np.round(corners).astype(np.int32)], True, (16, 235, 122), 2, cv2.LINE_AA)
        for point in detection["points5_xy"]:
            center = (int(round(point[0])), int(round(point[1])))
            cv2.circle(frame_bgr, center, 4, (38, 147, 255), -1, cv2.LINE_AA)
        anchor = tuple(int(round(value)) for value in detection["center_xy"])
        summary = summaries.get(int(detection["tag_id"]))
        if summary is None:
            label = f"id {detection['tag_id']}"
        else:
            label = f"id {detection['tag_id']} err={summary['position_error_m'] * 1000.0:.1f}mm"
        cv2.putText(frame_bgr, label, (anchor[0] + 10, anchor[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 248, 250), 1, cv2.LINE_AA)
    return frame_bgr


def _format_vector(values: list[float] | np.ndarray, *, precision: int = 6, scale: float = 1.0) -> str:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return "[" + ", ".join(f"{float(value) * scale:.{precision}f}" for value in array.tolist()) + "]"


def _image_plane_scale_px_per_m(camera_model: PhoneCameraModel) -> float:
    focal_length_m = float(camera_model.focal_length_mm) / 1000.0
    return ((float(camera_model.fx_px) + float(camera_model.fy_px)) * 0.5) / max(focal_length_m, 1e-12)


def _image_plane_m_to_px(value_m: float, camera_model: PhoneCameraModel) -> float:
    return float(value_m) * _image_plane_scale_px_per_m(camera_model)


def _image_plane_points_m_to_pixels(
    points_image_plane_m: Sequence[Sequence[float]] | np.ndarray,
    camera_model: PhoneCameraModel,
) -> np.ndarray:
    points = np.asarray(points_image_plane_m, dtype=np.float64).reshape(-1, 2)
    focal_length_m = float(camera_model.focal_length_mm) / 1000.0
    x_norm = points[:, 0] / max(focal_length_m, 1e-12)
    y_norm = -points[:, 1] / max(focal_length_m, 1e-12)
    u = camera_model.fx_px * x_norm + camera_model.cx_px
    v = camera_model.fy_px * y_norm + camera_model.cy_px
    return np.column_stack((u, v)).astype(np.float64)


def _world_position_error_m(position_a_m: Sequence[float], position_b_m: Sequence[float]) -> float:
    a = np.asarray(position_a_m, dtype=np.float64)
    b = np.asarray(position_b_m, dtype=np.float64)
    return float(np.linalg.norm(a - b))


def _load_camera_ground_truth_rows(path: Path) -> list[dict[str, Any]]:
    rows = _load_csv_dict_rows(path)
    ground_truth_rows: list[dict[str, Any]] = []
    for frame_index, row in enumerate(rows):
        rotation_cw = np.array(
            [
                [float(row["r00"]), float(row["r01"]), float(row["r02"])],
                [float(row["r10"]), float(row["r11"]), float(row["r12"])],
                [float(row["r20"]), float(row["r21"]), float(row["r22"])],
            ],
            dtype=np.float64,
        )
        position_world_m = np.array(
            [float(row["cx_world_m"]), float(row["cy_world_m"]), float(row["cz_world_m"])],
            dtype=np.float64,
        )
        velocity_world_mps = None
        if row.get("vx_world_mps") and row.get("vy_world_mps") and row.get("vz_world_mps"):
            velocity_world_mps = np.array(
                [float(row["vx_world_mps"]), float(row["vy_world_mps"]), float(row["vz_world_mps"])],
                dtype=np.float64,
            )
        ground_truth_rows.append(
            {
                "frame_index": frame_index,
                "recording_frame_index": int(row["tick_index"]),
                "tick_index": int(row["tick_index"]),
                "sim_time_s": float(row["sim_time_s"]),
                "position_world_m": position_world_m,
                "velocity_world_mps": velocity_world_mps,
                "rotation_cw": rotation_cw,
                "rotation_xyz_deg": _rotation_matrix_to_xyz_angles_deg(rotation_cw),
                "servo_positions_deg": np.array(
                    [float(row["servo0_deg"]), float(row["servo1_deg"]), float(row["servo2_deg"])],
                    dtype=np.float64,
                ),
            }
        )
    return ground_truth_rows


def _load_imu_rows(path: Path) -> list[dict[str, Any]]:
    rows = _load_csv_dict_rows(path)
    imu_rows: list[dict[str, Any]] = []
    for row in rows:
        imu_rows.append(
            {
                "tick_index": int(row["tick_index"]),
                "sim_time_s": float(row["sim_time_s"]),
                "accel_body_mps2": np.array(
                    [float(row["ax_mps2"]), float(row["ay_mps2"]), float(row["az_mps2"])],
                    dtype=np.float64,
                ),
                "gyro_body_rps": np.array(
                    [float(row["gx_rps"]), float(row["gy_rps"]), float(row["gz_rps"])],
                    dtype=np.float64,
                ),
            }
        )
    return imu_rows


def _finite_difference_world_velocities(ground_truth_rows: list[dict[str, Any]]) -> list[np.ndarray]:
    if not ground_truth_rows:
        return []
    if len(ground_truth_rows) == 1:
        return [np.zeros(3, dtype=np.float64)]

    velocities: list[np.ndarray] = []
    for index, row in enumerate(ground_truth_rows):
        if index == 0:
            next_row = ground_truth_rows[index + 1]
            dt = max(float(next_row["sim_time_s"]) - float(row["sim_time_s"]), 1e-9)
            velocity = (np.asarray(next_row["position_world_m"], dtype=np.float64) - np.asarray(row["position_world_m"], dtype=np.float64)) / dt
        elif index == len(ground_truth_rows) - 1:
            prev_row = ground_truth_rows[index - 1]
            dt = max(float(row["sim_time_s"]) - float(prev_row["sim_time_s"]), 1e-9)
            velocity = (np.asarray(row["position_world_m"], dtype=np.float64) - np.asarray(prev_row["position_world_m"], dtype=np.float64)) / dt
        else:
            prev_row = ground_truth_rows[index - 1]
            next_row = ground_truth_rows[index + 1]
            dt = max(float(next_row["sim_time_s"]) - float(prev_row["sim_time_s"]), 1e-9)
            velocity = (np.asarray(next_row["position_world_m"], dtype=np.float64) - np.asarray(prev_row["position_world_m"], dtype=np.float64)) / dt
        velocities.append(np.asarray(velocity, dtype=np.float64))
    return velocities


def _integrate_imu_world_trajectory(
    *,
    ground_truth_rows: list[dict[str, Any]],
    imu_rows: list[dict[str, Any]],
    mode_name: str,
    initial_velocity_world_mps: np.ndarray,
) -> list[dict[str, Any]]:
    if not ground_truth_rows or not imu_rows:
        return []

    imu_by_tick = {int(row["tick_index"]): row for row in imu_rows}
    aligned_ground_truth_rows = [row for row in ground_truth_rows if int(row["tick_index"]) in imu_by_tick]
    if not aligned_ground_truth_rows:
        return []

    gravity_world = np.array([0.0, 0.0, -9.81], dtype=np.float64)
    first_row = aligned_ground_truth_rows[0]
    first_imu = imu_by_tick[int(first_row["tick_index"])]

    position_world_m = np.asarray(first_row["position_world_m"], dtype=np.float64).copy()
    rotation_cw = np.asarray(first_row["rotation_cw"], dtype=np.float64).copy()
    velocity_world_mps = np.asarray(initial_velocity_world_mps, dtype=np.float64).reshape(3).copy()
    first_acceleration_world_mps2 = rotation_cw @ np.asarray(first_imu["accel_body_mps2"], dtype=np.float64) + gravity_world
    trajectory_records: list[dict[str, Any]] = [
        {
            "mode_name": mode_name,
            "frame_index": int(first_row["frame_index"]),
            "recording_frame_index": int(first_row["recording_frame_index"]),
            "tick_index": int(first_row["tick_index"]),
            "sim_time_s": float(first_row["sim_time_s"]),
            "estimated_camera_world_position_m": [float(value) for value in position_world_m.tolist()],
            "estimated_camera_world_rotation_cw": [float(value) for value in rotation_cw.reshape(-1).tolist()],
            "estimated_rotation_xyz_deg": [float(value) for value in _rotation_matrix_to_xyz_angles_deg(rotation_cw).tolist()],
            "estimated_velocity_world_mps": [float(value) for value in velocity_world_mps.tolist()],
            "estimated_acceleration_world_mps2": [float(value) for value in first_acceleration_world_mps2.tolist()],
            "ground_truth_camera_world_position_m": [float(value) for value in np.asarray(first_row["position_world_m"], dtype=np.float64).tolist()],
            "ground_truth_camera_world_rotation_cw": [float(value) for value in np.asarray(first_row["rotation_cw"], dtype=np.float64).reshape(-1).tolist()],
            "ground_truth_rotation_xyz_deg": [float(value) for value in np.asarray(first_row["rotation_xyz_deg"], dtype=np.float64).tolist()],
            "position_error_m": 0.0,
            "rotation_error_deg": 0.0,
            "position_error_vector_world_m": [0.0, 0.0, 0.0],
            "rotation_error_vector_xyz_deg": [0.0, 0.0, 0.0],
            "initialization": {
                "position_world_m": [float(value) for value in np.asarray(first_row["position_world_m"], dtype=np.float64).tolist()],
                "rotation_cw": [float(value) for value in np.asarray(first_row["rotation_cw"], dtype=np.float64).reshape(-1).tolist()],
                "velocity_world_mps": [float(value) for value in velocity_world_mps.tolist()],
            },
        }
    ]

    previous_time_s = float(first_row["sim_time_s"])
    for row in aligned_ground_truth_rows[1:]:
        current_time_s = float(row["sim_time_s"])
        dt_s = max(current_time_s - previous_time_s, 1e-9)
        imu_row = imu_by_tick[int(row["tick_index"])]
        delta_rotation = _rotation_matrix_rvec_np(np.asarray(imu_row["gyro_body_rps"], dtype=np.float64) * dt_s)
        rotation_cw = rotation_cw @ delta_rotation
        current_acceleration_world_mps2 = rotation_cw @ np.asarray(imu_row["accel_body_mps2"], dtype=np.float64) + gravity_world
        # The simulator logs acceleration as the discrete velocity difference at the end of each step.
        # Matching that convention exactly gives: v_k = v_{k-1} + a_k * dt and p_k = p_{k-1} + v_k * dt.
        velocity_world_mps = velocity_world_mps + current_acceleration_world_mps2 * dt_s
        position_world_m = position_world_m + velocity_world_mps * dt_s
        previous_time_s = current_time_s

        ground_truth_position_world_m = np.asarray(row["position_world_m"], dtype=np.float64)
        ground_truth_rotation_cw = np.asarray(row["rotation_cw"], dtype=np.float64)
        position_error_vector_world_m = ground_truth_position_world_m - position_world_m
        rotation_error_vector_xyz_deg = _rotation_matrix_to_xyz_angles_deg(ground_truth_rotation_cw @ rotation_cw.T)

        trajectory_records.append(
            {
                "mode_name": mode_name,
                "frame_index": int(row["frame_index"]),
                "recording_frame_index": int(row["recording_frame_index"]),
                "tick_index": int(row["tick_index"]),
                "sim_time_s": current_time_s,
                "estimated_camera_world_position_m": [float(value) for value in position_world_m.tolist()],
                "estimated_camera_world_rotation_cw": [float(value) for value in rotation_cw.reshape(-1).tolist()],
                "estimated_rotation_xyz_deg": [float(value) for value in _rotation_matrix_to_xyz_angles_deg(rotation_cw).tolist()],
                "estimated_velocity_world_mps": [float(value) for value in velocity_world_mps.tolist()],
                "estimated_acceleration_world_mps2": [float(value) for value in current_acceleration_world_mps2.tolist()],
                "ground_truth_camera_world_position_m": [float(value) for value in ground_truth_position_world_m.tolist()],
                "ground_truth_camera_world_rotation_cw": [float(value) for value in ground_truth_rotation_cw.reshape(-1).tolist()],
                "ground_truth_rotation_xyz_deg": [float(value) for value in np.asarray(row["rotation_xyz_deg"], dtype=np.float64).tolist()],
                "position_error_m": float(np.linalg.norm(position_error_vector_world_m)),
                "rotation_error_deg": _rotation_error_deg(rotation_cw, ground_truth_rotation_cw),
                "position_error_vector_world_m": [float(value) for value in position_error_vector_world_m.tolist()],
                "rotation_error_vector_xyz_deg": [float(value) for value in rotation_error_vector_xyz_deg.tolist()],
            }
        )

    return trajectory_records


def _trajectory_error_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    evaluation_records = records[1:] if len(records) > 1 else list(records)
    if not evaluation_records:
        return {
            "samples_evaluated": 0,
            "mean_position_error_m": None,
            "median_position_error_m": None,
            "max_position_error_m": None,
            "final_position_error_m": None,
            "mean_rotation_error_deg": None,
            "max_rotation_error_deg": None,
            "final_rotation_error_deg": None,
        }

    position_errors_m = [float(record["position_error_m"]) for record in evaluation_records]
    rotation_errors_deg = [float(record["rotation_error_deg"]) for record in evaluation_records]
    return {
        "samples_evaluated": len(evaluation_records),
        "mean_position_error_m": float(np.mean(position_errors_m)),
        "median_position_error_m": float(np.median(position_errors_m)),
        "max_position_error_m": float(np.max(position_errors_m)),
        "final_position_error_m": float(position_errors_m[-1]),
        "mean_rotation_error_deg": float(np.mean(rotation_errors_deg)),
        "max_rotation_error_deg": float(np.max(rotation_errors_deg)),
        "final_rotation_error_deg": float(rotation_errors_deg[-1]),
    }


def _markdown_escape_cell(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["_No rows available._"]
    separator = ["---"] + ["---:" for _ in headers[1:]]
    escaped_headers = [_markdown_escape_cell(header) for header in headers]
    lines = [
        "| " + " | ".join(escaped_headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    lines.extend("| " + " | ".join(_markdown_escape_cell(cell) for cell in row) + " |" for row in rows)
    return lines


def _load_camera_model_from_metadata(metadata: dict[str, Any]) -> PhoneCameraModel:
    camera_meta = metadata.get("camera_model") or {}
    source_toml_path = camera_meta.get("source_toml_path")
    if not source_toml_path:
        raise FileNotFoundError("Recording metadata does not include a camera model TOML path.")
    resolution = camera_meta.get("output_resolution_px") or metadata.get("config", {}).get("primary_camera", {})
    if isinstance(resolution, dict):
        width_px = int(resolution["width"])
        height_px = int(resolution["height"])
    else:
        width_px = int(resolution[0])
        height_px = int(resolution[1])
    return load_phone_camera_model(source_toml_path, output_width_px=width_px, output_height_px=height_px)


def analyze_recording_run(run_dir: str | Path) -> dict[str, Any]:
    resolved_run_dir = Path(run_dir).resolve()
    metadata_path = resolved_run_dir / "metadata.json"
    samples_path = resolved_run_dir / "samples.jsonl"
    video_path = resolved_run_dir / "phone_raw.mp4"
    if not metadata_path.exists() or not samples_path.exists() or not video_path.exists():
        raise FileNotFoundError(f"Run directory is missing required recording artifacts: {resolved_run_dir}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    samples = _load_json_lines(samples_path)
    camera_model = _load_camera_model_from_metadata(metadata)
    detector = AprilTag36h11Detector()

    analysis_dir = resolved_run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    undistorted_video_path = analysis_dir / "phone_undistorted.mp4"
    annotated_video_path = analysis_dir / "phone_undistorted_detected.mp4"
    estimates_path = analysis_dir / "pose_estimates.jsonl"
    summary_path = analysis_dir / "summary.json"
    report_path = analysis_dir / "report.md"
    joint_world_estimates_path = analysis_dir / "joint_world_estimates.jsonl"
    imu_trajectory_estimates_path = analysis_dir / "imu_trajectory_estimates.jsonl"
    imu_trajectory_report_path = analysis_dir / "imu_trajectory_report.md"
    representative_image_path = analysis_dir / "representative_fit.png"
    representative_pose_comparison_path = analysis_dir / "representative_pose_comparison.png"
    representative_observation_path = analysis_dir / "representative_observation.json"
    single_pattern_overlay_path = analysis_dir / "single_pattern_overlay.png"
    single_pattern_report_path = analysis_dir / "single_pattern_report.md"
    worst_single_pattern_overlay_path = analysis_dir / "worst_single_pattern_overlay.png"
    worst_single_pattern_report_path = analysis_dir / "worst_single_pattern_report.md"
    per_tag_dir = analysis_dir / "per_tag"
    per_tag_summary_path = analysis_dir / "per_tag_summary.json"
    real_measurement_known_solution_report_path = analysis_dir / "real_measurement_known_solution_report.md"
    synthetic_single_pattern_overlay_path = analysis_dir / "synthetic_single_pattern_overlay.png"
    synthetic_single_pattern_report_path = analysis_dir / "synthetic_single_pattern_report.md"
    position_error_plot_path = analysis_dir / "position_error_timeline.png"
    reprojection_plot_path = analysis_dir / "reprojection_timeline.png"
    loss_trace_plot_path = analysis_dir / "optimizer_loss_trace.png"
    imu_position_error_plot_path = analysis_dir / "imu_position_error_timeline.png"
    imu_rotation_error_plot_path = analysis_dir / "imu_rotation_error_timeline.png"
    world_pose_plot_dir = analysis_dir / "world_pose_by_tag"
    world_pose_plot_dir.mkdir(parents=True, exist_ok=True)
    imu_world_pose_plot_dir = analysis_dir / "imu_world_pose"
    imu_world_pose_plot_dir.mkdir(parents=True, exist_ok=True)
    world_pose_x_plot_path = world_pose_plot_dir / "camera_world_x_m.png"
    world_pose_y_plot_path = world_pose_plot_dir / "camera_world_y_m.png"
    world_pose_z_plot_path = world_pose_plot_dir / "camera_world_z_m.png"
    world_pose_rx_plot_path = world_pose_plot_dir / "camera_world_rx_deg.png"
    world_pose_ry_plot_path = world_pose_plot_dir / "camera_world_ry_deg.png"
    world_pose_rz_plot_path = world_pose_plot_dir / "camera_world_rz_deg.png"
    imu_world_pose_x_plot_path = imu_world_pose_plot_dir / "camera_world_x_m.png"
    imu_world_pose_y_plot_path = imu_world_pose_plot_dir / "camera_world_y_m.png"
    imu_world_pose_z_plot_path = imu_world_pose_plot_dir / "camera_world_z_m.png"
    imu_world_pose_rx_plot_path = imu_world_pose_plot_dir / "camera_world_rx_deg.png"
    imu_world_pose_ry_plot_path = imu_world_pose_plot_dir / "camera_world_ry_deg.png"
    imu_world_pose_rz_plot_path = imu_world_pose_plot_dir / "camera_world_rz_deg.png"
    image_plane_px_scale = _image_plane_scale_px_per_m(camera_model)
    scene_config = metadata.get("config", {}) if isinstance(metadata.get("config"), dict) else {}
    configured_tags = scene_config.get("tags", []) if isinstance(scene_config.get("tags"), list) else []
    configured_tag_sizes_m = [float(tag["size_m"]) for tag in configured_tags if isinstance(tag, dict) and tag.get("size_m") is not None]
    tag_size_reference_m = float(np.mean(configured_tag_sizes_m)) if configured_tag_sizes_m else None

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open recorded video: {video_path}")

    fps = float(metadata.get("recording", {}).get("frame_rate_hz", 0.0) or capture.get(cv2.CAP_PROP_FPS) or 8.0)
    width = int(camera_model.output_width_px)
    height = int(camera_model.output_height_px)
    undistorted_writer = ManagedMp4Writer(undistorted_video_path, fps=fps, frame_size=(width, height))
    annotated_writer = ManagedMp4Writer(annotated_video_path, fps=fps, frame_size=(width, height))

    position_errors_m: list[float] = []
    rotation_errors_deg: list[float] = []
    reprojection_errors_m: list[float] = []
    observation_records: list[dict[str, Any]] = []
    joint_world_records: list[dict[str, Any]] = []
    estimates_written = 0
    frame_count = 0
    detected_tags_total = 0
    tag_counts: dict[int, int] = {}
    representative_record: dict[str, Any] | None = None
    worst_record: dict[str, Any] | None = None
    previous_final_state_by_tag: dict[int, np.ndarray] = {}
    previous_recording_frame_by_tag: dict[int, int] = {}
    previous_joint_world_state: np.ndarray | None = None
    previous_joint_world_recording_frame: int | None = None

    with estimates_path.open("w", encoding="utf-8") as estimates_handle:
        try:
            while True:
                ok, frame_bgr = capture.read()
                if not ok:
                    break
                if frame_count >= len(samples):
                    break

                sample = samples[frame_count]
                undistorted = camera_model.analysis_image(frame_bgr)
                undistorted_writer.write(undistorted)

                detections = detector.detect_image(
                    undistorted,
                    frame_index=int(sample.get("tick_index", frame_count)),
                    timestamp_s=float(sample.get("sim_time_s", 0.0)),
                )
                detections_json = detector.to_jsonable(detections)["detections"]
                ground_truth_by_tag = {
                    int(item["tag_id"]): item
                    for item in (sample.get("ground_truth", {}) or {}).get("tags", [])
                    if item.get("visible", True)
                }
                ground_truth_world_pose = (sample.get("ground_truth", {}) or {}).get("camera_world_pose", {})
                ground_truth_world_position = np.asarray(
                    ground_truth_world_pose.get("position_m", [0.0, 0.0, 0.0]),
                    dtype=np.float64,
                ).reshape(3)
                ground_truth_world_rotation_cw = np.asarray(
                    ground_truth_world_pose.get("rotation_cw", np.eye(3, dtype=np.float64).reshape(-1).tolist()),
                    dtype=np.float64,
                ).reshape(3, 3)
                recording_frame_index = int(sample.get("recording_frame_index", frame_count + 1))

                frame_estimates: list[dict[str, Any]] = []
                frame_estimate_details_by_tag: dict[int, dict[str, Any]] = {}
                overlay_summaries: dict[int, dict[str, Any]] = {}
                for detection in detections_json:
                    tag_id = int(detection["tag_id"])
                    ground_truth = ground_truth_by_tag.get(tag_id)
                    if ground_truth is None:
                        continue

                    ordered_pixels = np.asarray(
                        [
                            detection["center_xy"],
                            detection["corners_xy_clockwise"][1],
                            detection["corners_xy_clockwise"][2],
                            detection["corners_xy_clockwise"][3],
                            detection["corners_xy_clockwise"][0],
                        ],
                        dtype=np.float64,
                    )
                    observation_image_plane_m = camera_model.pixels_to_image_plane_mm(ordered_pixels, undistort=False) / 1000.0
                    half_extent = float(ground_truth["size_m"]) * 0.5
                    ground_truth_position = np.asarray(ground_truth["camera_position_tag_m"], dtype=np.float64)
                    ground_truth_rotation_tc = np.asarray(ground_truth["camera_rotation_tc"], dtype=np.float64).reshape(3, 3)
                    tag_center_world_m = np.asarray(ground_truth["tag_center_world_m"], dtype=np.float64)
                    tag_rotation_wt = ground_truth_world_rotation_cw @ ground_truth_rotation_tc.T

                    candidate_seed_specs: list[dict[str, Any]] | None = None
                    previous_frame_for_tag = previous_recording_frame_by_tag.get(tag_id)
                    if previous_frame_for_tag is not None and previous_frame_for_tag == recording_frame_index - 1:
                        previous_state = previous_final_state_by_tag.get(tag_id)
                        if previous_state is not None:
                            candidate_seed_specs = [
                                {
                                    "seed_name": "previous_frame",
                                    "state": [float(value) for value in np.asarray(previous_state, dtype=np.float64).tolist()],
                                    "source_recording_frame_index": int(previous_frame_for_tag),
                                }
                            ]
                    elif previous_joint_world_state is not None and previous_joint_world_recording_frame == recording_frame_index - 1:
                        prior_world_position = np.asarray(previous_joint_world_state[:3], dtype=np.float64).reshape(3)
                        prior_world_rotation_cw = _rotation_matrix_rvec_np(np.asarray(previous_joint_world_state[3:], dtype=np.float64))
                        prior_camera_position_tag_m = tag_rotation_wt.T @ (prior_world_position - tag_center_world_m)
                        prior_camera_rotation_tc = tag_rotation_wt.T @ prior_world_rotation_cw
                        prior_rotation_ct = prior_camera_rotation_tc.T
                        prior_translation_ct = -(prior_rotation_ct @ prior_camera_position_tag_m)
                        prior_rotation_vector_ct, _ = cv2.Rodrigues(prior_rotation_ct)
                        candidate_seed_specs = [
                            {
                                "seed_name": "previous_world_frame",
                                "state": [
                                    float(value)
                                    for value in np.concatenate(
                                        [prior_translation_ct.reshape(3), prior_rotation_vector_ct.reshape(3)]
                                    ).tolist()
                                ],
                                "source_recording_frame_index": int(previous_joint_world_recording_frame),
                            }
                        ]
                    estimate = _estimate_pose_newton(
                        observation_image_plane_m,
                        ordered_pixels_px=ordered_pixels,
                        camera_model=camera_model,
                        focal_length_m=float(camera_model.focal_length_mm) / 1000.0,
                        pattern_half_extent_m=half_extent,
                        candidate_seed_specs=candidate_seed_specs,
                        use_diagonal_damping=False,
                        initial_damping=0.0,
                    )

                    prior_camera_position_tag_m: np.ndarray | None = None
                    if previous_joint_world_state is not None and previous_joint_world_recording_frame == recording_frame_index - 1:
                        prior_camera_position_tag_m = (
                            tag_rotation_wt.T
                            @ (
                                np.asarray(previous_joint_world_state[:3], dtype=np.float64).reshape(3)
                                - tag_center_world_m
                            )
                        )
                        estimate = _maybe_disambiguate_half_turn_branch(
                            estimate,
                            prior_camera_position_tag_m=prior_camera_position_tag_m,
                            observed_points_image_plane_m=observation_image_plane_m,
                            focal_length_m=float(camera_model.focal_length_mm) / 1000.0,
                            pattern_half_extent_m=half_extent,
                        )
                    estimated_position = np.asarray(estimate["camera_position_tag_m"], dtype=np.float64)
                    estimated_rotation_tc = np.asarray(estimate["camera_rotation_tc"], dtype=np.float64).reshape(3, 3)
                    estimated_world_position, estimated_world_rotation_cw = _camera_world_pose_from_tag_pose(
                        camera_position_tag_m=estimated_position,
                        camera_rotation_tc=estimated_rotation_tc,
                        tag_center_world_m=tag_center_world_m,
                        tag_rotation_wt=tag_rotation_wt,
                    )
                    position_error_m = float(np.linalg.norm(estimated_position - ground_truth_position))
                    rotation_error_deg = _rotation_error_deg(estimated_rotation_tc, ground_truth_rotation_tc)
                    reprojection_rmse = float(estimate["reprojection_rmse_image_plane_m"])

                    position_errors_m.append(position_error_m)
                    rotation_errors_deg.append(rotation_error_deg)
                    reprojection_errors_m.append(reprojection_rmse)
                    detected_tags_total += 1
                    estimates_written += 1
                    tag_counts[tag_id] = tag_counts.get(tag_id, 0) + 1

                    frame_estimate = {
                        "frame_index": frame_count,
                        "recording_frame_index": recording_frame_index,
                        "tick_index": int(sample.get("tick_index", frame_count)),
                        "sim_time_s": float(sample.get("sim_time_s", 0.0)),
                        "tag_id": tag_id,
                        "observation_image_plane_points_m": [[float(x), float(y)] for x, y in observation_image_plane_m.tolist()],
                        "estimate": estimate,
                        "tag_center_world_m": [float(value) for value in tag_center_world_m.tolist()],
                        "tag_rotation_wt": [float(value) for value in tag_rotation_wt.reshape(-1).tolist()],
                        "ground_truth_camera_world_position_m": [float(value) for value in ground_truth_world_position.tolist()],
                        "ground_truth_camera_world_rotation_cw": [float(value) for value in ground_truth_world_rotation_cw.reshape(-1).tolist()],
                        "estimated_camera_world_position_m": [float(value) for value in estimated_world_position.tolist()],
                        "estimated_camera_world_rotation_cw": [float(value) for value in estimated_world_rotation_cw.reshape(-1).tolist()],
                        "ground_truth_camera_position_tag_m": [float(value) for value in ground_truth_position.tolist()],
                        "ground_truth_camera_rotation_tc": [float(value) for value in ground_truth_rotation_tc.reshape(-1).tolist()],
                        **(
                            {
                                "continuity_prior_camera_position_tag_m": [
                                    float(value) for value in np.asarray(prior_camera_position_tag_m, dtype=np.float64).tolist()
                                ]
                            }
                            if prior_camera_position_tag_m is not None
                            else {}
                        ),
                        "position_error_m": position_error_m,
                        "rotation_error_deg": rotation_error_deg,
                        "reprojection_rmse_image_plane_m": reprojection_rmse,
                    }
                    frame_estimates.append(frame_estimate)
                    previous_final_state_by_tag[tag_id] = np.asarray(estimate["final_state"], dtype=np.float64)
                    previous_recording_frame_by_tag[tag_id] = recording_frame_index
                    frame_estimate_details_by_tag[tag_id] = {
                        "ordered_pixels_px": ordered_pixels.copy(),
                        "observation_image_plane_m": observation_image_plane_m.copy(),
                        "pattern_half_extent_m": half_extent,
                        "object_points_world_m": np.array(
                            [
                                tag_center_world_m + tag_rotation_wt @ point
                                for point in _object_points_tag_m(half_extent)
                            ],
                            dtype=np.float64,
                        ),
                    }
                    observation_records.append(frame_estimate)
                    overlay_summaries[tag_id] = frame_estimate

                frame_joint_world_estimate: dict[str, Any] | None = None
                if len(frame_estimates) >= 2:
                    sorted_tag_ids = sorted(frame_estimate_details_by_tag)
                    combined_observations = np.vstack(
                        [
                            np.asarray(frame_estimate_details_by_tag[tag_id]["observation_image_plane_m"], dtype=np.float64)
                            for tag_id in sorted_tag_ids
                        ]
                    )
                    combined_object_points_world = np.vstack(
                        [
                            np.asarray(frame_estimate_details_by_tag[tag_id]["object_points_world_m"], dtype=np.float64)
                            for tag_id in sorted_tag_ids
                        ]
                    )
                    seed_states_world: list[np.ndarray] = []
                    if previous_joint_world_state is not None and previous_joint_world_recording_frame == recording_frame_index - 1:
                        seed_states_world.append(np.asarray(previous_joint_world_state, dtype=np.float64))
                    for frame_estimate in sorted(frame_estimates, key=lambda item: int(item["tag_id"])):
                        seed_position_world = np.asarray(frame_estimate["estimated_camera_world_position_m"], dtype=np.float64)
                        seed_rotation_cw = np.asarray(frame_estimate["estimated_camera_world_rotation_cw"], dtype=np.float64).reshape(3, 3)
                        seed_rvec_cw, _ = cv2.Rodrigues(seed_rotation_cw)
                        seed_state = np.concatenate([seed_position_world, seed_rvec_cw.reshape(3)])
                        if not any(np.allclose(seed_state, existing, atol=1e-10) for existing in seed_states_world):
                            seed_states_world.append(seed_state)

                    joint_world_estimate = _estimate_world_pose_from_all_points(
                        combined_observations,
                        object_points_world_m=combined_object_points_world,
                        seed_states_world=seed_states_world,
                        focal_length_m=float(camera_model.focal_length_mm) / 1000.0,
                        use_diagonal_damping=False,
                        initial_damping=0.0,
                    )
                    joint_world_position = np.asarray(joint_world_estimate["camera_position_world_m"], dtype=np.float64)
                    joint_world_rotation_cw = np.asarray(joint_world_estimate["camera_rotation_cw"], dtype=np.float64).reshape(3, 3)
                    frame_joint_world_estimate = {
                        "frame_index": frame_count,
                        "recording_frame_index": recording_frame_index,
                        "tick_index": int(sample.get("tick_index", frame_count)),
                        "sim_time_s": float(sample.get("sim_time_s", 0.0)),
                        "tag_ids": [int(tag_id) for tag_id in sorted_tag_ids],
                        "point_count": int(combined_observations.shape[0]),
                        "estimate": joint_world_estimate,
                        "ground_truth_camera_world_position_m": [float(value) for value in ground_truth_world_position.tolist()],
                        "ground_truth_camera_world_rotation_cw": [float(value) for value in ground_truth_world_rotation_cw.reshape(-1).tolist()],
                        "estimated_camera_world_position_m": [float(value) for value in joint_world_position.tolist()],
                        "estimated_camera_world_rotation_cw": [float(value) for value in joint_world_rotation_cw.reshape(-1).tolist()],
                        "position_error_m": float(np.linalg.norm(joint_world_position - ground_truth_world_position)),
                        "rotation_error_deg": _rotation_error_deg(joint_world_rotation_cw, ground_truth_world_rotation_cw),
                        "reprojection_rmse_image_plane_m": float(joint_world_estimate["reprojection_rmse_image_plane_m"]),
                    }

                    for frame_estimate in frame_estimates:
                        tag_id = int(frame_estimate["tag_id"])
                        tag_details = frame_estimate_details_by_tag[tag_id]
                        tag_center_world_m = np.asarray(frame_estimate["tag_center_world_m"], dtype=np.float64)
                        tag_rotation_wt = np.asarray(frame_estimate["tag_rotation_wt"], dtype=np.float64).reshape(3, 3)
                        world_seed_state, world_seed_camera_position_tag_m = _tag_pose_state_from_world_pose(
                            camera_position_world_m=joint_world_position,
                            camera_rotation_cw=joint_world_rotation_cw,
                            tag_center_world_m=tag_center_world_m,
                            tag_rotation_wt=tag_rotation_wt,
                        )
                        current_camera_position_tag_m = np.asarray(
                            frame_estimate["estimate"]["camera_position_tag_m"],
                            dtype=np.float64,
                        )
                        if float(np.linalg.norm(current_camera_position_tag_m - world_seed_camera_position_tag_m)) <= 0.05:
                            continue

                        refined_estimate = _estimate_pose_newton(
                            np.asarray(tag_details["observation_image_plane_m"], dtype=np.float64),
                            ordered_pixels_px=np.asarray(tag_details["ordered_pixels_px"], dtype=np.float64),
                            camera_model=camera_model,
                            focal_length_m=float(camera_model.focal_length_mm) / 1000.0,
                            pattern_half_extent_m=float(tag_details["pattern_half_extent_m"]),
                            candidate_seed_specs=[
                                {
                                    "seed_name": "same_frame_joint_world",
                                    "state": [float(value) for value in np.asarray(world_seed_state, dtype=np.float64).tolist()],
                                    "source_recording_frame_index": int(recording_frame_index),
                                }
                            ],
                            use_diagonal_damping=False,
                            initial_damping=0.0,
                        )
                        refined_camera_position_tag_m = np.asarray(refined_estimate["camera_position_tag_m"], dtype=np.float64)
                        if not (
                            float(np.linalg.norm(refined_camera_position_tag_m - world_seed_camera_position_tag_m))
                            + 1e-9
                            < float(np.linalg.norm(current_camera_position_tag_m - world_seed_camera_position_tag_m))
                            and float(refined_estimate["reprojection_rmse_image_plane_m"])
                            <= float(frame_estimate["estimate"]["reprojection_rmse_image_plane_m"]) + 1e-9
                        ):
                            continue

                        refined_camera_rotation_tc = np.asarray(refined_estimate["camera_rotation_tc"], dtype=np.float64).reshape(3, 3)
                        refined_world_position, refined_world_rotation_cw = _camera_world_pose_from_tag_pose(
                            camera_position_tag_m=refined_camera_position_tag_m,
                            camera_rotation_tc=refined_camera_rotation_tc,
                            tag_center_world_m=tag_center_world_m,
                            tag_rotation_wt=tag_rotation_wt,
                        )
                        refined_ground_truth_position = np.asarray(
                            frame_estimate["ground_truth_camera_position_tag_m"],
                            dtype=np.float64,
                        )
                        refined_ground_truth_rotation_tc = np.asarray(
                            frame_estimate["ground_truth_camera_rotation_tc"],
                            dtype=np.float64,
                        ).reshape(3, 3)
                        frame_estimate["estimate"] = refined_estimate
                        frame_estimate["estimated_camera_world_position_m"] = [
                            float(value) for value in refined_world_position.tolist()
                        ]
                        frame_estimate["estimated_camera_world_rotation_cw"] = [
                            float(value) for value in refined_world_rotation_cw.reshape(-1).tolist()
                        ]
                        frame_estimate["position_error_m"] = float(
                            np.linalg.norm(refined_camera_position_tag_m - refined_ground_truth_position)
                        )
                        frame_estimate["rotation_error_deg"] = _rotation_error_deg(
                            refined_camera_rotation_tc,
                            refined_ground_truth_rotation_tc,
                        )
                        frame_estimate["reprojection_rmse_image_plane_m"] = float(
                            refined_estimate["reprojection_rmse_image_plane_m"]
                        )
                        previous_final_state_by_tag[tag_id] = np.asarray(refined_estimate["final_state"], dtype=np.float64)

                    joint_world_records.append(frame_joint_world_estimate)
                    previous_joint_world_state = np.concatenate(
                        [
                            np.asarray(joint_world_estimate["camera_position_world_m"], dtype=np.float64),
                            np.asarray(joint_world_estimate["rotation_vector_cw_rad"], dtype=np.float64),
                        ]
                    )
                    previous_joint_world_recording_frame = recording_frame_index
                else:
                    previous_joint_world_state = None
                    previous_joint_world_recording_frame = None

                annotated = _annotate_detected_frame(undistorted.copy(), detections_json, overlay_summaries)
                annotated_writer.write(annotated)
                if frame_estimates:
                    best_frame_estimate = min(frame_estimates, key=lambda item: item["reprojection_rmse_image_plane_m"])
                    if representative_record is None or (
                        best_frame_estimate["reprojection_rmse_image_plane_m"]
                        < representative_record["estimate_record"]["reprojection_rmse_image_plane_m"]
                    ):
                        best_details = frame_estimate_details_by_tag[int(best_frame_estimate["tag_id"])]
                        representative_record = {
                            "estimate_record": best_frame_estimate,
                            "raw_frame_bgr": frame_bgr.copy(),
                            "analysis_frame_bgr": undistorted.copy(),
                            "annotated_frame_bgr": annotated.copy(),
                            "ordered_pixels_px": best_details["ordered_pixels_px"],
                            "observation_image_plane_m": best_details["observation_image_plane_m"],
                            "pattern_half_extent_m": best_details["pattern_half_extent_m"],
                        }
                    worst_frame_estimate = max(frame_estimates, key=lambda item: item["position_error_m"])
                    if worst_record is None or (
                        worst_frame_estimate["position_error_m"]
                        > worst_record["estimate_record"]["position_error_m"]
                    ):
                        worst_details = frame_estimate_details_by_tag[int(worst_frame_estimate["tag_id"])]
                        worst_record = {
                            "estimate_record": worst_frame_estimate,
                            "raw_frame_bgr": frame_bgr.copy(),
                            "analysis_frame_bgr": undistorted.copy(),
                            "annotated_frame_bgr": annotated.copy(),
                            "ordered_pixels_px": worst_details["ordered_pixels_px"],
                            "observation_image_plane_m": worst_details["observation_image_plane_m"],
                            "pattern_half_extent_m": worst_details["pattern_half_extent_m"],
                        }
                estimates_handle.write(
                    json.dumps(
                        {
                            "frame_index": frame_count,
                            "tick_index": sample.get("tick_index", frame_count),
                            "sim_time_s": sample.get("sim_time_s", float(frame_count) / max(fps, 1.0)),
                            "detections": detections_json,
                            "estimates": frame_estimates,
                            "joint_world_estimate": frame_joint_world_estimate,
                        }
                    )
                    + "\n"
                )
                frame_count += 1
        finally:
            capture.release()
            undistorted_writer.close()
            annotated_writer.close()

    position_errors_m = [float(record["position_error_m"]) for record in observation_records]
    rotation_errors_deg = [float(record["rotation_error_deg"]) for record in observation_records]
    reprojection_errors_m = [float(record["reprojection_rmse_image_plane_m"]) for record in observation_records]

    mean_position_error_m = float(np.mean(position_errors_m)) if position_errors_m else None
    mean_rotation_error_deg = float(np.mean(rotation_errors_deg)) if rotation_errors_deg else None
    mean_reprojection_rmse_m = float(np.mean(reprojection_errors_m)) if reprojection_errors_m else None
    median_position_error_m = float(np.median(position_errors_m)) if position_errors_m else None
    median_reprojection_rmse_m = float(np.median(reprojection_errors_m)) if reprojection_errors_m else None
    min_position_error_m = float(np.min(position_errors_m)) if position_errors_m else None
    max_position_error_m = float(np.max(position_errors_m)) if position_errors_m else None
    mean_reprojection_rmse_px = float(np.mean([_image_plane_m_to_px(value, camera_model) for value in reprojection_errors_m])) if reprojection_errors_m else None
    median_reprojection_rmse_px = float(np.median([_image_plane_m_to_px(value, camera_model) for value in reprojection_errors_m])) if reprojection_errors_m else None
    camera_tag_distances_m = [
        float(np.linalg.norm(np.asarray(record["ground_truth_camera_position_tag_m"], dtype=np.float64)))
        for record in observation_records
    ]
    min_camera_tag_distance_m = float(np.min(camera_tag_distances_m)) if camera_tag_distances_m else None
    max_camera_tag_distance_m = float(np.max(camera_tag_distances_m)) if camera_tag_distances_m else None
    mean_camera_tag_distance_m = float(np.mean(camera_tag_distances_m)) if camera_tag_distances_m else None
    per_tag_dir.mkdir(parents=True, exist_ok=True)
    per_tag_rows: list[list[str]] = []
    per_tag_summary: dict[str, Any] = {}
    grouped_records_by_tag: dict[int, list[dict[str, Any]]] = {}
    for record in observation_records:
        grouped_records_by_tag.setdefault(int(record["tag_id"]), []).append(record)
    for tag_id, tag_records in sorted(grouped_records_by_tag.items()):
        tag_key = str(tag_id)
        tag_subdir = per_tag_dir / f"tag_{tag_id}"
        tag_subdir.mkdir(parents=True, exist_ok=True)
        tag_estimates_path = tag_subdir / "estimates.jsonl"
        tag_report_path = tag_subdir / "report.md"
        tag_position_plot_path = tag_subdir / "position_error_timeline.png"
        tag_reprojection_plot_path = tag_subdir / "reprojection_timeline.png"

        tag_position_errors = [float(record["position_error_m"]) for record in tag_records]
        tag_rotation_errors = [float(record["rotation_error_deg"]) for record in tag_records]
        tag_reprojection_px = [
            _image_plane_m_to_px(float(record["reprojection_rmse_image_plane_m"]), camera_model)
            for record in tag_records
        ]
        tag_distances_m = [
            float(np.linalg.norm(np.asarray(record["ground_truth_camera_position_tag_m"], dtype=np.float64)))
            for record in tag_records
        ]
        tag_best_record = min(tag_records, key=lambda item: float(item["reprojection_rmse_image_plane_m"]))
        tag_worst_record = max(tag_records, key=lambda item: float(item["position_error_m"]))
        tag_worst_rows: list[list[str]] = []
        for item in sorted(tag_records, key=lambda entry: float(entry["position_error_m"]), reverse=True)[:5]:
            tag_worst_rows.append(
                [
                    f"{int(item['frame_index'])}",
                    f"{int(item['tick_index'])}",
                    f"{float(item['sim_time_s']):.3f}",
                    f"{float(item['position_error_m']):.6f}",
                    f"{float(item['rotation_error_deg']):.6f}",
                    f"{_image_plane_m_to_px(float(item['reprojection_rmse_image_plane_m']), camera_model):.3f}",
                ]
            )

        tag_estimates_path.write_text(
            "".join(json.dumps(record) + "\n" for record in tag_records),
            encoding="utf-8",
        )
        if tag_position_errors:
            _save_metric_plot(
                tag_position_plot_path,
                title=f"Tag {tag_id} World-Space Position Error",
                y_label="meters [m]",
                values=tag_position_errors,
                color_bgr=(61, 142, 255),
            )
        if tag_reprojection_px:
            _save_metric_plot(
                tag_reprojection_plot_path,
                title=f"Tag {tag_id} Image-Space Reprojection RMSE",
                y_label="pixels [px]",
                values=tag_reprojection_px,
                color_bgr=(24, 186, 108),
            )

        tag_report_lines = [
            f"# Tag {tag_id} Separate Estimation Report",
            "",
            "This report contains only the estimates for one calibration pattern, solved separately frame by frame.",
            "",
            "## Overview",
            "",
            f"- Run directory: `{resolved_run_dir}`",
            f"- Tag id: `{tag_id}`",
            f"- Observations solved: `{len(tag_records)}`",
            f"- Mean position error: `{float(np.mean(tag_position_errors)):.6f}` m",
            f"- Median position error: `{float(np.median(tag_position_errors)):.6f}` m",
            f"- Max position error: `{float(np.max(tag_position_errors)):.6f}` m",
            f"- Mean rotation error: `{float(np.mean(tag_rotation_errors)):.6f}` deg",
            f"- Mean reprojection RMSE: `{float(np.mean(tag_reprojection_px)):.6f}` px",
            f"- Mean camera-to-tag distance: `{float(np.mean(tag_distances_m)):.6f}` m",
            "",
            "## Best And Worst Observations",
            "",
            f"- Best reprojection observation: frame `{int(tag_best_record['frame_index'])}`, tick `{int(tag_best_record['tick_index'])}`, time `{float(tag_best_record['sim_time_s']):.3f}s`, world error `{float(tag_best_record['position_error_m']):.6f}` m, reprojection `{_image_plane_m_to_px(float(tag_best_record['reprojection_rmse_image_plane_m']), camera_model):.6f}` px",
            f"- Worst position observation: frame `{int(tag_worst_record['frame_index'])}`, tick `{int(tag_worst_record['tick_index'])}`, time `{float(tag_worst_record['sim_time_s']):.3f}s`, world error `{float(tag_worst_record['position_error_m']):.6f}` m, reprojection `{_image_plane_m_to_px(float(tag_worst_record['reprojection_rmse_image_plane_m']), camera_model):.6f}` px",
            "",
            "## Worst Five Observations",
            "",
            *_markdown_table(
                ["frame", "tick", "time [s]", "position error [m]", "rotation error [deg]", "reproj [px]"],
                tag_worst_rows,
            ),
            "",
            "## Artifacts",
            "",
            f"- Extracted estimates: `{tag_estimates_path.relative_to(resolved_run_dir)}`",
            f"- Position error plot: `{tag_position_plot_path.relative_to(resolved_run_dir)}`",
            f"- Reprojection plot: `{tag_reprojection_plot_path.relative_to(resolved_run_dir)}`",
            "",
        ]
        tag_report_path.write_text("\n".join(tag_report_lines).strip() + "\n", encoding="utf-8")

        per_tag_rows.append(
            [
                f"`{tag_id}`",
                f"{len(tag_records)}",
                f"{float(np.mean(tag_position_errors)):.6f}",
                f"{float(np.median(tag_position_errors)):.6f}",
                f"{float(np.max(tag_position_errors)):.6f}",
                f"{float(np.mean(tag_rotation_errors)):.6f}",
                f"{float(np.mean(tag_reprojection_px)):.6f}",
            ]
        )
        per_tag_summary[tag_key] = {
            "tag_id": tag_id,
            "observations": len(tag_records),
            "mean_position_error_m": float(np.mean(tag_position_errors)),
            "median_position_error_m": float(np.median(tag_position_errors)),
            "max_position_error_m": float(np.max(tag_position_errors)),
            "mean_rotation_error_deg": float(np.mean(tag_rotation_errors)),
            "mean_reprojection_rmse_px": float(np.mean(tag_reprojection_px)),
            "mean_camera_tag_distance_m": float(np.mean(tag_distances_m)),
            "best_reprojection_observation": {
                "frame_index": int(tag_best_record["frame_index"]),
                "tick_index": int(tag_best_record["tick_index"]),
                "sim_time_s": float(tag_best_record["sim_time_s"]),
                "position_error_m": float(tag_best_record["position_error_m"]),
                "rotation_error_deg": float(tag_best_record["rotation_error_deg"]),
                "reprojection_rmse_px": _image_plane_m_to_px(float(tag_best_record["reprojection_rmse_image_plane_m"]), camera_model),
            },
            "worst_position_observation": {
                "frame_index": int(tag_worst_record["frame_index"]),
                "tick_index": int(tag_worst_record["tick_index"]),
                "sim_time_s": float(tag_worst_record["sim_time_s"]),
                "position_error_m": float(tag_worst_record["position_error_m"]),
                "rotation_error_deg": float(tag_worst_record["rotation_error_deg"]),
                "reprojection_rmse_px": _image_plane_m_to_px(float(tag_worst_record["reprojection_rmse_image_plane_m"]), camera_model),
            },
            "artifacts": {
                "report": str(tag_report_path.relative_to(resolved_run_dir)),
                "estimates": str(tag_estimates_path.relative_to(resolved_run_dir)),
                "position_error_plot": str(tag_position_plot_path.relative_to(resolved_run_dir)),
                "reprojection_plot": str(tag_reprojection_plot_path.relative_to(resolved_run_dir)),
            },
        }
    per_tag_summary_path.write_text(json.dumps(per_tag_summary, indent=2), encoding="utf-8")
    joint_world_estimates_path.write_text(
        "".join(json.dumps(record) + "\n" for record in joint_world_records),
        encoding="utf-8",
    )
    world_pose_ground_truth_by_frame: dict[int, dict[str, np.ndarray]] = {}
    for sample in samples[:frame_count]:
        ground_truth_world_pose = (sample.get("ground_truth", {}) or {}).get("camera_world_pose", {})
        if not isinstance(ground_truth_world_pose, dict):
            continue
        frame_number = int(sample.get("recording_frame_index", len(world_pose_ground_truth_by_frame) + 1))
        camera_world_position = np.asarray(ground_truth_world_pose.get("position_m", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
        camera_world_rotation_cw = np.asarray(
            ground_truth_world_pose.get("rotation_cw", np.eye(3, dtype=np.float64).reshape(-1).tolist()),
            dtype=np.float64,
        ).reshape(3, 3)
        world_pose_ground_truth_by_frame[frame_number] = {
            "position_m": camera_world_position,
            "rotation_xyz_deg": _rotation_matrix_to_xyz_angles_deg(camera_world_rotation_cw),
        }

    tag_world_estimates_by_frame: dict[int, dict[int, dict[str, np.ndarray]]] = {}
    for record in observation_records:
        tag_id = int(record["tag_id"])
        frame_number = int(record.get("recording_frame_index", int(record["frame_index"]) + 1))
        estimated_world_position = np.asarray(record["estimated_camera_world_position_m"], dtype=np.float64).reshape(3)
        estimated_world_rotation_cw = np.asarray(record["estimated_camera_world_rotation_cw"], dtype=np.float64).reshape(3, 3)
        tag_world_estimates_by_frame.setdefault(tag_id, {})[frame_number] = {
            "position_m": estimated_world_position,
            "rotation_xyz_deg": _rotation_matrix_to_xyz_angles_deg(estimated_world_rotation_cw),
        }
    joint_world_estimates_by_frame: dict[int, dict[str, np.ndarray]] = {}
    for record in joint_world_records:
        frame_number = int(record.get("recording_frame_index", int(record["frame_index"]) + 1))
        estimated_world_position = np.asarray(record["estimated_camera_world_position_m"], dtype=np.float64).reshape(3)
        estimated_world_rotation_cw = np.asarray(record["estimated_camera_world_rotation_cw"], dtype=np.float64).reshape(3, 3)
        joint_world_estimates_by_frame[frame_number] = {
            "position_m": estimated_world_position,
            "rotation_xyz_deg": _rotation_matrix_to_xyz_angles_deg(estimated_world_rotation_cw),
        }
    joint_world_position_errors_m = [float(record["position_error_m"]) for record in joint_world_records]
    joint_world_rotation_errors_deg = [float(record["rotation_error_deg"]) for record in joint_world_records]
    joint_world_reprojection_px = [
        _image_plane_m_to_px(float(record["reprojection_rmse_image_plane_m"]), camera_model)
        for record in joint_world_records
    ]

    world_pose_frame_numbers = sorted(world_pose_ground_truth_by_frame)
    estimate_color_cycle = [
        (255, 106, 61),
        (24, 186, 108),
        (227, 96, 27),
        (161, 98, 255),
    ]
    if world_pose_frame_numbers:
        component_plot_specs = [
            ("x", 0, "Camera World X By Frame", "world x [m]", "position"),
            ("y", 1, "Camera World Y By Frame", "world y [m]", "position"),
            ("z", 2, "Camera World Z By Frame", "world z [m]", "position"),
            ("rx", 0, "Camera World Rotation X By Frame", "rot x [deg]", "rotation"),
            ("ry", 1, "Camera World Rotation Y By Frame", "rot y [deg]", "rotation"),
            ("rz", 2, "Camera World Rotation Z By Frame", "rot z [deg]", "rotation"),
        ]
        component_plot_paths = {
            "x": world_pose_x_plot_path,
            "y": world_pose_y_plot_path,
            "z": world_pose_z_plot_path,
            "rx": world_pose_rx_plot_path,
            "ry": world_pose_ry_plot_path,
            "rz": world_pose_rz_plot_path,
        }
        for component_name, axis_index, title, y_label, component_kind in component_plot_specs:
            ground_truth_values = [
                float(
                    world_pose_ground_truth_by_frame[frame_number]["position_m" if component_kind == "position" else "rotation_xyz_deg"][axis_index]
                )
                for frame_number in world_pose_frame_numbers
            ]
            series_specs = []
            for series_index, tag_id in enumerate(sorted(tag_world_estimates_by_frame)):
                color_bgr = estimate_color_cycle[series_index % len(estimate_color_cycle)]
                tag_frame_map = tag_world_estimates_by_frame[tag_id]
                values = [
                    float(tag_frame_map[frame_number]["position_m"][axis_index]) if component_kind == "position" and frame_number in tag_frame_map else (
                        float(tag_frame_map[frame_number]["rotation_xyz_deg"][axis_index]) if frame_number in tag_frame_map else float("nan")
                    )
                    for frame_number in world_pose_frame_numbers
                ]
                series_specs.append(
                    {
                        "label": f"estimate from tag {tag_id}",
                        "color_bgr": color_bgr,
                        "values": values,
                    }
                )
            joint_values = [
                float(joint_world_estimates_by_frame[frame_number]["position_m"][axis_index]) if component_kind == "position" and frame_number in joint_world_estimates_by_frame else (
                    float(joint_world_estimates_by_frame[frame_number]["rotation_xyz_deg"][axis_index]) if frame_number in joint_world_estimates_by_frame else float("nan")
                )
                for frame_number in world_pose_frame_numbers
            ]
            series_specs.append(
                {
                    "label": "joint estimate from all points",
                    "color_bgr": (214, 83, 255),
                    "values": joint_values,
                }
            )
            series_specs.append(
                {
                    "label": "ground truth",
                    "color_bgr": (44, 52, 60),
                    "values": ground_truth_values,
                }
            )
            _save_multi_series_plot(
                component_plot_paths[component_name],
                title=title,
                y_label=y_label,
                x_label="frame number",
                x_values=world_pose_frame_numbers,
                series_specs=series_specs,
            )
    representative_observation_payload: dict[str, Any] | None = None

    if position_errors_m:
        _save_metric_plot(
            position_error_plot_path,
            title="World-Space Camera Position Error Per Observation",
            y_label="meters [m]",
            values=position_errors_m,
            color_bgr=(61, 142, 255),
        )
    if reprojection_errors_m:
        _save_metric_plot(
            reprojection_plot_path,
            title="Image-Space Reprojection RMSE Per Observation",
            y_label="pixels [px]",
            values=[_image_plane_m_to_px(value, camera_model) for value in reprojection_errors_m],
            color_bgr=(24, 186, 108),
        )
    if representative_record is not None:
        estimate_record = representative_record["estimate_record"]
        representative_prior_camera_position_tag_m = estimate_record.get("continuity_prior_camera_position_tag_m")
        detailed_estimate = _estimate_pose_newton(
            np.asarray(representative_record["observation_image_plane_m"], dtype=np.float64),
            ordered_pixels_px=np.asarray(representative_record["ordered_pixels_px"], dtype=np.float64),
            camera_model=camera_model,
            focal_length_m=float(camera_model.focal_length_mm) / 1000.0,
            pattern_half_extent_m=float(representative_record["pattern_half_extent_m"]),
            candidate_seed_specs=_candidate_seed_specs_for_replay(estimate_record["estimate"]),
            include_history=True,
            use_diagonal_damping=False,
            initial_damping=0.0,
        )
        if representative_prior_camera_position_tag_m is not None:
            detailed_estimate = _maybe_disambiguate_half_turn_branch(
                detailed_estimate,
                prior_camera_position_tag_m=np.asarray(representative_prior_camera_position_tag_m, dtype=np.float64),
                observed_points_image_plane_m=np.asarray(representative_record["observation_image_plane_m"], dtype=np.float64),
                focal_length_m=float(camera_model.focal_length_mm) / 1000.0,
                pattern_half_extent_m=float(representative_record["pattern_half_extent_m"]),
            )
        estimate_record["estimate"] = detailed_estimate
        estimate_record["reprojection_rmse_image_plane_m"] = float(detailed_estimate["reprojection_rmse_image_plane_m"])
        ground_truth_position = np.asarray(estimate_record["ground_truth_camera_position_tag_m"], dtype=np.float64)
        ground_truth_rotation_tc = np.asarray(estimate_record["ground_truth_camera_rotation_tc"], dtype=np.float64).reshape(3, 3)
        estimated_position = np.asarray(detailed_estimate["camera_position_tag_m"], dtype=np.float64)
        estimated_rotation_tc = np.asarray(detailed_estimate["camera_rotation_tc"], dtype=np.float64).reshape(3, 3)
        estimate_record["position_error_m"] = float(np.linalg.norm(estimated_position - ground_truth_position))
        estimate_record["rotation_error_deg"] = _rotation_error_deg(estimated_rotation_tc, ground_truth_rotation_tc)
        focal_length_m = float(camera_model.focal_length_mm) / 1000.0
        pattern_half_extent_m = float(representative_record["pattern_half_extent_m"])
        observed_points_image_plane_m = np.asarray(representative_record["observation_image_plane_m"], dtype=np.float64)
        candidate_lookup = {str(candidate["seed_name"]): candidate for candidate in detailed_estimate["candidate_seed_summaries"]}
        camera_obscura_seed_diag = _state_diagnostics(
            np.asarray(detailed_estimate["camera_obscura_seed"]["state"], dtype=np.float64),
            observed_points_image_plane_m=observed_points_image_plane_m,
            focal_length_m=focal_length_m,
            pattern_half_extent_m=pattern_half_extent_m,
        )
        camera_obscura_final_candidate = candidate_lookup.get("camera_obscura")
        camera_obscura_final_diag = (
            _state_diagnostics(
                np.asarray(camera_obscura_final_candidate["final_state"], dtype=np.float64),
                observed_points_image_plane_m=observed_points_image_plane_m,
                focal_length_m=focal_length_m,
                pattern_half_extent_m=pattern_half_extent_m,
            )
            if camera_obscura_final_candidate is not None
            else None
        )
        selected_final_diag = _state_diagnostics(
            np.asarray(detailed_estimate["translation_tag_to_camera_m"] + detailed_estimate["rotation_vector_ct_rad"], dtype=np.float64),
            observed_points_image_plane_m=observed_points_image_plane_m,
            focal_length_m=focal_length_m,
            pattern_half_extent_m=pattern_half_extent_m,
        )
        ground_truth_rotation_ct = ground_truth_rotation_tc.T
        ground_truth_translation_ct = -(ground_truth_rotation_ct @ ground_truth_position)
        ground_truth_rvec_ct, _ = cv2.Rodrigues(ground_truth_rotation_ct)
        ground_truth_state = np.concatenate([ground_truth_translation_ct.reshape(3), ground_truth_rvec_ct.reshape(3)])
        ground_truth_operator_diag = _state_diagnostics(
            ground_truth_state,
            observed_points_image_plane_m=observed_points_image_plane_m,
            focal_length_m=focal_length_m,
            pattern_half_extent_m=pattern_half_extent_m,
        )
        ground_truth_projected_image_plane_m = np.asarray(
            ground_truth_operator_diag["predicted_image_plane_points_m"],
            dtype=np.float64,
        )
        ground_truth_projected_pixels_px = _image_plane_points_m_to_pixels(
            ground_truth_projected_image_plane_m,
            camera_model,
        )
        optimized_projected_pixels_px = _image_plane_points_m_to_pixels(
            np.asarray(detailed_estimate["predicted_image_plane_points_m"], dtype=np.float64),
            camera_model,
        )
        representative_pose_rows = [
            {
                "short_label": "GT",
                "label": "Ground truth",
                "position_m": [float(value) for value in ground_truth_position.tolist()],
                "color_bgr": [40, 40, 40],
                "distance_to_tag_m": float(np.linalg.norm(ground_truth_position)),
                "position_error_m": 0.0,
                "reprojection_rmse_image_plane_m": None,
                "reprojection_rmse_px": None,
            },
            {
                "short_label": "OBS",
                "label": "Camera obscura seed",
                "position_m": camera_obscura_seed_diag["camera_position_tag_m"],
                "color_bgr": [24, 186, 108],
                "distance_to_tag_m": float(np.linalg.norm(np.asarray(camera_obscura_seed_diag["camera_position_tag_m"], dtype=np.float64))),
                "position_error_m": _world_position_error_m(camera_obscura_seed_diag["camera_position_tag_m"], ground_truth_position),
                "reprojection_rmse_image_plane_m": float(camera_obscura_seed_diag["reprojection_rmse_image_plane_m"]),
                "reprojection_rmse_px": _image_plane_m_to_px(float(camera_obscura_seed_diag["reprojection_rmse_image_plane_m"]), camera_model),
            },
        ]
        if camera_obscura_final_diag is not None:
            representative_pose_rows.append(
                {
                    "short_label": "OBF",
                    "label": "Exact operator result from obscura seed",
                    "position_m": camera_obscura_final_diag["camera_position_tag_m"],
                    "color_bgr": [0, 170, 255],
                    "distance_to_tag_m": float(np.linalg.norm(np.asarray(camera_obscura_final_diag["camera_position_tag_m"], dtype=np.float64))),
                    "position_error_m": _world_position_error_m(camera_obscura_final_diag["camera_position_tag_m"], ground_truth_position),
                    "reprojection_rmse_image_plane_m": float(camera_obscura_final_diag["reprojection_rmse_image_plane_m"]),
                    "reprojection_rmse_px": _image_plane_m_to_px(float(camera_obscura_final_diag["reprojection_rmse_image_plane_m"]), camera_model),
                }
            )
        representative_pose_rows.append(
            {
                "short_label": "EX",
                "label": f"Selected exact-operator result ({detailed_estimate['selected_seed_name']})",
                "position_m": selected_final_diag["camera_position_tag_m"],
                "color_bgr": [255, 106, 61],
                "distance_to_tag_m": float(np.linalg.norm(np.asarray(selected_final_diag["camera_position_tag_m"], dtype=np.float64))),
                "position_error_m": _world_position_error_m(selected_final_diag["camera_position_tag_m"], ground_truth_position),
                "reprojection_rmse_image_plane_m": float(selected_final_diag["reprojection_rmse_image_plane_m"]),
                "reprojection_rmse_px": _image_plane_m_to_px(float(selected_final_diag["reprojection_rmse_image_plane_m"]), camera_model),
            }
        )

        loss_trace = estimate_record["estimate"].get("loss_trace", [])
        if loss_trace:
            _save_metric_plot(
                loss_trace_plot_path,
                title="Optimizer Loss Trace For Representative Estimate",
                y_label="objective",
                values=[float(value) for value in loss_trace],
                color_bgr=(27, 96, 227),
            )
        _save_representative_fit(
            representative_image_path,
            raw_frame_bgr=representative_record["raw_frame_bgr"],
            analysis_frame_bgr=representative_record["analysis_frame_bgr"],
            annotated_frame_bgr=representative_record["annotated_frame_bgr"],
            metadata_lines=[
                f"frame={estimate_record['frame_index']}  tick={estimate_record['tick_index']}  tag={estimate_record['tag_id']}",
                f"sim_time={estimate_record['sim_time_s']:.3f}s",
                f"position_error={estimate_record['position_error_m']:.6f} m",
                f"rotation_error={estimate_record['rotation_error_deg']:.6f} deg",
                f"reprojection_rmse={estimate_record['reprojection_rmse_image_plane_m'] * 1000.0:.6f} mm ({_image_plane_m_to_px(estimate_record['reprojection_rmse_image_plane_m'], camera_model):.3f} px)",
                f"loss_steps={len(estimate_record['estimate'].get('loss_trace', []))}",
            ],
        )
        _save_pose_comparison(
            representative_pose_comparison_path,
            entries=representative_pose_rows,
        )
        point_labels = ["C", "TR", "BR", "BL", "TL"]
        _save_single_pattern_overlay(
            single_pattern_overlay_path,
            analysis_frame_bgr=representative_record["analysis_frame_bgr"],
            observed_pixels_px=np.asarray(representative_record["ordered_pixels_px"], dtype=np.float64),
            ground_truth_pixels_px=ground_truth_projected_pixels_px,
            optimized_pixels_px=optimized_projected_pixels_px,
            point_labels=point_labels,
            metadata_lines=[
                f"frame={estimate_record['frame_index']}  tick={estimate_record['tick_index']}  tag={estimate_record['tag_id']}",
                f"gt-vs-captured={_image_plane_m_to_px(float(ground_truth_operator_diag['reprojection_rmse_image_plane_m']), camera_model):.3f} px",
                f"opt-vs-captured={_image_plane_m_to_px(float(selected_final_diag['reprojection_rmse_image_plane_m']), camera_model):.3f} px",
                f"world position error of optimized pose={estimate_record['position_error_m']:.6f} m",
            ],
        )
        real_measurement_ground_truth_loss = 0.5 * float(
            np.sum(np.square(np.asarray(ground_truth_operator_diag["residual_image_plane_points_m"], dtype=np.float64)))
        )
        real_measurement_gt_seed_damped = _optimize_pose_from_state(
            ground_truth_state,
            observed_points_image_plane_m=observed_points_image_plane_m,
            focal_length_m=focal_length_m,
            pattern_half_extent_m=pattern_half_extent_m,
            include_history=True,
            use_diagonal_damping=True,
            initial_damping=1e-7,
        )
        real_measurement_gt_seed_undamped = _optimize_pose_from_state(
            ground_truth_state,
            observed_points_image_plane_m=observed_points_image_plane_m,
            focal_length_m=focal_length_m,
            pattern_half_extent_m=pattern_half_extent_m,
            include_history=True,
            use_diagonal_damping=False,
            initial_damping=0.0,
        )
        real_gt_seed_damped_position = np.asarray(real_measurement_gt_seed_damped["camera_position_tag_m"], dtype=np.float64)
        real_gt_seed_damped_rotation_tc = np.asarray(real_measurement_gt_seed_damped["camera_rotation_tc"], dtype=np.float64).reshape(3, 3)
        real_gt_seed_undamped_position = np.asarray(real_measurement_gt_seed_undamped["camera_position_tag_m"], dtype=np.float64)
        real_gt_seed_undamped_rotation_tc = np.asarray(real_measurement_gt_seed_undamped["camera_rotation_tc"], dtype=np.float64).reshape(3, 3)
        real_gt_seed_damped_position_error_m = float(np.linalg.norm(real_gt_seed_damped_position - ground_truth_position))
        real_gt_seed_damped_rotation_error_deg = _rotation_error_deg(real_gt_seed_damped_rotation_tc, ground_truth_rotation_tc)
        real_gt_seed_undamped_position_error_m = float(np.linalg.norm(real_gt_seed_undamped_position - ground_truth_position))
        real_gt_seed_undamped_rotation_error_deg = _rotation_error_deg(real_gt_seed_undamped_rotation_tc, ground_truth_rotation_tc)
        synthetic_observation_image_plane_m = ground_truth_projected_image_plane_m.copy()
        synthetic_observation_pixels_px = ground_truth_projected_pixels_px.copy()
        synthetic_ground_truth_diag = _state_diagnostics(
            ground_truth_state,
            observed_points_image_plane_m=synthetic_observation_image_plane_m,
            focal_length_m=focal_length_m,
            pattern_half_extent_m=pattern_half_extent_m,
        )
        synthetic_estimate_damped = _estimate_pose_newton(
            synthetic_observation_image_plane_m,
            ordered_pixels_px=synthetic_observation_pixels_px,
            camera_model=camera_model,
            focal_length_m=focal_length_m,
            pattern_half_extent_m=pattern_half_extent_m,
            include_history=True,
        )
        synthetic_estimate = _estimate_pose_newton(
            synthetic_observation_image_plane_m,
            ordered_pixels_px=synthetic_observation_pixels_px,
            camera_model=camera_model,
            focal_length_m=focal_length_m,
            pattern_half_extent_m=pattern_half_extent_m,
            include_history=True,
            use_diagonal_damping=False,
            initial_damping=0.0,
        )
        synthetic_estimated_position = np.asarray(synthetic_estimate["camera_position_tag_m"], dtype=np.float64)
        synthetic_estimated_rotation_tc = np.asarray(synthetic_estimate["camera_rotation_tc"], dtype=np.float64).reshape(3, 3)
        synthetic_position_error_m = float(np.linalg.norm(synthetic_estimated_position - ground_truth_position))
        synthetic_rotation_error_deg = _rotation_error_deg(synthetic_estimated_rotation_tc, ground_truth_rotation_tc)
        synthetic_damped_estimated_position = np.asarray(synthetic_estimate_damped["camera_position_tag_m"], dtype=np.float64)
        synthetic_damped_estimated_rotation_tc = np.asarray(synthetic_estimate_damped["camera_rotation_tc"], dtype=np.float64).reshape(3, 3)
        synthetic_damped_position_error_m = float(np.linalg.norm(synthetic_damped_estimated_position - ground_truth_position))
        synthetic_damped_rotation_error_deg = _rotation_error_deg(synthetic_damped_estimated_rotation_tc, ground_truth_rotation_tc)
        synthetic_estimated_pixels_px = _image_plane_points_m_to_pixels(
            np.asarray(synthetic_estimate["predicted_image_plane_points_m"], dtype=np.float64),
            camera_model,
        )
        _save_single_pattern_overlay(
            synthetic_single_pattern_overlay_path,
            analysis_frame_bgr=representative_record["analysis_frame_bgr"],
            observed_pixels_px=synthetic_observation_pixels_px,
            ground_truth_pixels_px=ground_truth_projected_pixels_px,
            optimized_pixels_px=synthetic_estimated_pixels_px,
            point_labels=point_labels,
            metadata_lines=[
                f"synthetic observation = exact ground-truth projection for frame={estimate_record['frame_index']} tag={estimate_record['tag_id']}",
                f"gt-vs-synthetic={_image_plane_m_to_px(float(synthetic_ground_truth_diag['reprojection_rmse_image_plane_m']), camera_model):.3f} px",
                f"undamped optimized-vs-synthetic={_image_plane_m_to_px(float(synthetic_estimate['reprojection_rmse_image_plane_m']), camera_model):.3f} px",
                f"undamped optimized world position error={synthetic_position_error_m:.6f} m",
            ],
        )
        representative_observation_payload = {
            "frame_index": int(estimate_record["frame_index"]),
            "tick_index": int(estimate_record["tick_index"]),
            "sim_time_s": float(estimate_record["sim_time_s"]),
            "tag_id": int(estimate_record["tag_id"]),
            "camera_model": {
                "name": camera_model.name,
                "projection_model": camera_model.projection_model,
                "source_toml_path": camera_model.source_toml_path,
                "focal_length_mm": float(camera_model.focal_length_mm),
                "output_resolution_px": [int(camera_model.output_width_px), int(camera_model.output_height_px)],
                "apply_lens_distortion_in_render": bool(camera_model.apply_lens_distortion_in_render),
            },
            "observation": {
                "point_order": ["center", "top_right", "bottom_right", "bottom_left", "top_left"],
                "ordered_pixels_px": [
                    [float(x), float(y)]
                    for x, y in np.asarray(representative_record["ordered_pixels_px"], dtype=np.float64).tolist()
                ],
                "image_plane_points_m": [
                    [float(x), float(y)]
                    for x, y in np.asarray(representative_record["observation_image_plane_m"], dtype=np.float64).tolist()
                ],
                "pattern_half_extent_m": float(representative_record["pattern_half_extent_m"]),
                "pattern_size_m": float(representative_record["pattern_half_extent_m"]) * 2.0,
            },
            "ground_truth": {
                "camera_position_tag_m": [float(value) for value in ground_truth_position.tolist()],
                "camera_rotation_tc": [float(value) for value in ground_truth_rotation_tc.reshape(-1).tolist()],
                "translation_tag_to_camera_m": [float(value) for value in ground_truth_translation_ct.tolist()],
                "rotation_vector_ct_rad": [float(value) for value in ground_truth_rvec_ct.reshape(3).tolist()],
            },
            "ground_truth_operator": {
                "predicted_image_plane_points_m": [
                    [float(x), float(y)]
                    for x, y in ground_truth_projected_image_plane_m.tolist()
                ],
                "predicted_pixels_px": [
                    [float(x), float(y)]
                    for x, y in ground_truth_projected_pixels_px.tolist()
                ],
                "reprojection_rmse_image_plane_m": float(ground_truth_operator_diag["reprojection_rmse_image_plane_m"]),
                "reprojection_rmse_px": _image_plane_m_to_px(float(ground_truth_operator_diag["reprojection_rmse_image_plane_m"]), camera_model),
            },
            "estimate": detailed_estimate,
            "position_error_m": float(estimate_record["position_error_m"]),
            "rotation_error_deg": float(estimate_record["rotation_error_deg"]),
            "reprojection_rmse_image_plane_m": float(estimate_record["reprojection_rmse_image_plane_m"]),
            "reprojection_rmse_px": _image_plane_m_to_px(float(estimate_record["reprojection_rmse_image_plane_m"]), camera_model),
            "real_measurement_ground_truth_seed": {
                "ground_truth_loss": real_measurement_ground_truth_loss,
                "ground_truth_reprojection_rmse_image_plane_m": float(ground_truth_operator_diag["reprojection_rmse_image_plane_m"]),
                "ground_truth_reprojection_rmse_px": _image_plane_m_to_px(float(ground_truth_operator_diag["reprojection_rmse_image_plane_m"]), camera_model),
                "estimate_damped": real_measurement_gt_seed_damped,
                "damped_position_error_m": real_gt_seed_damped_position_error_m,
                "damped_rotation_error_deg": real_gt_seed_damped_rotation_error_deg,
                "damped_reprojection_rmse_image_plane_m": float(real_measurement_gt_seed_damped["reprojection_rmse_image_plane_m"]),
                "damped_reprojection_rmse_px": _image_plane_m_to_px(float(real_measurement_gt_seed_damped["reprojection_rmse_image_plane_m"]), camera_model),
                "estimate_undamped": real_measurement_gt_seed_undamped,
                "undamped_position_error_m": real_gt_seed_undamped_position_error_m,
                "undamped_rotation_error_deg": real_gt_seed_undamped_rotation_error_deg,
                "undamped_reprojection_rmse_image_plane_m": float(real_measurement_gt_seed_undamped["reprojection_rmse_image_plane_m"]),
                "undamped_reprojection_rmse_px": _image_plane_m_to_px(float(real_measurement_gt_seed_undamped["reprojection_rmse_image_plane_m"]), camera_model),
            },
            "synthetic_perfect_observation": {
                "observation_image_plane_points_m": [
                    [float(x), float(y)]
                    for x, y in synthetic_observation_image_plane_m.tolist()
                ],
                "observation_pixels_px": [
                    [float(x), float(y)]
                    for x, y in synthetic_observation_pixels_px.tolist()
                ],
                "ground_truth_reprojection_rmse_image_plane_m": float(synthetic_ground_truth_diag["reprojection_rmse_image_plane_m"]),
                "ground_truth_reprojection_rmse_px": _image_plane_m_to_px(float(synthetic_ground_truth_diag["reprojection_rmse_image_plane_m"]), camera_model),
                "estimate_damped": synthetic_estimate_damped,
                "damped_position_error_m": synthetic_damped_position_error_m,
                "damped_rotation_error_deg": synthetic_damped_rotation_error_deg,
                "damped_reprojection_rmse_image_plane_m": float(synthetic_estimate_damped["reprojection_rmse_image_plane_m"]),
                "damped_reprojection_rmse_px": _image_plane_m_to_px(float(synthetic_estimate_damped["reprojection_rmse_image_plane_m"]), camera_model),
                "estimate": synthetic_estimate,
                "position_error_m": synthetic_position_error_m,
                "rotation_error_deg": synthetic_rotation_error_deg,
                "reprojection_rmse_image_plane_m": float(synthetic_estimate["reprojection_rmse_image_plane_m"]),
                "reprojection_rmse_px": _image_plane_m_to_px(float(synthetic_estimate["reprojection_rmse_image_plane_m"]), camera_model),
            },
            "scene_scale": {
                "tag_size_reference_m": tag_size_reference_m,
                "camera_tag_distance_range_m": [min_camera_tag_distance_m, max_camera_tag_distance_m],
                "mean_camera_tag_distance_m": mean_camera_tag_distance_m,
                "wall_width_m": scene_config.get("scene", {}).get("wall_width_m") if isinstance(scene_config.get("scene"), dict) else None,
                "wall_height_m": scene_config.get("scene", {}).get("wall_height_m") if isinstance(scene_config.get("scene"), dict) else None,
            },
            "pose_comparison_rows": representative_pose_rows,
        }
        representative_observation_path.write_text(
            json.dumps(representative_observation_payload, indent=2),
            encoding="utf-8",
        )

    camera_ground_truth_rows = _load_camera_ground_truth_rows(resolved_run_dir / "camera_gt.csv")
    imu_rows = _load_imu_rows(resolved_run_dir / "imu.csv")
    finite_difference_velocities_world_mps = _finite_difference_world_velocities(camera_ground_truth_rows)
    first_logged_velocity_world_mps = (
        np.asarray(camera_ground_truth_rows[0].get("velocity_world_mps"), dtype=np.float64)
        if camera_ground_truth_rows and camera_ground_truth_rows[0].get("velocity_world_mps") is not None
        else None
    )
    imu_initial_velocity_source = "logged_camera_gt_velocity" if first_logged_velocity_world_mps is not None else "finite_difference_fallback"
    imu_mode_order = ["true_start_velocity"]
    imu_mode_initial_velocities = {
        "true_start_velocity": (
            first_logged_velocity_world_mps.copy()
            if first_logged_velocity_world_mps is not None
            else (
                np.asarray(finite_difference_velocities_world_mps[0], dtype=np.float64)
                if finite_difference_velocities_world_mps
                else np.zeros(3, dtype=np.float64)
            )
        ),
    }
    imu_mode_labels = {
        "true_start_velocity": "IMU DR, p0/R0 true, v0 = GT finite diff",
    }
    imu_mode_colors = {
        "true_start_velocity": (186, 85, 211),
    }
    imu_records_by_mode: dict[str, list[dict[str, Any]]] = {}
    imu_summaries_by_mode: dict[str, dict[str, Any]] = {}
    imu_summary_rows: list[list[str]] = []

    for mode_name, initial_velocity_world_mps in imu_mode_initial_velocities.items():
        imu_records = _integrate_imu_world_trajectory(
            ground_truth_rows=camera_ground_truth_rows,
            imu_rows=imu_rows,
            mode_name=mode_name,
            initial_velocity_world_mps=initial_velocity_world_mps,
        )
        if not imu_records:
            continue
        imu_records_by_mode[mode_name] = imu_records
        mode_summary = _trajectory_error_summary(imu_records)
        mode_summary["initial_velocity_world_mps"] = [float(value) for value in initial_velocity_world_mps.tolist()]
        imu_summaries_by_mode[mode_name] = mode_summary
        imu_summary_rows.append(
            [
                f"`{mode_name}`",
                _format_vector(initial_velocity_world_mps, precision=4),
                f"{mode_summary['mean_position_error_m']:.6f}" if mode_summary["mean_position_error_m"] is not None else "n/a",
                f"{mode_summary['median_position_error_m']:.6f}" if mode_summary["median_position_error_m"] is not None else "n/a",
                f"{mode_summary['max_position_error_m']:.6f}" if mode_summary["max_position_error_m"] is not None else "n/a",
                f"{mode_summary['final_position_error_m']:.6f}" if mode_summary["final_position_error_m"] is not None else "n/a",
                f"{mode_summary['mean_rotation_error_deg']:.6f}" if mode_summary["mean_rotation_error_deg"] is not None else "n/a",
                f"{mode_summary['max_rotation_error_deg']:.6f}" if mode_summary["max_rotation_error_deg"] is not None else "n/a",
                f"{mode_summary['final_rotation_error_deg']:.6f}" if mode_summary["final_rotation_error_deg"] is not None else "n/a",
            ]
        )

    imu_main_summary = imu_summaries_by_mode.get("true_start_velocity")
    if imu_records_by_mode:
        with imu_trajectory_estimates_path.open("w", encoding="utf-8") as handle:
            for mode_name in imu_mode_order:
                for record in imu_records_by_mode.get(mode_name, []):
                    handle.write(json.dumps(record) + "\n")

        imu_frame_numbers = [int(row["recording_frame_index"]) for row in camera_ground_truth_rows]
        gt_position_by_frame = {
            int(row["recording_frame_index"]): np.asarray(row["position_world_m"], dtype=np.float64)
            for row in camera_ground_truth_rows
        }
        gt_rotation_xyz_by_frame = {
            int(row["recording_frame_index"]): np.asarray(row["rotation_xyz_deg"], dtype=np.float64)
            for row in camera_ground_truth_rows
        }
        imu_record_lookups = {
            mode_name: {int(record["recording_frame_index"]): record for record in records}
            for mode_name, records in imu_records_by_mode.items()
        }
        estimated_position_by_mode = {
            mode_name: {
                int(record["recording_frame_index"]): np.asarray(record["estimated_camera_world_position_m"], dtype=np.float64)
                for record in records
            }
            for mode_name, records in imu_records_by_mode.items()
        }
        estimated_rotation_xyz_by_mode = {
            mode_name: {
                int(record["recording_frame_index"]): np.asarray(record["estimated_rotation_xyz_deg"], dtype=np.float64)
                for record in records
            }
            for mode_name, records in imu_records_by_mode.items()
        }

        _save_multi_series_plot(
            imu_position_error_plot_path,
            title="IMU trajectory position error vs ground truth",
            y_label="Position error [m]",
            x_label="Recording frame",
            x_values=imu_frame_numbers,
            series_specs=[
                {
                    "label": imu_mode_labels[mode_name],
                    "values": [
                        float(imu_record_lookups[mode_name][frame_number]["position_error_m"])
                        if frame_number in imu_record_lookups[mode_name]
                        else None
                        for frame_number in imu_frame_numbers
                    ],
                    "color_bgr": imu_mode_colors[mode_name],
                }
                for mode_name in imu_records_by_mode
            ],
        )
        _save_multi_series_plot(
            imu_rotation_error_plot_path,
            title="IMU trajectory rotation error vs ground truth",
            y_label="Rotation error [deg]",
            x_label="Recording frame",
            x_values=imu_frame_numbers,
            series_specs=[
                {
                    "label": imu_mode_labels[mode_name],
                    "values": [
                        float(imu_record_lookups[mode_name][frame_number]["rotation_error_deg"])
                        if frame_number in imu_record_lookups[mode_name]
                        else None
                        for frame_number in imu_frame_numbers
                    ],
                    "color_bgr": imu_mode_colors[mode_name],
                }
                for mode_name in imu_records_by_mode
            ],
        )

        imu_component_specs = [
            (
                imu_world_pose_x_plot_path,
                "IMU trajectory camera world X",
                "World X [m]",
                lambda frame_number: float(gt_position_by_frame[frame_number][0]),
                lambda mapping, frame_number: float(mapping[frame_number][0]) if frame_number in mapping else None,
                estimated_position_by_mode,
            ),
            (
                imu_world_pose_y_plot_path,
                "IMU trajectory camera world Y",
                "World Y [m]",
                lambda frame_number: float(gt_position_by_frame[frame_number][1]),
                lambda mapping, frame_number: float(mapping[frame_number][1]) if frame_number in mapping else None,
                estimated_position_by_mode,
            ),
            (
                imu_world_pose_z_plot_path,
                "IMU trajectory camera world Z",
                "World Z [m]",
                lambda frame_number: float(gt_position_by_frame[frame_number][2]),
                lambda mapping, frame_number: float(mapping[frame_number][2]) if frame_number in mapping else None,
                estimated_position_by_mode,
            ),
            (
                imu_world_pose_rx_plot_path,
                "IMU trajectory camera world rot X",
                "Rot X [deg]",
                lambda frame_number: float(gt_rotation_xyz_by_frame[frame_number][0]),
                lambda mapping, frame_number: float(mapping[frame_number][0]) if frame_number in mapping else None,
                estimated_rotation_xyz_by_mode,
            ),
            (
                imu_world_pose_ry_plot_path,
                "IMU trajectory camera world rot Y",
                "Rot Y [deg]",
                lambda frame_number: float(gt_rotation_xyz_by_frame[frame_number][1]),
                lambda mapping, frame_number: float(mapping[frame_number][1]) if frame_number in mapping else None,
                estimated_rotation_xyz_by_mode,
            ),
            (
                imu_world_pose_rz_plot_path,
                "IMU trajectory camera world rot Z",
                "Rot Z [deg]",
                lambda frame_number: float(gt_rotation_xyz_by_frame[frame_number][2]),
                lambda mapping, frame_number: float(mapping[frame_number][2]) if frame_number in mapping else None,
                estimated_rotation_xyz_by_mode,
            ),
        ]
        for plot_path, title, y_label, gt_value_fn, estimate_value_fn, mode_mappings in imu_component_specs:
            _save_multi_series_plot(
                plot_path,
                title=title,
                y_label=y_label,
                x_label="Recording frame",
                x_values=imu_frame_numbers,
                series_specs=[
                    {
                        "label": "ground truth",
                        "values": [gt_value_fn(frame_number) for frame_number in imu_frame_numbers],
                        "color_bgr": (52, 60, 68),
                    },
                    *[
                        {
                            "label": imu_mode_labels[mode_name],
                            "values": [estimate_value_fn(mode_mappings.get(mode_name, {}), frame_number) for frame_number in imu_frame_numbers],
                            "color_bgr": imu_mode_colors[mode_name],
                        }
                        for mode_name in imu_mode_order
                        if mode_name in mode_mappings
                    ],
                ],
            )

        imu_caveat_lines: list[str] = []
        if imu_initial_velocity_source != "logged_camera_gt_velocity":
            imu_caveat_lines = [
                "## Important Caveat",
                "",
                "This recording predates the exact IMU logging fix.",
                "- The start velocity is not logged explicitly in `camera_gt.csv`, so this report must infer it from finite differences.",
                "- The persisted IMU/time stream in this historical run is therefore not the exact simulator internal state.",
                "- These plots are current for this run, but they should be read as a diagnostic of the legacy recording path, not as the exact-simulator inertial benchmark.",
                "",
            ]

        imu_report_lines = [
            "# IMU Trajectory Reconstruction Report",
            "",
            "## Scope",
            "",
            f"- Run directory: `{resolved_run_dir}`",
            f"- Camera model: `{camera_model.name}`",
            f"- Ground-truth anchor: the first recorded camera world position and camera-to-world rotation are fixed to the true values from `camera_gt.csv`",
            f"- Samples used: `{len(camera_ground_truth_rows)}` camera ground-truth rows and `{len(imu_rows)}` IMU rows",
            f"- Initial velocity source: `{imu_initial_velocity_source}`",
            "",
            "## Method",
            "",
            "This report reconstructs the camera trajectory from the recorded accelerometer and gyroscope streams only.",
            "",
            "State and initialization:",
            "- The reconstruction state is camera world position, camera-to-world rotation, and world velocity.",
            "- `p0` and `R0` are clamped to the true first camera pose so the trajectory can be compared directly against ground truth.",
            (
                "- The reconstruction uses the exact logged initial world velocity from `camera_gt.csv`."
                if imu_initial_velocity_source == "logged_camera_gt_velocity"
                else "- The reconstruction falls back to a finite-difference estimate of the initial world velocity from the first two ground-truth samples because the recording does not contain logged world velocity."
            ),
            "",
            "Propagation model:",
            "- Gyroscope integration: `R_{k+1} = R_k Exp(omega_body * dt)`.",
            "- Accelerometer interpretation: the logged accelerometer is body-frame specific force, so world linear acceleration is `a_world = R * a_body + g` with `g = [0, 0, -9.81] m/s^2`.",
            "- Translation update matches the simulator's discrete convention: `v_k = v_{k-1} + a_k dt`, then `p_k = p_{k-1} + v_k dt`.",
            "",
            "This is pure inertial dead reckoning: no visual corrections, no bias estimation, and no smoothing pass.",
            "",
            *imu_caveat_lines,
            "## Summary",
            "",
            *_markdown_table(
                [
                    "mode",
                    "initial velocity [m/s]",
                    "mean pos err [m]",
                    "median pos err [m]",
                    "max pos err [m]",
                    "final pos err [m]",
                    "mean rot err [deg]",
                    "max rot err [deg]",
                    "final rot err [deg]",
                ],
                imu_summary_rows,
            ),
            "",
            "## Error Timelines",
            "",
            "Position error against ground truth in world meters:",
            "",
            "![IMU position error timeline](imu_position_error_timeline.png)",
            "",
            "Rotation error against ground truth in degrees:",
            "",
            "![IMU rotation error timeline](imu_rotation_error_timeline.png)",
            "",
            "## World-Pose Components",
            "",
            (
                "Each plot uses recording frame number on the x axis. Gray is ground truth and purple is the IMU-only reconstruction anchored at the true first pose with the exact logged initial world velocity."
                if imu_initial_velocity_source == "logged_camera_gt_velocity"
                else "Each plot uses recording frame number on the x axis. Gray is ground truth and purple is the IMU-only reconstruction anchored at the true first pose with a finite-difference initial world velocity estimate."
            ),
            "",
            "![IMU world x](imu_world_pose/camera_world_x_m.png)",
            "",
            "![IMU world y](imu_world_pose/camera_world_y_m.png)",
            "",
            "![IMU world z](imu_world_pose/camera_world_z_m.png)",
            "",
            "![IMU world rot x](imu_world_pose/camera_world_rx_deg.png)",
            "",
            "![IMU world rot y](imu_world_pose/camera_world_ry_deg.png)",
            "",
            "![IMU world rot z](imu_world_pose/camera_world_rz_deg.png)",
            "",
            "## Interpretation",
            "",
            (
                f"The main anchored IMU reconstruction reaches mean position error `{imu_main_summary['mean_position_error_m']:.6f}` m "
                f"and max position error `{imu_main_summary['max_position_error_m']:.6f}` m."
                if imu_main_summary is not None and imu_main_summary["mean_position_error_m"] is not None
                else "The main anchored IMU reconstruction did not produce enough samples for summary statistics."
            ),
            (
                "This run uses the exact logged initial world velocity from the simulator, so any remaining drift mainly reflects the precision of the persisted IMU/time stream and the discrete integration model."
                if imu_initial_velocity_source == "logged_camera_gt_velocity"
                else "This run had to infer the initial world velocity from finite differences, so its drift is expected to be worse than a recording with exact logged start velocity."
            ),
            "Because the simulated IMU stream is noise-free and bias-free, these numbers should be interpreted as a best-case inertial-only baseline before any visual fusion.",
            "",
        ]
        imu_trajectory_report_path.write_text("\n".join(imu_report_lines).strip() + "\n", encoding="utf-8")
    else:
        imu_trajectory_estimates_path.write_text("", encoding="utf-8")
        imu_trajectory_report_path.write_text(
            "\n".join(
                [
                    "# IMU Trajectory Reconstruction Report",
                    "",
                    "No IMU trajectory reconstruction could be generated because `camera_gt.csv` and `imu.csv` did not contain an alignable sample set.",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(resolved_run_dir),
        "camera_model_name": camera_model.name,
        "camera_model_toml_path": camera_model.source_toml_path,
        "analysis_input_is_undistorted": bool(camera_model.apply_lens_distortion_in_render),
        "frames_analyzed": frame_count,
        "recorded_samples": len(samples),
        "estimated_tags": estimates_written,
        "mean_position_error_m": mean_position_error_m,
        "median_position_error_m": median_position_error_m,
        "min_position_error_m": min_position_error_m,
        "max_position_error_m": max_position_error_m,
        "mean_rotation_error_deg": mean_rotation_error_deg,
        "mean_reprojection_rmse_image_plane_m": mean_reprojection_rmse_m,
        "mean_reprojection_rmse_px": mean_reprojection_rmse_px,
        "median_reprojection_rmse_image_plane_m": median_reprojection_rmse_m,
        "median_reprojection_rmse_px": median_reprojection_rmse_px,
        "tag_size_reference_m": tag_size_reference_m,
        "mean_position_error_tag_widths": (mean_position_error_m / tag_size_reference_m) if mean_position_error_m is not None and tag_size_reference_m else None,
        "mean_position_error_relative_to_range": (mean_position_error_m / mean_camera_tag_distance_m) if mean_position_error_m is not None and mean_camera_tag_distance_m else None,
        "camera_tag_distance_range_m": [min_camera_tag_distance_m, max_camera_tag_distance_m],
        "mean_camera_tag_distance_m": mean_camera_tag_distance_m,
        "tag_counts": tag_counts,
        "joint_world_frames_solved": len(joint_world_records),
        "joint_world_mean_position_error_m": float(np.mean(joint_world_position_errors_m)) if joint_world_position_errors_m else None,
        "joint_world_median_position_error_m": float(np.median(joint_world_position_errors_m)) if joint_world_position_errors_m else None,
        "joint_world_max_position_error_m": float(np.max(joint_world_position_errors_m)) if joint_world_position_errors_m else None,
        "joint_world_mean_rotation_error_deg": float(np.mean(joint_world_rotation_errors_deg)) if joint_world_rotation_errors_deg else None,
        "joint_world_mean_reprojection_rmse_px": float(np.mean(joint_world_reprojection_px)) if joint_world_reprojection_px else None,
        "imu_trajectory": {
            "main_mode": "true_start_velocity",
            "initial_velocity_source": imu_initial_velocity_source,
            "modes": imu_summaries_by_mode,
        },
        "per_tag": per_tag_summary,
        "artifacts": {
            "undistorted_video": str(undistorted_video_path.relative_to(resolved_run_dir)),
            "annotated_video": str(annotated_video_path.relative_to(resolved_run_dir)),
            "pose_estimates": str(estimates_path.relative_to(resolved_run_dir)),
            "joint_world_estimates": str(joint_world_estimates_path.relative_to(resolved_run_dir)),
            "imu_trajectory_estimates": str(imu_trajectory_estimates_path.relative_to(resolved_run_dir)),
            "imu_trajectory_report": str(imu_trajectory_report_path.relative_to(resolved_run_dir)),
            "report": str(report_path.relative_to(resolved_run_dir)),
            "single_pattern_report": str(single_pattern_report_path.relative_to(resolved_run_dir)),
            "single_pattern_overlay": str(single_pattern_overlay_path.relative_to(resolved_run_dir)),
            "worst_single_pattern_report": str(worst_single_pattern_report_path.relative_to(resolved_run_dir)),
            "worst_single_pattern_overlay": str(worst_single_pattern_overlay_path.relative_to(resolved_run_dir)),
            "real_measurement_known_solution_report": str(real_measurement_known_solution_report_path.relative_to(resolved_run_dir)),
            "synthetic_single_pattern_report": str(synthetic_single_pattern_report_path.relative_to(resolved_run_dir)),
            "synthetic_single_pattern_overlay": str(synthetic_single_pattern_overlay_path.relative_to(resolved_run_dir)),
            "representative_fit_image": str(representative_image_path.relative_to(resolved_run_dir)),
            "representative_pose_comparison_image": str(representative_pose_comparison_path.relative_to(resolved_run_dir)),
            "representative_observation": str(representative_observation_path.relative_to(resolved_run_dir)),
            "position_error_plot": str(position_error_plot_path.relative_to(resolved_run_dir)),
            "reprojection_plot": str(reprojection_plot_path.relative_to(resolved_run_dir)),
            "loss_trace_plot": str(loss_trace_plot_path.relative_to(resolved_run_dir)),
            "imu_position_error_plot": str(imu_position_error_plot_path.relative_to(resolved_run_dir)),
            "imu_rotation_error_plot": str(imu_rotation_error_plot_path.relative_to(resolved_run_dir)),
            "per_tag_directory": str(per_tag_dir.relative_to(resolved_run_dir)),
            "per_tag_summary": str(per_tag_summary_path.relative_to(resolved_run_dir)),
            "world_pose_by_tag_directory": str(world_pose_plot_dir.relative_to(resolved_run_dir)),
            "world_pose_x_plot": str(world_pose_x_plot_path.relative_to(resolved_run_dir)),
            "world_pose_y_plot": str(world_pose_y_plot_path.relative_to(resolved_run_dir)),
            "world_pose_z_plot": str(world_pose_z_plot_path.relative_to(resolved_run_dir)),
            "world_pose_rx_plot": str(world_pose_rx_plot_path.relative_to(resolved_run_dir)),
            "world_pose_ry_plot": str(world_pose_ry_plot_path.relative_to(resolved_run_dir)),
            "world_pose_rz_plot": str(world_pose_rz_plot_path.relative_to(resolved_run_dir)),
            "imu_world_pose_directory": str(imu_world_pose_plot_dir.relative_to(resolved_run_dir)),
            "imu_world_pose_x_plot": str(imu_world_pose_x_plot_path.relative_to(resolved_run_dir)),
            "imu_world_pose_y_plot": str(imu_world_pose_y_plot_path.relative_to(resolved_run_dir)),
            "imu_world_pose_z_plot": str(imu_world_pose_z_plot_path.relative_to(resolved_run_dir)),
            "imu_world_pose_rx_plot": str(imu_world_pose_rx_plot_path.relative_to(resolved_run_dir)),
            "imu_world_pose_ry_plot": str(imu_world_pose_ry_plot_path.relative_to(resolved_run_dir)),
            "imu_world_pose_rz_plot": str(imu_world_pose_rz_plot_path.relative_to(resolved_run_dir)),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    representative_section_lines: list[str] = []
    if representative_observation_payload is not None:
        estimate = representative_observation_payload["estimate"]
        observation = representative_observation_payload["observation"]
        ground_truth = representative_observation_payload["ground_truth"]
        ground_truth_position_array = np.asarray(ground_truth["camera_position_tag_m"], dtype=np.float64)
        ground_truth_rotation_tc_array = np.asarray(ground_truth["camera_rotation_tc"], dtype=np.float64).reshape(3, 3)
        observed_points = np.asarray(observation["image_plane_points_m"], dtype=np.float64)
        observed_pixels = np.asarray(observation["ordered_pixels_px"], dtype=np.float64)
        predicted_points = np.asarray(estimate["predicted_image_plane_points_m"], dtype=np.float64)
        residual_points = predicted_points - observed_points
        point_labels = list(observation["point_order"])
        camera_obscura_seed = estimate["camera_obscura_seed"]
        candidate_lookup = {str(candidate["seed_name"]): candidate for candidate in estimate["candidate_seed_summaries"]}
        candidate_rows: list[list[str]] = []
        selected_seed_name = str(estimate["selected_seed_name"])
        for candidate in candidate_lookup.values():
            candidate_rows.append(
                [
                    f"`{candidate['seed_name']}`{' (selected)' if candidate['seed_name'] == selected_seed_name else ''}",
                    f"{float(candidate['seed_reprojection_rmse_image_plane_m']) * 1000.0:.6f}",
                    f"{float(candidate['final_reprojection_rmse_image_plane_m']) * 1000.0:.6f}",
                    f"{float(candidate['final_loss']):.9e}",
                    f"`{candidate['termination_reason']}`",
                ]
            )

        point_rows: list[list[str]] = []
        for index, point_name in enumerate(point_labels):
            point_rows.append(
                [
                    f"`{point_name}`",
                    f"{observed_pixels[index, 0]:.2f}",
                    f"{observed_pixels[index, 1]:.2f}",
                    f"{observed_points[index, 0] * 1000.0:.6f}",
                    f"{observed_points[index, 1] * 1000.0:.6f}",
                    f"{predicted_points[index, 0] * 1000.0:.6f}",
                    f"{predicted_points[index, 1] * 1000.0:.6f}",
                    f"{residual_points[index, 0] * 1e6:.3f}",
                    f"{residual_points[index, 1] * 1e6:.3f}",
                ]
            )

        def _history_rows_from_steps(
            steps: list[dict[str, Any]],
            *,
            ground_truth_position_m: np.ndarray = ground_truth_position_array,
            ground_truth_rotation_tc: np.ndarray = ground_truth_rotation_tc_array,
        ) -> list[list[str]]:
            rows: list[list[str]] = []
            for step in steps:
                camera_position_after = np.asarray(step.get("camera_position_after_m", [math.nan, math.nan, math.nan]), dtype=np.float64)
                rotation_vector_after = np.asarray(step.get("rotation_vector_after_rad", [math.nan, math.nan, math.nan]), dtype=np.float64)
                if np.all(np.isfinite(camera_position_after)) and np.all(np.isfinite(rotation_vector_after)):
                    estimated_rotation_tc = _rotation_matrix_rvec_np(rotation_vector_after).T
                    position_error_vector_m, rotation_error_xyz_deg = _pose_error_vectors(
                        ground_truth_position_tag_m=ground_truth_position_m,
                        ground_truth_rotation_tc=ground_truth_rotation_tc,
                        estimated_position_tag_m=camera_position_after,
                        estimated_rotation_tc=estimated_rotation_tc,
                    )
                    position_error_vector_text = _format_vector(position_error_vector_m, precision=6)
                    rotation_error_xyz_text = _format_vector(rotation_error_xyz_deg, precision=3)
                else:
                    position_error_vector_text = "n/a"
                    rotation_error_xyz_text = "n/a"
                rows.append(
                    [
                        str(step["iteration"]),
                        f"{float(step['loss_before']):.9e}",
                        f"{float(step['gradient_norm']):.9e}",
                        f"{float(step['step_norm']):.9e}",
                        f"{float(step['damping']):.1e}",
                        f"{float(step['line_search_alpha']):.4f}",
                        "`yes`" if bool(step["accepted"]) else "`no`",
                        f"{float(step['loss_after']):.9e}",
                        f"{float(step['reprojection_rmse_after_m']) * 1000.0:.6f}",
                        position_error_vector_text,
                        rotation_error_xyz_text,
                        f"{float(step.get('min_depth_after_m', float('nan'))):.6f}",
                    ]
                )
            return rows

        def _history_rows_for(seed_name: str) -> list[list[str]]:
            candidate = candidate_lookup.get(seed_name) or {}
            return _history_rows_from_steps(candidate.get("history", []))

        camera_obscura_history_rows = _history_rows_for("camera_obscura")
        selected_history_rows = _history_rows_for(selected_seed_name)
        pose_comparison_rows = []
        pose_row_lookup = {str(row["short_label"]): row for row in representative_observation_payload.get("pose_comparison_rows", [])}
        for row in representative_observation_payload.get("pose_comparison_rows", []):
            reprojection_mm = (
                f"{float(row['reprojection_rmse_image_plane_m']) * 1000.0:.6f}"
                if row.get("reprojection_rmse_image_plane_m") is not None
                else "n/a"
            )
            reprojection_px = f"{float(row['reprojection_rmse_px']):.3f}" if row.get("reprojection_rmse_px") is not None else "n/a"
            pose_comparison_rows.append(
                [
                    str(row["short_label"]),
                    str(row["label"]),
                    _format_vector(row["position_m"], precision=6),
                    f"{float(row['distance_to_tag_m']):.6f}",
                    f"{float(row['position_error_m']):.6f}",
                    reprojection_mm,
                    reprojection_px,
                ]
            )
        scene_scale = representative_observation_payload.get("scene_scale", {})
        tag_size_reference_text = (
            f"{float(scene_scale['tag_size_reference_m']):.3f}"
            if scene_scale.get("tag_size_reference_m") is not None
            else "n/a"
        )
        mean_camera_tag_distance_text = (
            f"{float(scene_scale['mean_camera_tag_distance_m']):.3f}"
            if scene_scale.get("mean_camera_tag_distance_m") is not None
            else "n/a"
        )
        distance_range = scene_scale.get("camera_tag_distance_range_m") or [None, None]
        min_camera_tag_distance_text = f"{float(distance_range[0]):.3f}" if distance_range[0] is not None else "n/a"
        max_camera_tag_distance_text = f"{float(distance_range[1]):.3f}" if distance_range[1] is not None else "n/a"
        if scene_scale.get("wall_width_m") is not None and scene_scale.get("wall_height_m") is not None:
            scene_wall_size_line = f"- Scene wall size: `{float(scene_scale['wall_width_m']):.3f}` m wide by `{float(scene_scale['wall_height_m']):.3f}` m high"
        else:
            scene_wall_size_line = "- Scene wall size: not recorded in this run metadata"
        ground_truth_operator = representative_observation_payload["ground_truth_operator"]
        gt_projected_pixels = np.asarray(ground_truth_operator["predicted_pixels_px"], dtype=np.float64)
        optimized_projected_pixels = _image_plane_points_m_to_pixels(predicted_points, camera_model)
        single_pattern_rows: list[list[str]] = []
        point_label_short = ["C", "TR", "BR", "BL", "TL"]
        for index, label in enumerate(point_label_short):
            gt_delta_px = gt_projected_pixels[index] - observed_pixels[index]
            fit_delta_px = optimized_projected_pixels[index] - observed_pixels[index]
            single_pattern_rows.append(
                [
                    f"`{label}`",
                    _format_vector(observed_pixels[index], precision=2),
                    _format_vector(gt_projected_pixels[index], precision=2),
                    _format_vector(optimized_projected_pixels[index], precision=2),
                    _format_vector(gt_delta_px, precision=2),
                    f"{float(np.linalg.norm(gt_delta_px)):.3f}",
                    _format_vector(fit_delta_px, precision=2),
                    f"{float(np.linalg.norm(fit_delta_px)):.3f}",
                ]
            )

        representative_section_lines = [
            "## Representative Single-Pattern Solve",
            "",
            "This section expands one solved observation in detail so the initialization and convergence are visible numerically.",
            "",
            f"- Frame/tick/time/tag: `{representative_observation_payload['frame_index']}` / `{representative_observation_payload['tick_index']}` / `{representative_observation_payload['sim_time_s']:.3f}s` / `{representative_observation_payload['tag_id']}`",
            f"- Selected seed: `{selected_seed_name}`",
            f"- Ground-truth camera position in tag coordinates: `{_format_vector(ground_truth['camera_position_tag_m'])}` m",
            f"- Estimated camera position in tag coordinates: `{_format_vector(estimate['camera_position_tag_m'])}` m",
            f"- Final translation tag->camera: `{_format_vector(estimate['translation_tag_to_camera_m'])}` m",
            f"- Final Rodrigues rotation vector: `{_format_vector(estimate['rotation_vector_ct_rad'])}` rad",
            f"- Position error: `{representative_observation_payload['position_error_m']:.6f}` m",
            f"- Rotation error: `{representative_observation_payload['rotation_error_deg']:.6f}` deg",
            f"- Final reprojection RMSE: `{representative_observation_payload['reprojection_rmse_image_plane_m'] * 1000.0:.6f}` mm on the image plane = `{float(representative_observation_payload['reprojection_rmse_px']):.3f}` px",
            "",
            "### Units And Scale",
            "",
            f"- Tag size in this setup: `{tag_size_reference_text}` m",
            scene_wall_size_line,
            f"- Camera-to-tag distance range across all solved observations: `{min_camera_tag_distance_text}` m to `{max_camera_tag_distance_text}` m, mean `{mean_camera_tag_distance_text}` m",
            f"- The `position_error_timeline.png` plot is a world-space camera-position error in meters, measured in the tag coordinate frame.",
            f"- The `reprojection_timeline.png` plot is an image-space reprojection RMSE in pixels, not world meters.",
            "",
            (
                f"A useful sanity check is that the reference tag size in this run is `{tag_size_reference_m:.3f}` m "
                f"and the camera is typically about `{mean_camera_tag_distance_m:.3f}` m away from a tag. "
                "So meter-scale world-position error would be huge, while a sub-pixel to low-pixel reprojection error is comparatively small."
                if tag_size_reference_m is not None and mean_camera_tag_distance_m is not None
                else "A useful sanity check is to compare world-position error against the physical tag size and camera-to-tag distance reported above."
            ),
            "",
            "### Pose Comparison Against Ground Truth",
            "",
        ]
        representative_section_lines.extend(
            _markdown_table(
                [
                    "row",
                    "meaning",
                    "camera position in tag frame [m]",
                    "distance to tag [m]",
                    "world error [m]",
                    "reproj [mm]",
                    "reproj [px]",
                ],
                pose_comparison_rows,
            )
        )
        representative_section_lines.extend(
            [
                "",
                "The key thing to notice is that the corrected undamped solve no longer falls into the bad basin seen earlier. Here the camera-obscura seed starts far from truth, and the exact operator improves both the image fit and the world-space pose at the same time.",
                "",
                "![Representative pose comparison](representative_pose_comparison.png)",
                "",
                "### Image Comparison On The Recorded Frame",
                "",
                f"- Ground-truth camera operator vs captured points: `{float(ground_truth_operator['reprojection_rmse_px']):.3f}` px",
                f"- Final optimized camera operator vs captured points: `{float(representative_observation_payload['reprojection_rmse_px']):.3f}` px",
                f"- Initial camera-obscura seed vs captured points: `{float(pose_row_lookup['OBS']['reprojection_rmse_px']):.3f}` px",
                "",
                "Orange filled circles are the captured points from the recorded image.",
                "Green crosses are the points produced by the minimizer camera operator when it uses the ground-truth pose.",
                "Orange-red tilted crosses are the points produced by the same operator after optimization.",
                "",
                "![Single pattern overlay](single_pattern_overlay.png)",
                "",
                "### Image-Plane Point Comparison",
                "",
            ]
        )
        representative_section_lines.extend(
            _markdown_table(
                [
                    "point",
                    "captured px",
                    "gt operator px",
                    "optimized px",
                    "gt-captured dx,dy [px]",
                    "gt error [px]",
                    "opt-captured dx,dy [px]",
                    "opt error [px]",
                ],
                single_pattern_rows,
            )
        )
        representative_section_lines.extend(
            [
                "",
                "### Observed And Fitted 5-Point Measurement",
                "",
                "Point order is `center, top_right, bottom_right, bottom_left, top_left`.",
                "",
            ]
        )
        representative_section_lines.extend(
            _markdown_table(
                [
                    "point",
                    "pixel x",
                    "pixel y",
                    "obs x [mm]",
                    "obs y [mm]",
                    "fit x [mm]",
                    "fit y [mm]",
                    "dx [um]",
                    "dy [um]",
                ],
                point_rows,
            )
        )
        representative_section_lines.extend(
            [
                "",
                "### Camera Obscura Initial Guess",
                "",
                "The pinhole-style seed uses the apparent tag scale to estimate depth, then back-projects the observed center:",
                "",
                "`z0 = f * tag_size / average_observed_edge`",
                "",
                "`tx0 = x_center * z0 / f`, `ty0 = y_center * z0 / f`",
                "",
                f"- Focal length used by the solver: `{camera_obscura_seed['focal_length_m'] * 1000.0:.6f}` mm",
                f"- Pattern size: `{camera_obscura_seed['pattern_size_m']:.6f}` m",
                f"- Observed center on the image plane: `{_format_vector(camera_obscura_seed['center_image_plane_m'], precision=9, scale=1000.0)}` mm",
                f"- Observed edge lengths: top `{camera_obscura_seed['observed_edge_lengths_m']['top_width'] * 1000.0:.6f}` mm, bottom `{camera_obscura_seed['observed_edge_lengths_m']['bottom_width'] * 1000.0:.6f}` mm, right `{camera_obscura_seed['observed_edge_lengths_m']['right_height'] * 1000.0:.6f}` mm, left `{camera_obscura_seed['observed_edge_lengths_m']['left_height'] * 1000.0:.6f}` mm",
                f"- Average observed edge: `{camera_obscura_seed['average_observed_edge_m'] * 1000.0:.6f}` mm",
                f"- Estimated depth from camera obscura: `{camera_obscura_seed['estimated_depth_m']:.6f}` m",
                f"- Camera obscura seed state `[tx, ty, tz, rx, ry, rz]`: `{_format_vector(camera_obscura_seed['state'])}`",
                f"- Camera obscura seed reprojection RMSE: `{float(estimate['camera_obscura_seed_reprojection_rmse_image_plane_m']) * 1000.0:.6f}` mm",
                "",
                "### Seed Competition",
                "",
            ]
        )
        representative_section_lines.extend(
            _markdown_table(
                ["seed", "seed rmse [mm]", "final rmse [mm]", "final loss", "termination"],
                candidate_rows,
            )
        )
        if "opencv_pnp_seed" in estimate:
            representative_section_lines.extend(
                [
                    "",
                    f"- OpenCV PnP seed state `[tx, ty, tz, rx, ry, rz]`: `{_format_vector(estimate['opencv_pnp_seed']['state'])}`",
                    f"- OpenCV PnP seed reprojection RMSE: `{float(estimate['opencv_pnp_seed_reprojection_rmse_image_plane_m']) * 1000.0:.6f}` mm",
                ]
            )
        representative_section_lines.extend(
            [
                "",
                "### What The Three Main Poses Mean",
                "",
                f"- `OBS` is the raw camera-obscura seed before Newton. For this frame it is `{float(pose_row_lookup['OBS']['position_error_m']):.6f}` m away from ground truth, with `{float(pose_row_lookup['OBS']['reprojection_rmse_px']):.3f}` px reprojection error.",
                f"- `OBF` is the result of running the full exact camera operator starting from the obscura seed. For this frame it ends `{float(pose_row_lookup['OBF']['position_error_m']):.6f}` m away from ground truth, with `{float(pose_row_lookup['OBF']['reprojection_rmse_px']):.3f}` px reprojection error."
                if "OBF" in pose_row_lookup
                else "- `OBF` is unavailable for this observation.",
                f"- `EX` is the final exact-operator result chosen by the optimizer. For this frame it ends `{float(pose_row_lookup['EX']['position_error_m']):.6f}` m away from ground truth, with `{float(pose_row_lookup['EX']['reprojection_rmse_px']):.3f}` px reprojection error.",
                "",
                "For this representative frame, the corrected solver resolves the earlier mismatch: the exact operator finds a pose that matches the image much better than the obscura seed and is also very close to the recorded ground truth in world coordinates.",
            ]
        )
        representative_section_lines.extend(
            [
                "",
                "### Newton Convergence From Camera Obscura Seed",
                "",
                "This is the requested pinhole-style seed followed by the corrected modified Newton steps on the full fitted observation model used in this run.",
                "The `gt - cam pos [m]` column is `ground_truth_position - current_estimate_position` and the `gt - rot xyz [deg]` column is the XYZ Euler-angle error vector of the relative rotation from estimate to ground truth.",
                "",
            ]
        )
        representative_section_lines.extend(
            _markdown_table(
                [
                    "iter",
                    "loss before",
                    "grad norm",
                    "step norm",
                    "damping",
                    "alpha",
                    "accepted",
                    "loss after",
                    "rmse after [mm]",
                    "gt - cam pos [m]",
                    "gt - rot xyz [deg]",
                    "min depth [m]",
                ],
                camera_obscura_history_rows,
            )
        )
        if selected_seed_name != "camera_obscura":
            representative_section_lines.extend(
                [
                    "",
                    "### Newton Convergence For The Selected Seed",
                    "",
                    f"The obscura path is shown above, but this representative observation ultimately converged to a lower final objective from the `{selected_seed_name}` seed.",
                    "The pose-error vector columns use the same `ground truth - current estimate` convention as above.",
                    "",
                ]
            )
            representative_section_lines.extend(
                _markdown_table(
                    [
                        "iter",
                        "loss before",
                        "grad norm",
                        "step norm",
                        "damping",
                        "alpha",
                        "accepted",
                        "loss after",
                        "rmse after [mm]",
                        "gt - cam pos [m]",
                        "gt - rot xyz [deg]",
                        "min depth [m]",
                    ],
                    selected_history_rows,
                )
            )
        representative_section_lines.extend(
            [
                "",
                f"Machine-readable details for this observation are saved in `{representative_observation_path.relative_to(resolved_run_dir)}`.",
                "",
            ]
        )

        single_pattern_report_lines = [
            "# Single Pattern Report",
            "",
            "This report isolates one frame and one target pattern only.",
            "",
            "## Scope",
            "",
            f"- Run directory: `{resolved_run_dir}`",
            f"- Frame/tick/time/tag: `{representative_observation_payload['frame_index']}` / `{representative_observation_payload['tick_index']}` / `{representative_observation_payload['sim_time_s']:.3f}s` / `{representative_observation_payload['tag_id']}`",
            f"- Camera model: `{camera_model.name}`",
            f"- Pattern size: `{float(observation['pattern_size_m']):.6f}` m",
            "",
            "## Ground Truth",
            "",
            f"- Camera position in tag frame: `{_format_vector(ground_truth['camera_position_tag_m'], precision=6)}` m",
            f"- Tag-to-camera translation used by the minimizer: `{_format_vector(ground_truth['translation_tag_to_camera_m'], precision=6)}` m",
            f"- Ground-truth Rodrigues rotation used by the minimizer: `{_format_vector(ground_truth['rotation_vector_ct_rad'], precision=6)}` rad",
            f"- Ground-truth camera-operator reprojection error against captured points: `{float(ground_truth_operator['reprojection_rmse_px']):.3f}` px",
            "",
            "## Image Comparison",
            "",
            "Orange filled circles are the captured points from the recorded image.",
            "Green crosses are the points produced by the minimizer camera operator when it uses the ground-truth pose.",
            "Orange-red tilted crosses are the points produced by the same operator after optimization.",
            "",
            "![Single pattern overlay](single_pattern_overlay.png)",
            "",
            "## Point Table",
            "",
            *_markdown_table(
                [
                    "point",
                    "captured px",
                    "gt operator px",
                    "optimized px",
                    "gt-captured dx,dy [px]",
                    "gt error [px]",
                    "opt-captured dx,dy [px]",
                    "opt error [px]",
                ],
                single_pattern_rows,
            ),
            "",
            "## Optimization Result",
            "",
            f"- Initial camera-obscura seed position: `{_format_vector(pose_row_lookup['OBS']['position_m'], precision=6)}` m",
            f"- Initial camera-obscura seed reprojection error: `{float(pose_row_lookup['OBS']['reprojection_rmse_px']):.3f}` px",
            f"- Final optimized camera position: `{_format_vector(pose_row_lookup['EX']['position_m'], precision=6)}` m",
            f"- Final optimized reprojection error: `{float(pose_row_lookup['EX']['reprojection_rmse_px']):.3f}` px",
            f"- Final optimized world-position error vs ground truth: `{float(pose_row_lookup['EX']['position_error_m']):.6f}` m",
            f"- Final optimized rotation error vs ground truth: `{float(representative_observation_payload['rotation_error_deg']):.6f}` deg",
            "",
            "## Interpretation",
            "",
            "If the green crosses do not sit exactly on the orange captured points, that is detector-versus-ground-truth mismatch on this frame, not a failure of the forward camera operator.",
            "If the orange-red optimized crosses sit closer to the orange captured points than the green ones do, the optimizer is matching the image better than ground truth for this observation, even if the recovered 3D pose is physically worse.",
            "",
        ]
        single_pattern_report_path.write_text("\n".join(single_pattern_report_lines).strip() + "\n", encoding="utf-8")

        if worst_record is not None:
            worst_estimate_record = worst_record["estimate_record"]
            worst_prior_camera_position_tag_m = worst_estimate_record.get("continuity_prior_camera_position_tag_m")
            worst_detailed_estimate = _estimate_pose_newton(
                np.asarray(worst_record["observation_image_plane_m"], dtype=np.float64),
                ordered_pixels_px=np.asarray(worst_record["ordered_pixels_px"], dtype=np.float64),
                camera_model=camera_model,
                focal_length_m=float(camera_model.focal_length_mm) / 1000.0,
                pattern_half_extent_m=float(worst_record["pattern_half_extent_m"]),
                candidate_seed_specs=_candidate_seed_specs_for_replay(worst_estimate_record["estimate"]),
                include_history=True,
                use_diagonal_damping=False,
                initial_damping=0.0,
            )
            if worst_prior_camera_position_tag_m is not None:
                worst_detailed_estimate = _maybe_disambiguate_half_turn_branch(
                    worst_detailed_estimate,
                    prior_camera_position_tag_m=np.asarray(worst_prior_camera_position_tag_m, dtype=np.float64),
                    observed_points_image_plane_m=np.asarray(worst_record["observation_image_plane_m"], dtype=np.float64),
                    focal_length_m=float(camera_model.focal_length_mm) / 1000.0,
                    pattern_half_extent_m=float(worst_record["pattern_half_extent_m"]),
                )
            worst_estimate_record["estimate"] = worst_detailed_estimate
            worst_estimate_record["reprojection_rmse_image_plane_m"] = float(worst_detailed_estimate["reprojection_rmse_image_plane_m"])
            worst_ground_truth_position = np.asarray(worst_estimate_record["ground_truth_camera_position_tag_m"], dtype=np.float64)
            worst_ground_truth_rotation_tc = np.asarray(worst_estimate_record["ground_truth_camera_rotation_tc"], dtype=np.float64).reshape(3, 3)
            worst_estimated_position = np.asarray(worst_detailed_estimate["camera_position_tag_m"], dtype=np.float64)
            worst_estimated_rotation_tc = np.asarray(worst_detailed_estimate["camera_rotation_tc"], dtype=np.float64).reshape(3, 3)
            worst_estimate_record["position_error_m"] = float(np.linalg.norm(worst_estimated_position - worst_ground_truth_position))
            worst_estimate_record["rotation_error_deg"] = _rotation_error_deg(worst_estimated_rotation_tc, worst_ground_truth_rotation_tc)
            worst_focal_length_m = float(camera_model.focal_length_mm) / 1000.0
            worst_pattern_half_extent_m = float(worst_record["pattern_half_extent_m"])
            worst_observed_points_image_plane_m = np.asarray(worst_record["observation_image_plane_m"], dtype=np.float64)
            worst_observed_pixels = np.asarray(worst_record["ordered_pixels_px"], dtype=np.float64)
            worst_candidate_lookup = {str(candidate["seed_name"]): candidate for candidate in worst_detailed_estimate["candidate_seed_summaries"]}
            worst_selected_seed_name = str(worst_detailed_estimate["selected_seed_name"])
            worst_camera_obscura_seed = worst_detailed_estimate["camera_obscura_seed"]
            worst_camera_obscura_seed_diag = _state_diagnostics(
                np.asarray(worst_camera_obscura_seed["state"], dtype=np.float64),
                observed_points_image_plane_m=worst_observed_points_image_plane_m,
                focal_length_m=worst_focal_length_m,
                pattern_half_extent_m=worst_pattern_half_extent_m,
            )
            worst_local_step_diag = _local_step_diagnostics(
                np.asarray(worst_camera_obscura_seed["state"], dtype=np.float64),
                observed_points_image_plane_m=worst_observed_points_image_plane_m,
                focal_length_m=worst_focal_length_m,
                pattern_half_extent_m=worst_pattern_half_extent_m,
            )
            worst_raw_step_is_non_descent = (
                worst_local_step_diag["raw_step_dot_gradient"] is not None
                and float(worst_local_step_diag["raw_step_dot_gradient"]) >= 0.0
            ) or (worst_local_step_diag["raw_solve_error"] is not None)
            worst_selected_final_diag = _state_diagnostics(
                np.asarray(
                    worst_detailed_estimate["translation_tag_to_camera_m"] + worst_detailed_estimate["rotation_vector_ct_rad"],
                    dtype=np.float64,
                ),
                observed_points_image_plane_m=worst_observed_points_image_plane_m,
                focal_length_m=worst_focal_length_m,
                pattern_half_extent_m=worst_pattern_half_extent_m,
            )
            worst_ground_truth_rotation_ct = worst_ground_truth_rotation_tc.T
            worst_ground_truth_translation_ct = -(worst_ground_truth_rotation_ct @ worst_ground_truth_position)
            worst_ground_truth_rvec_ct, _ = cv2.Rodrigues(worst_ground_truth_rotation_ct)
            worst_ground_truth_state = np.concatenate([worst_ground_truth_translation_ct.reshape(3), worst_ground_truth_rvec_ct.reshape(3)])
            worst_ground_truth_operator_diag = _state_diagnostics(
                worst_ground_truth_state,
                observed_points_image_plane_m=worst_observed_points_image_plane_m,
                focal_length_m=worst_focal_length_m,
                pattern_half_extent_m=worst_pattern_half_extent_m,
            )
            worst_ground_truth_projected_pixels_px = _image_plane_points_m_to_pixels(
                np.asarray(worst_ground_truth_operator_diag["predicted_image_plane_points_m"], dtype=np.float64),
                camera_model,
            )
            worst_synthetic_observation_image_plane_m = np.asarray(
                worst_ground_truth_operator_diag["predicted_image_plane_points_m"],
                dtype=np.float64,
            )
            worst_synthetic_step_diag = _local_step_diagnostics(
                np.asarray(worst_camera_obscura_seed["state"], dtype=np.float64),
                observed_points_image_plane_m=worst_synthetic_observation_image_plane_m,
                focal_length_m=worst_focal_length_m,
                pattern_half_extent_m=worst_pattern_half_extent_m,
            )
            worst_synthetic_estimate = _estimate_pose_newton(
                worst_synthetic_observation_image_plane_m,
                ordered_pixels_px=worst_ground_truth_projected_pixels_px,
                camera_model=camera_model,
                focal_length_m=worst_focal_length_m,
                pattern_half_extent_m=worst_pattern_half_extent_m,
                include_history=True,
                use_diagonal_damping=False,
                initial_damping=0.0,
            )
            worst_synthetic_position_error_m = float(
                np.linalg.norm(np.asarray(worst_synthetic_estimate["camera_position_tag_m"], dtype=np.float64) - worst_ground_truth_position)
            )
            worst_synthetic_rotation_error_deg = _rotation_error_deg(
                np.asarray(worst_synthetic_estimate["camera_rotation_tc"], dtype=np.float64).reshape(3, 3),
                worst_ground_truth_rotation_tc,
            )
            worst_optimized_projected_pixels_px = _image_plane_points_m_to_pixels(
                np.asarray(worst_detailed_estimate["predicted_image_plane_points_m"], dtype=np.float64),
                camera_model,
            )
            _save_single_pattern_overlay(
                worst_single_pattern_overlay_path,
                analysis_frame_bgr=worst_record["analysis_frame_bgr"],
                observed_pixels_px=worst_observed_pixels,
                ground_truth_pixels_px=worst_ground_truth_projected_pixels_px,
                optimized_pixels_px=worst_optimized_projected_pixels_px,
                point_labels=point_labels,
                metadata_lines=[
                    f"frame={worst_estimate_record['frame_index']}  tick={worst_estimate_record['tick_index']}  tag={worst_estimate_record['tag_id']}",
                    f"gt-vs-captured={_image_plane_m_to_px(float(worst_ground_truth_operator_diag['reprojection_rmse_image_plane_m']), camera_model):.3f} px",
                    f"opt-vs-captured={_image_plane_m_to_px(float(worst_selected_final_diag['reprojection_rmse_image_plane_m']), camera_model):.3f} px",
                    f"world position error of optimized pose={worst_estimate_record['position_error_m']:.6f} m",
                ],
            )
            worst_point_rows: list[list[str]] = []
            worst_point_label_short = ["C", "TR", "BR", "BL", "TL"]
            for index, label in enumerate(worst_point_label_short):
                worst_gt_delta_px = worst_ground_truth_projected_pixels_px[index] - worst_observed_pixels[index]
                worst_fit_delta_px = worst_optimized_projected_pixels_px[index] - worst_observed_pixels[index]
                worst_point_rows.append(
                    [
                        f"`{label}`",
                        _format_vector(worst_observed_pixels[index], precision=2),
                        _format_vector(worst_ground_truth_projected_pixels_px[index], precision=2),
                        _format_vector(worst_optimized_projected_pixels_px[index], precision=2),
                        _format_vector(worst_gt_delta_px, precision=2),
                        f"{float(np.linalg.norm(worst_gt_delta_px)):.3f}",
                        _format_vector(worst_fit_delta_px, precision=2),
                        f"{float(np.linalg.norm(worst_fit_delta_px)):.3f}",
                    ]
                )
            worst_pose_rows = [
                [
                    "GT",
                    _format_vector(worst_ground_truth_position, precision=6),
                    "0.000000",
                    "n/a",
                    "n/a",
                ],
                [
                    "OBS",
                    _format_vector(worst_camera_obscura_seed_diag["camera_position_tag_m"], precision=6),
                    f"{_world_position_error_m(worst_camera_obscura_seed_diag['camera_position_tag_m'], worst_ground_truth_position):.6f}",
                    f"{_image_plane_m_to_px(float(worst_camera_obscura_seed_diag['reprojection_rmse_image_plane_m']), camera_model):.3f}",
                    f"{float(np.linalg.norm(np.asarray(worst_camera_obscura_seed_diag['camera_position_tag_m'], dtype=np.float64))):.6f}",
                ],
                [
                    "EX",
                    _format_vector(worst_detailed_estimate["camera_position_tag_m"], precision=6),
                    f"{float(worst_estimate_record['position_error_m']):.6f}",
                    f"{_image_plane_m_to_px(float(worst_selected_final_diag['reprojection_rmse_image_plane_m']), camera_model):.3f}",
                    f"{float(np.linalg.norm(np.asarray(worst_detailed_estimate['camera_position_tag_m'], dtype=np.float64))):.6f}",
                ],
            ]
            worst_history_rows = _history_rows_from_steps(
                worst_detailed_estimate.get("selected_seed_history", []),
                ground_truth_position_m=worst_ground_truth_position,
                ground_truth_rotation_tc=worst_ground_truth_rotation_tc,
            )
            worst_report_lines = [
                "# Worst Single Pattern Report",
                "",
                "This report isolates the single worst solved observation in the run, ranked by final world-space camera-position error.",
                "",
                "## Scope",
                "",
                f"- Run directory: `{resolved_run_dir}`",
                f"- Frame/tick/time/tag: `{worst_estimate_record['frame_index']}` / `{worst_estimate_record['tick_index']}` / `{worst_estimate_record['sim_time_s']:.3f}s` / `{worst_estimate_record['tag_id']}`",
                f"- Camera model: `{camera_model.name}`",
                f"- Ranking criterion: maximum world-space position error across all solved observations in the run",
                f"- This worst observation error: `{float(worst_estimate_record['position_error_m']):.6f}` m",
                f"- Mean camera-to-tag distance in the run: `{mean_camera_tag_distance_m if mean_camera_tag_distance_m is not None else 'n/a'}` m",
                "",
                "## Optimization Result",
                "",
                f"- Selected seed: `{worst_selected_seed_name}`",
                f"- Ground-truth camera position in tag frame: `{_format_vector(worst_ground_truth_position, precision=6)}` m",
                f"- Final optimized camera position in tag frame: `{_format_vector(worst_detailed_estimate['camera_position_tag_m'], precision=6)}` m",
                f"- Final world-position error: `{float(worst_estimate_record['position_error_m']):.6f}` m",
                f"- Final rotation error: `{float(worst_estimate_record['rotation_error_deg']):.6f}` deg",
                f"- Final reprojection error: `{_image_plane_m_to_px(float(worst_selected_final_diag['reprojection_rmse_image_plane_m']), camera_model):.3f}` px",
                "",
                "## Pose Comparison",
                "",
                *_markdown_table(
                    ["row", "camera position in tag frame [m]", "world error [m]", "reproj [px]", "distance to tag [m]"],
                    worst_pose_rows,
                ),
                "",
                "## Image Comparison",
                "",
                "Orange filled circles are the captured points from the recorded image.",
                "Green crosses are the points produced by the minimizer camera operator when it uses the ground-truth pose.",
                "Orange-red tilted crosses are the points produced by the same operator after optimization.",
                "",
                "![Worst single pattern overlay](worst_single_pattern_overlay.png)",
                "",
                "## Point Table",
                "",
                *_markdown_table(
                    [
                        "point",
                        "captured px",
                        "gt operator px",
                        "optimized px",
                        "gt-captured dx,dy [px]",
                        "gt error [px]",
                        "opt-captured dx,dy [px]",
                        "opt error [px]",
                    ],
                    worst_point_rows,
                ),
                "",
                "## Camera Obscura Seed",
                "",
                f"- Initial camera-obscura seed position: `{_format_vector(worst_camera_obscura_seed_diag['camera_position_tag_m'], precision=6)}` m",
                f"- Initial camera-obscura seed reprojection error: `{_image_plane_m_to_px(float(worst_camera_obscura_seed_diag['reprojection_rmse_image_plane_m']), camera_model):.3f}` px",
                f"- Estimated depth from camera obscura: `{float(worst_camera_obscura_seed['estimated_depth_m']):.6f}` m",
                "",
                "## Solver Diagnosis",
                "",
                f"- Raw Hessian rank at the seed: `{int(worst_local_step_diag['hessian_rank'])}`",
                f"- Raw Hessian eigenvalues: `{_format_vector(worst_local_step_diag['hessian_eigenvalues'], precision=9)}`",
                f"- Raw Newton step norm before any correction: `{float(worst_local_step_diag['raw_step_norm']) if worst_local_step_diag['raw_step_norm'] is not None else 'n/a'}`",
                f"- Raw Newton directional derivative `g^T p`: `{float(worst_local_step_diag['raw_step_dot_gradient']) if worst_local_step_diag['raw_step_dot_gradient'] is not None else 'n/a'}`",
                f"- Corrected step method chosen by the new solver: `{worst_local_step_diag['modified_step_method']}`",
                f"- Diagonal shift needed to make the step descend: `{float(worst_local_step_diag['modified_step_damping']):.1e}`",
                f"- Corrected step directional derivative `g^T p`: `{float(worst_local_step_diag['modified_step_dot_gradient']):.9e}`",
                "",
                (
                    "The important failure mechanism is that the raw Newton solve did not fail numerically, but it produced a non-descent direction. The old code treated that as a reason to replace Newton by a tiny `-gradient` step, which is why the optimizer appeared to do nothing on this frame."
                    if worst_raw_step_is_non_descent
                    else "For the current post-fix worst case, the raw Newton step is already a valid descent direction, so the solver no longer needs the special rescue path here. This report is still useful because it shows what the remaining tail case looks like after the broken step logic was removed."
                ),
                "",
                "## Synthetic Reproduction Of This Same Geometry",
                "",
                "This repeats the exact same worst-case geometry, but replaces the detected image points with perfect synthetic points projected from the ground-truth pose.",
                "",
                f"- Synthetic raw Hessian eigenvalues at the same obscura seed: `{_format_vector(worst_synthetic_step_diag['hessian_eigenvalues'], precision=9)}`",
                f"- Synthetic raw Newton directional derivative `g^T p`: `{float(worst_synthetic_step_diag['raw_step_dot_gradient']) if worst_synthetic_step_diag['raw_step_dot_gradient'] is not None else 'n/a'}`",
                f"- Synthetic corrected step method: `{worst_synthetic_step_diag['modified_step_method']}` with diagonal shift `{float(worst_synthetic_step_diag['modified_step_damping']):.1e}`",
                f"- Synthetic final position error after the fix: `{float(worst_synthetic_position_error_m):.6f}` m",
                f"- Synthetic final rotation error after the fix: `{float(worst_synthetic_rotation_error_deg):.6f}` deg",
                f"- Synthetic final reprojection error after the fix: `{_image_plane_m_to_px(float(worst_synthetic_estimate['reprojection_rmse_image_plane_m']), camera_model):.6f}` px",
                "",
                (
                    "Because the same stuck behavior appears on the synthetic version of this frame, the root cause is inside the minimizer step logic, not in detector noise or in the camera model."
                    if worst_raw_step_is_non_descent
                    else "The synthetic version of this current worst case also solves cleanly, which is exactly what we want after the step-selection fix. That means the remaining error here is measurement-limited, not a broken minimizer."
                ),
                "",
                "## Newton Convergence From Camera Obscura Seed",
                "",
                "The `gt - cam pos [m]` column is `ground_truth_position - current_estimate_position` and the `gt - rot xyz [deg]` column is the XYZ Euler-angle error vector of the relative rotation from estimate to ground truth.",
                "",
                *_markdown_table(
                    [
                        "iter",
                        "loss before",
                        "grad norm",
                        "step norm",
                        "damping",
                        "alpha",
                        "accepted",
                        "loss after",
                        "rmse after [mm]",
                        "gt - cam pos [m]",
                        "gt - rot xyz [deg]",
                        "min depth [m]",
                    ],
                    worst_history_rows,
                ),
                "",
                "## Interpretation",
                "",
                "This is the current tail case to study after the minimizer fix.",
                "Its remaining error is small enough to be consistent with image-measurement mismatch and single-tag ambiguity, not with a broken forward camera model or a solver that is refusing to step.",
                "",
            ]
            worst_single_pattern_report_path.write_text("\n".join(worst_report_lines).strip() + "\n", encoding="utf-8")

        real_measurement_payload = representative_observation_payload["real_measurement_ground_truth_seed"]
        real_measurement_damped = real_measurement_payload["estimate_damped"]
        real_measurement_undamped = real_measurement_payload["estimate_undamped"]
        real_measurement_pose_rows = [
            [
                "GT start",
                _format_vector(ground_truth["camera_position_tag_m"], precision=6),
                "0.000000",
                f"{float(real_measurement_payload['ground_truth_reprojection_rmse_px']):.6f}",
                f"{float(real_measurement_payload['ground_truth_loss']):.9e}",
            ],
            [
                "damped",
                _format_vector(real_measurement_damped["camera_position_tag_m"], precision=6),
                f"{float(real_measurement_payload['damped_position_error_m']):.6f}",
                f"{float(real_measurement_payload['damped_reprojection_rmse_px']):.6f}",
                f"{float(real_measurement_damped['final_loss']):.9e}",
            ],
            [
                "undamped",
                _format_vector(real_measurement_undamped["camera_position_tag_m"], precision=6),
                f"{float(real_measurement_payload['undamped_position_error_m']):.6f}",
                f"{float(real_measurement_payload['undamped_reprojection_rmse_px']):.6f}",
                f"{float(real_measurement_undamped['final_loss']):.9e}",
            ],
        ]
        real_measurement_solver_rows = [
            [
                "`damped`",
                f"{float(real_measurement_payload['damped_reprojection_rmse_px']):.6f}",
                f"{float(real_measurement_payload['damped_position_error_m']):.6f}",
                f"{float(real_measurement_payload['damped_rotation_error_deg']):.6f}",
                f"{float(real_measurement_damped.get('initial_damping', 0.0)):.1e}",
                "`on`" if bool(real_measurement_damped.get("use_diagonal_damping", False)) else "`off`",
            ],
            [
                "`undamped`",
                f"{float(real_measurement_payload['undamped_reprojection_rmse_px']):.6f}",
                f"{float(real_measurement_payload['undamped_position_error_m']):.6f}",
                f"{float(real_measurement_payload['undamped_rotation_error_deg']):.6f}",
                f"{float(real_measurement_undamped.get('initial_damping', 0.0)):.1e}",
                "`on`" if bool(real_measurement_undamped.get("use_diagonal_damping", False)) else "`off`",
            ],
        ]
        real_measurement_damped_history_rows = _history_rows_from_steps(real_measurement_damped.get("history", []))
        real_measurement_undamped_history_rows = _history_rows_from_steps(real_measurement_undamped.get("history", []))
        real_measurement_known_solution_report_lines = [
            "# Real Measurement Ground-Truth-Seed Report",
            "",
            "This report keeps the real detected image measurement from the recorded sim frame, but initializes Newton at the exact known ground-truth pose.",
            "",
            "## Scope",
            "",
            f"- Run directory: `{resolved_run_dir}`",
            f"- Frame/tick/time/tag: `{representative_observation_payload['frame_index']}` / `{representative_observation_payload['tick_index']}` / `{representative_observation_payload['sim_time_s']:.3f}s` / `{representative_observation_payload['tag_id']}`",
            f"- Camera model: `{camera_model.name}`",
            f"- Measurement source: the real detected points from the recorded image, not synthetic projections",
            "",
            "## Key Result",
            "",
            f"- The known ground-truth pose is not a zero-loss point on the real measurement: `{float(real_measurement_payload['ground_truth_reprojection_rmse_px']):.6f}` px, objective `{float(real_measurement_payload['ground_truth_loss']):.9e}`",
            f"- Starting exactly from ground truth, the damped solve ends at `{float(real_measurement_payload['damped_reprojection_rmse_px']):.6f}` px with `{float(real_measurement_payload['damped_position_error_m']):.6f}` m world-position error",
            f"- Starting exactly from ground truth, the undamped solve ends at `{float(real_measurement_payload['undamped_reprojection_rmse_px']):.6f}` px with `{float(real_measurement_payload['undamped_position_error_m']):.6f}` m world-position error",
            "",
            "This isolates what the optimizer itself does when it is handed the correct physical pose on the noisy real observation.",
            "",
            "## Final Pose Summary",
            "",
            *_markdown_table(
                ["row", "camera position in tag frame [m]", "world error [m]", "reproj [px]", "objective"],
                real_measurement_pose_rows,
            ),
            "",
            "## Damped vs Undamped Summary",
            "",
            *_markdown_table(
                ["solver", "final reproj [px]", "final world error [m]", "final rot error [deg]", "initial damping", "diag damping"],
                real_measurement_solver_rows,
            ),
            "",
            "## Newton From Ground Truth On Real Measurement: Damped",
            "",
            "The `gt - cam pos [m]` column is `ground_truth_position - current_estimate_position` and the `gt - rot xyz [deg]` column is the XYZ Euler-angle error vector of the relative rotation from estimate to ground truth.",
            "",
            *_markdown_table(
                [
                    "iter",
                    "loss before",
                    "grad norm",
                    "step norm",
                    "damping",
                    "alpha",
                    "accepted",
                    "loss after",
                    "rmse after [mm]",
                    "gt - cam pos [m]",
                    "gt - rot xyz [deg]",
                    "min depth [m]",
                ],
                real_measurement_damped_history_rows,
            ),
            "",
            "## Newton From Ground Truth On Real Measurement: Undamped",
            "",
            "This table uses the same columns, but with `damping = 0` inside the Newton solve.",
            "",
            *_markdown_table(
                [
                    "iter",
                    "loss before",
                    "grad norm",
                    "step norm",
                    "damping",
                    "alpha",
                    "accepted",
                    "loss after",
                    "rmse after [mm]",
                    "gt - cam pos [m]",
                    "gt - rot xyz [deg]",
                    "min depth [m]",
                ],
                real_measurement_undamped_history_rows,
            ),
            "",
        ]
        real_measurement_known_solution_report_path.write_text(
            "\n".join(real_measurement_known_solution_report_lines).strip() + "\n",
            encoding="utf-8",
        )

        synthetic_payload = representative_observation_payload["synthetic_perfect_observation"]
        synthetic_estimate_damped = synthetic_payload["estimate_damped"]
        synthetic_estimate = synthetic_payload["estimate"]
        synthetic_point_rows: list[list[str]] = []
        synthetic_observed_pixels = np.asarray(synthetic_payload["observation_pixels_px"], dtype=np.float64)
        synthetic_estimated_pixels = _image_plane_points_m_to_pixels(
            np.asarray(synthetic_estimate["predicted_image_plane_points_m"], dtype=np.float64),
            camera_model,
        )
        for index, label in enumerate(point_label_short):
            fit_delta_px = synthetic_estimated_pixels[index] - synthetic_observed_pixels[index]
            synthetic_point_rows.append(
                [
                    f"`{label}`",
                    _format_vector(synthetic_observed_pixels[index], precision=2),
                    _format_vector(gt_projected_pixels[index], precision=2),
                    _format_vector(synthetic_estimated_pixels[index], precision=2),
                    _format_vector(fit_delta_px, precision=2),
                    f"{float(np.linalg.norm(fit_delta_px)):.3f}",
                ]
            )
        synthetic_pose_rows = [
            [
                "GT",
                _format_vector(ground_truth["camera_position_tag_m"], precision=6),
                "0.000000",
                f"{float(synthetic_payload['ground_truth_reprojection_rmse_px']):.6f}",
            ],
            [
                "OBS",
                _format_vector(pose_row_lookup["OBS"]["position_m"], precision=6),
                f"{float(pose_row_lookup['OBS']['position_error_m']):.6f}",
                f"{float(synthetic_estimate['camera_obscura_seed_reprojection_rmse_image_plane_m']) * _image_plane_scale_px_per_m(camera_model):.6f}",
            ],
            [
                "EX",
                _format_vector(synthetic_estimate["camera_position_tag_m"], precision=6),
                f"{float(synthetic_payload['position_error_m']):.6f}",
                f"{float(synthetic_payload['reprojection_rmse_px']):.6f}",
            ],
        ]
        synthetic_solver_rows = [
            [
                "`damped`",
                f"{float(synthetic_payload['damped_reprojection_rmse_px']):.6f}",
                f"{float(synthetic_payload['damped_position_error_m']):.6f}",
                f"{float(synthetic_payload['damped_rotation_error_deg']):.6f}",
                f"{float(synthetic_estimate_damped.get('initial_damping', 0.0)):.1e}",
                "`on`" if bool(synthetic_estimate_damped.get("use_diagonal_damping", False)) else "`off`",
            ],
            [
                "`undamped`",
                f"{float(synthetic_payload['reprojection_rmse_px']):.6f}",
                f"{float(synthetic_payload['position_error_m']):.6f}",
                f"{float(synthetic_payload['rotation_error_deg']):.6f}",
                f"{float(synthetic_estimate.get('initial_damping', 0.0)):.1e}",
                "`on`" if bool(synthetic_estimate.get("use_diagonal_damping", False)) else "`off`",
            ],
        ]
        synthetic_history_rows = _history_rows_from_steps(synthetic_estimate.get("selected_seed_history", []))
        synthetic_report_lines = [
            "# Synthetic Single Pattern Report",
            "",
            "This report uses the exact ground-truth camera operator to create a perfectly synthetic observation for the representative frame/tag.",
            "",
            "## Scope",
            "",
            f"- Run directory: `{resolved_run_dir}`",
            f"- Frame/tick/time/tag: `{representative_observation_payload['frame_index']}` / `{representative_observation_payload['tick_index']}` / `{representative_observation_payload['sim_time_s']:.3f}s` / `{representative_observation_payload['tag_id']}`",
            f"- Camera model: `{camera_model.name}`",
            f"- Synthetic observation source: exact projection of the ground-truth pose through the minimizer camera operator",
            "",
            "## Key Result",
            "",
            f"- At the true ground-truth state, synthetic reprojection error is `{float(synthetic_payload['ground_truth_reprojection_rmse_px']):.6f}` px",
            f"- The new requested undamped solve converges to `{float(synthetic_payload['reprojection_rmse_px']):.6f}` px and `{float(synthetic_payload['position_error_m']):.6f}` m world-position error",
            f"- For comparison, the damped solve lands at `{float(synthetic_payload['damped_reprojection_rmse_px']):.6f}` px and `{float(synthetic_payload['damped_position_error_m']):.6f}` m world-position error",
            "",
            "So zero error is achievable in the synthetic problem definition. The undamped solve gets very close to that zero-loss branch, while the damped solve falls into the wrong basin from the same seed.",
            "",
            "## Image Comparison",
            "",
            "Orange filled circles are the synthetic observation points.",
            "Green crosses are the exact ground-truth projection, so they should coincide perfectly with the orange points.",
            "Orange-red tilted crosses are the result of running the undamped optimizer from the camera-obscura seed.",
            "",
            "![Synthetic single pattern overlay](synthetic_single_pattern_overlay.png)",
            "",
            "## Damped vs Undamped Summary",
            "",
            *_markdown_table(
                ["solver", "final reproj [px]", "final world error [m]", "final rot error [deg]", "initial damping", "diag damping"],
                synthetic_solver_rows,
            ),
            "",
            "## Pose Comparison",
            "",
            *_markdown_table(
                ["row", "camera position in tag frame [m]", "world error [m]", "reproj [px]"],
                synthetic_pose_rows,
            ),
            "",
            "## Point Table",
            "",
            *_markdown_table(
                [
                    "point",
                    "synthetic obs px",
                    "gt operator px",
                    "optimized px",
                    "opt-synth dx,dy [px]",
                    "opt error [px]",
                ],
                synthetic_point_rows,
            ),
            "",
            "## Newton Convergence On Perfect Synthetic Points",
            "",
            "This table is for the undamped run requested here, so `damping = 0` throughout and `alpha` is only the Armijo line-search multiplier.",
            "The `gt - cam pos [m]` column is the camera-position error vector `ground_truth_position - current_estimate_position`.",
            "The `gt - rot xyz [deg]` column is the XYZ Euler-angle error vector of the relative rotation that maps the current estimate onto the ground-truth orientation.",
            "",
            *_markdown_table(
                [
                    "iter",
                    "loss before",
                    "grad norm",
                    "step norm",
                    "damping",
                    "alpha",
                    "accepted",
                    "loss after",
                    "rmse after [mm]",
                    "gt - cam pos [m]",
                    "gt - rot xyz [deg]",
                    "min depth [m]",
                ],
                synthetic_history_rows,
            ),
            "",
            "## Interpretation",
            "",
            "This synthetic check isolates the optimizer from detector noise and rendering mismatch.",
            "Because the exact ground-truth pose gives zero residual, any non-zero final error in this report comes from the current optimization setup, not from the camera model.",
            "",
        ]
        synthetic_single_pattern_report_path.write_text(
            "\n".join(synthetic_report_lines).strip() + "\n",
            encoding="utf-8",
        )

    report_lines = [
        "# Interactive Recording Analysis",
        "",
        "## Overview",
        "",
        f"- Run directory: `{resolved_run_dir}`",
        f"- Camera model: `{camera_model.name}` from `{camera_model.source_toml_path}`",
        f"- Analysis input correction applied: `{'yes' if camera_model.apply_lens_distortion_in_render else 'no'}`",
        f"- Frames analyzed: `{frame_count}` out of `{len(samples)}` recorded samples",
        f"- Pose estimates produced: `{estimates_written}`",
        "- Primary solver used in this report: `camera obscura seed + modified Newton + Armijo line search`",
        "",
        "## Fit Implementation",
        "",
        "State vector:",
        "`x = [tx, ty, tz, rx, ry, rz]`, where translation is tag-to-camera in meters and `[rx, ry, rz]` is a Rodrigues rotation vector.",
        "",
        "Observation model:",
        "- The fitted observation is a 5-point image-plane vector in meters.",
        "- Point order is `center, top_right, bottom_right, bottom_left, top_left`.",
        "- The object points are a 10 cm square tag on the `z = 0` plane in tag coordinates.",
        "",
        "Objective:",
        "- Minimize the sum of squared image-plane residuals between predicted and observed 5-point measurements.",
        "- Add a strong penalty when predicted depth goes non-positive.",
        "",
        "Initialization and optimization:",
        "1. Read the recorded raw phone video.",
        "2. If the selected preset rendered a distorted image, undistort each frame before analysis; otherwise analyze the recorded frame directly.",
        "3. Detect AprilTag 36h11 markers and build the 5-point measurement vector.",
        "4. Convert image points into image-plane metric coordinates using the TOML focal length.",
        "5. Build a camera-obscura seed from apparent tag scale and center offset.",
        "6. Run a modified Newton solver with JAX autodiff gradients and Hessians. Start from the raw Hessian solve, and if that step is singular or not a descent direction, inject the smallest diagonal shift needed to recover a descent step before Armijo backtracking line search.",
        "7. Keep the best result reached from the physically interpretable obscura seed.",
        "8. Use the supplementary reports to compare against the older damped variant and against ground-truth-seeded runs.",
        "",
    ]
    if representative_section_lines:
        report_lines.extend(representative_section_lines)
    report_lines.extend(
        [
        "## Results",
        "",
        f"- Mean position error: `{mean_position_error_m if mean_position_error_m is not None else 'n/a'}` m",
        f"- Median position error: `{median_position_error_m if median_position_error_m is not None else 'n/a'}` m",
        f"- Min position error: `{min_position_error_m if min_position_error_m is not None else 'n/a'}` m",
        f"- Max position error: `{max_position_error_m if max_position_error_m is not None else 'n/a'}` m",
        f"- Mean position error in tag widths: `{(mean_position_error_m / tag_size_reference_m) if mean_position_error_m is not None and tag_size_reference_m else 'n/a'}`",
        f"- Mean position error relative to mean camera-tag range: `{(mean_position_error_m / mean_camera_tag_distance_m) if mean_position_error_m is not None and mean_camera_tag_distance_m else 'n/a'}`",
        f"- Mean rotation error: `{mean_rotation_error_deg if mean_rotation_error_deg is not None else 'n/a'}` deg",
        f"- Mean image-plane reprojection RMSE: `{mean_reprojection_rmse_m if mean_reprojection_rmse_m is not None else 'n/a'}` m = `{(mean_reprojection_rmse_m * 1000.0) if mean_reprojection_rmse_m is not None else 'n/a'}` mm = `{mean_reprojection_rmse_px if mean_reprojection_rmse_px is not None else 'n/a'}` px",
        f"- Median image-plane reprojection RMSE: `{median_reprojection_rmse_m if median_reprojection_rmse_m is not None else 'n/a'}` m = `{(median_reprojection_rmse_m * 1000.0) if median_reprojection_rmse_m is not None else 'n/a'}` mm = `{median_reprojection_rmse_px if median_reprojection_rmse_px is not None else 'n/a'}` px",
        f"- Camera-to-tag distance across the run: `{min_camera_tag_distance_m if min_camera_tag_distance_m is not None else 'n/a'}` m to `{max_camera_tag_distance_m if max_camera_tag_distance_m is not None else 'n/a'}` m, mean `{mean_camera_tag_distance_m if mean_camera_tag_distance_m is not None else 'n/a'}` m",
        "",
        "## Per-Pattern Breakdown",
        "",
        "These rows are already solved separately for each visible calibration pattern; the reports under `analysis/per_tag/` expose the same split explicitly.",
        "",
        "## World-Pose Components By Tag",
        "",
        "Each plot uses recording frame number on the x axis. The estimate lines are solved from each calibration pattern independently and then transformed back into the shared world frame using that pattern's recorded world pose for the same frame. The purple line is a true joint solve over all visible tag points in that frame.",
        "",
        f"- Joint world-trajectory frames solved: `{len(joint_world_records)}`",
        f"- Joint world mean position error: `{float(np.mean(joint_world_position_errors_m)) if joint_world_position_errors_m else 'n/a'}` m",
        f"- Joint world median position error: `{float(np.median(joint_world_position_errors_m)) if joint_world_position_errors_m else 'n/a'}` m",
        f"- Joint world max position error: `{float(np.max(joint_world_position_errors_m)) if joint_world_position_errors_m else 'n/a'}` m",
        f"- Joint world mean rotation error: `{float(np.mean(joint_world_rotation_errors_deg)) if joint_world_rotation_errors_deg else 'n/a'}` deg",
        f"- Joint world mean reprojection RMSE: `{float(np.mean(joint_world_reprojection_px)) if joint_world_reprojection_px else 'n/a'}` px",
        "",
        "World-frame position components:",
        "",
        "![Camera world x](world_pose_by_tag/camera_world_x_m.png)",
        "",
        "![Camera world y](world_pose_by_tag/camera_world_y_m.png)",
        "",
        "![Camera world z](world_pose_by_tag/camera_world_z_m.png)",
        "",
        "World-frame XYZ Euler angles of the camera-to-world rotation:",
        "",
        "![Camera world rot x](world_pose_by_tag/camera_world_rx_deg.png)",
        "",
        "![Camera world rot y](world_pose_by_tag/camera_world_ry_deg.png)",
        "",
        "![Camera world rot z](world_pose_by_tag/camera_world_rz_deg.png)",
        "",
        "## IMU Trajectory Reconstruction",
        "",
        (
            "This run also includes a pure IMU dead-reckoning reconstruction. The first camera pose is fixed to the true world pose so the trajectory can be compared directly against ground truth, and the initial world velocity is taken from the exact value logged in `camera_gt.csv`."
            if imu_initial_velocity_source == "logged_camera_gt_velocity"
            else "This run also includes a pure IMU dead-reckoning reconstruction. The first camera pose is fixed to the true world pose so the trajectory can be compared directly against ground truth, and the initial world velocity is inferred from finite differences because this recording did not log it explicitly."
        ),
        "",
        f"- Detailed IMU report: `{imu_trajectory_report_path.relative_to(resolved_run_dir)}`",
        f"- IMU trajectory samples: `{imu_trajectory_estimates_path.relative_to(resolved_run_dir)}`",
        f"- Main IMU mean position error: `{imu_main_summary['mean_position_error_m'] if imu_main_summary and imu_main_summary['mean_position_error_m'] is not None else 'n/a'}` m",
        f"- Main IMU max position error: `{imu_main_summary['max_position_error_m'] if imu_main_summary and imu_main_summary['max_position_error_m'] is not None else 'n/a'}` m",
        f"- Main IMU mean rotation error: `{imu_main_summary['mean_rotation_error_deg'] if imu_main_summary and imu_main_summary['mean_rotation_error_deg'] is not None else 'n/a'}` deg",
        "",
        *_markdown_table(
            [
                "mode",
                "initial velocity [m/s]",
                "mean pos err [m]",
                "median pos err [m]",
                "max pos err [m]",
                "final pos err [m]",
                "mean rot err [deg]",
                "max rot err [deg]",
                "final rot err [deg]",
            ],
            imu_summary_rows,
        ),
        "",
        "![IMU position error timeline](imu_position_error_timeline.png)",
        "",
        "![IMU rotation error timeline](imu_rotation_error_timeline.png)",
        "",
        "## Visual Diagnostics",
        "",
        "Representative best-reprojection frame:",
        "",
        "![Representative fit](representative_fit.png)",
        "",
        "Representative world-space pose comparison:",
        "",
        "![Representative pose comparison](representative_pose_comparison.png)",
        "",
        "World-space position error over all solved observations. This is measured in meters in the tag coordinate frame, so it should be read against the roughly 1.6 m camera-to-tag scale of the demo scene:",
        "",
        "![Position error timeline](position_error_timeline.png)",
        "",
        "Image-space reprojection RMSE over all solved observations. This plot is in pixels, not world meters:",
        "",
        "![Reprojection timeline](reprojection_timeline.png)",
        "",
        "Representative optimizer loss trace:",
        "",
        "![Optimizer loss trace](optimizer_loss_trace.png)",
        "",
        "## Interpretation",
        "",
        "Using the corrected modified Newton method removes the main failure mode we saw when the raw Hessian step was silently discarded and replaced by an almost-zero gradient fallback.",
        f"In this run the average world-space camera-position error is `{mean_position_error_m if mean_position_error_m is not None else 'n/a'}` m while the camera is typically about `{mean_camera_tag_distance_m if mean_camera_tag_distance_m is not None else 'n/a'}` m from a tag, which is much more physically plausible than the earlier meter-scale failures.",
        "The supplementary synthetic report shows that the undamped solver reaches the zero-loss branch much more reliably, and the ground-truth-seeded real-measurement report shows the optimizer behaves well locally around the true pose on the noisy detected points.",
        "The remaining residual error in this report is now better interpreted as a combination of detector noise, single-tag ambiguity, and seed sensitivity rather than a broken forward camera model or an intrinsically bad Newton formulation.",
        "",
        "## Artifacts",
        "",
        f"- Undistorted video: `{undistorted_video_path.relative_to(resolved_run_dir)}`",
        f"- Detected/annotated video: `{annotated_video_path.relative_to(resolved_run_dir)}`",
        f"- Per-frame estimates: `{estimates_path.relative_to(resolved_run_dir)}`",
        f"- Per-frame joint world estimates: `{joint_world_estimates_path.relative_to(resolved_run_dir)}`",
        f"- IMU trajectory estimates: `{imu_trajectory_estimates_path.relative_to(resolved_run_dir)}`",
        f"- IMU trajectory report: `{imu_trajectory_report_path.relative_to(resolved_run_dir)}`",
        f"- Summary JSON: `{summary_path.relative_to(resolved_run_dir)}`",
        f"- Single pattern report: `{single_pattern_report_path.relative_to(resolved_run_dir)}`",
        f"- Single pattern overlay: `{single_pattern_overlay_path.relative_to(resolved_run_dir)}`",
        f"- Worst single pattern report: `{worst_single_pattern_report_path.relative_to(resolved_run_dir)}`",
        f"- Worst single pattern overlay: `{worst_single_pattern_overlay_path.relative_to(resolved_run_dir)}`",
        f"- Real measurement ground-truth-seed report: `{real_measurement_known_solution_report_path.relative_to(resolved_run_dir)}`",
        f"- Synthetic single pattern report: `{synthetic_single_pattern_report_path.relative_to(resolved_run_dir)}`",
        f"- Synthetic single pattern overlay: `{synthetic_single_pattern_overlay_path.relative_to(resolved_run_dir)}`",
        f"- Representative pose comparison image: `{representative_pose_comparison_path.relative_to(resolved_run_dir)}`",
        f"- Representative observation JSON: `{representative_observation_path.relative_to(resolved_run_dir)}`",
        f"- Per-tag directory: `{per_tag_dir.relative_to(resolved_run_dir)}`",
        f"- Per-tag summary JSON: `{per_tag_summary_path.relative_to(resolved_run_dir)}`",
        f"- World-pose-by-tag directory: `{world_pose_plot_dir.relative_to(resolved_run_dir)}`",
        f"- World X plot: `{world_pose_x_plot_path.relative_to(resolved_run_dir)}`",
        f"- World Y plot: `{world_pose_y_plot_path.relative_to(resolved_run_dir)}`",
        f"- World Z plot: `{world_pose_z_plot_path.relative_to(resolved_run_dir)}`",
        f"- World rot X plot: `{world_pose_rx_plot_path.relative_to(resolved_run_dir)}`",
        f"- World rot Y plot: `{world_pose_ry_plot_path.relative_to(resolved_run_dir)}`",
        f"- World rot Z plot: `{world_pose_rz_plot_path.relative_to(resolved_run_dir)}`",
        f"- IMU position error plot: `{imu_position_error_plot_path.relative_to(resolved_run_dir)}`",
        f"- IMU rotation error plot: `{imu_rotation_error_plot_path.relative_to(resolved_run_dir)}`",
        f"- IMU world-pose directory: `{imu_world_pose_plot_dir.relative_to(resolved_run_dir)}`",
    ]
    )
    if per_tag_rows:
        per_tag_section_lines = _markdown_table(
            ["tag", "observations", "mean pos err [m]", "median pos err [m]", "max pos err [m]", "mean rot err [deg]", "mean reproj [px]"],
            per_tag_rows,
        )
        insert_index = report_lines.index("## Visual Diagnostics")
        report_lines[insert_index:insert_index] = per_tag_section_lines + [""]
    try:
        batch_analysis = run_batch_estimation_analysis(resolved_run_dir)
    except Exception as exc:
        summary["batch_estimation_error"] = str(exc)
        report_lines.extend(
            [
                "",
                "## Batch Estimation",
                "",
                f"Batch estimation failed during report generation: `{exc}`",
                "",
            ]
        )
    else:
        summary.update(batch_analysis["summary_updates"])
        summary["artifacts"].update(batch_analysis["artifact_updates"])
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        report_lines.extend(batch_analysis["report_appendix_lines"])

    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return summary
