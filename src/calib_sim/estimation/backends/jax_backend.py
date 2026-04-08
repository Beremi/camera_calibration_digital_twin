"""JAX-backed block LM solvers for visual and visual-inertial batch estimation."""

from __future__ import annotations

import math
import time
from functools import lru_cache
from typing import Any

import numpy as np

from calib_sim.common.inertial import camera_pose_from_imu_pose, imu_pose_from_camera_pose
from calib_sim.estimation._geometry import normalize_rvec, pose_components_from_vector, pose_vector_from_components, rotation_matrix_from_rpy_deg
from calib_sim.estimation.graph_build import BatchGraphDefinition, ImuFactorEntry
from calib_sim.estimation.noise_models import ImuNoisePreset, VisionNoisePreset
from calib_sim.estimation.types import BatchCalibrationDataset, BatchSolveResult, InitialGuess


def _import_jax() -> tuple[Any, Any]:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    return jax, jnp


def _rotation_matrix_from_rvec_np(rvec: np.ndarray) -> np.ndarray:
    import cv2

    rotation_matrix, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return rotation_matrix.astype(np.float64)


def _pose_update(pose_vector: np.ndarray, delta_vector: np.ndarray) -> np.ndarray:
    updated = np.asarray(pose_vector, dtype=np.float64).reshape(6).copy()
    updated[:3] += np.asarray(delta_vector[:3], dtype=np.float64)
    updated[3:] = normalize_rvec(updated[3:] + np.asarray(delta_vector[3:], dtype=np.float64))
    return updated


@lru_cache(maxsize=32)
def _compiled_visual_factor(
    fx_px: float,
    fy_px: float,
    cx_px: float,
    cy_px: float,
    apply_distortion: bool,
    distortion_coefficients: tuple[float, float, float, float, float],
    tag_size_m: float,
) -> tuple[Any, Any]:
    jax, jnp = _import_jax()
    tag_local_corners = jnp.asarray(
        np.array(
            [
                [-tag_size_m * 0.5, -tag_size_m * 0.5, 0.0],
                [tag_size_m * 0.5, -tag_size_m * 0.5, 0.0],
                [tag_size_m * 0.5, tag_size_m * 0.5, 0.0],
                [-tag_size_m * 0.5, tag_size_m * 0.5, 0.0],
            ],
            dtype=np.float64,
        ),
        dtype=jnp.float64,
    )
    k1, k2, k3, p1, p2 = [float(value) for value in distortion_coefficients]

    def skew(vector: Any) -> Any:
        vx, vy, vz = vector
        return jnp.array([[0.0, -vz, vy], [vz, 0.0, -vx], [-vy, vx, 0.0]], dtype=jnp.float64)

    def rotation_matrix_rvec(rvec_rad: Any) -> Any:
        theta2 = jnp.dot(rvec_rad, rvec_rad)
        theta = jnp.sqrt(theta2 + 1e-12)
        k_matrix = skew(rvec_rad)
        identity = jnp.eye(3, dtype=jnp.float64)
        a_coeff = jnp.sin(theta) / theta
        b_coeff = (1.0 - jnp.cos(theta)) / (theta * theta)
        return identity + a_coeff * k_matrix + b_coeff * (k_matrix @ k_matrix)

    def project(points_camera_xyz: Any) -> Any:
        z = jnp.maximum(points_camera_xyz[:, 2], 1e-6)
        x = points_camera_xyz[:, 0] / z
        y_down = -(points_camera_xyz[:, 1] / z)
        if apply_distortion:
            r2 = x * x + y_down * y_down
            radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
            x_dist = x * radial + 2.0 * p1 * x * y_down + p2 * (r2 + 2.0 * x * x)
            y_dist = y_down * radial + p1 * (r2 + 2.0 * y_down * y_down) + 2.0 * p2 * x * y_down
        else:
            x_dist = x
            y_dist = y_down
        u = fx_px * x_dist + cx_px
        v = fy_px * y_dist + cy_px
        return jnp.stack((u, v), axis=1)

    def residual_fn(local_state: Any, observed_corners_px: Any) -> Any:
        camera_position = local_state[:3]
        camera_rotation = rotation_matrix_rvec(local_state[3:6])
        tag_position = local_state[6:9]
        tag_rotation = rotation_matrix_rvec(local_state[9:12])
        world_points = tag_position + (tag_rotation @ tag_local_corners.T).T
        points_camera = (camera_rotation.T @ (world_points - camera_position).T).T
        predicted_corners = project(points_camera)
        return (predicted_corners - observed_corners_px).reshape(-1)

    return jax.jit(residual_fn), jax.jit(jax.jacfwd(residual_fn))


@lru_cache(maxsize=32)
def _compiled_visual_imu_factor(
    fx_px: float,
    fy_px: float,
    cx_px: float,
    cy_px: float,
    apply_distortion: bool,
    distortion_coefficients: tuple[float, float, float, float, float],
    tag_size_m: float,
    imu_translation_m: tuple[float, float, float],
    imu_rpy_deg: tuple[float, float, float],
) -> tuple[Any, Any]:
    jax, jnp = _import_jax()
    tag_local_corners = jnp.asarray(
        np.array(
            [
                [-tag_size_m * 0.5, -tag_size_m * 0.5, 0.0],
                [tag_size_m * 0.5, -tag_size_m * 0.5, 0.0],
                [tag_size_m * 0.5, tag_size_m * 0.5, 0.0],
                [-tag_size_m * 0.5, tag_size_m * 0.5, 0.0],
            ],
            dtype=np.float64,
        ),
        dtype=jnp.float64,
    )
    k1, k2, k3, p1, p2 = [float(value) for value in distortion_coefficients]
    rotation_ci = rotation_matrix_from_rpy_deg(imu_rpy_deg)
    rotation_ic = rotation_ci.T
    translation_ic = -(rotation_ic @ np.asarray(imu_translation_m, dtype=np.float64).reshape(3))
    rotation_ic_jax = jnp.asarray(rotation_ic, dtype=jnp.float64)
    translation_ic_jax = jnp.asarray(translation_ic, dtype=jnp.float64)

    def skew(vector: Any) -> Any:
        vx, vy, vz = vector
        return jnp.array([[0.0, -vz, vy], [vz, 0.0, -vx], [-vy, vx, 0.0]], dtype=jnp.float64)

    def rotation_matrix_rvec(rvec_rad: Any) -> Any:
        theta2 = jnp.dot(rvec_rad, rvec_rad)
        theta = jnp.sqrt(theta2 + 1e-12)
        k_matrix = skew(rvec_rad)
        identity = jnp.eye(3, dtype=jnp.float64)
        a_coeff = jnp.sin(theta) / theta
        b_coeff = (1.0 - jnp.cos(theta)) / (theta * theta)
        return identity + a_coeff * k_matrix + b_coeff * (k_matrix @ k_matrix)

    def project(points_camera_xyz: Any) -> Any:
        z = jnp.maximum(points_camera_xyz[:, 2], 1e-6)
        x = points_camera_xyz[:, 0] / z
        y_down = -(points_camera_xyz[:, 1] / z)
        if apply_distortion:
            r2 = x * x + y_down * y_down
            radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
            x_dist = x * radial + 2.0 * p1 * x * y_down + p2 * (r2 + 2.0 * x * x)
            y_dist = y_down * radial + p1 * (r2 + 2.0 * y_down * y_down) + 2.0 * p2 * x * y_down
        else:
            x_dist = x
            y_dist = y_down
        u = fx_px * x_dist + cx_px
        v = fy_px * y_dist + cy_px
        return jnp.stack((u, v), axis=1)

    def residual_fn(local_state: Any, observed_corners_px: Any) -> Any:
        imu_position = local_state[:3]
        imu_rotation = rotation_matrix_rvec(local_state[3:6])
        tag_pose = local_state[9:15]
        camera_position = imu_position + imu_rotation @ translation_ic_jax
        camera_rotation = imu_rotation @ rotation_ic_jax
        tag_position = tag_pose[:3]
        tag_rotation = rotation_matrix_rvec(tag_pose[3:6])
        world_points = tag_position + (tag_rotation @ tag_local_corners.T).T
        points_camera = (camera_rotation.T @ (world_points - camera_position).T).T
        predicted_corners = project(points_camera)
        return (predicted_corners - observed_corners_px).reshape(-1)

    return jax.jit(residual_fn), jax.jit(jax.jacfwd(residual_fn))


@lru_cache(maxsize=32)
def _compiled_imu_factor(sequence_length: int) -> tuple[Any, Any]:
    jax, jnp = _import_jax()

    def skew(vector: Any) -> Any:
        vx, vy, vz = vector
        return jnp.array([[0.0, -vz, vy], [vz, 0.0, -vx], [-vy, vx, 0.0]], dtype=jnp.float64)

    def rotation_matrix_rvec(rvec_rad: Any) -> Any:
        theta2 = jnp.dot(rvec_rad, rvec_rad)
        theta = jnp.sqrt(theta2 + 1e-12)
        k_matrix = skew(rvec_rad)
        identity = jnp.eye(3, dtype=jnp.float64)
        a_coeff = jnp.sin(theta) / theta
        b_coeff = (1.0 - jnp.cos(theta)) / (theta * theta)
        return identity + a_coeff * k_matrix + b_coeff * (k_matrix @ k_matrix)

    def rvec_from_rotation_matrix(rotation_matrix: Any) -> Any:
        trace = jnp.trace(rotation_matrix)
        cosine = jnp.clip((trace - 1.0) * 0.5, -1.0, 1.0)
        theta = jnp.arccos(cosine)
        sine = jnp.sqrt(jnp.maximum(1.0 - cosine * cosine, 1e-12))
        vee = jnp.array(
            [
                rotation_matrix[2, 1] - rotation_matrix[1, 2],
                rotation_matrix[0, 2] - rotation_matrix[2, 0],
                rotation_matrix[1, 0] - rotation_matrix[0, 1],
            ],
            dtype=jnp.float64,
        )
        scale = jnp.where(theta < 1e-6, 0.5 + (theta * theta) / 12.0, theta / (2.0 * sine))
        return vee * scale

    def residual_fn(local_state: Any, gyro_packets: Any, accel_packets: Any, dt_packets: Any, gravity_world: Any) -> Any:
        position_i = local_state[:3]
        rotation_i = rotation_matrix_rvec(local_state[3:6])
        velocity_i = local_state[6:9]
        position_j = local_state[9:12]
        rotation_j = rotation_matrix_rvec(local_state[12:15])
        velocity_j = local_state[15:18]
        gyro_bias = local_state[18:21]
        accel_bias = local_state[21:24]
        propagated_rotation = rotation_i
        total_dt = jnp.sum(dt_packets)
        accumulated_world_acceleration = jnp.zeros(3, dtype=jnp.float64)

        def body(loop_index: int, carry: Any) -> Any:
            current_rotation, current_world_acceleration = carry
            dt_s = dt_packets[loop_index]
            corrected_gyro = gyro_packets[loop_index] - gyro_bias
            corrected_accel = accel_packets[loop_index] - accel_bias
            delta_rotation = rotation_matrix_rvec(corrected_gyro * dt_s)
            next_rotation = current_rotation @ delta_rotation
            acceleration_world = next_rotation @ corrected_accel + gravity_world
            next_world_acceleration = current_world_acceleration + acceleration_world * dt_s
            return next_rotation, next_world_acceleration

        propagated_rotation, accumulated_world_acceleration = jax.lax.fori_loop(
            0,
            sequence_length,
            body,
            (propagated_rotation, accumulated_world_acceleration),
        )
        mean_world_acceleration = accumulated_world_acceleration / jnp.maximum(total_dt, 1e-9)
        propagated_velocity = velocity_i + mean_world_acceleration * total_dt
        propagated_position = position_i + propagated_velocity * total_dt
        rotation_residual = rvec_from_rotation_matrix(propagated_rotation.T @ rotation_j)
        return jnp.concatenate(
            (
                propagated_position - position_j,
                rotation_residual,
                propagated_velocity - velocity_j,
            )
        )

    return jax.jit(residual_fn), jax.jit(jax.jacfwd(residual_fn))


def _visual_factor_weighted(
    residual: np.ndarray,
    jacobian: np.ndarray,
    *,
    sigma_px: float,
    huber_delta_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    residual_pairs = np.asarray(residual, dtype=np.float64).reshape(-1, 2)
    jacobian_pairs = np.asarray(jacobian, dtype=np.float64).reshape(-1, 2, jacobian.shape[1])
    weighted_residual_pairs = []
    weighted_jacobian_pairs = []
    sigma_px = max(float(sigma_px), 1e-6)
    for pair_index in range(residual_pairs.shape[0]):
        pair_residual = residual_pairs[pair_index]
        pair_jacobian = jacobian_pairs[pair_index]
        residual_norm = float(np.linalg.norm(pair_residual))
        huber_weight = 1.0 if residual_norm <= huber_delta_px else max(huber_delta_px / max(residual_norm, 1e-9), 1e-9)
        scale = math.sqrt(huber_weight) / sigma_px
        weighted_residual_pairs.append(pair_residual * scale)
        weighted_jacobian_pairs.append(pair_jacobian * scale)
    return np.asarray(weighted_residual_pairs, dtype=np.float64).reshape(-1), np.asarray(weighted_jacobian_pairs, dtype=np.float64).reshape(jacobian.shape)


def _imu_sqrt_information(delta_time_s: float, imu_noise: ImuNoisePreset) -> np.ndarray:
    delta_time_s = max(float(delta_time_s), 1e-9)
    gyro_noise = float(np.mean(np.asarray(imu_noise.gyro_noise_std, dtype=np.float64)))
    accel_noise = float(np.mean(np.asarray(imu_noise.accel_noise_std, dtype=np.float64)))
    likelihood_scale = max(float(getattr(imu_noise, "likelihood_scale", 1.0)), 1e-9)
    # The simulator and solver use a simplified discrete IMU model rather than a
    # full continuous-time rigid-body model with lever-arm terms. Floors keep the
    # ideal preset from becoming unrealistically overconfident relative to that
    # approximation error.
    rotation_std = max(
        gyro_noise * math.sqrt(delta_time_s),
        float(getattr(imu_noise, "rotation_std_floor", 5.0e-2)),
    ) * likelihood_scale
    velocity_std = max(
        accel_noise * math.sqrt(delta_time_s),
        float(getattr(imu_noise, "velocity_std_floor", 2.0e-1)),
    ) * likelihood_scale
    position_std = max(
        velocity_std * delta_time_s + 0.5 * accel_noise * delta_time_s * delta_time_s * likelihood_scale,
        float(getattr(imu_noise, "position_std_floor", 2.0e-2)) * likelihood_scale,
    )
    diagonal = np.array(
        [
            1.0 / position_std,
            1.0 / position_std,
            1.0 / position_std,
            1.0 / rotation_std,
            1.0 / rotation_std,
            1.0 / rotation_std,
            1.0 / velocity_std,
            1.0 / velocity_std,
            1.0 / velocity_std,
        ],
        dtype=np.float64,
    )
    return np.diag(diagonal)


def _frame_and_tag_index_maps(graph: BatchGraphDefinition) -> tuple[dict[int, int], dict[int, int]]:
    return (
        {int(frame_index): idx for idx, frame_index in enumerate(graph.frame_indices)},
        {int(tag_id): idx for idx, tag_id in enumerate(graph.tag_ids)},
    )


def _pack_visual_state(graph: BatchGraphDefinition, init: InitialGuess, *, fixed_tag_pose_vectors: dict[int, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
    camera_state = np.stack([np.asarray(init.camera_pose_vectors[int(frame_index)], dtype=np.float64).reshape(6) for frame_index in graph.frame_indices], axis=0)
    if fixed_tag_pose_vectors is None:
        tag_state = np.stack([np.asarray(init.tag_pose_vectors[int(tag_id)], dtype=np.float64).reshape(6) for tag_id in graph.tag_ids], axis=0)
    else:
        tag_state = np.stack([np.asarray(fixed_tag_pose_vectors[int(tag_id)], dtype=np.float64).reshape(6) for tag_id in graph.tag_ids], axis=0)
    return camera_state, tag_state


def _pack_imu_state(
    dataset: BatchCalibrationDataset,
    graph: BatchGraphDefinition,
    init: InitialGuess,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rotation_ci = rotation_matrix_from_rpy_deg(tuple(float(value) for value in dataset.device_config.mount.imu_rpy_deg))
    imu_state = []
    for frame_index in graph.frame_indices:
        camera_pose_vector = np.asarray(init.camera_pose_vectors[int(frame_index)], dtype=np.float64).reshape(6)
        camera_position_world_m, camera_rotation_wc = pose_components_from_vector(camera_pose_vector)
        imu_position_world_m, imu_rotation_wi = imu_pose_from_camera_pose(
            camera_position_world_m=camera_position_world_m,
            camera_rotation_wc=camera_rotation_wc,
            imu_translation_camera_m=dataset.device_config.mount.imu_translation_m,
            rotation_ci=rotation_ci,
        )
        pose_vector = pose_vector_from_components(imu_position_world_m, imu_rotation_wi)
        velocity = np.asarray(init.velocity_world_mps_by_frame.get(int(frame_index), np.zeros(3, dtype=np.float64)), dtype=np.float64).reshape(3)
        imu_state.append(np.concatenate((pose_vector, velocity), axis=0))
    tag_state = np.stack([np.asarray(init.tag_pose_vectors[int(tag_id)], dtype=np.float64).reshape(6) for tag_id in graph.tag_ids], axis=0)
    global_bias = np.concatenate(
        (
            np.asarray(init.global_gyro_bias_rps, dtype=np.float64).reshape(3),
            np.asarray(init.global_accel_bias_mps2, dtype=np.float64).reshape(3),
        ),
        axis=0,
    )
    return np.asarray(imu_state, dtype=np.float64), tag_state, global_bias


def solve_visual_block_lm(
    *,
    dataset: BatchCalibrationDataset,
    graph: BatchGraphDefinition,
    init: InitialGuess,
    vision_noise: VisionNoisePreset,
    fixed_tag_pose_vectors: dict[int, np.ndarray] | None = None,
    max_iterations: int = 18,
    initial_damping: float = 1e-3,
    variant: str = "visual_only_unknown_map",
) -> BatchSolveResult:
    camera_state, tag_state = _pack_visual_state(graph, init, fixed_tag_pose_vectors=fixed_tag_pose_vectors)
    frame_index_map, tag_index_map = _frame_and_tag_index_maps(graph)
    state_dimension = camera_state.shape[0] * 6 + (0 if fixed_tag_pose_vectors is not None else tag_state.shape[0] * 6)
    initial_cost = None
    damping = float(initial_damping)
    iteration_timings_ms: list[float] = []
    diagnostics: dict[str, Any] = {"iteration_costs": [], "variant": variant}
    final_hessian = None
    final_gradient = None
    final_whitened_visual_sq_per_corner = None

    for iteration_index in range(max_iterations):
        started_at = time.perf_counter()
        hessian = np.zeros((state_dimension, state_dimension), dtype=np.float64)
        gradient = np.zeros(state_dimension, dtype=np.float64)
        total_cost = 0.0
        visual_residual_norms_px: list[float] = []
        visual_whitened_sq = 0.0
        visual_corner_count = 0

        for factor in graph.visual_factors:
            frame_slot = frame_index_map[int(factor.frame_index)]
            tag_slot = tag_index_map[int(factor.tag_id)]
            local_state = np.concatenate((camera_state[frame_slot], tag_state[tag_slot]), axis=0)
            residual_fn, jacobian_fn = _compiled_visual_factor(
                float(dataset.camera_model.fx_px),
                float(dataset.camera_model.fy_px),
                float(dataset.camera_model.cx_px),
                float(dataset.camera_model.cy_px),
                bool(dataset.camera_model.apply_lens_distortion_in_render),
                tuple(float(value) for value in dataset.camera_model.distortion_coefficients),
                float(factor.tag_size_m),
            )
            observed = np.asarray(factor.observed_corners_px, dtype=np.float64)
            residual = np.asarray(residual_fn(local_state, observed), dtype=np.float64)
            jacobian = np.asarray(jacobian_fn(local_state, observed), dtype=np.float64)
            weighted_residual, weighted_jacobian = _visual_factor_weighted(
                residual,
                jacobian,
                sigma_px=float(vision_noise.corner_noise_std_px),
                huber_delta_px=max(float(vision_noise.corner_noise_std_px) * 3.0, 1.0),
            )
            total_cost += 0.5 * float(weighted_residual @ weighted_residual)
            visual_whitened_sq += float(weighted_residual @ weighted_residual)
            visual_corner_count += residual.reshape(-1, 2).shape[0]
            residual_pairs = residual.reshape(-1, 2)
            visual_residual_norms_px.extend(float(np.linalg.norm(pair)) for pair in residual_pairs)

            frame_slice = slice(frame_slot * 6, (frame_slot + 1) * 6)
            jacobian_camera = weighted_jacobian[:, :6]
            hessian[frame_slice, frame_slice] += jacobian_camera.T @ jacobian_camera
            gradient[frame_slice] += jacobian_camera.T @ weighted_residual

            if fixed_tag_pose_vectors is None:
                tag_base = camera_state.shape[0] * 6
                tag_slice = slice(tag_base + tag_slot * 6, tag_base + (tag_slot + 1) * 6)
                jacobian_tag = weighted_jacobian[:, 6:12]
                hessian[tag_slice, tag_slice] += jacobian_tag.T @ jacobian_tag
                gradient[tag_slice] += jacobian_tag.T @ weighted_residual
                cross_term = jacobian_camera.T @ jacobian_tag
                hessian[frame_slice, tag_slice] += cross_term
                hessian[tag_slice, frame_slice] += cross_term.T

        for prior in graph.pose_priors:
            if prior.variable_kind != "camera_pose":
                continue
            frame_slot = frame_index_map[int(prior.index)]
            frame_slice = slice(frame_slot * 6, (frame_slot + 1) * 6)
            residual = np.asarray(prior.sqrt_information @ (camera_state[frame_slot] - prior.mean_vector), dtype=np.float64)
            jacobian = np.asarray(prior.sqrt_information, dtype=np.float64)
            hessian[frame_slice, frame_slice] += jacobian.T @ jacobian
            gradient[frame_slice] += jacobian.T @ residual
            total_cost += 0.5 * float(residual @ residual)

        if initial_cost is None:
            initial_cost = total_cost
        diagnostics["iteration_costs"].append(total_cost)
        final_hessian = hessian.copy()
        final_gradient = gradient.copy()
        final_whitened_visual_sq_per_corner = visual_whitened_sq / max(visual_corner_count, 1)

        step_success = False
        for _attempt in range(20):
            trial_hessian = hessian + damping * np.eye(state_dimension, dtype=np.float64)
            delta = -np.linalg.solve(trial_hessian, gradient)
            trial_camera_state = camera_state.copy()
            trial_tag_state = tag_state.copy()
            for frame_slot in range(camera_state.shape[0]):
                frame_slice = slice(frame_slot * 6, (frame_slot + 1) * 6)
                trial_camera_state[frame_slot] = _pose_update(camera_state[frame_slot], delta[frame_slice])
            if fixed_tag_pose_vectors is None:
                tag_base = camera_state.shape[0] * 6
                for tag_slot in range(tag_state.shape[0]):
                    tag_slice = slice(tag_base + tag_slot * 6, tag_base + (tag_slot + 1) * 6)
                    trial_tag_state[tag_slot] = _pose_update(tag_state[tag_slot], delta[tag_slice])

            trial_cost = 0.0
            for factor in graph.visual_factors:
                frame_slot = frame_index_map[int(factor.frame_index)]
                tag_slot = tag_index_map[int(factor.tag_id)]
                local_state = np.concatenate((trial_camera_state[frame_slot], trial_tag_state[tag_slot]), axis=0)
                residual_fn, jacobian_fn = _compiled_visual_factor(
                    float(dataset.camera_model.fx_px),
                    float(dataset.camera_model.fy_px),
                    float(dataset.camera_model.cx_px),
                    float(dataset.camera_model.cy_px),
                    bool(dataset.camera_model.apply_lens_distortion_in_render),
                    tuple(float(value) for value in dataset.camera_model.distortion_coefficients),
                    float(factor.tag_size_m),
                )
                residual = np.asarray(residual_fn(local_state, np.asarray(factor.observed_corners_px, dtype=np.float64)), dtype=np.float64)
                jacobian = np.asarray(jacobian_fn(local_state, np.asarray(factor.observed_corners_px, dtype=np.float64)), dtype=np.float64)
                weighted_residual, _ = _visual_factor_weighted(
                    residual,
                    jacobian,
                    sigma_px=float(vision_noise.corner_noise_std_px),
                    huber_delta_px=max(float(vision_noise.corner_noise_std_px) * 3.0, 1.0),
                )
                trial_cost += 0.5 * float(weighted_residual @ weighted_residual)
            for prior in graph.pose_priors:
                if prior.variable_kind != "camera_pose":
                    continue
                frame_slot = frame_index_map[int(prior.index)]
                residual = np.asarray(prior.sqrt_information @ (trial_camera_state[frame_slot] - prior.mean_vector), dtype=np.float64)
                trial_cost += 0.5 * float(residual @ residual)
            if trial_cost < total_cost:
                camera_state = trial_camera_state
                tag_state = trial_tag_state
                damping = max(damping * 0.5, 1e-6)
                step_success = True
                break
            damping *= 10.0
        iteration_timings_ms.append((time.perf_counter() - started_at) * 1000.0)
        if not step_success:
            break
        if float(np.linalg.norm(delta)) < 1e-6:
            break

    camera_pose_vectors = {int(frame_index): camera_state[frame_index_map[int(frame_index)]].copy() for frame_index in graph.frame_indices}
    tag_pose_vectors = {int(tag_id): tag_state[tag_index_map[int(tag_id)]].copy() for tag_id in graph.tag_ids}
    return BatchSolveResult(
        stage="visual_only",
        variant=variant,
        success=True,
        solver_name="jax_block_lm_visual",
        iterations=len(diagnostics["iteration_costs"]),
        final_cost=float(diagnostics["iteration_costs"][-1]),
        initial_cost=float(initial_cost if initial_cost is not None else 0.0),
        damping=float(damping),
        camera_pose_vectors=camera_pose_vectors,
        tag_pose_vectors=tag_pose_vectors,
        hessian=final_hessian,
        gradient=final_gradient,
        diagnostics={
            **diagnostics,
            "mean_iteration_time_ms": float(np.mean(iteration_timings_ms)) if iteration_timings_ms else None,
            "mean_visual_residual_norm_px": float(np.mean(visual_residual_norms_px)) if visual_residual_norms_px else None,
            "mean_whitened_visual_sq_residual_per_corner": final_whitened_visual_sq_per_corner,
            "factor_breakdown": {
                "visual_factor_count": len(graph.visual_factors),
                "visual_corner_count": len(graph.visual_factors) * 4,
                "mean_whitened_visual_sq_residual_per_corner": final_whitened_visual_sq_per_corner,
            },
            "state_layout": {
                "frame_block_size": 6,
                "tag_block_size": 6,
                "frame_state_count": camera_state.shape[0],
                "tag_state_count": tag_state.shape[0],
                "tag_base": camera_state.shape[0] * 6,
            },
        },
    )


def solve_visual_inertial_block_lm(
    *,
    dataset: BatchCalibrationDataset,
    graph: BatchGraphDefinition,
    init: InitialGuess,
    vision_noise: VisionNoisePreset,
    imu_noise: ImuNoisePreset,
    max_iterations: int = 14,
    initial_damping: float = 1e-3,
    variant: str = "visual_inertial_fused",
) -> BatchSolveResult:
    frame_state, tag_state, global_bias = _pack_imu_state(dataset, graph, init)
    frame_index_map, tag_index_map = _frame_and_tag_index_maps(graph)
    frame_block_size = 9
    tag_block_size = 6
    frame_state_count = frame_state.shape[0]
    tag_state_count = tag_state.shape[0]
    tag_base = frame_state_count * frame_block_size
    global_bias_base = tag_base + tag_state_count * tag_block_size
    state_dimension = global_bias_base + 6
    initial_cost = None
    damping = float(initial_damping)
    iteration_timings_ms: list[float] = []
    diagnostics: dict[str, Any] = {"iteration_costs": [], "variant": variant}
    final_hessian = None
    final_gradient = None
    final_visual_whitened_sq_per_corner = None
    final_imu_whitened_sq_per_factor = None
    final_imu_position_sq_per_factor = None
    final_imu_rotation_sq_per_factor = None
    final_imu_velocity_sq_per_factor = None
    imu_translation_m = tuple(float(value) for value in dataset.device_config.mount.imu_translation_m)
    imu_rpy_deg = tuple(float(value) for value in dataset.device_config.mount.imu_rpy_deg)
    gravity_world = np.array([0.0, 0.0, -9.81], dtype=np.float64)

    for _iteration in range(max_iterations):
        started_at = time.perf_counter()
        hessian = np.zeros((state_dimension, state_dimension), dtype=np.float64)
        gradient = np.zeros(state_dimension, dtype=np.float64)
        total_cost = 0.0
        visual_whitened_sq = 0.0
        visual_corner_count = 0
        imu_whitened_sq = 0.0
        imu_factor_count = 0
        imu_position_whitened_sq = 0.0
        imu_rotation_whitened_sq = 0.0
        imu_velocity_whitened_sq = 0.0

        for factor in graph.visual_factors:
            frame_slot = frame_index_map[int(factor.frame_index)]
            tag_slot = tag_index_map[int(factor.tag_id)]
            local_state = np.concatenate((frame_state[frame_slot], tag_state[tag_slot]), axis=0)
            residual_fn, jacobian_fn = _compiled_visual_imu_factor(
                float(dataset.camera_model.fx_px),
                float(dataset.camera_model.fy_px),
                float(dataset.camera_model.cx_px),
                float(dataset.camera_model.cy_px),
                bool(dataset.camera_model.apply_lens_distortion_in_render),
                tuple(float(value) for value in dataset.camera_model.distortion_coefficients),
                float(factor.tag_size_m),
                imu_translation_m,
                imu_rpy_deg,
            )
            observed = np.asarray(factor.observed_corners_px, dtype=np.float64)
            residual = np.asarray(residual_fn(local_state, observed), dtype=np.float64)
            jacobian = np.asarray(jacobian_fn(local_state, observed), dtype=np.float64)
            weighted_residual, weighted_jacobian = _visual_factor_weighted(
                residual,
                jacobian,
                sigma_px=float(vision_noise.corner_noise_std_px),
                huber_delta_px=max(float(vision_noise.corner_noise_std_px) * 3.0, 1.0),
            )
            total_cost += 0.5 * float(weighted_residual @ weighted_residual)
            visual_whitened_sq += float(weighted_residual @ weighted_residual)
            visual_corner_count += residual.reshape(-1, 2).shape[0]

            frame_slice = slice(frame_slot * frame_block_size, (frame_slot + 1) * frame_block_size)
            tag_slice = slice(tag_base + tag_slot * tag_block_size, tag_base + (tag_slot + 1) * tag_block_size)
            jacobian_frame = weighted_jacobian[:, :frame_block_size]
            jacobian_tag = weighted_jacobian[:, frame_block_size : frame_block_size + tag_block_size]
            hessian[frame_slice, frame_slice] += jacobian_frame.T @ jacobian_frame
            hessian[tag_slice, tag_slice] += jacobian_tag.T @ jacobian_tag
            hessian[frame_slice, tag_slice] += jacobian_frame.T @ jacobian_tag
            hessian[tag_slice, frame_slice] += jacobian_tag.T @ jacobian_frame
            gradient[frame_slice] += jacobian_frame.T @ weighted_residual
            gradient[tag_slice] += jacobian_tag.T @ weighted_residual

        for factor in graph.imu_factors:
            start_slot = frame_index_map[int(factor.start_frame_index)]
            end_slot = frame_index_map[int(factor.end_frame_index)]
            packet_sequence = [dataset.imu_packets[packet_index] for packet_index in factor.packet_indices]
            gyro_packets = np.asarray([packet.gyro_body_rps for packet in packet_sequence], dtype=np.float64)
            accel_packets = np.asarray([packet.accel_body_mps2 for packet in packet_sequence], dtype=np.float64)
            dt_packets = np.asarray(factor.dt_s, dtype=np.float64)
            residual_fn, jacobian_fn = _compiled_imu_factor(len(packet_sequence))
            local_state = np.concatenate((frame_state[start_slot], frame_state[end_slot], global_bias), axis=0)
            residual = np.asarray(residual_fn(local_state, gyro_packets, accel_packets, dt_packets, gravity_world), dtype=np.float64)
            jacobian = np.asarray(jacobian_fn(local_state, gyro_packets, accel_packets, dt_packets, gravity_world), dtype=np.float64)
            sqrt_information = _imu_sqrt_information(float(np.sum(dt_packets)), imu_noise)
            weighted_residual = sqrt_information @ residual
            weighted_jacobian = sqrt_information @ jacobian
            total_cost += 0.5 * float(weighted_residual @ weighted_residual)
            imu_whitened_sq += float(weighted_residual @ weighted_residual)
            imu_factor_count += 1
            imu_position_whitened_sq += float(weighted_residual[:3] @ weighted_residual[:3])
            imu_rotation_whitened_sq += float(weighted_residual[3:6] @ weighted_residual[3:6])
            imu_velocity_whitened_sq += float(weighted_residual[6:9] @ weighted_residual[6:9])

            start_slice = slice(start_slot * frame_block_size, (start_slot + 1) * frame_block_size)
            end_slice = slice(end_slot * frame_block_size, (end_slot + 1) * frame_block_size)
            bias_slice = slice(global_bias_base, global_bias_base + 6)
            jacobian_start = weighted_jacobian[:, :frame_block_size]
            jacobian_end = weighted_jacobian[:, frame_block_size : 2 * frame_block_size]
            jacobian_bias = weighted_jacobian[:, 2 * frame_block_size : 2 * frame_block_size + 6]
            hessian[start_slice, start_slice] += jacobian_start.T @ jacobian_start
            hessian[end_slice, end_slice] += jacobian_end.T @ jacobian_end
            hessian[bias_slice, bias_slice] += jacobian_bias.T @ jacobian_bias
            cross_term = jacobian_start.T @ jacobian_end
            hessian[start_slice, end_slice] += cross_term
            hessian[end_slice, start_slice] += cross_term.T
            cross_start_bias = jacobian_start.T @ jacobian_bias
            cross_end_bias = jacobian_end.T @ jacobian_bias
            hessian[start_slice, bias_slice] += cross_start_bias
            hessian[bias_slice, start_slice] += cross_start_bias.T
            hessian[end_slice, bias_slice] += cross_end_bias
            hessian[bias_slice, end_slice] += cross_end_bias.T
            gradient[start_slice] += jacobian_start.T @ weighted_residual
            gradient[end_slice] += jacobian_end.T @ weighted_residual
            gradient[bias_slice] += jacobian_bias.T @ weighted_residual

        for prior in graph.pose_priors:
            if prior.variable_kind == "imu_state":
                state_slot = frame_index_map[int(prior.index)]
                state_slice = slice(state_slot * frame_block_size, (state_slot + 1) * frame_block_size)
                residual = np.asarray(prior.sqrt_information @ (frame_state[state_slot] - prior.mean_vector), dtype=np.float64)
                jacobian = np.asarray(prior.sqrt_information, dtype=np.float64)
                hessian[state_slice, state_slice] += jacobian.T @ jacobian
                gradient[state_slice] += jacobian.T @ residual
                total_cost += 0.5 * float(residual @ residual)
            elif prior.variable_kind == "imu_global_bias":
                bias_slice = slice(global_bias_base, global_bias_base + 6)
                residual = np.asarray(prior.sqrt_information @ (global_bias - prior.mean_vector), dtype=np.float64)
                jacobian = np.asarray(prior.sqrt_information, dtype=np.float64)
                hessian[bias_slice, bias_slice] += jacobian.T @ jacobian
                gradient[bias_slice] += jacobian.T @ residual
                total_cost += 0.5 * float(residual @ residual)

        if initial_cost is None:
            initial_cost = total_cost
        diagnostics["iteration_costs"].append(total_cost)
        final_hessian = hessian.copy()
        final_gradient = gradient.copy()
        final_visual_whitened_sq_per_corner = visual_whitened_sq / max(visual_corner_count, 1)
        final_imu_whitened_sq_per_factor = imu_whitened_sq / max(imu_factor_count, 1)
        final_imu_position_sq_per_factor = imu_position_whitened_sq / max(imu_factor_count, 1)
        final_imu_rotation_sq_per_factor = imu_rotation_whitened_sq / max(imu_factor_count, 1)
        final_imu_velocity_sq_per_factor = imu_velocity_whitened_sq / max(imu_factor_count, 1)

        step_success = False
        for _attempt in range(20):
            trial_hessian = hessian + damping * np.eye(state_dimension, dtype=np.float64)
            delta = -np.linalg.solve(trial_hessian, gradient)
            trial_frame_state = frame_state.copy()
            trial_tag_state = tag_state.copy()
            trial_global_bias = global_bias.copy()
            for frame_slot in range(frame_state.shape[0]):
                state_slice = slice(frame_slot * frame_block_size, (frame_slot + 1) * frame_block_size)
                state_delta = delta[state_slice]
                updated = trial_frame_state[frame_slot].copy()
                updated[:6] = _pose_update(updated[:6], state_delta[:6])
                updated[6:9] += state_delta[6:9]
                trial_frame_state[frame_slot] = updated
            for tag_slot in range(tag_state.shape[0]):
                tag_slice = slice(tag_base + tag_slot * tag_block_size, tag_base + (tag_slot + 1) * tag_block_size)
                trial_tag_state[tag_slot] = _pose_update(trial_tag_state[tag_slot], delta[tag_slice])
            trial_global_bias += delta[global_bias_base : global_bias_base + 6]

            trial_cost = 0.0
            for factor in graph.visual_factors:
                frame_slot = frame_index_map[int(factor.frame_index)]
                tag_slot = tag_index_map[int(factor.tag_id)]
                local_state = np.concatenate((trial_frame_state[frame_slot], trial_tag_state[tag_slot]), axis=0)
                residual_fn, jacobian_fn = _compiled_visual_imu_factor(
                    float(dataset.camera_model.fx_px),
                    float(dataset.camera_model.fy_px),
                    float(dataset.camera_model.cx_px),
                    float(dataset.camera_model.cy_px),
                    bool(dataset.camera_model.apply_lens_distortion_in_render),
                    tuple(float(value) for value in dataset.camera_model.distortion_coefficients),
                    float(factor.tag_size_m),
                    imu_translation_m,
                    imu_rpy_deg,
                )
                residual = np.asarray(residual_fn(local_state, np.asarray(factor.observed_corners_px, dtype=np.float64)), dtype=np.float64)
                jacobian = np.asarray(jacobian_fn(local_state, np.asarray(factor.observed_corners_px, dtype=np.float64)), dtype=np.float64)
                weighted_residual, _ = _visual_factor_weighted(
                    residual,
                    jacobian,
                    sigma_px=float(vision_noise.corner_noise_std_px),
                    huber_delta_px=max(float(vision_noise.corner_noise_std_px) * 3.0, 1.0),
                )
                trial_cost += 0.5 * float(weighted_residual @ weighted_residual)
            for factor in graph.imu_factors:
                start_slot = frame_index_map[int(factor.start_frame_index)]
                end_slot = frame_index_map[int(factor.end_frame_index)]
                packet_sequence = [dataset.imu_packets[packet_index] for packet_index in factor.packet_indices]
                gyro_packets = np.asarray([packet.gyro_body_rps for packet in packet_sequence], dtype=np.float64)
                accel_packets = np.asarray([packet.accel_body_mps2 for packet in packet_sequence], dtype=np.float64)
                dt_packets = np.asarray(factor.dt_s, dtype=np.float64)
                residual_fn, jacobian_fn = _compiled_imu_factor(len(packet_sequence))
                residual = np.asarray(
                    residual_fn(
                        np.concatenate((trial_frame_state[start_slot], trial_frame_state[end_slot], trial_global_bias), axis=0),
                        gyro_packets,
                        accel_packets,
                        dt_packets,
                        gravity_world,
                    ),
                    dtype=np.float64,
                )
                jacobian = np.asarray(
                    jacobian_fn(
                        np.concatenate((trial_frame_state[start_slot], trial_frame_state[end_slot], trial_global_bias), axis=0),
                        gyro_packets,
                        accel_packets,
                        dt_packets,
                        gravity_world,
                    ),
                    dtype=np.float64,
                )
                sqrt_information = _imu_sqrt_information(float(np.sum(dt_packets)), imu_noise)
                weighted_residual = sqrt_information @ residual
                trial_cost += 0.5 * float(weighted_residual @ weighted_residual)
            for prior in graph.pose_priors:
                if prior.variable_kind == "imu_state":
                    state_slot = frame_index_map[int(prior.index)]
                    residual = np.asarray(prior.sqrt_information @ (trial_frame_state[state_slot] - prior.mean_vector), dtype=np.float64)
                    trial_cost += 0.5 * float(residual @ residual)
                elif prior.variable_kind == "imu_global_bias":
                    residual = np.asarray(prior.sqrt_information @ (trial_global_bias - prior.mean_vector), dtype=np.float64)
                    trial_cost += 0.5 * float(residual @ residual)
            if trial_cost < total_cost:
                frame_state = trial_frame_state
                tag_state = trial_tag_state
                global_bias = trial_global_bias
                damping = max(damping * 0.5, 1e-6)
                step_success = True
                break
            damping *= 10.0
        iteration_timings_ms.append((time.perf_counter() - started_at) * 1000.0)
        if not step_success:
            break
        if float(np.linalg.norm(delta)) < 1e-6:
            break

    camera_pose_vectors = {}
    velocity_world_mps_by_frame = {}
    gyro_bias_rps_by_frame = {}
    accel_bias_mps2_by_frame = {}
    rotation_ci = rotation_matrix_from_rpy_deg(imu_rpy_deg)
    global_gyro_bias_rps = global_bias[:3].copy()
    global_accel_bias_mps2 = global_bias[3:6].copy()
    for frame_index in graph.frame_indices:
        state = frame_state[frame_index_map[int(frame_index)]]
        imu_position_world_m, imu_rotation_wi = pose_components_from_vector(state[:6])
        camera_position_world_m, camera_rotation_wc = camera_pose_from_imu_pose(
            imu_position_world_m=imu_position_world_m,
            imu_rotation_wi=imu_rotation_wi,
            imu_translation_camera_m=imu_translation_m,
            rotation_ci=rotation_ci,
        )
        camera_pose_vectors[int(frame_index)] = pose_vector_from_components(camera_position_world_m, camera_rotation_wc)
        velocity_world_mps_by_frame[int(frame_index)] = state[6:9].copy()
        gyro_bias_rps_by_frame[int(frame_index)] = global_gyro_bias_rps.copy()
        accel_bias_mps2_by_frame[int(frame_index)] = global_accel_bias_mps2.copy()
    tag_pose_vectors = {int(tag_id): tag_state[tag_index_map[int(tag_id)]].copy() for tag_id in graph.tag_ids}
    return BatchSolveResult(
        stage="visual_inertial",
        variant=variant,
        success=True,
        solver_name="jax_block_lm_visual_inertial",
        iterations=len(diagnostics["iteration_costs"]),
        final_cost=float(diagnostics["iteration_costs"][-1]),
        initial_cost=float(initial_cost if initial_cost is not None else 0.0),
        damping=float(damping),
        camera_pose_vectors=camera_pose_vectors,
        tag_pose_vectors=tag_pose_vectors,
        velocity_world_mps_by_frame=velocity_world_mps_by_frame,
        gyro_bias_rps_by_frame=gyro_bias_rps_by_frame,
        accel_bias_mps2_by_frame=accel_bias_mps2_by_frame,
        global_gyro_bias_rps=global_gyro_bias_rps,
        global_accel_bias_mps2=global_accel_bias_mps2,
        hessian=final_hessian,
        gradient=final_gradient,
        diagnostics={
            **diagnostics,
            "mean_iteration_time_ms": float(np.mean(iteration_timings_ms)) if iteration_timings_ms else None,
            "imu_factor_count": len(graph.imu_factors),
            "mean_whitened_visual_sq_residual_per_corner": final_visual_whitened_sq_per_corner,
            "mean_whitened_imu_sq_residual_per_factor": final_imu_whitened_sq_per_factor,
            "factor_breakdown": {
                "visual_factor_count": len(graph.visual_factors),
                "visual_corner_count": len(graph.visual_factors) * 4,
                "imu_factor_count": len(graph.imu_factors),
                "mean_whitened_visual_sq_residual_per_corner": final_visual_whitened_sq_per_corner,
                "mean_whitened_imu_sq_residual_per_factor": final_imu_whitened_sq_per_factor,
                "mean_whitened_imu_sq_residual_position_per_factor": final_imu_position_sq_per_factor,
                "mean_whitened_imu_sq_residual_rotation_per_factor": final_imu_rotation_sq_per_factor,
                "mean_whitened_imu_sq_residual_velocity_per_factor": final_imu_velocity_sq_per_factor,
            },
            "state_layout": {
                "frame_block_size": frame_block_size,
                "tag_block_size": tag_block_size,
                "frame_state_count": frame_state_count,
                "tag_state_count": tag_state_count,
                "tag_base": tag_base,
                "global_bias_base": global_bias_base,
            },
        },
    )
