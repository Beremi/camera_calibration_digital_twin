"""Bootstrap routines for the batch visual and visual-inertial estimators."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from calib_sim.common.inertial import imu_pose_from_camera_pose
from calib_sim.estimation._geometry import (
    apply_world_alignment_to_pose_vector,
    camera_pose_from_tag_observation,
    compose_rvec,
    pose_components_from_vector,
    pose_vector_from_components,
    project_world_points_to_pixels,
    rvec_from_rotation_matrix,
    rotation_matrix_from_rpy_deg,
    tag_pose_from_camera_pixels,
)
from calib_sim.estimation.types import BatchCalibrationDataset, InitialGuess


@dataclass(slots=True)
class BootstrapFrameScore:
    frame_index: int
    visible_tags: int
    border_margin_score: float
    image_spread_score: float


def _rotation_align_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    source_norm = np.linalg.norm(source)
    target_norm = np.linalg.norm(target)
    if source_norm < 1e-12 or target_norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    source = source / source_norm
    target = target / target_norm
    cross = np.cross(source, target)
    cross_norm = np.linalg.norm(cross)
    dot = float(np.clip(source @ target, -1.0, 1.0))
    if cross_norm < 1e-12:
        if dot > 0.0:
            return np.eye(3, dtype=np.float64)
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(source @ axis)) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = axis - source * float(source @ axis)
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        return pose_components_from_vector(np.concatenate((np.zeros(3, dtype=np.float64), axis * np.pi), axis=0))[1]
    axis = cross / cross_norm
    angle = math.acos(dot)
    return pose_components_from_vector(np.concatenate((np.zeros(3, dtype=np.float64), axis * angle), axis=0))[1]


def _quaternion_wxyz_from_rotation(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = 2.0 * math.sqrt(trace + 1.0)
        return np.array(
            [
                0.25 * s,
                (matrix[2, 1] - matrix[1, 2]) / s,
                (matrix[0, 2] - matrix[2, 0]) / s,
                (matrix[1, 0] - matrix[0, 1]) / s,
            ],
            dtype=np.float64,
        )
    diagonal = np.diag(matrix)
    if diagonal[0] > diagonal[1] and diagonal[0] > diagonal[2]:
        s = 2.0 * math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
        return np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / s,
                0.25 * s,
                (matrix[0, 1] + matrix[1, 0]) / s,
                (matrix[0, 2] + matrix[2, 0]) / s,
            ],
            dtype=np.float64,
        )
    if diagonal[1] > diagonal[2]:
        s = 2.0 * math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
        return np.array(
            [
                (matrix[0, 2] - matrix[2, 0]) / s,
                (matrix[0, 1] + matrix[1, 0]) / s,
                0.25 * s,
                (matrix[1, 2] + matrix[2, 1]) / s,
            ],
            dtype=np.float64,
        )
    s = 2.0 * math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
    return np.array(
        [
            (matrix[1, 0] - matrix[0, 1]) / s,
            (matrix[0, 2] + matrix[2, 0]) / s,
            (matrix[1, 2] + matrix[2, 1]) / s,
            0.25 * s,
        ],
        dtype=np.float64,
    )


def _rotation_matrix_from_quaternion_wxyz(quaternion_wxyz: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
    return np.array(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float64,
    )


def _slerp_rotation(start_rotation: np.ndarray, end_rotation: np.ndarray, alpha: float) -> np.ndarray:
    start = _quaternion_wxyz_from_rotation(start_rotation)
    end = _quaternion_wxyz_from_rotation(end_rotation)
    dot = float(np.dot(start, end))
    if dot < 0.0:
        end = -end
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        blended = start + float(alpha) * (end - start)
        blended /= max(float(np.linalg.norm(blended)), 1e-12)
        return _rotation_matrix_from_quaternion_wxyz(blended)
    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * float(alpha)
    sin_theta = math.sin(theta)
    scale_start = math.sin(theta_0 - theta) / max(sin_theta_0, 1e-12)
    scale_end = sin_theta / max(sin_theta_0, 1e-12)
    blended = scale_start * start + scale_end * end
    blended /= max(float(np.linalg.norm(blended)), 1e-12)
    return _rotation_matrix_from_quaternion_wxyz(blended)


def _fill_missing_camera_pose_vectors(dataset: BatchCalibrationDataset, camera_pose_vectors: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    if not camera_pose_vectors:
        return camera_pose_vectors
    frame_by_index = {int(frame.frame_index): frame for frame in dataset.camera_frames}
    known_indices = sorted(int(index) for index in camera_pose_vectors)
    for frame in dataset.camera_frames:
        frame_index = int(frame.frame_index)
        if frame_index in camera_pose_vectors:
            continue
        prev_candidates = [index for index in known_indices if index < frame_index]
        next_candidates = [index for index in known_indices if index > frame_index]
        if prev_candidates and next_candidates:
            prev_index = prev_candidates[-1]
            next_index = next_candidates[0]
            prev_position_world_m, prev_rotation_wc = pose_components_from_vector(camera_pose_vectors[prev_index])
            next_position_world_m, next_rotation_wc = pose_components_from_vector(camera_pose_vectors[next_index])
            prev_timestamp_s = float(frame_by_index[prev_index].timestamp_s)
            next_timestamp_s = float(frame_by_index[next_index].timestamp_s)
            alpha = 0.0 if next_timestamp_s <= prev_timestamp_s else (
                float(frame.timestamp_s) - prev_timestamp_s
            ) / max(next_timestamp_s - prev_timestamp_s, 1e-9)
            alpha = float(np.clip(alpha, 0.0, 1.0))
            position_world_m = (1.0 - alpha) * prev_position_world_m + alpha * next_position_world_m
            rotation_wc = _slerp_rotation(prev_rotation_wc, next_rotation_wc, alpha)
            camera_pose_vectors[frame_index] = pose_vector_from_components(position_world_m, rotation_wc)
            continue
        source_index = prev_candidates[-1] if prev_candidates else next_candidates[0]
        source_position_world_m, source_rotation_wc = pose_components_from_vector(camera_pose_vectors[source_index])
        camera_pose_vectors[frame_index] = pose_vector_from_components(source_position_world_m.copy(), source_rotation_wc.copy())
    return camera_pose_vectors


def _gravity_alignment_rotation_from_visual_and_imu(
    dataset: BatchCalibrationDataset,
    camera_pose_vectors: dict[int, np.ndarray],
) -> np.ndarray:
    frame_lookup = {int(frame.frame_index): frame for frame in dataset.camera_frames}
    packets_by_interval_end: dict[int, list] = {}
    for start_frame, end_frame in zip(dataset.camera_frames[:-1], dataset.camera_frames[1:]):
        packets_by_interval_end[int(end_frame.frame_index)] = [
            packet
            for packet in dataset.imu_packets
            if float(start_frame.timestamp_s) < float(packet.timestamp_s) <= float(end_frame.timestamp_s) + 1e-12
        ]

    rotation_ci = rotation_matrix_from_rpy_deg(tuple(float(value) for value in dataset.device_config.mount.imu_rpy_deg))
    world_specific_force_samples: list[np.ndarray] = []
    for frame_index, packets in packets_by_interval_end.items():
        if not packets or frame_index not in camera_pose_vectors:
            continue
        camera_position_world_m, camera_rotation_wc = pose_components_from_vector(camera_pose_vectors[frame_index])
        _, imu_rotation_wi = imu_pose_from_camera_pose(
            camera_position_world_m=camera_position_world_m,
            camera_rotation_wc=camera_rotation_wc,
            imu_translation_camera_m=dataset.device_config.mount.imu_translation_m,
            rotation_ci=rotation_ci,
        )
        accel_body_mps2 = np.asarray(packets[-1].accel_body_mps2, dtype=np.float64).reshape(3)
        world_specific_force_samples.append(imu_rotation_wi @ accel_body_mps2)
    if not world_specific_force_samples:
        return np.eye(3, dtype=np.float64)
    mean_world_specific_force = np.mean(np.asarray(world_specific_force_samples, dtype=np.float64), axis=0)
    return _rotation_align_vectors(mean_world_specific_force, np.array([0.0, 0.0, 9.81], dtype=np.float64))


def choose_anchor_frame(dataset: BatchCalibrationDataset) -> BootstrapFrameScore:
    detections_by_frame: dict[int, list[np.ndarray]] = {}
    for detection in dataset.tag_detections:
        detections_by_frame.setdefault(int(detection.frame_index), []).append(
            np.asarray(detection.corners_xy_clockwise_px, dtype=np.float64).reshape(4, 2)
        )

    best_score: BootstrapFrameScore | None = None
    for frame in dataset.camera_frames:
        corners = detections_by_frame.get(int(frame.frame_index), [])
        visible_tags = len(corners)
        if visible_tags <= 0:
            continue
        stacked = np.vstack(corners)
        border_margin = float(
            min(
                np.min(stacked[:, 0]),
                np.min(stacked[:, 1]),
                float(dataset.camera_model.output_width_px) - np.max(stacked[:, 0]),
                float(dataset.camera_model.output_height_px) - np.max(stacked[:, 1]),
            )
        )
        spread = float((np.max(stacked[:, 0]) - np.min(stacked[:, 0])) * (np.max(stacked[:, 1]) - np.min(stacked[:, 1])))
        score = BootstrapFrameScore(
            frame_index=int(frame.frame_index),
            visible_tags=visible_tags,
            border_margin_score=border_margin,
            image_spread_score=spread,
        )
        if best_score is None:
            best_score = score
            continue
        if (score.visible_tags, score.border_margin_score, score.image_spread_score, -score.frame_index) > (
            best_score.visible_tags,
            best_score.border_margin_score,
            best_score.image_spread_score,
            -best_score.frame_index,
        ):
            best_score = score
    if best_score is None:
        raise RuntimeError("Cannot choose an anchor frame because the dataset contains no tag detections.")
    return best_score


def _detections_by_frame(dataset: BatchCalibrationDataset) -> dict[int, list]:
    mapping: dict[int, list] = {}
    for detection in dataset.tag_detections:
        mapping.setdefault(int(detection.frame_index), []).append(detection)
    for frame_detections in mapping.values():
        frame_detections.sort(key=lambda item: int(item.tag_id))
    return mapping


def _camera_pose_residual(
    pose_vector: np.ndarray,
    *,
    dataset: BatchCalibrationDataset,
    detections: list,
    tag_pose_vectors: dict[int, np.ndarray],
) -> np.ndarray:
    camera_position_world_m, camera_rotation_wc = pose_components_from_vector(pose_vector)
    residual_terms: list[np.ndarray] = []
    for detection in detections:
        tag_pose_vector = tag_pose_vectors.get(int(detection.tag_id))
        if tag_pose_vector is None:
            continue
        tag_position_world_m, tag_rotation_wt = pose_components_from_vector(tag_pose_vector)
        world_points = tag_position_world_m.reshape(1, 3) + (
            tag_rotation_wt @ dataset.tag_catalog[int(detection.tag_id)].corner_points_local_m().T
        ).T
        predicted_pixels_px, visible_mask = project_world_points_to_pixels(
            camera_model=dataset.camera_model,
            camera_position_world_m=camera_position_world_m,
            camera_rotation_wc=camera_rotation_wc,
            world_points_m=world_points,
        )
        if not bool(np.all(visible_mask)):
            predicted_pixels_px = np.asarray(predicted_pixels_px, dtype=np.float64)
        observed_pixels_px = np.asarray(detection.corners_xy_clockwise_px, dtype=np.float64).reshape(4, 2)
        residual_terms.append((predicted_pixels_px - observed_pixels_px).reshape(-1))
    if not residual_terms:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(residual_terms).astype(np.float64)


def _numerical_jacobian(pose_vector: np.ndarray, *, residual_fn, eps_translation: float = 1e-5, eps_rotation: float = 1e-5) -> np.ndarray:
    reference = residual_fn(pose_vector)
    jacobian = np.zeros((reference.size, pose_vector.size), dtype=np.float64)
    for index in range(pose_vector.size):
        step = np.zeros_like(pose_vector)
        step[index] = eps_translation if index < 3 else eps_rotation
        plus = residual_fn(pose_vector + step)
        minus = residual_fn(pose_vector - step)
        jacobian[:, index] = (plus - minus) / max(2.0 * step[index], 1e-12)
    return jacobian


def _refine_camera_pose(
    initial_pose_vector: np.ndarray,
    *,
    dataset: BatchCalibrationDataset,
    detections: list,
    tag_pose_vectors: dict[int, np.ndarray],
    max_iterations: int = 18,
) -> np.ndarray:
    pose_vector = np.asarray(initial_pose_vector, dtype=np.float64).reshape(6).copy()
    damping = 1e-3

    def residual_fn(candidate: np.ndarray) -> np.ndarray:
        return _camera_pose_residual(candidate, dataset=dataset, detections=detections, tag_pose_vectors=tag_pose_vectors)

    residual = residual_fn(pose_vector)
    if residual.size == 0:
        return pose_vector
    best_cost = 0.5 * float(residual @ residual)

    for _ in range(max_iterations):
        jacobian = _numerical_jacobian(pose_vector, residual_fn=residual_fn)
        normal_matrix = jacobian.T @ jacobian
        gradient = jacobian.T @ residual
        trial_success = False
        for _attempt in range(10):
            step = -np.linalg.solve(normal_matrix + damping * np.eye(6, dtype=np.float64), gradient)
            trial_pose = pose_vector + step
            trial_residual = residual_fn(trial_pose)
            trial_cost = 0.5 * float(trial_residual @ trial_residual)
            if trial_cost < best_cost:
                pose_vector = trial_pose
                residual = trial_residual
                best_cost = trial_cost
                damping = max(damping * 0.5, 1e-6)
                trial_success = True
                break
            damping *= 4.0
        if not trial_success or float(np.linalg.norm(step)) < 1e-7:
            break
    return pose_vector


def _bootstrap_tag_pose_from_frame(
    *,
    dataset: BatchCalibrationDataset,
    frame_pose_vector: np.ndarray,
    detection,
) -> np.ndarray:
    camera_position_world_m, camera_rotation_wc = pose_components_from_vector(frame_pose_vector)
    rotation_ct, translation_ct = tag_pose_from_camera_pixels(
        camera_model=dataset.camera_model,
        tag_size_m=float(detection.physical_edge_length_m),
        ordered_corners_xy_px=np.asarray(detection.corners_xy_clockwise_px, dtype=np.float64).reshape(4, 2),
    )
    tag_rotation_wt = camera_rotation_wc @ rotation_ct
    tag_position_world_m = camera_position_world_m + camera_rotation_wc @ np.asarray(translation_ct, dtype=np.float64).reshape(3)
    return pose_vector_from_components(tag_position_world_m, tag_rotation_wt)


def bootstrap_initial_guess(dataset: BatchCalibrationDataset, *, stage: str) -> InitialGuess:
    detections_by_frame = _detections_by_frame(dataset)
    anchor = choose_anchor_frame(dataset)
    camera_pose_vectors: dict[int, np.ndarray] = {int(anchor.frame_index): np.zeros(6, dtype=np.float64)}
    tag_pose_vectors: dict[int, np.ndarray] = {}
    notes = [
        f"anchor_frame_index={anchor.frame_index}",
        f"anchor_visible_tags={anchor.visible_tags}",
        f"anchor_border_margin={anchor.border_margin_score:.2f}",
        f"anchor_image_spread={anchor.image_spread_score:.2f}",
    ]

    for detection in detections_by_frame.get(int(anchor.frame_index), []):
        rotation_ct, translation_ct = tag_pose_from_camera_pixels(
            camera_model=dataset.camera_model,
            tag_size_m=float(detection.physical_edge_length_m),
            ordered_corners_xy_px=np.asarray(detection.corners_xy_clockwise_px, dtype=np.float64).reshape(4, 2),
        )
        tag_pose_vectors[int(detection.tag_id)] = pose_vector_from_components(translation_ct, rotation_ct)

    frame_indices = [int(frame.frame_index) for frame in dataset.camera_frames]
    progress = True
    while progress:
        progress = False
        for frame_index in frame_indices:
            if frame_index in camera_pose_vectors:
                continue
            detections = detections_by_frame.get(frame_index, [])
            known_detections = [detection for detection in detections if int(detection.tag_id) in tag_pose_vectors]
            if not known_detections:
                continue
            seed_pose_vector: np.ndarray | None = None
            nearest_initialized = [
                (abs(other_frame_index - frame_index), other_frame_index)
                for other_frame_index in camera_pose_vectors
            ]
            if nearest_initialized:
                seed_pose_vector = camera_pose_vectors[min(nearest_initialized)[1]].copy()
            if seed_pose_vector is None:
                first_detection = known_detections[0]
                tag_pose_vector = tag_pose_vectors[int(first_detection.tag_id)]
                tag_position_world_m, tag_rotation_wt = pose_components_from_vector(tag_pose_vector)
                position_world_m, rotation_wc = camera_pose_from_tag_observation(
                    camera_model=dataset.camera_model,
                    tag_size_m=float(first_detection.physical_edge_length_m),
                    ordered_corners_xy_px=np.asarray(first_detection.corners_xy_clockwise_px, dtype=np.float64).reshape(4, 2),
                    tag_position_world_m=tag_position_world_m,
                    tag_rotation_wt=tag_rotation_wt,
                )
                seed_pose_vector = pose_vector_from_components(position_world_m, rotation_wc)
            refined_pose_vector = _refine_camera_pose(
                seed_pose_vector,
                dataset=dataset,
                detections=known_detections,
                tag_pose_vectors=tag_pose_vectors,
            )
            camera_pose_vectors[frame_index] = refined_pose_vector
            progress = True

        for frame_index, detections in detections_by_frame.items():
            frame_pose_vector = camera_pose_vectors.get(int(frame_index))
            if frame_pose_vector is None:
                continue
            for detection in detections:
                if int(detection.tag_id) in tag_pose_vectors:
                    continue
                tag_pose_vectors[int(detection.tag_id)] = _bootstrap_tag_pose_from_frame(
                    dataset=dataset,
                    frame_pose_vector=frame_pose_vector,
                    detection=detection,
                )
                progress = True

    first_frame_index = frame_indices[0]
    reference_pose_vector = camera_pose_vectors.get(first_frame_index)
    if reference_pose_vector is None:
        reference_pose_vector = camera_pose_vectors[int(anchor.frame_index)]
    ref_position_world_m, ref_rotation_wc = pose_components_from_vector(reference_pose_vector)
    alignment_rotation = ref_rotation_wc.T
    alignment_translation = -(alignment_rotation @ ref_position_world_m)
    camera_pose_vectors = {
        frame_index: apply_world_alignment_to_pose_vector(alignment_rotation, alignment_translation, pose_vector)
        for frame_index, pose_vector in camera_pose_vectors.items()
    }
    tag_pose_vectors = {
        tag_id: apply_world_alignment_to_pose_vector(alignment_rotation, alignment_translation, pose_vector)
        for tag_id, pose_vector in tag_pose_vectors.items()
    }
    camera_pose_vectors = _fill_missing_camera_pose_vectors(dataset, camera_pose_vectors)

    velocity_world_mps_by_frame: dict[int, np.ndarray] = {}
    gyro_bias_rps_by_frame: dict[int, np.ndarray] = {}
    accel_bias_mps2_by_frame: dict[int, np.ndarray] = {}
    if stage != "visual_only":
        sorted_frame_indices = sorted(camera_pose_vectors)
        rotation_ci = rotation_matrix_from_rpy_deg(tuple(float(value) for value in dataset.device_config.mount.imu_rpy_deg))
        imu_positions_world_m = {}
        for frame_index in sorted_frame_indices:
            camera_position_world_m, camera_rotation_wc = pose_components_from_vector(camera_pose_vectors[frame_index])
            imu_position_world_m, _ = imu_pose_from_camera_pose(
                camera_position_world_m=camera_position_world_m,
                camera_rotation_wc=camera_rotation_wc,
                imu_translation_camera_m=dataset.device_config.mount.imu_translation_m,
                rotation_ci=rotation_ci,
            )
            imu_positions_world_m[frame_index] = imu_position_world_m
        for index, frame_index in enumerate(sorted_frame_indices):
            frame = next(item for item in dataset.camera_frames if int(item.frame_index) == frame_index)
            if index == 0:
                next_frame_index = sorted_frame_indices[min(index + 1, len(sorted_frame_indices) - 1)]
                next_frame = next(item for item in dataset.camera_frames if int(item.frame_index) == next_frame_index)
                current_position = imu_positions_world_m[frame_index]
                next_position = imu_positions_world_m[next_frame_index]
                dt_s = max(float(next_frame.timestamp_s) - float(frame.timestamp_s), 1e-9)
                velocity_world_mps_by_frame[frame_index] = (next_position - current_position) / dt_s
            else:
                previous_frame_index = sorted_frame_indices[index - 1]
                previous_frame = next(item for item in dataset.camera_frames if int(item.frame_index) == previous_frame_index)
                current_position = imu_positions_world_m[frame_index]
                previous_position = imu_positions_world_m[previous_frame_index]
                dt_s = max(float(frame.timestamp_s) - float(previous_frame.timestamp_s), 1e-9)
                velocity_world_mps_by_frame[frame_index] = (current_position - previous_position) / dt_s
            gyro_bias_rps_by_frame[frame_index] = np.zeros(3, dtype=np.float64)
            accel_bias_mps2_by_frame[frame_index] = np.zeros(3, dtype=np.float64)

    return InitialGuess(
        stage=stage,
        anchor_frame_index=int(anchor.frame_index),
        camera_pose_vectors=camera_pose_vectors,
        tag_pose_vectors=tag_pose_vectors,
        velocity_world_mps_by_frame=velocity_world_mps_by_frame,
        gyro_bias_rps_by_frame=gyro_bias_rps_by_frame,
        accel_bias_mps2_by_frame=accel_bias_mps2_by_frame,
        global_gyro_bias_rps=np.zeros(3, dtype=np.float64),
        global_accel_bias_mps2=np.zeros(3, dtype=np.float64),
        notes=tuple(notes),
    )


def inertial_initial_guess_from_visual_solution(
    dataset: BatchCalibrationDataset,
    visual_result,
    *,
    anchor_frame_index: int,
) -> InitialGuess:
    camera_pose_vectors = {
        int(frame_index): np.asarray(pose_vector, dtype=np.float64).reshape(6).copy()
        for frame_index, pose_vector in visual_result.camera_pose_vectors.items()
    }
    tag_pose_vectors = {
        int(tag_id): np.asarray(pose_vector, dtype=np.float64).reshape(6).copy()
        for tag_id, pose_vector in visual_result.tag_pose_vectors.items()
    }
    gravity_alignment_rotation = _gravity_alignment_rotation_from_visual_and_imu(dataset, camera_pose_vectors)
    if not np.allclose(gravity_alignment_rotation, np.eye(3, dtype=np.float64), atol=1e-12):
        camera_pose_vectors = {
            int(frame_index): apply_world_alignment_to_pose_vector(
                gravity_alignment_rotation,
                np.zeros(3, dtype=np.float64),
                pose_vector,
            )
            for frame_index, pose_vector in camera_pose_vectors.items()
        }
        tag_pose_vectors = {
            int(tag_id): apply_world_alignment_to_pose_vector(
                gravity_alignment_rotation,
                np.zeros(3, dtype=np.float64),
                pose_vector,
            )
            for tag_id, pose_vector in tag_pose_vectors.items()
        }
    frame_lookup = {int(frame.frame_index): frame for frame in dataset.camera_frames}
    sorted_frame_indices = sorted(camera_pose_vectors)
    velocity_world_mps_by_frame: dict[int, np.ndarray] = {}
    gyro_bias_rps_by_frame: dict[int, np.ndarray] = {}
    accel_bias_mps2_by_frame: dict[int, np.ndarray] = {}
    rotation_ci = rotation_matrix_from_rpy_deg(tuple(float(value) for value in dataset.device_config.mount.imu_rpy_deg))
    imu_positions_world_m = {}
    for frame_index in sorted_frame_indices:
        camera_position_world_m, camera_rotation_wc = pose_components_from_vector(camera_pose_vectors[frame_index])
        imu_position_world_m, _ = imu_pose_from_camera_pose(
            camera_position_world_m=camera_position_world_m,
            camera_rotation_wc=camera_rotation_wc,
            imu_translation_camera_m=dataset.device_config.mount.imu_translation_m,
            rotation_ci=rotation_ci,
        )
        imu_positions_world_m[frame_index] = imu_position_world_m
    for index, frame_index in enumerate(sorted_frame_indices):
        if index == 0:
            next_frame_index = sorted_frame_indices[min(index + 1, len(sorted_frame_indices) - 1)]
            current_position_world_m = imu_positions_world_m[frame_index]
            next_position_world_m = imu_positions_world_m[next_frame_index]
            dt_s = max(float(frame_lookup[next_frame_index].timestamp_s) - float(frame_lookup[frame_index].timestamp_s), 1e-9)
            velocity_world_mps_by_frame[frame_index] = (next_position_world_m - current_position_world_m) / dt_s
        else:
            prev_frame_index = sorted_frame_indices[index - 1]
            prev_position_world_m = imu_positions_world_m[prev_frame_index]
            current_position_world_m = imu_positions_world_m[frame_index]
            dt_s = max(float(frame_lookup[frame_index].timestamp_s) - float(frame_lookup[prev_frame_index].timestamp_s), 1e-9)
            velocity_world_mps_by_frame[frame_index] = (current_position_world_m - prev_position_world_m) / dt_s
        gyro_bias_rps_by_frame[frame_index] = np.zeros(3, dtype=np.float64)
        accel_bias_mps2_by_frame[frame_index] = np.zeros(3, dtype=np.float64)

    return InitialGuess(
        stage="visual_inertial",
        anchor_frame_index=int(anchor_frame_index),
        camera_pose_vectors=camera_pose_vectors,
        tag_pose_vectors=tag_pose_vectors,
        velocity_world_mps_by_frame=velocity_world_mps_by_frame,
        gyro_bias_rps_by_frame=gyro_bias_rps_by_frame,
        accel_bias_mps2_by_frame=accel_bias_mps2_by_frame,
        global_gyro_bias_rps=np.zeros(3, dtype=np.float64),
        global_accel_bias_mps2=np.zeros(3, dtype=np.float64),
        notes=("initialized_from_visual_result", "gravity_aligned_from_async_imu"),
    )
