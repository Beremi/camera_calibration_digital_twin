"""Evaluation helpers for batch solver outputs."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from calib_sim.common.inertial import imu_pose_from_camera_pose
from calib_sim.estimation._geometry import (
    pose_components_from_vector,
    pose_vector_from_components,
    project_world_points_to_pixels,
    relative_rotation_error_deg,
    rotation_matrix_from_rpy_deg,
    rvec_from_rotation_matrix,
    tag_rotation_from_mount,
    world_alignment_from_reference_pose,
)
from calib_sim.estimation.types import BatchCalibrationDataset, BatchCalibrationEvaluationData, BatchSolveResult, EvalSummary, UncertaintySummary


def _camera_truth_by_frame(evaluation_data: BatchCalibrationEvaluationData) -> dict[int, object]:
    return {int(sample.frame_index): sample for sample in evaluation_data.camera_truth}


def _tag_truth_pose_by_id(evaluation_data: BatchCalibrationEvaluationData, camera_truth_by_frame: dict[int, object]) -> dict[int, np.ndarray]:
    tag_truth_pose_vectors: dict[int, np.ndarray] = {}
    for frame_index, tag_samples in evaluation_data.tag_truth_by_frame.items():
        camera_truth = camera_truth_by_frame.get(int(frame_index))
        if camera_truth is None:
            continue
        camera_position_world_m = np.asarray(camera_truth.position_world_m, dtype=np.float64)
        camera_rotation_cw = np.asarray(camera_truth.rotation_cw, dtype=np.float64).reshape(3, 3)
        for tag_sample in tag_samples:
            if int(tag_sample.tag_id) in tag_truth_pose_vectors:
                continue
            camera_rotation_tc = np.asarray(tag_sample.camera_rotation_tc, dtype=np.float64).reshape(3, 3)
            tag_rotation_wt = camera_rotation_cw @ camera_rotation_tc.T
            tag_truth_pose_vectors[int(tag_sample.tag_id)] = pose_vector_from_components(
                np.asarray(tag_sample.tag_center_world_m, dtype=np.float64),
                tag_rotation_wt,
            )
    return tag_truth_pose_vectors


def _imu_truth_velocity_by_frame(
    evaluation_data: BatchCalibrationEvaluationData,
    dataset: BatchCalibrationDataset,
) -> dict[int, np.ndarray]:
    rotation_ci = rotation_matrix_from_rpy_deg(tuple(float(value) for value in dataset.device_config.mount.imu_rpy_deg))
    sorted_truth = sorted(evaluation_data.camera_truth, key=lambda item: int(item.frame_index))
    imu_positions_world_m: dict[int, np.ndarray] = {}
    for truth in sorted_truth:
        imu_position_world_m, _ = imu_pose_from_camera_pose(
            camera_position_world_m=np.asarray(truth.position_world_m, dtype=np.float64),
            camera_rotation_wc=np.asarray(truth.rotation_cw, dtype=np.float64).reshape(3, 3),
            imu_translation_camera_m=dataset.device_config.mount.imu_translation_m,
            rotation_ci=rotation_ci,
        )
        imu_positions_world_m[int(truth.frame_index)] = imu_position_world_m
    velocities: dict[int, np.ndarray] = {}
    sorted_frame_indices = [int(item.frame_index) for item in sorted_truth]
    truth_lookup = {int(item.frame_index): item for item in sorted_truth}
    for index, frame_index in enumerate(sorted_frame_indices):
        if index == 0:
            next_frame_index = sorted_frame_indices[min(index + 1, len(sorted_frame_indices) - 1)]
            dt_s = max(float(truth_lookup[next_frame_index].timestamp_s) - float(truth_lookup[frame_index].timestamp_s), 1e-9)
            velocities[frame_index] = (imu_positions_world_m[next_frame_index] - imu_positions_world_m[frame_index]) / dt_s
        else:
            previous_frame_index = sorted_frame_indices[index - 1]
            dt_s = max(float(truth_lookup[frame_index].timestamp_s) - float(truth_lookup[previous_frame_index].timestamp_s), 1e-9)
            velocities[frame_index] = (imu_positions_world_m[frame_index] - imu_positions_world_m[previous_frame_index]) / dt_s
    return velocities


def evaluate_batch_result(
    result: BatchSolveResult,
    evaluation_data: BatchCalibrationEvaluationData,
    dataset: BatchCalibrationDataset | None = None,
    uncertainty_summary: UncertaintySummary | None = None,
) -> EvalSummary:
    camera_truth_by_frame = _camera_truth_by_frame(evaluation_data)
    common_frame_indices = sorted(set(int(index) for index in result.camera_pose_vectors) & set(camera_truth_by_frame))
    if not common_frame_indices:
        return EvalSummary(
            variant=result.variant,
            frames_analyzed=0,
            estimated_tags=0,
            mean_position_error_m=None,
            median_position_error_m=None,
            p95_position_error_m=None,
            max_position_error_m=None,
            mean_rotation_error_deg=None,
            mean_reprojection_rmse_px=None,
            joint_world_mean_position_error_m=None,
            joint_world_mean_rotation_error_deg=None,
            solve_rate=0.0,
            runtime_ms_per_iteration=result.diagnostics.get("mean_iteration_time_ms"),
        )

    reference_frame_index = common_frame_indices[0]
    reference_truth = camera_truth_by_frame[reference_frame_index]
    reference_truth_pose = pose_vector_from_components(
        np.asarray(reference_truth.position_world_m, dtype=np.float64),
        np.asarray(reference_truth.rotation_cw, dtype=np.float64).reshape(3, 3),
    )
    alignment_rotation, alignment_translation = world_alignment_from_reference_pose(
        np.asarray(result.camera_pose_vectors[reference_frame_index], dtype=np.float64),
        reference_truth_pose,
    )

    per_frame_records = []
    position_errors_m = []
    rotation_errors_deg = []
    axis_errors_by_level: dict[float, list[tuple[float, float]]] = defaultdict(list)
    camera_uncertainty_lookup = (
        {int(item.index): item for item in uncertainty_summary.camera_pose_uncertainty}
        if uncertainty_summary is not None
        else {}
    )
    velocity_uncertainty_lookup = (
        {int(item.index): item for item in uncertainty_summary.velocity_uncertainty}
        if uncertainty_summary is not None
        else {}
    )
    imu_truth_velocity_lookup = (
        _imu_truth_velocity_by_frame(evaluation_data, dataset)
        if dataset is not None and result.stage == "visual_inertial"
        else {}
    )
    position_nees_values: list[float] = []
    rotation_nees_values: list[float] = []
    rotation_sigma_pairs: list[tuple[float, float]] = []
    velocity_nees_values: list[float] = []
    for frame_index in common_frame_indices:
        estimated_position_world_m, estimated_rotation_wc = pose_components_from_vector(np.asarray(result.camera_pose_vectors[frame_index], dtype=np.float64))
        aligned_position_world_m = alignment_rotation @ estimated_position_world_m + alignment_translation
        aligned_rotation_wc = alignment_rotation @ estimated_rotation_wc
        truth = camera_truth_by_frame[frame_index]
        truth_position_world_m = np.asarray(truth.position_world_m, dtype=np.float64)
        truth_rotation_wc = np.asarray(truth.rotation_cw, dtype=np.float64).reshape(3, 3)
        position_error_vector_world_m = truth_position_world_m - aligned_position_world_m
        position_error_m = float(np.linalg.norm(position_error_vector_world_m))
        rotation_error_deg = relative_rotation_error_deg(aligned_rotation_wc, truth_rotation_wc)
        position_errors_m.append(position_error_m)
        rotation_errors_deg.append(rotation_error_deg)
        per_frame_records.append(
            {
                "frame_index": int(frame_index),
                "timestamp_s": float(truth.timestamp_s),
                "position_error_m": position_error_m,
                "rotation_error_deg": rotation_error_deg,
                "position_error_vector_world_m": [float(value) for value in position_error_vector_world_m.tolist()],
                "estimated_camera_world_position_m": [float(value) for value in aligned_position_world_m.tolist()],
            }
        )
        if uncertainty_summary is not None:
            uncertainty = camera_uncertainty_lookup.get(int(frame_index))
            if uncertainty is not None:
                axis_error = np.abs(position_error_vector_world_m)
                axis_sigma = np.asarray(uncertainty.position_std_m, dtype=np.float64)
                for error_value, sigma_value in zip(axis_error.tolist(), axis_sigma.tolist()):
                    axis_errors_by_level[1.0].append((float(error_value), float(sigma_value)))
                safe_sigma = np.maximum(axis_sigma, 1e-12)
                position_nees_values.append(float(np.sum(np.square(position_error_vector_world_m / safe_sigma))))
                rotation_error_rvec = rvec_from_rotation_matrix(aligned_rotation_wc.T @ truth_rotation_wc)
                rotation_sigma_rad = np.maximum(np.radians(np.asarray(uncertainty.rotation_std_deg, dtype=np.float64)), 1e-12)
                rotation_nees_values.append(float(np.sum(np.square(rotation_error_rvec / rotation_sigma_rad))))
                rotation_sigma_pairs.append(
                    (
                        float(np.linalg.norm(rotation_error_rvec)),
                        float(np.linalg.norm(rotation_sigma_rad)),
                    )
                )
            velocity_uncertainty = velocity_uncertainty_lookup.get(int(frame_index))
            if velocity_uncertainty is not None and int(frame_index) in result.velocity_world_mps_by_frame:
                truth_velocity = np.asarray(
                    imu_truth_velocity_lookup.get(int(frame_index), np.asarray(truth.velocity_world_mps, dtype=np.float64)),
                    dtype=np.float64,
                )
                estimated_velocity = np.asarray(result.velocity_world_mps_by_frame[int(frame_index)], dtype=np.float64)
                velocity_error = truth_velocity - estimated_velocity
                velocity_sigma = np.maximum(np.asarray(velocity_uncertainty.std, dtype=np.float64), 1e-12)
                velocity_nees_values.append(float(np.sum(np.square(velocity_error / velocity_sigma))))

    tag_truth_pose_vectors = _tag_truth_pose_by_id(evaluation_data, camera_truth_by_frame)
    per_tag_records = []
    for tag_id in sorted(int(index) for index in result.tag_pose_vectors):
        tag_truth_pose = tag_truth_pose_vectors.get(int(tag_id))
        if tag_truth_pose is None:
            continue
        estimated_position_world_m, estimated_rotation_wt = pose_components_from_vector(np.asarray(result.tag_pose_vectors[tag_id], dtype=np.float64))
        aligned_position_world_m = alignment_rotation @ estimated_position_world_m + alignment_translation
        aligned_rotation_wt = alignment_rotation @ estimated_rotation_wt
        truth_position_world_m, truth_rotation_wt = pose_components_from_vector(tag_truth_pose)
        translation_error_m = float(np.linalg.norm(truth_position_world_m - aligned_position_world_m))
        rotation_error_deg = relative_rotation_error_deg(aligned_rotation_wt, truth_rotation_wt)
        per_tag_records.append(
            {
                "tag_id": int(tag_id),
                "translation_error_m": translation_error_m,
                "rotation_error_deg": rotation_error_deg,
                "estimated_tag_center_world_m": [float(value) for value in aligned_position_world_m.tolist()],
            }
        )

    mean_reprojection_rmse_px = None
    if dataset is not None:
        detection_groups: dict[int, list] = defaultdict(list)
        for detection in dataset.tag_detections:
            detection_groups[int(detection.frame_index)].append(detection)
        reprojection_errors = []
        for frame_index in common_frame_indices:
            estimated_position_world_m, estimated_rotation_wc = pose_components_from_vector(np.asarray(result.camera_pose_vectors[frame_index], dtype=np.float64))
            aligned_camera_position_world_m = alignment_rotation @ estimated_position_world_m + alignment_translation
            aligned_camera_rotation_wc = alignment_rotation @ estimated_rotation_wc
            for detection in detection_groups.get(int(frame_index), []):
                if int(detection.tag_id) not in result.tag_pose_vectors:
                    continue
                estimated_tag_position_world_m, estimated_tag_rotation_wt = pose_components_from_vector(np.asarray(result.tag_pose_vectors[int(detection.tag_id)], dtype=np.float64))
                aligned_tag_position_world_m = alignment_rotation @ estimated_tag_position_world_m + alignment_translation
                aligned_tag_rotation_wt = alignment_rotation @ estimated_tag_rotation_wt
                world_points = aligned_tag_position_world_m.reshape(1, 3) + (
                    aligned_tag_rotation_wt @ dataset.tag_catalog[int(detection.tag_id)].corner_points_local_m().T
                ).T
                predicted_pixels_px, _ = project_world_points_to_pixels(
                    camera_model=dataset.camera_model,
                    camera_position_world_m=aligned_camera_position_world_m,
                    camera_rotation_wc=aligned_camera_rotation_wc,
                    world_points_m=world_points,
                )
                observed_pixels_px = np.asarray(detection.corners_xy_clockwise_px, dtype=np.float64).reshape(4, 2)
                reprojection_errors.append(float(np.sqrt(np.mean(np.sum(np.square(predicted_pixels_px - observed_pixels_px), axis=1)))))
        if reprojection_errors:
            mean_reprojection_rmse_px = float(np.mean(reprojection_errors))

    if uncertainty_summary is not None and axis_errors_by_level:
        axis_sigma_pairs = axis_errors_by_level[1.0]
        sigma_values = np.asarray([sigma for _, sigma in axis_sigma_pairs], dtype=np.float64)
        error_values = np.asarray([error for error, _ in axis_sigma_pairs], dtype=np.float64)
        uncertainty_summary.coverage_by_level = {
            "50": float(np.mean(error_values <= 0.674 * sigma_values)) if sigma_values.size else 0.0,
            "68": float(np.mean(error_values <= 1.0 * sigma_values)) if sigma_values.size else 0.0,
            "90": float(np.mean(error_values <= 1.645 * sigma_values)) if sigma_values.size else 0.0,
            "95": float(np.mean(error_values <= 1.96 * sigma_values)) if sigma_values.size else 0.0,
        }
        if sigma_values.size > 1:
            uncertainty_summary.metadata["position_sigma_error_corr"] = float(np.corrcoef(sigma_values, error_values)[0, 1])
        if position_nees_values:
            uncertainty_summary.metadata["position_nees_mean"] = float(np.mean(np.asarray(position_nees_values, dtype=np.float64)))
            uncertainty_summary.metadata["position_whitened_sq_error_mean"] = float(
                np.mean(np.asarray(position_nees_values, dtype=np.float64)) / 3.0
            )
        if rotation_nees_values:
            uncertainty_summary.metadata["rotation_nees_mean"] = float(np.mean(np.asarray(rotation_nees_values, dtype=np.float64)))
            uncertainty_summary.metadata["rotation_whitened_sq_error_mean"] = float(
                np.mean(np.asarray(rotation_nees_values, dtype=np.float64)) / 3.0
            )
        if len(rotation_sigma_pairs) > 1:
            rotation_errors = np.asarray([error for error, _ in rotation_sigma_pairs], dtype=np.float64)
            rotation_sigmas = np.asarray([sigma for _, sigma in rotation_sigma_pairs], dtype=np.float64)
            uncertainty_summary.metadata["rotation_sigma_error_corr"] = float(np.corrcoef(rotation_sigmas, rotation_errors)[0, 1])
        if velocity_nees_values:
            uncertainty_summary.metadata["velocity_nees_mean"] = float(np.mean(np.asarray(velocity_nees_values, dtype=np.float64)))
            uncertainty_summary.metadata["velocity_whitened_sq_error_mean"] = float(
                np.mean(np.asarray(velocity_nees_values, dtype=np.float64)) / 3.0
            )
        if uncertainty_summary.gyro_bias_uncertainty is not None and evaluation_data.global_gyro_bias_rps_truth is not None:
            gyro_error = np.asarray(evaluation_data.global_gyro_bias_rps_truth, dtype=np.float64) - np.asarray(result.global_gyro_bias_rps, dtype=np.float64)
            gyro_sigma = np.maximum(np.asarray(uncertainty_summary.gyro_bias_uncertainty.std, dtype=np.float64), 1e-12)
            gyro_nees = float(np.sum(np.square(gyro_error / gyro_sigma)))
            uncertainty_summary.metadata["gyro_bias_nees"] = gyro_nees
            uncertainty_summary.metadata["gyro_bias_whitened_sq_error_mean"] = gyro_nees / 3.0
        if uncertainty_summary.accel_bias_uncertainty is not None and evaluation_data.global_accel_bias_mps2_truth is not None:
            accel_error = np.asarray(evaluation_data.global_accel_bias_mps2_truth, dtype=np.float64) - np.asarray(result.global_accel_bias_mps2, dtype=np.float64)
            accel_sigma = np.maximum(np.asarray(uncertainty_summary.accel_bias_uncertainty.std, dtype=np.float64), 1e-12)
            accel_nees = float(np.sum(np.square(accel_error / accel_sigma)))
            uncertainty_summary.metadata["accel_bias_nees"] = accel_nees
            uncertainty_summary.metadata["accel_bias_whitened_sq_error_mean"] = accel_nees / 3.0

    return EvalSummary(
        variant=result.variant,
        frames_analyzed=len(common_frame_indices),
        estimated_tags=len(dataset.tag_detections) if dataset is not None else len(result.tag_pose_vectors),
        mean_position_error_m=float(np.mean(position_errors_m)) if position_errors_m else None,
        median_position_error_m=float(np.median(position_errors_m)) if position_errors_m else None,
        p95_position_error_m=float(np.percentile(position_errors_m, 95.0)) if position_errors_m else None,
        max_position_error_m=float(np.max(position_errors_m)) if position_errors_m else None,
        mean_rotation_error_deg=float(np.mean(rotation_errors_deg)) if rotation_errors_deg else None,
        mean_reprojection_rmse_px=mean_reprojection_rmse_px,
        joint_world_mean_position_error_m=float(np.mean(position_errors_m)) if position_errors_m else None,
        joint_world_mean_rotation_error_deg=float(np.mean(rotation_errors_deg)) if rotation_errors_deg else None,
        solve_rate=float(len(result.camera_pose_vectors)) / max(len(common_frame_indices), 1),
        runtime_ms_per_iteration=result.diagnostics.get("mean_iteration_time_ms"),
        per_frame_records=tuple(per_frame_records),
        per_tag_records=tuple(per_tag_records),
        metadata={
            "alignment_frame_index": int(reference_frame_index),
            "camera_count": len(result.camera_pose_vectors),
            "tag_count": len(result.tag_pose_vectors),
        },
    )
