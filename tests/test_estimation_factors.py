"""Focused tests for reusable estimation factor utilities."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from calib_sim.common.inertial import (
    accelerometer_specific_force_body,
    camera_pose_from_imu_pose,
    gyroscope_measurement_body,
    imu_pose_from_camera_pose,
    inertial_transition_residual_components,
    integrate_interval_constant_world_acceleration,
)
from calib_sim.common.models import TagDetection
from calib_sim.estimation import load_batch_dataset, load_evaluation_data
from calib_sim.estimation._geometry import rotation_matrix_from_rpy_deg
from calib_sim.estimation.backends.jax_validation import project_tag_corners_jax, validate_tag_corner_projection
from calib_sim.estimation.factors.imu_preintegration import ImuSample, integrate_discrete_imu_sequence, preintegrate_imu_samples
from calib_sim.estimation.factors.priors import gaussian_prior_residual, pose_prior_residual
from calib_sim.estimation.factors.tag_corner_factor import (
    canonical_tag_corner_order,
    canonical_tag_corner_pixels,
    canonical_tag_object_points_m,
    project_tag_corners_pixels,
    tag_corner_pixel_residual,
)
from calib_sim.estimation.graph_build import build_visual_inertial_graph
from calib_sim.interactive.camera_model import load_phone_camera_model


ASYNC_RUN_DIR = Path("output/interactive_runs/run_20260407_143339")


def _identity_rotation() -> np.ndarray:
    return np.eye(3, dtype=np.float64)


def test_tag_corner_order_and_projection_match_current_camera_model() -> None:
    model = load_phone_camera_model("config/camera/pixel_9a_main.toml", output_width_px=960, output_height_px=540)
    assert canonical_tag_corner_order() == ("top_right", "bottom_right", "bottom_left", "top_left")

    detection = TagDetection(
        family="36h11",
        tag_id=42,
        corners_xy_clockwise=[
            (10.0, 20.0),
            (30.0, 20.0),
            (30.0, 60.0),
            (10.0, 60.0),
        ],
        center_xy=(20.0, 40.0),
        points5_xy=[],
    )
    assert np.allclose(
        canonical_tag_corner_pixels(detection),
        np.array([[30.0, 20.0], [30.0, 60.0], [10.0, 60.0], [10.0, 20.0]], dtype=np.float64),
    )

    camera_rotation_cw = _identity_rotation()
    camera_position_world_m = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    tag_center_world_m = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    tag_rotation_wt = _identity_rotation()
    projected = project_tag_corners_pixels(
        camera_model=model,
        camera_position_world_m=camera_position_world_m,
        camera_rotation_cw=camera_rotation_cw,
        tag_center_world_m=tag_center_world_m,
        tag_rotation_wt=tag_rotation_wt,
        tag_size_m=0.1,
        apply_distortion=False,
    )
    expected = np.array(
        [
            [512.4754199999999, 304.145364],
            [512.4754199999999, 239.486796],
            [447.816852, 239.486796],
            [447.816852, 304.145364],
        ],
        dtype=np.float64,
    )
    assert np.allclose(projected, expected, atol=1e-6)

    residual = tag_corner_pixel_residual(
        camera_model=model,
        observed_corner_pixels_px=projected,
        camera_position_world_m=camera_position_world_m,
        camera_rotation_cw=camera_rotation_cw,
        tag_center_world_m=tag_center_world_m,
        tag_rotation_wt=tag_rotation_wt,
        tag_size_m=0.1,
        apply_distortion=False,
    )
    assert np.allclose(residual, np.zeros(8, dtype=np.float64))

    jax_projected = project_tag_corners_jax(
        camera_model=model,
        camera_position_world_m=camera_position_world_m,
        camera_rotation_cw=camera_rotation_cw,
        tag_center_world_m=tag_center_world_m,
        tag_rotation_wt=tag_rotation_wt,
        tag_size_m=0.1,
    )
    assert np.allclose(jax_projected, projected, atol=1e-9)
    assert validate_tag_corner_projection(
        camera_model=model,
        camera_position_world_m=camera_position_world_m,
        camera_rotation_cw=camera_rotation_cw,
        tag_center_world_m=tag_center_world_m,
        tag_rotation_wt=tag_rotation_wt,
        tag_size_m=0.1,
    )["max_abs_error_px"] < 1e-9


def test_gaussian_and_pose_priors_are_linear_residuals() -> None:
    sqrt_information = np.diag([2.0, 3.0, 4.0]).astype(np.float64)
    residual = gaussian_prior_residual([1.0, 2.0, 3.0], [0.5, 1.0, 2.0], sqrt_information)
    assert np.allclose(residual, np.array([1.0, 3.0, 4.0], dtype=np.float64))
    assert np.allclose(residual, pose_prior_residual([1.0, 2.0, 3.0], [0.5, 1.0, 2.0], sqrt_information))


def test_discrete_imu_integration_matches_repo_convention_exactly() -> None:
    samples = [
        ImuSample(
            timestamp_s=0.1,
            dt_s=0.1,
            accel_body_mps2=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            gyro_body_rps=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        ),
        ImuSample(
            timestamp_s=0.2,
            dt_s=0.1,
            accel_body_mps2=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            gyro_body_rps=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        ),
    ]
    result = integrate_discrete_imu_sequence(
        samples,
        initial_position_world_m=[0.0, 0.0, 0.0],
        initial_rotation_cw=_identity_rotation(),
        initial_velocity_world_mps=[0.0, 0.0, 0.0],
        gravity_world_mps2=[0.0, 0.0, -9.81],
    )
    assert np.allclose(result["final_rotation_cw"], np.eye(3), atol=1e-12)
    assert np.allclose(result["final_velocity_world_mps"], np.array([0.0, 0.0, -1.762], dtype=np.float64), atol=1e-12)
    assert np.allclose(result["final_position_world_m"], np.array([0.0, 0.0, -0.3524], dtype=np.float64), atol=1e-12)

    delta = preintegrate_imu_samples(
        samples,
        initial_rotation_cw=_identity_rotation(),
        initial_velocity_world_mps=[0.0, 0.0, 0.0],
        gravity_world_mps2=[0.0, 0.0, -9.81],
    )
    assert math.isclose(delta.delta_time_s, 0.2, rel_tol=0.0, abs_tol=1e-12)
    assert np.allclose(delta.delta_rotation_cw, np.eye(3), atol=1e-12)
    assert np.allclose(delta.delta_velocity_world_mps, np.array([0.0, 0.0, -1.762], dtype=np.float64), atol=1e-12)
    assert np.allclose(delta.delta_position_world_m, np.array([0.0, 0.0, -0.3524], dtype=np.float64), atol=1e-12)


def test_camera_and_imu_pose_transforms_are_inverse_consistent() -> None:
    camera_position_world_m = np.array([0.2, -0.1, 0.7], dtype=np.float64)
    camera_rotation_wc = rotation_matrix_from_rpy_deg((12.0, -7.0, 18.0))
    imu_translation_camera_m = (0.01, -0.02, 0.03)
    rotation_ci = rotation_matrix_from_rpy_deg((4.0, -3.0, 8.0))

    imu_position_world_m, imu_rotation_wi = imu_pose_from_camera_pose(
        camera_position_world_m=camera_position_world_m,
        camera_rotation_wc=camera_rotation_wc,
        imu_translation_camera_m=imu_translation_camera_m,
        rotation_ci=rotation_ci,
    )
    recovered_camera_position_world_m, recovered_camera_rotation_wc = camera_pose_from_imu_pose(
        imu_position_world_m=imu_position_world_m,
        imu_rotation_wi=imu_rotation_wi,
        imu_translation_camera_m=imu_translation_camera_m,
        rotation_ci=rotation_ci,
    )
    assert np.allclose(recovered_camera_position_world_m, camera_position_world_m, atol=1e-12)
    assert np.allclose(recovered_camera_rotation_wc, camera_rotation_wc, atol=1e-12)


def test_specific_force_helper_matches_documented_equation() -> None:
    rotation_wi = np.eye(3, dtype=np.float64)
    acceleration_world_mps2 = np.array([0.3, -0.2, 9.9], dtype=np.float64)
    gravity_world_mps2 = np.array([0.0, 0.0, -9.81], dtype=np.float64)
    accel_bias_mps2 = np.array([0.01, -0.02, 0.03], dtype=np.float64)

    measured = accelerometer_specific_force_body(
        rotation_wi=rotation_wi,
        acceleration_world_mps2=acceleration_world_mps2,
        gravity_world_mps2=gravity_world_mps2,
        accel_bias_mps2=accel_bias_mps2,
    )

    expected = acceleration_world_mps2 - gravity_world_mps2 + accel_bias_mps2
    assert np.allclose(measured, expected, atol=1e-12)


def test_known_constant_bias_is_explained_by_matching_inertial_model() -> None:
    start_position_world_m = np.zeros(3, dtype=np.float64)
    start_rotation_wi = np.eye(3, dtype=np.float64)
    start_velocity_world_mps = np.zeros(3, dtype=np.float64)
    acceleration_world_mps2 = np.array([0.5, -0.1, 0.2], dtype=np.float64)
    gravity_world_mps2 = np.array([0.0, 0.0, -9.81], dtype=np.float64)
    gyro_bias_rps = np.array([0.01, -0.015, 0.005], dtype=np.float64)
    accel_bias_mps2 = np.array([0.12, -0.08, 0.04], dtype=np.float64)
    dt_packets = np.array([0.1, 0.1], dtype=np.float64)
    accel_packets = np.vstack(
        [
            accelerometer_specific_force_body(
                rotation_wi=start_rotation_wi,
                acceleration_world_mps2=acceleration_world_mps2,
                gravity_world_mps2=gravity_world_mps2,
                accel_bias_mps2=accel_bias_mps2,
            )
            for _ in range(2)
        ]
    )
    gyro_packets = np.vstack(
        [
            gyroscope_measurement_body(
                start_rotation_wi=start_rotation_wi,
                end_rotation_wi=start_rotation_wi,
                delta_time_s=float(dt_packets[0]),
                gyro_bias_rps=gyro_bias_rps,
            )
            for _ in range(2)
        ]
    )
    propagated = integrate_interval_constant_world_acceleration(
        start_position_world_m=start_position_world_m,
        start_rotation_wi=start_rotation_wi,
        start_velocity_world_mps=start_velocity_world_mps,
        gyro_packets_body_rps=gyro_packets,
        accel_packets_body_mps2=accel_packets,
        dt_packets_s=dt_packets,
        gravity_world_mps2=gravity_world_mps2,
        accel_bias_mps2=accel_bias_mps2,
        gyro_bias_rps=gyro_bias_rps,
    )
    matching = inertial_transition_residual_components(
        start_position_world_m=start_position_world_m,
        start_rotation_wi=start_rotation_wi,
        start_velocity_world_mps=start_velocity_world_mps,
        end_position_world_m=propagated.final_position_world_m,
        end_rotation_wi=propagated.final_rotation_wi,
        end_velocity_world_mps=propagated.final_velocity_world_mps,
        gyro_packets_body_rps=gyro_packets,
        accel_packets_body_mps2=accel_packets,
        dt_packets_s=dt_packets,
        gravity_world_mps2=gravity_world_mps2,
        accel_bias_mps2=accel_bias_mps2,
        gyro_bias_rps=gyro_bias_rps,
    )
    mismatched = inertial_transition_residual_components(
        start_position_world_m=start_position_world_m,
        start_rotation_wi=start_rotation_wi,
        start_velocity_world_mps=start_velocity_world_mps,
        end_position_world_m=propagated.final_position_world_m,
        end_rotation_wi=propagated.final_rotation_wi,
        end_velocity_world_mps=propagated.final_velocity_world_mps,
        gyro_packets_body_rps=gyro_packets,
        accel_packets_body_mps2=accel_packets,
        dt_packets_s=dt_packets,
        gravity_world_mps2=gravity_world_mps2,
        accel_bias_mps2=(0.0, 0.0, 0.0),
        gyro_bias_rps=(0.0, 0.0, 0.0),
    )
    assert np.linalg.norm(matching.stacked()) < 1e-10
    assert np.linalg.norm(mismatched.stacked()) > 1e-3


def test_async_headline_gt_imu_packets_match_interval_factor_convention() -> None:
    dataset = load_batch_dataset(ASYNC_RUN_DIR)
    evaluation = load_evaluation_data(ASYNC_RUN_DIR)
    rotation_ci = rotation_matrix_from_rpy_deg(tuple(float(value) for value in dataset.device_config.mount.imu_rpy_deg))
    graph = build_visual_inertial_graph(dataset, anchor_frame_index=0)
    truth_lookup = {int(sample.frame_index): sample for sample in evaluation.camera_truth}
    sorted_truth = sorted(evaluation.camera_truth, key=lambda item: int(item.frame_index))

    imu_positions_world_m: dict[int, np.ndarray] = {}
    for truth in sorted_truth:
        imu_position_world_m, _ = imu_pose_from_camera_pose(
            camera_position_world_m=np.asarray(truth.position_world_m, dtype=np.float64),
            camera_rotation_wc=np.asarray(truth.rotation_cw, dtype=np.float64).reshape(3, 3),
            imu_translation_camera_m=dataset.device_config.mount.imu_translation_m,
            rotation_ci=rotation_ci,
        )
        imu_positions_world_m[int(truth.frame_index)] = imu_position_world_m
    imu_velocities_world_mps: dict[int, np.ndarray] = {}
    frame_indices = [int(item.frame_index) for item in sorted_truth]
    for slot, frame_index in enumerate(frame_indices):
        if slot == 0:
            next_frame_index = frame_indices[min(slot + 1, len(frame_indices) - 1)]
            dt_s = max(float(truth_lookup[next_frame_index].timestamp_s) - float(truth_lookup[frame_index].timestamp_s), 1e-9)
            imu_velocities_world_mps[frame_index] = (imu_positions_world_m[next_frame_index] - imu_positions_world_m[frame_index]) / dt_s
        else:
            prev_frame_index = frame_indices[slot - 1]
            dt_s = max(float(truth_lookup[frame_index].timestamp_s) - float(truth_lookup[prev_frame_index].timestamp_s), 1e-9)
            imu_velocities_world_mps[frame_index] = (imu_positions_world_m[frame_index] - imu_positions_world_m[prev_frame_index]) / dt_s

    position_norms = []
    rotation_norms = []
    velocity_norms = []
    accel_bias_truth = np.asarray(evaluation.global_accel_bias_mps2_truth or (0.0, 0.0, 0.0), dtype=np.float64)
    gyro_bias_truth = np.asarray(evaluation.global_gyro_bias_rps_truth or (0.0, 0.0, 0.0), dtype=np.float64)
    for factor in graph.imu_factors[: min(32, len(graph.imu_factors))]:
        start_truth = truth_lookup[int(factor.start_frame_index)]
        end_truth = truth_lookup[int(factor.end_frame_index)]
        start_imu_position_world_m, start_imu_rotation_wi = imu_pose_from_camera_pose(
            camera_position_world_m=np.asarray(start_truth.position_world_m, dtype=np.float64),
            camera_rotation_wc=np.asarray(start_truth.rotation_cw, dtype=np.float64).reshape(3, 3),
            imu_translation_camera_m=dataset.device_config.mount.imu_translation_m,
            rotation_ci=rotation_ci,
        )
        end_imu_position_world_m, end_imu_rotation_wi = imu_pose_from_camera_pose(
            camera_position_world_m=np.asarray(end_truth.position_world_m, dtype=np.float64),
            camera_rotation_wc=np.asarray(end_truth.rotation_cw, dtype=np.float64).reshape(3, 3),
            imu_translation_camera_m=dataset.device_config.mount.imu_translation_m,
            rotation_ci=rotation_ci,
        )
        packet_sequence = [dataset.imu_packets[packet_index] for packet_index in factor.packet_indices]
        residual = inertial_transition_residual_components(
            start_position_world_m=start_imu_position_world_m,
            start_rotation_wi=start_imu_rotation_wi,
            start_velocity_world_mps=imu_velocities_world_mps[int(factor.start_frame_index)],
            end_position_world_m=end_imu_position_world_m,
            end_rotation_wi=end_imu_rotation_wi,
            end_velocity_world_mps=imu_velocities_world_mps[int(factor.end_frame_index)],
            gyro_packets_body_rps=np.asarray([packet.gyro_body_rps for packet in packet_sequence], dtype=np.float64),
            accel_packets_body_mps2=np.asarray([packet.accel_body_mps2 for packet in packet_sequence], dtype=np.float64),
            dt_packets_s=np.asarray(factor.dt_s, dtype=np.float64),
            accel_bias_mps2=accel_bias_truth,
            gyro_bias_rps=gyro_bias_truth,
        )
        position_norms.append(float(np.linalg.norm(residual.position)))
        rotation_norms.append(float(np.linalg.norm(residual.rotation)))
        velocity_norms.append(float(np.linalg.norm(residual.velocity)))

    assert float(np.mean(np.asarray(position_norms, dtype=np.float64))) < 2.5e-3
    assert float(np.mean(np.asarray(rotation_norms, dtype=np.float64))) < 2.5e-3
    assert float(np.mean(np.asarray(velocity_norms, dtype=np.float64))) < 3.5e-3
