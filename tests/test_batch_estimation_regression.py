"""Regression tests for the batch MAP estimation pipeline."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from calib_sim.common.inertial import accelerometer_specific_force_body, gyroscope_measurement_body, imu_pose_from_camera_pose
from calib_sim.estimation import (
    bootstrap_initial_guess,
    compute_laplace_posterior,
    evaluate_batch_result,
    inertial_initial_guess_from_visual_solution,
    load_batch_dataset,
    load_evaluation_data,
    solve_visual_batch_map,
    solve_visual_inertial_batch_map,
)
from calib_sim.estimation._geometry import pose_components_from_vector, pose_vector_from_components, project_world_points_to_pixels, rotation_matrix_from_rpy_deg
from calib_sim.estimation.graph_build import build_visual_graph, build_visual_inertial_graph
from calib_sim.estimation.types import CameraFrame, ImuPacket, TagDetectionObservation, TagSpec


RUN_DIR = Path("output/interactive_runs/run_20260406_083611")
HEADLINE_RUN_DIR = Path("output/interactive_runs/run_20260407_143339")


def _build_exact_two_frame_dataset():
    base = load_batch_dataset(HEADLINE_RUN_DIR)
    tag_spec = TagSpec(
        tag_id=999,
        family="36h11",
        size_m=0.06,
        position_m=(0.0, 0.0, 1.0),
        mount="wall",
        orientation_rpy_deg=(0.0, 0.0, 0.0),
    )
    tag_rotation_wt = np.eye(3, dtype=np.float64)
    frame_pose_vectors = {
        0: pose_vector_from_components(np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64)),
        1: pose_vector_from_components(np.array([0.05, 0.0, 0.0], dtype=np.float64), np.eye(3, dtype=np.float64)),
    }
    frame_times = {0: 0.0, 1: 0.1}
    frames = tuple(
        CameraFrame(
            frame_index=frame_index,
            recording_frame_index=frame_index,
            tick_index=frame_index,
            timestamp_s=frame_times[frame_index],
            servo_positions_deg=(0.0, 0.0, 0.0),
            servo_targets_deg=(0.0, 0.0, 0.0),
            auto_demo_enabled=False,
            camera_model_name=base.camera_model.name,
        )
        for frame_index in (0, 1)
    )
    world_points = np.asarray(tag_spec.position_m, dtype=np.float64).reshape(1, 3) + (
        tag_rotation_wt @ tag_spec.corner_points_local_m().T
    ).T
    detections = []
    for frame_index in (0, 1):
        camera_position_world_m, camera_rotation_wc = pose_components_from_vector(frame_pose_vectors[frame_index])
        projected_pixels_px, visible_mask = project_world_points_to_pixels(
            camera_model=base.camera_model,
            camera_position_world_m=camera_position_world_m,
            camera_rotation_wc=camera_rotation_wc,
            world_points_m=world_points,
        )
        assert bool(np.all(visible_mask))
        center_xy_px = np.mean(projected_pixels_px, axis=0)
        detections.append(
            TagDetectionObservation(
                frame_index=frame_index,
                recording_frame_index=frame_index,
                tick_index=frame_index,
                timestamp_s=frame_times[frame_index],
                tag_id=tag_spec.tag_id,
                family=tag_spec.family,
                corners_xy_clockwise_px=tuple((float(point[0]), float(point[1])) for point in projected_pixels_px.tolist()),
                center_xy_px=(float(center_xy_px[0]), float(center_xy_px[1])),
                points5_xy_px=(),
                physical_edge_length_m=tag_spec.size_m,
                camera_model_name=base.camera_model.name,
            )
        )

    rotation_ci = rotation_matrix_from_rpy_deg(tuple(float(value) for value in base.device_config.mount.imu_rpy_deg))
    _, start_rotation_wc = pose_components_from_vector(frame_pose_vectors[0])
    _, end_rotation_wc = pose_components_from_vector(frame_pose_vectors[1])
    start_imu_position_world_m, start_imu_rotation_wi = imu_pose_from_camera_pose(
        camera_position_world_m=np.zeros(3, dtype=np.float64),
        camera_rotation_wc=start_rotation_wc,
        imu_translation_camera_m=base.device_config.mount.imu_translation_m,
        rotation_ci=rotation_ci,
    )
    end_imu_position_world_m, end_imu_rotation_wi = imu_pose_from_camera_pose(
        camera_position_world_m=np.array([0.05, 0.0, 0.0], dtype=np.float64),
        camera_rotation_wc=end_rotation_wc,
        imu_translation_camera_m=base.device_config.mount.imu_translation_m,
        rotation_ci=rotation_ci,
    )
    del start_imu_position_world_m, end_imu_position_world_m
    start_velocity_world_mps = np.array([0.5, 0.0, 0.0], dtype=np.float64)
    end_velocity_world_mps = np.array([0.5, 0.0, 0.0], dtype=np.float64)
    dt_s = 0.1
    accel_packet = accelerometer_specific_force_body(
        rotation_wi=end_imu_rotation_wi,
        acceleration_world_mps2=np.zeros(3, dtype=np.float64),
    )
    gyro_packet = gyroscope_measurement_body(
        start_rotation_wi=start_imu_rotation_wi,
        end_rotation_wi=end_imu_rotation_wi,
        delta_time_s=dt_s,
    )
    imu_packets = (
        ImuPacket(
            tick_index=1,
            timestamp_s=frame_times[1],
            accel_body_mps2=tuple(float(value) for value in accel_packet.tolist()),
            gyro_body_rps=tuple(float(value) for value in gyro_packet.tolist()),
        ),
    )
    dataset = replace(
        base,
        run_name="test_exact_two_frame",
        config_name="test_exact_two_frame",
        tag_catalog={tag_spec.tag_id: tag_spec},
        camera_frames=frames,
        tag_detections=tuple(detections),
        imu_packets=imu_packets,
    )
    truth = {
        "camera_pose_vectors": frame_pose_vectors,
        "tag_pose_vector": pose_vector_from_components(np.asarray(tag_spec.position_m, dtype=np.float64), tag_rotation_wt),
        "velocity_world_mps": end_velocity_world_mps,
    }
    return dataset, truth


def test_visual_graph_prior_uses_bootstrap_anchor_frame() -> None:
    dataset = load_batch_dataset(RUN_DIR)
    init = bootstrap_initial_guess(dataset, stage="visual_only")

    graph = build_visual_graph(dataset, anchor_frame_index=init.anchor_frame_index)
    default_graph = build_visual_graph(dataset)
    inertial_graph = build_visual_inertial_graph(dataset, anchor_frame_index=init.anchor_frame_index)

    assert init.anchor_frame_index == 119
    assert graph.pose_priors[0].index == 119
    assert default_graph.pose_priors[0].index == 1
    imu_prior = next(prior for prior in inertial_graph.pose_priors if prior.variable_kind == "imu_state")
    assert imu_prior.index == 119


def test_preserved_checkpoint_visual_and_fused_batch_solves_stay_accurate() -> None:
    dataset = load_batch_dataset(RUN_DIR)
    evaluation_data = load_evaluation_data(RUN_DIR)

    visual_init = bootstrap_initial_guess(dataset, stage="visual_only")
    visual_result = solve_visual_batch_map(dataset, visual_init, "vision_nominal")
    visual_eval = evaluate_batch_result(visual_result, evaluation_data, dataset=dataset)

    assert visual_result.success is True
    assert visual_result.iterations >= 4
    assert visual_result.final_cost < 250.0
    assert visual_eval.mean_position_error_m is not None and visual_eval.mean_position_error_m < 0.002
    assert visual_eval.mean_rotation_error_deg is not None and visual_eval.mean_rotation_error_deg < 0.10
    assert visual_eval.mean_reprojection_rmse_px is not None and visual_eval.mean_reprojection_rmse_px < 0.20

    fused_init = inertial_initial_guess_from_visual_solution(
        dataset,
        visual_result,
        anchor_frame_index=int(visual_init.anchor_frame_index),
    )
    fused_result = solve_visual_inertial_batch_map(dataset, fused_init, noise_cfg=None)
    fused_posterior = compute_laplace_posterior(fused_result)
    fused_eval = evaluate_batch_result(fused_result, evaluation_data, dataset=dataset, uncertainty_summary=fused_posterior)

    assert fused_result.success is True
    assert fused_result.iterations >= 3
    # The preserved checkpoint IMU stream predates the IMU-frame lever-arm fix and is now
    # treated as a compatibility baseline rather than the scientific headline benchmark.
    assert fused_eval.mean_position_error_m is not None and fused_eval.mean_position_error_m < 0.02
    assert fused_eval.mean_rotation_error_deg is not None and fused_eval.mean_rotation_error_deg < 0.25
    assert fused_eval.mean_reprojection_rmse_px is not None and fused_eval.mean_reprojection_rmse_px < 0.20
    assert fused_result.diagnostics["state_layout"]["frame_block_size"] == 9
    assert fused_result.diagnostics["mean_whitened_visual_sq_residual_per_corner"] is not None
    assert fused_result.diagnostics["mean_whitened_imu_sq_residual_per_factor"] is not None
    assert fused_posterior.gyro_bias_uncertainty is not None
    assert fused_posterior.accel_bias_uncertainty is not None
    first_frame_index = sorted(fused_result.camera_pose_vectors)[0]
    assert fused_result.gyro_bias_rps_by_frame[first_frame_index].tolist() == fused_result.global_gyro_bias_rps.tolist()
    assert fused_result.accel_bias_mps2_by_frame[first_frame_index].tolist() == fused_result.global_accel_bias_mps2.tolist()


def test_visual_only_solver_recovers_exact_zero_noise_unknown_map_dataset() -> None:
    dataset, truth = _build_exact_two_frame_dataset()
    init = bootstrap_initial_guess(dataset, stage="visual_only")
    result = solve_visual_batch_map(dataset, init, "vision_nominal", variant="test_exact_visual")

    assert result.success is True
    assert result.diagnostics["factor_breakdown"]["mean_whitened_visual_sq_residual_per_corner"] < 1.0e-8
    for frame_index, truth_pose in truth["camera_pose_vectors"].items():
        assert np.allclose(result.camera_pose_vectors[frame_index], truth_pose, atol=1.0e-6)
    assert np.allclose(result.tag_pose_vectors[999], truth["tag_pose_vector"], atol=1.0e-6)


def test_fused_solver_matches_exact_zero_noise_visual_inertial_dataset() -> None:
    dataset, truth = _build_exact_two_frame_dataset()
    visual_init = bootstrap_initial_guess(dataset, stage="visual_only")
    visual_result = solve_visual_batch_map(dataset, visual_init, "vision_nominal", variant="test_exact_visual_seed")
    fused_init = inertial_initial_guess_from_visual_solution(
        dataset,
        visual_result,
        anchor_frame_index=int(visual_init.anchor_frame_index),
    )
    fused_result = solve_visual_inertial_batch_map(dataset, fused_init, noise_cfg=None, variant="test_exact_fused")

    assert fused_result.success is True
    assert fused_result.diagnostics["mean_whitened_imu_sq_residual_per_factor"] < 1.0e-8
    assert np.linalg.norm(fused_result.global_gyro_bias_rps) < 1.0e-8
    assert np.linalg.norm(fused_result.global_accel_bias_mps2) < 1.0e-8
    for frame_index, truth_pose in truth["camera_pose_vectors"].items():
        assert np.allclose(fused_result.camera_pose_vectors[frame_index], truth_pose, atol=1.0e-6)
    assert np.allclose(fused_result.tag_pose_vectors[999], truth["tag_pose_vector"], atol=1.0e-6)
    assert np.allclose(fused_result.velocity_world_mps_by_frame[1], truth["velocity_world_mps"], atol=1.0e-6)


def test_async_headline_clean_run_fused_matches_or_beats_visual() -> None:
    dataset = load_batch_dataset(HEADLINE_RUN_DIR)
    evaluation_data = load_evaluation_data(HEADLINE_RUN_DIR)

    visual_init = bootstrap_initial_guess(dataset, stage="visual_only")
    visual_result = solve_visual_batch_map(dataset, visual_init, "vision_nominal")
    visual_eval = evaluate_batch_result(visual_result, evaluation_data, dataset=dataset)

    fused_init = inertial_initial_guess_from_visual_solution(
        dataset,
        visual_result,
        anchor_frame_index=int(visual_init.anchor_frame_index),
    )
    fused_result = solve_visual_inertial_batch_map(dataset, fused_init, noise_cfg=None)
    fused_posterior = compute_laplace_posterior(fused_result)
    fused_eval = evaluate_batch_result(fused_result, evaluation_data, dataset=dataset, uncertainty_summary=fused_posterior)

    assert fused_eval.mean_position_error_m is not None
    assert fused_eval.mean_rotation_error_deg is not None
    assert visual_eval.mean_position_error_m is not None
    assert visual_eval.mean_rotation_error_deg is not None
    assert fused_eval.mean_position_error_m <= visual_eval.mean_position_error_m
    assert fused_eval.mean_rotation_error_deg <= visual_eval.mean_rotation_error_deg
    assert fused_result.diagnostics["mean_whitened_imu_sq_residual_per_factor"] < 12.0
    assert np.linalg.norm(np.asarray(fused_result.global_accel_bias_mps2, dtype=np.float64)) < 2.0
