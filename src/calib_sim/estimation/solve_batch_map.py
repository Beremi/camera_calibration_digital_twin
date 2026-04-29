"""Public solver entrypoints for the batch MAP estimators."""

from __future__ import annotations

from typing import Any

import numpy as np

from calib_sim.common.inertial import imu_pose_from_camera_pose
from calib_sim.estimation._geometry import pose_components_from_vector, pose_vector_from_components, rotation_matrix_from_rpy_deg
from calib_sim.estimation.backends.jax_backend import solve_visual_block_lm, solve_visual_inertial_block_lm
from calib_sim.estimation.graph_build import build_visual_graph, build_visual_inertial_graph
from calib_sim.estimation.noise_models import (
    ImuNoisePreset,
    VisionNoisePreset,
    load_imu_noise_preset,
    load_vision_noise_preset,
)
from calib_sim.estimation.types import BatchCalibrationDataset, BatchSolveResult, InitialGuess


def _resolve_vision_noise(noise_cfg: VisionNoisePreset | str | dict[str, Any] | None) -> VisionNoisePreset:
    if noise_cfg is None:
        return load_vision_noise_preset("vision_nominal")
    if isinstance(noise_cfg, VisionNoisePreset):
        return noise_cfg
    if isinstance(noise_cfg, str):
        return load_vision_noise_preset(noise_cfg)
    return VisionNoisePreset(
        name=str(noise_cfg.get("name", "vision_custom")),
        corner_noise_std_px=float(noise_cfg.get("corner_noise_std_px", 0.55)),
        detection_drop_probability=float(noise_cfg.get("detection_drop_probability", 0.0)),
        quality_scale=float(noise_cfg.get("quality_scale", 1.0)),
        visibility_failure_probability=float(noise_cfg.get("visibility_failure_probability", 0.0)),
        outlier_probability=float(noise_cfg.get("outlier_probability", 0.0)),
    )


def _resolve_imu_noise(noise_cfg: ImuNoisePreset | str | dict[str, Any] | None) -> ImuNoisePreset:
    if noise_cfg is None:
        return load_imu_noise_preset("imu_nominal_phone")
    if isinstance(noise_cfg, ImuNoisePreset):
        return noise_cfg
    if isinstance(noise_cfg, str):
        return load_imu_noise_preset(noise_cfg)
    return ImuNoisePreset(
        name=str(noise_cfg.get("name", "imu_custom")),
        accel_noise_std=tuple(float(value) for value in noise_cfg.get("accel_noise_std", [0.0, 0.0, 0.0])),
        gyro_noise_std=tuple(float(value) for value in noise_cfg.get("gyro_noise_std", [0.0, 0.0, 0.0])),
        accel_bias_random_walk_std=tuple(float(value) for value in noise_cfg.get("accel_bias_random_walk_std", [0.0, 0.0, 0.0])),
        gyro_bias_random_walk_std=tuple(float(value) for value in noise_cfg.get("gyro_bias_random_walk_std", [0.0, 0.0, 0.0])),
        accel_bias_mps2=tuple(float(value) for value in noise_cfg.get("accel_bias_mps2", [0.0, 0.0, 0.0])),
        gyro_bias_rps=tuple(float(value) for value in noise_cfg.get("gyro_bias_rps", [0.0, 0.0, 0.0])),
        sample_drop_probability=float(noise_cfg.get("sample_drop_probability", 0.0)),
        likelihood_scale=float(noise_cfg.get("likelihood_scale", 1.0)),
        position_std_floor=float(noise_cfg.get("position_std_floor", 2.0e-2)),
        velocity_std_floor=float(noise_cfg.get("velocity_std_floor", 2.0e-1)),
        rotation_std_floor=float(noise_cfg.get("rotation_std_floor", 5.0e-2)),
    )


def solve_visual_batch_map(
    dataset: BatchCalibrationDataset,
    init: InitialGuess,
    noise_cfg: VisionNoisePreset | str | dict[str, Any] | None = None,
    *,
    fixed_tag_pose_vectors: dict[int, np.ndarray] | None = None,
    solver_options: dict[str, Any] | None = None,
    variant: str = "visual_only_unknown_map",
) -> BatchSolveResult:
    vision_noise = _resolve_vision_noise(noise_cfg)
    graph = build_visual_graph(dataset, anchor_frame_index=int(init.anchor_frame_index))
    solver_options = {} if solver_options is None else dict(solver_options)
    return solve_visual_block_lm(
        dataset=dataset,
        graph=graph,
        init=init,
        vision_noise=vision_noise,
        fixed_tag_pose_vectors=fixed_tag_pose_vectors,
        max_iterations=int(solver_options.get("max_iterations", 18)),
        initial_damping=float(solver_options.get("initial_damping", 1e-3)),
        variant=variant,
    )


def solve_visual_inertial_batch_map(
    dataset: BatchCalibrationDataset,
    init: InitialGuess,
    noise_cfg: dict[str, Any] | tuple[VisionNoisePreset | str | dict[str, Any] | None, ImuNoisePreset | str | dict[str, Any] | None] | None = None,
    *,
    solver_options: dict[str, Any] | None = None,
    variant: str = "visual_inertial_fused",
) -> BatchSolveResult:
    if isinstance(noise_cfg, tuple):
        vision_noise_cfg, imu_noise_cfg = noise_cfg
    elif isinstance(noise_cfg, dict):
        vision_noise_cfg = noise_cfg.get("vision")
        imu_noise_cfg = noise_cfg.get("imu")
    else:
        vision_noise_cfg = None
        imu_noise_cfg = None
    anchor_camera_pose = np.asarray(init.camera_pose_vectors[int(init.anchor_frame_index)], dtype=np.float64).reshape(6)
    camera_position_world_m, camera_rotation_wc = pose_components_from_vector(anchor_camera_pose)
    rotation_ci = rotation_matrix_from_rpy_deg(tuple(float(value) for value in dataset.device_config.mount.imu_rpy_deg))
    anchor_imu_position_world_m, anchor_imu_rotation_wi = imu_pose_from_camera_pose(
        camera_position_world_m=camera_position_world_m,
        camera_rotation_wc=camera_rotation_wc,
        imu_translation_camera_m=dataset.device_config.mount.imu_translation_m,
        rotation_ci=rotation_ci,
    )
    anchor_imu_pose = pose_vector_from_components(anchor_imu_position_world_m, anchor_imu_rotation_wi)
    graph = build_visual_inertial_graph(
        dataset,
        anchor_frame_index=int(init.anchor_frame_index),
        first_pose_mean_vector=anchor_imu_pose,
        first_velocity_mean_mps=np.asarray(
            init.velocity_world_mps_by_frame.get(int(init.anchor_frame_index), np.zeros(3, dtype=np.float64)),
            dtype=np.float64,
        ),
    )
    solver_options = {} if solver_options is None else dict(solver_options)
    return solve_visual_inertial_block_lm(
        dataset=dataset,
        graph=graph,
        init=init,
        vision_noise=_resolve_vision_noise(vision_noise_cfg),
        imu_noise=_resolve_imu_noise(imu_noise_cfg),
        max_iterations=int(solver_options.get("max_iterations", 14)),
        initial_damping=float(solver_options.get("initial_damping", 1e-3)),
        variant=variant,
    )
