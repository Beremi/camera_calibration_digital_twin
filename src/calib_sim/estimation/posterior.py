"""Laplace posterior approximations for the batch estimators."""

from __future__ import annotations

import math

import numpy as np

from calib_sim.estimation.types import (
    BatchSolveResult,
    GlobalVectorUncertaintySummary,
    PoseUncertaintySummary,
    UncertaintySummary,
    VectorUncertaintySummary,
)


def _radius95(std_values: np.ndarray) -> float:
    return float(1.96 * np.sqrt(np.sum(np.square(std_values))))


def compute_laplace_posterior(
    result: BatchSolveResult,
    *,
    blocks: tuple[str, ...] = ("camera_poses", "tag_poses", "velocity", "biases"),
) -> UncertaintySummary:
    hessian = result.hessian
    if hessian is None or hessian.size == 0:
        return UncertaintySummary(camera_pose_uncertainty=(), tag_pose_uncertainty=(), covariance_condition_number=None)

    regularized_hessian = np.asarray(hessian, dtype=np.float64) + 1e-9 * np.eye(hessian.shape[0], dtype=np.float64)
    covariance = np.linalg.pinv(regularized_hessian)
    covariance_condition_number = float(np.linalg.cond(regularized_hessian))

    camera_indices = sorted(int(index) for index in result.camera_pose_vectors)
    tag_indices = sorted(int(index) for index in result.tag_pose_vectors)
    camera_pose_uncertainty = []
    tag_pose_uncertainty = []
    velocity_uncertainty = []

    state_layout = dict(result.diagnostics.get("state_layout", {}))
    camera_block_size = int(state_layout.get("frame_block_size", 6 if result.stage == "visual_only" else 15))
    tag_block_size = int(state_layout.get("tag_block_size", 6))
    tag_base = int(state_layout.get("tag_base", len(camera_indices) * camera_block_size))
    global_bias_base = state_layout.get("global_bias_base")
    for slot, frame_index in enumerate(camera_indices):
        base = slot * camera_block_size
        block = covariance[base : base + 6, base : base + 6]
        position_std = np.sqrt(np.maximum(np.diag(block[:3, :3]), 0.0))
        rotation_std_deg = np.degrees(np.sqrt(np.maximum(np.diag(block[3:6, 3:6]), 0.0)))
        camera_pose_uncertainty.append(
            PoseUncertaintySummary(
                index=int(frame_index),
                position_std_m=tuple(float(value) for value in position_std.tolist()),
                rotation_std_deg=tuple(float(value) for value in rotation_std_deg.tolist()),
                position_radius_95_m=_radius95(position_std),
                rotation_radius_95_deg=_radius95(rotation_std_deg),
            )
        )
        if result.stage == "visual_inertial" and camera_block_size >= 9 and "velocity" in blocks:
            velocity_block = covariance[base + 6 : base + 9, base + 6 : base + 9]
            velocity_std = np.sqrt(np.maximum(np.diag(velocity_block), 0.0))
            velocity_uncertainty.append(
                VectorUncertaintySummary(
                    index=int(frame_index),
                    std=tuple(float(value) for value in velocity_std.tolist()),
                    radius_95=_radius95(velocity_std),
                )
            )

    if "tag_poses" in blocks:
        for slot, tag_index in enumerate(tag_indices):
            base = tag_base + slot * tag_block_size
            block = covariance[base : base + 6, base : base + 6]
            position_std = np.sqrt(np.maximum(np.diag(block[:3, :3]), 0.0))
            rotation_std_deg = np.degrees(np.sqrt(np.maximum(np.diag(block[3:6, 3:6]), 0.0)))
            tag_pose_uncertainty.append(
                PoseUncertaintySummary(
                    index=int(tag_index),
                    position_std_m=tuple(float(value) for value in position_std.tolist()),
                    rotation_std_deg=tuple(float(value) for value in rotation_std_deg.tolist()),
                    position_radius_95_m=_radius95(position_std),
                    rotation_radius_95_deg=_radius95(rotation_std_deg),
                )
            )

    gyro_bias_uncertainty = None
    accel_bias_uncertainty = None
    if result.stage == "visual_inertial" and global_bias_base is not None and "biases" in blocks:
        bias_base = int(global_bias_base)
        gyro_block = covariance[bias_base : bias_base + 3, bias_base : bias_base + 3]
        accel_block = covariance[bias_base + 3 : bias_base + 6, bias_base + 3 : bias_base + 6]
        gyro_std = np.sqrt(np.maximum(np.diag(gyro_block), 0.0))
        accel_std = np.sqrt(np.maximum(np.diag(accel_block), 0.0))
        gyro_bias_uncertainty = GlobalVectorUncertaintySummary(
            name="global_gyro_bias_rps",
            std=tuple(float(value) for value in gyro_std.tolist()),
            radius_95=_radius95(gyro_std),
        )
        accel_bias_uncertainty = GlobalVectorUncertaintySummary(
            name="global_accel_bias_mps2",
            std=tuple(float(value) for value in accel_std.tolist()),
            radius_95=_radius95(accel_std),
        )

    return UncertaintySummary(
        camera_pose_uncertainty=tuple(camera_pose_uncertainty),
        tag_pose_uncertainty=tuple(tag_pose_uncertainty),
        velocity_uncertainty=tuple(velocity_uncertainty),
        gyro_bias_uncertainty=gyro_bias_uncertainty,
        accel_bias_uncertainty=accel_bias_uncertainty,
        covariance_condition_number=covariance_condition_number,
        metadata={
            "camera_block_size": camera_block_size,
            "tag_block_size": tag_block_size,
            "tag_count": len(tag_indices),
            "state_layout": state_layout,
        },
    )
