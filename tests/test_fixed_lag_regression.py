"""Regression checks for the fixed-lag smoother."""

from __future__ import annotations

import numpy as np

from calib_sim.isaac.estimation.fixed_lag_smoother import FixedLagSmoother
from calib_sim.isaac.estimation.state_defs import FilterStateSnapshot
from calib_sim.isaac.estimation.uncertainty import is_positive_semidefinite


def _snapshot(index: int) -> FilterStateSnapshot:
    return FilterStateSnapshot(
        timestamp_s=0.1 * index,
        sim_time_s=0.1 * index,
        rotation_wi=np.eye(3, dtype=np.float64),
        position_world_m=np.array([0.1 * index, 0.0, 0.0], dtype=np.float64),
        velocity_world_mps=np.zeros(3, dtype=np.float64),
        gyro_bias_rps=np.zeros(3, dtype=np.float64),
        accel_bias_mps2=np.zeros(3, dtype=np.float64),
        covariance=np.eye(15, dtype=np.float64) * (1e-3 + index * 1e-4),
        innovation_diagnostics={"last_innovation_norm": 0.01 * index},
    )


def test_fixed_lag_window_limits_clone_count_and_keeps_psd_covariance() -> None:
    smoother = FixedLagSmoother(lag_size=2)
    smoother.push_snapshot(_snapshot(0))
    smoother.push_snapshot(_snapshot(1))
    smoother.push_snapshot(_snapshot(2))
    smoother.refine_tag_pose(tag_id=42, pose_wt=np.eye(4, dtype=np.float64))
    smoother.refine_tag_pose(tag_id=42, pose_wt=np.eye(4, dtype=np.float64) * 2.0)
    solved = smoother.solve()

    assert len(solved.cloned_positions_world_m) == 2
    assert 42 in solved.active_tag_poses
    assert solved.active_tag_poses[42].shape == (4, 4)
    assert np.all(np.isfinite(solved.covariance))
    assert is_positive_semidefinite(solved.covariance)
