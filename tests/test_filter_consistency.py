"""Consistency checks for the anchored online filter."""

from __future__ import annotations

import numpy as np

from calib_sim.isaac.estimation.online_filter import AnchoredOnlineFilter
from calib_sim.isaac.estimation.uncertainty import is_positive_semidefinite
from calib_sim.isaac.logging.schemas import IsaacImuPacket


def _stationary_packets(count: int) -> tuple[IsaacImuPacket, ...]:
    return tuple(
        IsaacImuPacket(
            packet_index=index,
            timestamp_s=0.01 * (index + 1),
            sim_time_s=0.01 * (index + 1),
            dt_s=0.01,
            wx=0.0,
            wy=0.0,
            wz=0.0,
            ax=0.0,
            ay=0.0,
            az=9.81,
            imu_frame="I",
            imu_semantics="specific_force",
            noise_preset="ideal",
        )
        for index in range(count)
    )


def test_zero_noise_stationary_propagation_stays_near_origin() -> None:
    online_filter = AnchoredOnlineFilter.identity_initialized()
    online_filter.propagate(_stationary_packets(5))

    assert np.allclose(online_filter.state.position_world_m, 0.0, atol=1e-9)
    assert np.allclose(online_filter.state.velocity_world_mps, 0.0, atol=1e-9)
    assert np.allclose(online_filter.state.rotation_wi, np.eye(3), atol=1e-9)


def test_anchor_update_reduces_covariance_and_keeps_psd() -> None:
    online_filter = AnchoredOnlineFilter.identity_initialized()
    before_trace = float(np.trace(online_filter.covariance))
    online_filter.anchor_visual_update(
        measured_position_world_m=np.zeros(3, dtype=np.float64),
        measured_rotation_wi=np.eye(3, dtype=np.float64),
        measurement_std_m=0.01,
    )
    after_trace = float(np.trace(online_filter.covariance))
    snapshot = online_filter.export_snapshot(sim_time_s=0.1)

    assert after_trace < before_trace
    assert is_positive_semidefinite(snapshot.covariance)
    assert snapshot.innovation_diagnostics["position_radius_95_m"] > 0.0
