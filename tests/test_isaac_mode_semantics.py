"""Mode-splitting regression tests for the first-pass Isaac runtime."""

from __future__ import annotations

import numpy as np

from calib_sim.isaac.app import IsaacAppBootstrapConfig, create_runtime
from calib_sim.isaac.estimation.online_filter import AnchoredOnlineFilter
from calib_sim.isaac.logging.schemas import IsaacImuPacket


def _bootstrap(tmp_path, *, estimator_mode: str, controller_mode: str, bootstrap_control_policy: str = "hold_until_first_detection"):
    return IsaacAppBootstrapConfig(
        scene_config_path="config/isaac/scene/anchor_room.yaml",
        robot_config_path="config/isaac/robot/franka_phone_head.yaml",
        camera_config_path="config/isaac/camera/phone_main.yaml",
        imu_config_path="config/isaac/imu/phone_nominal.yaml",
        actuation_config_path="config/isaac/actuation/servo_nominal.yaml",
        estimation_config_path="config/isaac/estimation/anchored_vio.yaml",
        control_config_path="config/isaac/control/path_tracking.yaml",
        run_id="mode_test",
        run_dir=str((tmp_path / "isaac_runs" / "mode_test").resolve()),
        headless=True,
        duration_s=0.1,
        max_steps=1,
        estimator_mode=estimator_mode,
        controller_mode=controller_mode,
        bootstrap_control_policy=bootstrap_control_policy,
    )


def _rotating_packets() -> tuple[IsaacImuPacket, ...]:
    return (
        IsaacImuPacket(
            packet_index=0,
            timestamp_s=0.01,
            sim_time_s=0.01,
            dt_s=0.01,
            wx=0.0,
            wy=0.0,
            wz=1.0,
            ax=0.0,
            ay=0.0,
            az=9.81,
            imu_frame="I",
            imu_semantics="specific_force",
            noise_preset="ideal",
        ),
    )


def test_visual_mode_disables_imu_prediction_while_fused_mode_uses_it() -> None:
    visual_filter = AnchoredOnlineFilter.identity_initialized(estimator_mode="visual")
    fused_filter = AnchoredOnlineFilter.identity_initialized(estimator_mode="fused")

    visual_filter.predict(_rotating_packets())
    fused_filter.predict(_rotating_packets())

    assert np.allclose(visual_filter.state.rotation_wi, np.eye(3), atol=1e-9)
    assert not np.allclose(fused_filter.state.rotation_wi, np.eye(3), atol=1e-6)


def test_open_loop_control_state_uses_memory_not_estimate(tmp_path) -> None:
    runtime = create_runtime(_bootstrap(tmp_path, estimator_mode="fused", controller_mode="open-loop"))
    runtime._last_open_loop_target_position = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    runtime._filter = AnchoredOnlineFilter.identity_initialized(estimator_mode="fused")
    runtime._filter.state.position_world_m = np.array([9.0, 9.0, 9.0], dtype=np.float64)
    runtime._latest_filter_snapshot = object()

    position, source, hold = runtime._resolve_control_state(ee_position_world_m=np.zeros(3, dtype=np.float64))

    assert source == "open_loop_memory"
    assert hold is False
    assert np.allclose(position, np.array([1.0, 2.0, 3.0], dtype=np.float64))


def test_open_loop_control_state_initializes_from_control_frame_when_available(tmp_path) -> None:
    class _CameraBinding:
        def get_world_pose(self):
            return np.array([0.55, -0.40, 1.16], dtype=np.float64), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    runtime = create_runtime(_bootstrap(tmp_path, estimator_mode="fused", controller_mode="open-loop"))
    runtime._camera_binding = _CameraBinding()

    position, source, hold = runtime._resolve_control_state(ee_position_world_m=np.array([0.0, 0.0, 0.46], dtype=np.float64))

    assert source == "open_loop_memory"
    assert hold is False
    assert np.allclose(position, np.array([0.55, -0.40, 1.16], dtype=np.float64))


def test_closed_loop_control_uses_estimate_after_first_detection(tmp_path) -> None:
    runtime = create_runtime(_bootstrap(tmp_path, estimator_mode="visual", controller_mode="closed-loop"))
    runtime._counts["detections"] = 3
    runtime._filter = AnchoredOnlineFilter.identity_initialized(estimator_mode="visual")
    runtime._filter.state.position_world_m = np.array([0.4, -0.2, 1.1], dtype=np.float64)
    runtime._latest_filter_snapshot = object()

    position, source, hold = runtime._resolve_control_state(ee_position_world_m=np.zeros(3, dtype=np.float64))

    assert source == "estimate"
    assert hold is False
    assert np.allclose(position, np.array([0.4, -0.2, 1.1], dtype=np.float64))


def test_publication_bootstrap_holds_until_first_detection(tmp_path) -> None:
    runtime = create_runtime(
        _bootstrap(
            tmp_path,
            estimator_mode="fused",
            controller_mode="closed-loop",
            bootstrap_control_policy="hold_until_first_detection",
        )
    )

    position, source, hold = runtime._resolve_control_state(ee_position_world_m=np.array([0.1, 0.2, 0.3], dtype=np.float64))

    assert position is None
    assert source == "bootstrap_hold"
    assert hold is True


def test_debug_bootstrap_can_use_gt_before_first_detection(tmp_path) -> None:
    runtime = create_runtime(
        _bootstrap(
            tmp_path,
            estimator_mode="fused",
            controller_mode="closed-loop",
            bootstrap_control_policy="gt_until_first_detection",
        )
    )

    position, source, hold = runtime._resolve_control_state(ee_position_world_m=np.array([0.1, 0.2, 0.3], dtype=np.float64))

    assert source == "gt_debug"
    assert hold is False
    assert np.allclose(position, np.array([0.1, 0.2, 0.3], dtype=np.float64))


def test_control_targets_are_mapped_by_delta_into_ee_frame(tmp_path) -> None:
    runtime = create_runtime(_bootstrap(tmp_path, estimator_mode="fused", controller_mode="closed-loop"))

    ik_target = runtime._control_target_to_ik_target(
        current_control_position_world_m=np.array([0.55, -0.40, 1.16], dtype=np.float64),
        desired_control_position_world_m=np.array([0.61, -0.36, 1.16], dtype=np.float64),
        current_ee_position_world_m=np.array([0.60, -0.30, 1.15], dtype=np.float64),
    )

    assert np.allclose(ik_target, np.array([0.66, -0.26, 1.15], dtype=np.float64))
