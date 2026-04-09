"""Mode-splitting regression tests for the first-pass Isaac runtime."""

from __future__ import annotations

import numpy as np

from calib_sim.isaac.app import IsaacAppBootstrapConfig, create_runtime
from calib_sim.isaac.estimation.online_filter import AnchoredOnlineFilter
from calib_sim.isaac.logging.schemas import IsaacCameraFramePacket, IsaacImuPacket


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


def _imu_packet(*, ax: float, ay: float, az: float, wx: float = 0.0, wy: float = 0.0, wz: float = 0.0) -> IsaacImuPacket:
    return IsaacImuPacket(
        packet_index=0,
        timestamp_s=0.01,
        sim_time_s=0.01,
        dt_s=0.01,
        wx=wx,
        wy=wy,
        wz=wz,
        ax=ax,
        ay=ay,
        az=az,
        imu_frame="I",
        imu_semantics="specific_force",
        noise_preset="ideal",
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


def test_imu_sampling_uses_camera_pose_when_available(tmp_path) -> None:
    class _CameraBinding:
        def get_world_pose(self):
            return np.array([0.55, -0.40, 1.16], dtype=np.float64), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    class _Tick:
        sim_time_s = 0.01
        dt_s = 0.01

    class _ImuBinding:
        def __init__(self):
            self.positions: list[np.ndarray] = []
            self.rotations: list[np.ndarray] = []

        def due_ticks(self, sim_time_s: float):
            assert abs(sim_time_s - 0.01) < 1e-9
            return [_Tick()]

        def sample(self, *, tick, timestamps, position_world_m, rotation_wi):
            del tick, timestamps
            self.positions.append(np.asarray(position_world_m, dtype=np.float64).reshape(3))
            self.rotations.append(np.asarray(rotation_wi, dtype=np.float64).reshape(3, 3))
            return IsaacImuPacket(
                packet_index=0,
                timestamp_s=0.01,
                sim_time_s=0.01,
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

    runtime = create_runtime(_bootstrap(tmp_path, estimator_mode="fused", controller_mode="closed-loop"))
    runtime._sim_time_s = 0.01
    runtime._camera_binding = _CameraBinding()
    runtime._imu_binding = _ImuBinding()

    packets = runtime._process_imu(
        runtime._timestamps(),
        ee_position_world_m=np.array([0.60, -0.30, 0.46], dtype=np.float64),
        ee_orientation_wxyz=np.array([0.92387953, 0.0, 0.38268343, 0.0], dtype=np.float64),
    )

    assert len(packets) == 1
    assert len(runtime._imu_binding.positions) == 1
    assert np.allclose(runtime._imu_binding.positions[0], np.array([0.55, -0.40, 1.16], dtype=np.float64))
    assert np.allclose(runtime._imu_binding.rotations[0], np.eye(3), atol=1e-9)


def test_dropout_debug_frame_logging_writes_csv_row(tmp_path) -> None:
    runtime = create_runtime(_bootstrap(tmp_path, estimator_mode="fused", controller_mode="closed-loop"))
    runtime._filter = AnchoredOnlineFilter.identity_initialized(estimator_mode="fused")
    runtime._last_anchor_visible_raw = True
    runtime._last_anchor_visible = True
    runtime._camera_interval_imu_packets = [object(), object()]
    frame_packet = IsaacCameraFramePacket(
        frame_index=3,
        timestamp_s=0.5,
        sim_time_s=0.5,
        sensor_time_s=0.5,
        host_time_s=0.5,
        rgb_path="raw/rgb/frame_000003.png",
        intrinsics_snapshot={"fx_px": 1.0, "fy_px": 1.0, "cx_px": 0.0, "cy_px": 0.0},
        extrinsics_snapshot={
            "frame_id": "C",
            "position_world_m": [0.0, 0.0, 0.0],
            "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        image_width_px=1280,
        image_height_px=720,
        visible_gt_tag_ids=[0],
    )

    runtime._write_dropout_debug_frame(
        frame_index=3,
        timestamp_s=0.5,
        frame_packet=frame_packet,
        suppression_active=True,
        imu_prediction_disabled=False,
    )

    debug_csv = runtime.writer.raw_dir / "dropout_debug_frames.csv"
    assert debug_csv.exists()
    text = debug_csv.read_text(encoding="utf-8")
    assert "position_error_norm_m" in text
    assert "0.5" in text


def test_suppression_specific_force_gate_filters_implausible_imu_packets(tmp_path) -> None:
    runtime = create_runtime(_bootstrap(tmp_path, estimator_mode="fused", controller_mode="closed-loop"))
    runtime._suppression_imu_specific_force_gate_mps2 = 20.0

    packets = (
        _imu_packet(ax=0.0, ay=0.0, az=9.81),
        _imu_packet(ax=18.0, ay=0.0, az=5.0),
        _imu_packet(ax=25.0, ay=0.0, az=30.0),
    )

    filtered = runtime._filter_imu_packets_for_prediction(packets, suppression_active=True)

    assert len(filtered) == 2
    assert runtime._last_imu_packets_used_for_prediction == 2
    assert runtime._last_imu_packets_rejected_for_prediction == 1


def test_gyro_only_suppression_mode_updates_orientation_without_accel_translation(tmp_path) -> None:
    runtime = create_runtime(_bootstrap(tmp_path, estimator_mode="fused", controller_mode="closed-loop"))
    runtime._filter = AnchoredOnlineFilter.identity_initialized(estimator_mode="fused")
    runtime._filter.state.velocity_world_mps = np.array([0.5, 0.0, 0.0], dtype=np.float64)
    runtime._suppression_propagation_mode = "gyro_only"

    runtime._propagate_filter_from_imu(_rotating_packets(), suppression_active=True)

    assert not np.allclose(runtime._filter.state.rotation_wi, np.eye(3), atol=1e-6)
    assert np.allclose(runtime._filter.state.velocity_world_mps, np.array([0.5, 0.0, 0.0], dtype=np.float64))
    assert float(runtime._filter.state.position_world_m[0]) > 0.0


def test_constant_velocity_suppression_mode_advances_position_but_holds_orientation(tmp_path) -> None:
    runtime = create_runtime(_bootstrap(tmp_path, estimator_mode="fused", controller_mode="closed-loop"))
    runtime._filter = AnchoredOnlineFilter.identity_initialized(estimator_mode="fused")
    runtime._filter.state.velocity_world_mps = np.array([0.5, 0.0, 0.0], dtype=np.float64)
    runtime._suppression_propagation_mode = "constant_velocity"

    runtime._propagate_filter_from_imu(_rotating_packets(), suppression_active=True)

    assert np.allclose(runtime._filter.state.rotation_wi, np.eye(3), atol=1e-9)
    assert np.allclose(runtime._filter.state.velocity_world_mps, np.array([0.5, 0.0, 0.0], dtype=np.float64))
    assert float(runtime._filter.state.position_world_m[0]) > 0.0


def test_freeze_suppression_mode_holds_mean_state_fixed(tmp_path) -> None:
    runtime = create_runtime(_bootstrap(tmp_path, estimator_mode="fused", controller_mode="closed-loop"))
    runtime._filter = AnchoredOnlineFilter.identity_initialized(estimator_mode="fused")
    runtime._filter.state.position_world_m = np.array([0.1, -0.2, 0.3], dtype=np.float64)
    runtime._filter.state.velocity_world_mps = np.array([0.5, 0.0, 0.0], dtype=np.float64)
    runtime._suppression_propagation_mode = "freeze"

    runtime._propagate_filter_from_imu(_rotating_packets(), suppression_active=True)

    assert np.allclose(runtime._filter.state.rotation_wi, np.eye(3), atol=1e-9)
    assert np.allclose(runtime._filter.state.position_world_m, np.array([0.1, -0.2, 0.3], dtype=np.float64))
    assert np.allclose(runtime._filter.state.velocity_world_mps, np.array([0.5, 0.0, 0.0], dtype=np.float64))
