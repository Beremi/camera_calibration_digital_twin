"""Second-pass uncertainty calibration knob regressions."""

from __future__ import annotations

import numpy as np

from calib_sim.isaac.estimation.online_filter import AnchoredOnlineFilter
from calib_sim.isaac.logging.schemas import IsaacImuPacket, IsaacTagDetectionPacket
from calib_sim.isaac.tag_builder import TagPoseSpec


def test_uncertainty_scaling_knobs_are_reflected_in_filter_outputs() -> None:
    filter_model = AnchoredOnlineFilter.identity_initialized(estimator_mode="fused")
    filter_model.vision_covariance_scale = 2.0
    filter_model.anchor_vision_covariance_scale = 5.0
    filter_model.aux_vision_covariance_scale = 6.0
    filter_model.imu_process_covariance_scale = 3.0
    filter_model.post_relocalization_covariance_scale = 4.0

    initial_covariance = filter_model.covariance.copy()
    filter_model.predict(
        (
            IsaacImuPacket(
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
                noise_preset="phone_nominal",
            ),
        )
    )
    assert np.trace(filter_model.covariance) > np.trace(initial_covariance)

    detection = IsaacTagDetectionPacket(
        timestamp_s=0.02,
        sim_time_s=0.02,
        frame_index=0,
        tag_id=0,
        family="apriltag36h11",
        tag_size_m=0.10,
        pnp_tag_size_m=0.10,
        corners_xy=((40.0, 50.0), (60.0, 50.0), (60.0, 70.0), (40.0, 70.0)),
        local_tag_points_m=(
            (-0.05, 0.05, 0.0),
            (0.05, 0.05, 0.0),
            (0.05, -0.05, 0.0),
            (-0.05, -0.05, 0.0),
        ),
        corner_order="clockwise_top_left_first",
        score=1.0,
        is_anchor=True,
        pose_camera_rvec=(0.0, 0.0, 0.0),
        pose_camera_tvec_m=(0.5, 0.0, 1.0),
        detector_backend="opencv_aruco_apriltag36h11",
        visibility_flags={"detected": True, "native_backend": True, "pose_ready": True},
    )
    tag_pose = TagPoseSpec(
        tag_id=0,
        size_m=0.10,
        position_world_m=(0.0, 0.0, 1.0),
        rotation_wt=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        is_anchor=True,
    )
    filter_model.update_anchor(
        tag_detection=detection,
        tag_pose=tag_pose,
        measurement_std_m=0.01,
        relocalization_threshold_m=0.01,
    )
    uncertainty = filter_model.current_uncertainty_summary(sim_time_s=0.02)

    assert uncertainty.diagnostics["vision_covariance_scale"] == 2.0
    assert uncertainty.diagnostics["anchor_vision_covariance_scale"] == 5.0
    assert uncertainty.diagnostics["aux_vision_covariance_scale"] == 6.0
    assert uncertainty.diagnostics["imu_process_covariance_scale"] == 3.0
    assert uncertainty.diagnostics["post_relocalization_covariance_scale"] == 4.0
    assert uncertainty.diagnostics["last_anchor_nis"] is not None
    assert uncertainty.position_radius_95_m > 0.0
