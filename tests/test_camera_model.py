"""Tests for the TOML-driven phone camera model."""

from __future__ import annotations

import numpy as np

from calib_sim.common.models import TagDetection
from calib_sim.interactive.camera_model import load_phone_camera_model


def test_pixel_9a_preview_intrinsics_are_derived_from_active_array_crop() -> None:
    """The preview intrinsics should preserve the Pixel 9a 16:9 crop geometry."""
    model = load_phone_camera_model(
        "config/camera/pixel_9a_main.toml",
        output_width_px=960,
        output_height_px=540,
    )

    assert model.name == "pixel_9a_main"
    assert model.output_width_px == 960
    assert model.output_height_px == 540
    assert abs(model.fx_px - 646.58568) < 1e-4
    assert abs(model.fy_px - 646.58568) < 1e-4
    assert abs(model.cx_px - 480.146136) < 1e-4
    assert abs(model.cy_px - 271.81608) < 1e-4
    assert model.distortion_coefficients[:3] == (0.15084998, -0.44805366, 0.38709834)
    assert model.projection_model == "pixel_processed_video_default"
    assert model.apply_lens_distortion_in_render is False


def test_camera_pose_estimation_bridge_uses_center_then_clockwise_pattern_order() -> None:
    """The exported 5-point vector should match the external repo ordering."""
    model = load_phone_camera_model(
        "config/camera/pixel_9a_main.toml",
        output_width_px=960,
        output_height_px=540,
    )

    measurement = model.pixels_to_image_plane_mm(
        [
            [model.cx_px, model.cy_px],
            [model.cx_px + model.fx_px * 0.1, model.cy_px - model.fy_px * 0.1],
            [model.cx_px + model.fx_px * 0.1, model.cy_px + model.fy_px * 0.1],
            [model.cx_px - model.fx_px * 0.1, model.cy_px + model.fy_px * 0.1],
            [model.cx_px - model.fx_px * 0.1, model.cy_px - model.fy_px * 0.1],
        ],
        undistort=False,
    )

    assert np.allclose(
        measurement,
        np.array(
            [
                [0.0, 0.0],
                [0.453, 0.453],
                [0.453, -0.453],
                [-0.453, -0.453],
                [-0.453, 0.453],
            ],
            dtype=float,
        ),
        atol=1e-6,
    )


def test_raw_distorted_preset_keeps_distortion_in_render_and_export() -> None:
    """The explicit raw-like preset should preserve lens distortion in the rendered output."""
    model = load_phone_camera_model(
        "config/camera/pixel_9a_raw_distorted.toml",
        output_width_px=960,
        output_height_px=540,
    )

    assert model.projection_model == "pixel_metadata_raw_distorted"
    assert model.apply_lens_distortion_in_render is True
    assert model.pose_bridge.undistort_before_export is True
    assert model.distortion_coefficients_for_rendered_output() == model.distortion_coefficients


def test_exact_camera_pose_estimation_mode_disables_render_distortion() -> None:
    """The exact repo-style mode should keep the rendered image undistorted."""
    model = load_phone_camera_model(
        "config/camera/pixel_9a_camera_pose_estimation_exact.toml",
        output_width_px=960,
        output_height_px=540,
    )

    assert model.projection_model == "camera_pose_estimation_exact"
    assert model.apply_lens_distortion_in_render is False
    assert model.pose_bridge.undistort_before_export is False


def test_camera_pose_export_reorders_detector_corners_into_measurement_order() -> None:
    """The export should map the detector's synthetic clockwise order into the expected object-point order."""
    model = load_phone_camera_model(
        "config/camera/pixel_9a_main.toml",
        output_width_px=960,
        output_height_px=540,
    )
    detection = TagDetection(
        family="36h11",
        tag_id=7,
        corners_xy_clockwise=[
            (model.cx_px - 10.0, model.cy_px - 20.0),
            (model.cx_px + 10.0, model.cy_px - 20.0),
            (model.cx_px + 10.0, model.cy_px + 20.0),
            (model.cx_px - 10.0, model.cy_px + 20.0),
        ],
        center_xy=(model.cx_px, model.cy_px),
        points5_xy=[],
    )

    measurement = model.camera_pose_estimation_measurement(detection)

    assert measurement["order"] == ["center", "top_right", "bottom_right", "bottom_left", "top_left"]
    pixels = np.array(
        [
            [model.cx_px, model.cy_px],
            [model.cx_px + 10.0, model.cy_px - 20.0],
            [model.cx_px + 10.0, model.cy_px + 20.0],
            [model.cx_px - 10.0, model.cy_px + 20.0],
            [model.cx_px - 10.0, model.cy_px - 20.0],
        ],
        dtype=float,
    )
    expected_measurement = model.pixels_to_image_plane_mm(
        pixels,
        undistort=model.pose_bridge.undistort_before_export,
    )
    assert np.allclose(np.asarray(measurement["image_plane_points_mm"]), expected_measurement)
