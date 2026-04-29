from __future__ import annotations

import math

import numpy as np

from calib_sim.interactive.camera_model import load_phone_camera_model


def test_pixel_9a_landscape_video_model_matches_sim_export() -> None:
    model = load_phone_camera_model("config/camera/pixel_9a_main.toml")

    assert model.output_width_px == 1280
    assert model.output_height_px == 720
    assert model.encoded_rotation_degrees == 0
    assert math.isclose(model.target_frames_per_second, 30.0)
    assert math.isclose(model.crop_x_px, 0.0)
    assert math.isclose(model.crop_y_px, 375.0)
    assert math.isclose(model.crop_width_px, 4000.0)
    assert math.isclose(model.crop_height_px, 2250.0)
    assert math.isclose(model.fx_px, 862.11424)
    assert math.isclose(model.fy_px, 862.11424)
    assert math.isclose(model.cx_px, 640.194848)
    assert math.isclose(model.cy_px, 362.42144)
    assert model.apply_lens_distortion_in_render is False


def test_distorted_projection_preserves_repo_camera_y_up_convention() -> None:
    model = load_phone_camera_model("config/camera/pixel_9a_main.toml")

    points = np.asarray(
        [
            [0.0, 0.10, 1.0],
            [0.0, -0.10, 1.0],
        ],
        dtype=np.float64,
    )
    projected, visible = model.project_camera_points(points, apply_distortion=True)

    assert visible.tolist() == [True, True]
    assert projected[0, 1] < model.cy_px
    assert projected[1, 1] > model.cy_px
