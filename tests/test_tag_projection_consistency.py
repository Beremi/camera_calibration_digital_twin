"""Projection and anchor-frame checks for Isaac tags."""

from __future__ import annotations

import numpy as np

from calib_sim.isaac.estimation.factors import project_tag_corners
from calib_sim.isaac.tag_builder import anchor_pose_identity


def test_anchor_tag_identity_pose_is_fixed_in_world() -> None:
    tag = anchor_pose_identity(tag_id=0, size_m=0.10)
    assert tag.is_anchor is True
    assert tag.position_world_m == (0.0, 0.0, 0.0)
    assert tag.rotation_wt == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def test_tag_corner_projection_matches_expected_pixels() -> None:
    tag = anchor_pose_identity(tag_id=0, size_m=0.10)
    pixels = project_tag_corners(
        fx_px=100.0,
        fy_px=100.0,
        cx_px=50.0,
        cy_px=60.0,
        camera_from_tag_rotation=np.eye(3, dtype=np.float64),
        camera_from_tag_translation_m=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        local_corners_tag_m=tag.corners_local_m(),
    )
    expected = np.array(
        [
            [45.0, 55.0],
            [55.0, 55.0],
            [55.0, 65.0],
            [45.0, 65.0],
        ],
        dtype=np.float64,
    )
    assert np.allclose(pixels, expected, atol=1e-6)
