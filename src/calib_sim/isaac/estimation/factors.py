"""Geometric helpers for the Isaac estimator tests."""

from __future__ import annotations

import numpy as np


def project_tag_corners(
    *,
    fx_px: float,
    fy_px: float,
    cx_px: float,
    cy_px: float,
    camera_from_tag_rotation: np.ndarray,
    camera_from_tag_translation_m: np.ndarray,
    local_corners_tag_m: np.ndarray,
) -> np.ndarray:
    rotation_ct = np.asarray(camera_from_tag_rotation, dtype=np.float64).reshape(3, 3)
    translation_ct = np.asarray(camera_from_tag_translation_m, dtype=np.float64).reshape(3)
    corners = np.asarray(local_corners_tag_m, dtype=np.float64).reshape(-1, 3)
    camera_points = (rotation_ct @ corners.T).T + translation_ct
    depths = np.maximum(camera_points[:, 2:3], 1e-9)
    pixels = np.empty((camera_points.shape[0], 2), dtype=np.float64)
    pixels[:, 0] = float(cx_px) + float(fx_px) * camera_points[:, 0] / depths[:, 0]
    pixels[:, 1] = float(cy_px) + float(fy_px) * camera_points[:, 1] / depths[:, 0]
    return pixels
