"""Simple Gaussian prior residual helpers."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def gaussian_prior_residual(
    value: Sequence[float],
    mean: Sequence[float],
    sqrt_information: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    value_arr = np.asarray(value, dtype=np.float64).reshape(-1)
    mean_arr = np.asarray(mean, dtype=np.float64).reshape(-1)
    sqrt_info = np.asarray(sqrt_information, dtype=np.float64)
    return sqrt_info @ (value_arr - mean_arr)


def pose_prior_residual(
    pose_vector: Sequence[float],
    mean_pose_vector: Sequence[float],
    sqrt_information: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    return gaussian_prior_residual(pose_vector, mean_pose_vector, sqrt_information)
