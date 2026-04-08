"""Anchor and auxiliary-tag helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class TagPoseSpec:
    tag_id: int
    size_m: float
    position_world_m: tuple[float, float, float]
    rotation_wt: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    is_anchor: bool = False

    def corners_local_m(self) -> np.ndarray:
        half = float(self.size_m) * 0.5
        return np.array(
            [
                [-half, -half, 0.0],
                [half, -half, 0.0],
                [half, half, 0.0],
                [-half, half, 0.0],
            ],
            dtype=np.float64,
        )


def anchor_pose_identity(tag_id: int, size_m: float) -> TagPoseSpec:
    return TagPoseSpec(
        tag_id=int(tag_id),
        size_m=float(size_m),
        position_world_m=(0.0, 0.0, 0.0),
        rotation_wt=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        is_anchor=True,
    )
