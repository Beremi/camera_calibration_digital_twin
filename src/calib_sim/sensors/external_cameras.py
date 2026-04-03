"""Scene-camera management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from calib_sim.common.models import CameraIntrinsics, Pose


@dataclass
class SceneCameraSpec:
    """Configurable external observer camera."""

    name: str
    pose_world: Pose
    intrinsics: CameraIntrinsics
    namespace: str


class SceneCameraRegistry:
    """Tracks scene cameras for publishing and recording."""

    def __init__(self) -> None:
        self.cameras: List[SceneCameraSpec] = []

    def add(self, camera: SceneCameraSpec) -> None:
        self.cameras.append(camera)
