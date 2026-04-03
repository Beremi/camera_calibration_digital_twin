"""AprilTag layout helpers for the simulator scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from calib_sim.common.models import Pose


@dataclass(slots=True)
class TagSpec:
    """One simulator-side tag placement specification."""

    family: str
    tag_id: int
    edge_length_m: float
    pose_world: Pose
    image_path: str


class TagLayoutManager:
    """Keeps the simulator's truth model for spawned tags.

    This module should be the authoritative source for:
    - which tag ids exist in the scene
    - where they are in world coordinates
    - what physical size they have
    """

    def __init__(self) -> None:
        self.tags: List[TagSpec] = []

    def add_tag(self, tag: TagSpec) -> None:
        """Register a tag in the layout."""
        self.tags.append(tag)

    def get_truth_table(self) -> List[dict]:
        """Return a JSON-friendly summary for recording/debugging."""
        return [
            {
                "family": tag.family,
                "id": tag.tag_id,
                "edge_length_m": tag.edge_length_m,
                "image_path": tag.image_path,
                "pose_world": {"xyz_m": tag.pose_world.xyz_m, "rpy_deg": tag.pose_world.rpy_deg},
            }
            for tag in self.tags
        ]
