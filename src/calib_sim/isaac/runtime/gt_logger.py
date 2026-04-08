"""Ground-truth logging facade kept separate from estimator-facing writers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from calib_sim.isaac.logging.writer import IsaacRunWriter


@dataclass(slots=True)
class GroundTruthLogger:
    writer: IsaacRunWriter

    def log_camera(self, payload: dict[str, Any]) -> None:
        self.writer.append_gt_camera(payload)

    def log_imu(self, payload: dict[str, Any]) -> None:
        self.writer.append_gt_imu(payload)

    def log_joint(self, payload: dict[str, Any]) -> None:
        self.writer.append_gt_joint(payload)

    def log_tags(self, payload: dict[str, Any]) -> None:
        self.writer.write_tag_gt(payload)
