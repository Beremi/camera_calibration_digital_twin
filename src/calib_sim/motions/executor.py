"""Motion executor."""

from __future__ import annotations

from calib_sim.common.models import MotionPreset
from calib_sim.robot.controller import RobotController


class MotionExecutor:
    """Bridges a preset with the robot controller."""

    def __init__(self, controller: RobotController) -> None:
        self.controller = controller

    def execute(self, preset: MotionPreset) -> int:
        """Execute the preset and return the number of accepted samples."""
        return self.controller.execute_motion_samples(preset.samples)
