"""Robot control abstraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from calib_sim.common.models import MotionSample


@dataclass
class JointCommand:
    """Simple joint-target command."""

    joint_positions: Sequence[float]
    joint_velocities: Sequence[float] | None = None
    joint_efforts: Sequence[float] | None = None


class RobotController:
    """Facade for joint-space and task-space control.

    A production implementation would delegate to Isaac Sim articulation control
    and possibly a motion-generation layer.
    """

    def __init__(self, robot_name: str) -> None:
        self.robot_name = robot_name
        self.last_command: JointCommand | None = None

    def send_joint_command(self, command: JointCommand) -> None:
        """Send a low-level joint command.

        The scaffold stores the last command to make unit testing and API wiring
        easier before the simulator adapter is attached.
        """
        self.last_command = command

    def execute_motion_samples(self, samples: Iterable[MotionSample]) -> int:
        """Execute a list of task-space-like motion samples.

        Return the number of samples accepted. Replace this with true task-space
        interpolation and servo-loop dispatch once the motion layer is integrated.
        """
        count = 0
        for _sample in samples:
            count += 1
        return count
