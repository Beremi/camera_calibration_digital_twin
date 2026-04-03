"""Phone-like sensor rig mounted on the robot wrist."""

from __future__ import annotations

from dataclasses import dataclass

from calib_sim.common.models import PhoneRigConfig


@dataclass
class PhoneRig:
    """Logical phone rig.

    This object represents a rigid assembly:
    - one RGB camera
    - one IMU
    - a common namespace
    - known mount frames
    """

    config: PhoneRigConfig
    parent_frame: str

    def topic_base(self) -> str:
        """Return the namespace prefix used by publishers and recorders."""
        return self.config.namespace.rstrip("/")
