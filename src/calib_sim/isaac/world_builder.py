"""World-frame conventions for the Isaac benchmark."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class FrameConvention:
    """Explicit frame naming for the anchored Isaac benchmark."""

    world_frame: str = "W"
    anchor_frame: str = "A"
    camera_frame: str = "C"
    imu_frame: str = "I"
    base_frame: str = "B"

    def validate(self) -> None:
        if self.world_frame != "W" or self.anchor_frame != "A":
            raise ValueError("The Isaac benchmark fixes the world and anchor notation to W and A.")

    def summary(self) -> dict[str, str]:
        self.validate()
        return {
            "world_frame": self.world_frame,
            "anchor_frame": self.anchor_frame,
            "camera_frame": self.camera_frame,
            "imu_frame": self.imu_frame,
            "base_frame": self.base_frame,
            "gauge_policy": "W_equals_A",
        }
