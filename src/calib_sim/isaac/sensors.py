"""Sensor specifications for the Isaac runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class CameraSensorSpec:
    name: str
    width_px: int
    height_px: int
    rate_hz: float
    frame_id: str
    distortion_model: str = "opencv_pinhole"
    distortion_coefficients: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "width_px": int(self.width_px),
            "height_px": int(self.height_px),
            "rate_hz": float(self.rate_hz),
            "frame_id": self.frame_id,
            "distortion_model": self.distortion_model,
            "distortion_coefficients": [float(value) for value in self.distortion_coefficients],
        }


@dataclass(slots=True)
class ImuSensorSpec:
    name: str
    rate_hz: float
    frame_id: str
    read_gravity: bool = True
    gravity_world_mps2: tuple[float, float, float] = (0.0, 0.0, -9.81)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rate_hz": float(self.rate_hz),
            "frame_id": self.frame_id,
            "read_gravity": bool(self.read_gravity),
            "gravity_world_mps2": [float(value) for value in self.gravity_world_mps2],
        }
