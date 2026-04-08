"""Per-frame measurement packets passed into the estimator."""

from __future__ import annotations

from dataclasses import dataclass, field

from calib_sim.isaac.logging.schemas import IsaacCameraFramePacket, IsaacImuPacket, IsaacTagDetectionPacket


@dataclass(slots=True)
class FrameMeasurementPack:
    camera_frame: IsaacCameraFramePacket
    detections: tuple[IsaacTagDetectionPacket, ...]
    imu_packets: tuple[IsaacImuPacket, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def anchor_detections(self) -> tuple[IsaacTagDetectionPacket, ...]:
        return tuple(detection for detection in self.detections if detection.is_anchor)
