"""AprilTag front end for Isaac camera frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from calib_sim.isaac.frontend.measurement_pack import FrameMeasurementPack
from calib_sim.isaac.logging.schemas import IsaacCameraFramePacket, IsaacTagDetectionPacket
from calib_sim.tag_service.detector import AprilTag36h11Detector


@dataclass(slots=True)
class IsaacAprilTagFrontend:
    anchor_tag_id: int
    detector: AprilTag36h11Detector | None = None

    def __post_init__(self) -> None:
        if self.detector is None:
            self.detector = AprilTag36h11Detector()

    def process_bgr_frame(
        self,
        image_bgr: np.ndarray,
        *,
        frame_packet: IsaacCameraFramePacket,
        tag_size_by_id: dict[int, float] | None = None,
    ) -> FrameMeasurementPack:
        tag_size_by_id = {} if tag_size_by_id is None else dict(tag_size_by_id)
        detections = self.detector.detect_image(
            image_bgr,
            frame_index=int(frame_packet.frame_index),
            timestamp_s=float(frame_packet.timestamp_s),
        )
        packets: list[IsaacTagDetectionPacket] = []
        for detection in detections.detections:
            packets.append(
                IsaacTagDetectionPacket(
                    timestamp_s=float(frame_packet.timestamp_s),
                    sim_time_s=float(frame_packet.sim_time_s),
                    frame_index=int(frame_packet.frame_index),
                    tag_id=int(detection.tag_id),
                    corners_xy=tuple((float(x), float(y)) for x, y in detection.corners_xy_clockwise),
                    corner_order="clockwise_top_left_first",
                    score=float(detection.quality.get("decision_margin", 1.0)),
                    is_anchor=int(detection.tag_id) == int(self.anchor_tag_id),
                    visibility_flags={
                        "detected": True,
                        "known_size": int(detection.tag_id) in tag_size_by_id,
                    },
                )
            )
        return FrameMeasurementPack(camera_frame=frame_packet, detections=tuple(packets))
