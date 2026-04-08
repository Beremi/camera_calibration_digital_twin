"""AprilTag front end for Isaac camera frames."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from calib_sim.isaac.frontend.measurement_pack import FrameMeasurementPack
from calib_sim.isaac.logging.schemas import IsaacCameraFramePacket, IsaacTagDetectionPacket
from calib_sim.tag_service.detector import AprilTag36h11Detector


_APRILTAG_CODED_SIDE_TO_BOARD_SIDE = 512.0 / 624.0


def _canonical_apriltag_object_points_m(tag_side_m: float) -> tuple[tuple[float, float, float], ...]:
    half = 0.5 * float(tag_side_m)
    return (
        (-half, half, 0.0),
        (half, half, 0.0),
        (half, -half, 0.0),
        (-half, -half, 0.0),
    )


def _native_backend(backend: str) -> bool:
    lowered = str(backend).strip().lower()
    if not lowered:
        return False
    return "rejected_candidate_match" not in lowered and "bright_quad_match" not in lowered


@dataclass(slots=True)
class IsaacAprilTagFrontend:
    anchor_tag_id: int
    detector: AprilTag36h11Detector | None = None
    frames_processed: int = 0
    detection_failures: int = 0
    total_detections: int = 0
    anchor_visible_frames: int = 0

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
        tag_size_by_id = {} if tag_size_by_id is None else {int(key): float(value) for key, value in tag_size_by_id.items()}
        intrinsics = frame_packet.intrinsics_snapshot
        detections = self.detector.detect_image(
            image_bgr,
            frame_index=int(frame_packet.frame_index),
            timestamp_s=float(frame_packet.timestamp_s),
            candidate_tag_ids=tuple(sorted(tag_size_by_id)),
        )
        fx = float(intrinsics.get("fx_px", 0.0))
        fy = float(intrinsics.get("fy_px", 0.0))
        cx = float(intrinsics.get("cx_px", 0.0))
        cy = float(intrinsics.get("cy_px", 0.0))
        camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        dist_coeffs = np.asarray(frame_packet.intrinsics_snapshot.get("distortion_coefficients", [0.0, 0.0, 0.0, 0.0, 0.0]), dtype=np.float64)
        packets: list[IsaacTagDetectionPacket] = []
        anchor_visible = False
        for detection in detections.detections:
            tag_size_m = float(tag_size_by_id.get(int(detection.tag_id), 0.0))
            detector_backend = str(detection.quality.get("detector_backend", ""))
            native_backend = _native_backend(detector_backend)
            pnp_tag_size_m = None
            local_points: tuple[tuple[float, float, float], ...] = ()
            pose_camera_rvec = None
            pose_camera_tvec_m = None
            if tag_size_m > 0.0:
                pnp_tag_size_m = float(tag_size_m) * _APRILTAG_CODED_SIDE_TO_BOARD_SIDE if native_backend else None
                local_points = () if pnp_tag_size_m is None else _canonical_apriltag_object_points_m(pnp_tag_size_m)
            if pnp_tag_size_m is not None and fx > 0.0 and fy > 0.0:
                success, rvec, tvec = cv2.solvePnP(
                    objectPoints=np.asarray(local_points, dtype=np.float64),
                    imagePoints=np.asarray(detection.corners_xy_clockwise, dtype=np.float64),
                    cameraMatrix=camera_matrix,
                    distCoeffs=dist_coeffs,
                    flags=getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE),
                )
                if success:
                    pose_camera_rvec = tuple(float(value) for value in np.asarray(rvec, dtype=np.float64).reshape(3))
                    pose_camera_tvec_m = tuple(float(value) for value in np.asarray(tvec, dtype=np.float64).reshape(3))
            packet = IsaacTagDetectionPacket(
                timestamp_s=float(frame_packet.timestamp_s),
                sim_time_s=float(frame_packet.sim_time_s),
                frame_index=int(frame_packet.frame_index),
                tag_id=int(detection.tag_id),
                family=str(detection.family),
                tag_size_m=tag_size_m,
                pnp_tag_size_m=pnp_tag_size_m,
                corners_xy=tuple((float(x), float(y)) for x, y in detection.corners_xy_clockwise),
                local_tag_points_m=tuple(tuple(float(value) for value in row) for row in local_points),
                corner_order="clockwise_top_left_first",
                score=float(
                    detection.quality.get(
                        "decision_margin",
                        detection.quality.get(
                            "adjusted_template_agreement",
                            detection.quality.get("template_agreement", 0.0),
                        ),
                    )
                ),
                is_anchor=int(detection.tag_id) == int(self.anchor_tag_id),
                pose_camera_rvec=pose_camera_rvec,
                pose_camera_tvec_m=pose_camera_tvec_m,
                detector_backend=detector_backend,
                visibility_flags={
                    "detected": True,
                    "known_size": int(detection.tag_id) in tag_size_by_id,
                    "native_backend": native_backend,
                    "pose_ready": pose_camera_tvec_m is not None,
                },
            )
            anchor_visible = anchor_visible or bool(packet.is_anchor)
            packets.append(packet)
        self.frames_processed += 1
        self.total_detections += len(packets)
        if anchor_visible:
            self.anchor_visible_frames += 1
        if not packets:
            self.detection_failures += 1
        return FrameMeasurementPack(
            camera_frame=frame_packet,
            detections=tuple(packets),
            metadata={
                "anchor_visible": anchor_visible,
                "detected_tag_ids": [int(packet.tag_id) for packet in packets],
                "detections_per_frame": len(packets),
                "tag_size_lookup_m": {str(tag_id): float(size_m) for tag_id, size_m in tag_size_by_id.items()},
            },
        )

    def frontend_summary(self) -> dict[str, Any]:
        anchor_visible_ratio = 0.0 if self.frames_processed == 0 else float(self.anchor_visible_frames / self.frames_processed)
        mean_detections = 0.0 if self.frames_processed == 0 else float(self.total_detections / self.frames_processed)
        return {
            "frames_processed": int(self.frames_processed),
            "mean_detections_per_frame": mean_detections,
            "anchor_visible_ratio": anchor_visible_ratio,
            "detection_failures": int(self.detection_failures),
        }


__all__ = ["IsaacAprilTagFrontend"]
