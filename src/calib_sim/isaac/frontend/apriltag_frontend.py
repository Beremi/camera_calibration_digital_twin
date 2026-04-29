"""AprilTag front end for Isaac camera frames using the Pupil detector path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from calib_sim.common.models import TagDetection
from calib_sim.isaac.frontend.measurement_pack import FrameMeasurementPack
from calib_sim.isaac.logging.schemas import IsaacCameraFramePacket, IsaacTagDetectionPacket
from calib_sim.tag_service.detector import (
    AprilTag36h11Detector,
    apriltag_marker_side_m,
    rendered_apriltag_local_corners_m,
)


def _measurement_source_kind(backend: str) -> str:
    lowered = str(backend).strip().lower()
    if not lowered:
        return "unknown"
    if "roi_recovered" in lowered:
        return "roi_recovered"
    if "temporal_track" in lowered:
        return "temporal_tracked"
    if "retry_upsampled" in lowered:
        return "retry_detected"
    return "native_detected"


def _measurement_pose_source(backend: str) -> bool:
    return _measurement_source_kind(backend) in {"native_detected", "retry_detected", "roi_recovered"}


def _mean_center_xy(corners_xy: np.ndarray) -> tuple[float, float]:
    center_xy = np.mean(np.asarray(corners_xy, dtype=np.float64).reshape(-1, 2), axis=0)
    return float(center_xy[0]), float(center_xy[1])


def _edge_ratio(corners_xy: np.ndarray) -> float:
    corners = np.asarray(corners_xy, dtype=np.float64).reshape(4, 2)
    edges = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
    min_edge = float(np.min(edges))
    if min_edge <= 1e-9:
        return float("inf")
    return float(np.max(edges) / min_edge)


def _quad_area_px2(corners_xy: np.ndarray) -> float:
    corners = np.asarray(corners_xy, dtype=np.float64).reshape(4, 2)
    return abs(float(cv2.contourArea(corners.astype(np.float32))))


def _best_cyclic_corner_error_px(reference_xy: np.ndarray, candidate_xy: np.ndarray) -> tuple[float, np.ndarray]:
    reference = np.asarray(reference_xy, dtype=np.float64).reshape(4, 2)
    candidate = np.asarray(candidate_xy, dtype=np.float64).reshape(4, 2)
    best_error = float("inf")
    best_candidate = candidate
    for shift in range(4):
        rotated = np.roll(candidate, shift=shift, axis=0)
        error = float(np.mean(np.linalg.norm(rotated - reference, axis=1)))
        if error < best_error:
            best_error = error
            best_candidate = rotated
    return best_error, best_candidate


@dataclass(slots=True)
class IsaacAprilTagFrontend:
    anchor_tag_id: int
    detector: AprilTag36h11Detector | None = None
    detector_backend: str = "pupil_apriltags"
    direct_detection_retry_scale: float = 1.75
    direct_detection_retry_scales: tuple[float, ...] = ()
    pupil_nthreads: int = 8
    pupil_quad_decimate: float = 1.0
    pupil_quad_sigma: float = 0.0
    pupil_refine_edges: bool = True
    pupil_decode_sharpening: float = 0.25
    solve_tag_pose: bool = True
    roi_recovery_enabled: bool = True
    roi_recovery_max_gap_frames: int = 1
    roi_padding_px: int = 48
    roi_prediction_flow_win_size: int = 27
    roi_accept_max_corner_error_px: float = 10.0
    roi_accept_max_edge_ratio: float = 3.0
    roi_accept_max_area_delta_ratio: float = 0.40
    temporal_tracking_enabled: bool = True
    temporal_max_gap_frames: int = 3
    temporal_flow_max_error_px: float = 5.0
    anchor_temporal_clean_max_gap_frames: int = 1
    anchor_temporal_flow_max_error_px: float = 3.0
    anchor_temporal_max_edge_ratio: float = 2.4
    anchor_temporal_max_area_delta_ratio: float = 0.28
    frames_processed: int = 0
    detection_failures: int = 0
    total_detections: int = 0
    anchor_visible_frames: int = 0
    _last_gray_image: np.ndarray | None = field(default=None, init=False, repr=False)
    _last_frame_index: int | None = field(default=None, init=False, repr=False)
    _last_native_corners_by_tag: dict[int, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.detector is None:
            self.detector = AprilTag36h11Detector(
                detector_backend="pupil_apriltags",
                direct_detection_retry_scale=float(self.direct_detection_retry_scale),
                direct_detection_retry_scales=tuple(float(value) for value in self.direct_detection_retry_scales),
                pupil_nthreads=int(self.pupil_nthreads),
                pupil_quad_decimate=float(self.pupil_quad_decimate),
                pupil_quad_sigma=float(self.pupil_quad_sigma),
                pupil_refine_edges=bool(self.pupil_refine_edges),
                pupil_decode_sharpening=float(self.pupil_decode_sharpening),
            )

    def _recover_with_roi(
        self,
        *,
        gray_image: np.ndarray,
        frame_index: int,
        timestamp_s: float,
        detections: list[TagDetection],
        candidate_tag_ids: tuple[int, ...],
    ) -> list[TagDetection]:
        if (
            not self.roi_recovery_enabled
            or self._last_gray_image is None
            or self._last_frame_index is None
            or not self._last_native_corners_by_tag
        ):
            return detections
        frame_gap = int(frame_index) - int(self._last_frame_index)
        if frame_gap <= 0 or frame_gap > max(int(self.roi_recovery_max_gap_frames), 1):
            return detections
        width_px = int(gray_image.shape[1])
        height_px = int(gray_image.shape[0])
        claimed_tag_ids = {int(detection.tag_id) for detection in detections}
        mapped_tag_ids = {int(tag_id) for tag_id in candidate_tag_ids}
        augmented = list(detections)
        lk_window = max(int(self.roi_prediction_flow_win_size), 5)
        if lk_window % 2 == 0:
            lk_window += 1
        for tag_id, previous_state in sorted(self._last_native_corners_by_tag.items()):
            if int(tag_id) in claimed_tag_ids or int(tag_id) not in mapped_tag_ids:
                continue
            previous_corners = np.asarray(previous_state.get("corners_xy", ()), dtype=np.float32).reshape(-1, 1, 2)
            if previous_corners.shape != (4, 1, 2):
                continue
            next_corners, status, errors = cv2.calcOpticalFlowPyrLK(
                self._last_gray_image,
                gray_image,
                previous_corners,
                None,
                winSize=(lk_window, lk_window),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
            )
            if next_corners is None or status is None:
                continue
            status_mask = np.asarray(status, dtype=np.uint8).reshape(-1)
            if status_mask.size != 4 or not np.all(status_mask == 1):
                continue
            predicted_corners = np.asarray(next_corners, dtype=np.float64).reshape(4, 2)
            predicted_area_px2 = _quad_area_px2(predicted_corners)
            if predicted_area_px2 < 60.0:
                continue
            x0 = max(int(np.floor(np.min(predicted_corners[:, 0]) - float(self.roi_padding_px))), 0)
            y0 = max(int(np.floor(np.min(predicted_corners[:, 1]) - float(self.roi_padding_px))), 0)
            x1 = min(int(np.ceil(np.max(predicted_corners[:, 0]) + float(self.roi_padding_px))), width_px)
            y1 = min(int(np.ceil(np.max(predicted_corners[:, 1]) + float(self.roi_padding_px))), height_px)
            if x1 - x0 < 16 or y1 - y0 < 16:
                continue
            roi_gray = np.asarray(gray_image[y0:y1, x0:x1], dtype=np.uint8)
            roi_frame = self.detector.detect_image(
                roi_gray,
                frame_index=int(frame_index),
                timestamp_s=float(timestamp_s),
                candidate_tag_ids=(int(tag_id),),
            )
            roi_candidates = [item for item in roi_frame.detections if int(item.tag_id) == int(tag_id)]
            if len(roi_candidates) != 1:
                continue
            recovered = roi_candidates[0]
            recovered_corners = np.asarray(recovered.corners_xy_clockwise, dtype=np.float64).reshape(4, 2)
            recovered_corners[:, 0] += float(x0)
            recovered_corners[:, 1] += float(y0)
            mean_corner_error_px, aligned_corners = _best_cyclic_corner_error_px(predicted_corners, recovered_corners)
            is_anchor = int(tag_id) == int(self.anchor_tag_id)
            corner_limit_px = float(self.roi_accept_max_corner_error_px) * (0.75 if is_anchor else 1.0)
            if mean_corner_error_px > corner_limit_px:
                continue
            recovered_area_px2 = _quad_area_px2(aligned_corners)
            if recovered_area_px2 < 60.0:
                continue
            area_delta_ratio = 0.0
            if predicted_area_px2 > 1e-6:
                area_delta_ratio = abs(recovered_area_px2 - predicted_area_px2) / predicted_area_px2
            area_limit = float(self.roi_accept_max_area_delta_ratio) * (0.85 if is_anchor else 1.0)
            if area_delta_ratio > area_limit:
                continue
            edge_ratio = _edge_ratio(aligned_corners)
            if not np.isfinite(edge_ratio) or edge_ratio > float(self.roi_accept_max_edge_ratio):
                continue
            center_xy = _mean_center_xy(aligned_corners)
            quality = dict(recovered.quality)
            base_backend = str(quality.get("detector_backend", "pupil_apriltags_apriltag36h11"))
            quality.update(
                {
                    "detector_backend": f"{base_backend}_roi_recovered",
                    "roi_recovered": True,
                    "roi_corner_error_px": float(mean_corner_error_px),
                    "roi_area_delta_ratio": float(area_delta_ratio),
                    "roi_edge_ratio": float(edge_ratio),
                    "roi_bounds_xyxy": [int(x0), int(y0), int(x1), int(y1)],
                    "roi_source_backend": str(previous_state.get("detector_backend", "")),
                }
            )
            augmented.append(
                TagDetection(
                    family=str(recovered.family),
                    tag_id=int(recovered.tag_id),
                    corners_xy_clockwise=[tuple(float(value) for value in row) for row in aligned_corners.tolist()],
                    center_xy=center_xy,
                    points5_xy=[
                        tuple(float(value) for value in row)
                        for row in np.vstack([aligned_corners, np.asarray(center_xy, dtype=np.float64).reshape(1, 2)]).tolist()
                    ],
                    quality=quality,
                )
            )
            claimed_tag_ids.add(int(tag_id))
        return augmented

    def _augment_with_temporal_tracks(
        self,
        *,
        gray_image: np.ndarray,
        frame_index: int,
        detections: list[TagDetection],
    ) -> list[TagDetection]:
        if (
            not self.temporal_tracking_enabled
            or self._last_gray_image is None
            or self._last_frame_index is None
            or not self._last_native_corners_by_tag
        ):
            return detections
        frame_gap = int(frame_index) - int(self._last_frame_index)
        if frame_gap <= 0 or frame_gap > max(int(self.temporal_max_gap_frames), 1):
            return detections
        claimed_tag_ids = {int(detection.tag_id) for detection in detections}
        augmented = list(detections)
        for tag_id, previous_state in sorted(self._last_native_corners_by_tag.items()):
            if int(tag_id) in claimed_tag_ids:
                continue
            previous_corners = np.asarray(previous_state.get("corners_xy", ()), dtype=np.float32).reshape(-1, 1, 2)
            if previous_corners.shape != (4, 1, 2):
                continue
            next_corners, status, errors = cv2.calcOpticalFlowPyrLK(
                self._last_gray_image,
                gray_image,
                previous_corners,
                None,
                winSize=(27, 27),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
            )
            if next_corners is None or status is None:
                continue
            status_mask = np.asarray(status, dtype=np.uint8).reshape(-1)
            if status_mask.size != 4 or not np.all(status_mask == 1):
                continue
            tracked_corners = np.asarray(next_corners, dtype=np.float64).reshape(4, 2)
            area_px2 = abs(float(cv2.contourArea(tracked_corners.astype(np.float32))))
            if area_px2 < 60.0:
                continue
            edge_ratio = _edge_ratio(tracked_corners)
            if not np.isfinite(edge_ratio) or edge_ratio > 3.6:
                continue
            previous_area_px2 = abs(float(cv2.contourArea(previous_corners.reshape(4, 2).astype(np.float32))))
            if previous_area_px2 > 1e-6 and abs(area_px2 - previous_area_px2) / previous_area_px2 > 0.55:
                continue
            mean_track_error_px = 0.0
            if errors is not None and errors.size > 0:
                mean_track_error_px = float(np.mean(np.asarray(errors, dtype=np.float64).reshape(-1)))
            if mean_track_error_px > float(self.temporal_flow_max_error_px):
                continue
            width_px = float(gray_image.shape[1])
            height_px = float(gray_image.shape[0])
            if (
                np.min(tracked_corners[:, 0]) < -8.0
                or np.max(tracked_corners[:, 0]) > width_px + 8.0
                or np.min(tracked_corners[:, 1]) < -8.0
                or np.max(tracked_corners[:, 1]) > height_px + 8.0
            ):
                continue
            temporal_gap_frames = int(previous_state.get("temporal_gap_frames", 0)) + 1
            is_anchor = int(tag_id) == int(self.anchor_tag_id)
            anchor_temporal_degraded = bool(
                is_anchor
                and (
                    temporal_gap_frames > max(int(self.anchor_temporal_clean_max_gap_frames), 0)
                    or mean_track_error_px > float(self.anchor_temporal_flow_max_error_px)
                    or edge_ratio > float(self.anchor_temporal_max_edge_ratio)
                    or (
                        previous_area_px2 > 1e-6
                        and abs(area_px2 - previous_area_px2) / previous_area_px2
                        > float(self.anchor_temporal_max_area_delta_ratio)
                    )
                )
            )
            center_xy = _mean_center_xy(tracked_corners)
            augmented.append(
                TagDetection(
                    family="36h11",
                    tag_id=int(tag_id),
                    corners_xy_clockwise=[tuple(float(value) for value in row) for row in tracked_corners.tolist()],
                    center_xy=center_xy,
                    points5_xy=[
                        tuple(float(value) for value in row)
                        for row in np.vstack([tracked_corners, np.asarray(center_xy, dtype=np.float64).reshape(1, 2)]).tolist()
                    ],
                    quality={
                        "detector_backend": "pupil_apriltags_apriltag36h11_temporal_track",
                        "temporal_track_error_px": float(mean_track_error_px),
                        "temporal_source_backend": str(previous_state.get("detector_backend", "")),
                        "temporal_source_gap_frames": int(frame_gap),
                        "temporal_chain_gap_frames": int(temporal_gap_frames),
                        "anchor_temporal_degraded": bool(anchor_temporal_degraded),
                        "candidate_area_px2": float(area_px2),
                        "candidate_edge_ratio": float(edge_ratio),
                    },
                )
            )
            claimed_tag_ids.add(int(tag_id))
        return augmented

    def _update_temporal_state(
        self,
        *,
        gray_image: np.ndarray,
        frame_index: int,
        packets: list[IsaacTagDetectionPacket],
    ) -> None:
        tracked_packets: dict[int, dict[str, Any]] = {}
        for packet in packets:
            if str(getattr(packet, "measurement_source", "")) not in {"native_detected", "retry_detected", "roi_recovered"}:
                continue
            temporal_backend = bool(packet.visibility_flags.get("temporal_backend", False))
            if temporal_backend and bool(packet.is_anchor):
                continue
            tracked_packets[int(packet.tag_id)] = {
                "corners_xy": tuple(tuple(float(value) for value in row) for row in packet.corners_xy),
                "detector_backend": str(packet.detector_backend),
                "temporal_gap_frames": int(packet.temporal_gap_frames),
            }
        self._last_gray_image = np.asarray(gray_image, dtype=np.uint8).copy()
        self._last_frame_index = int(frame_index)
        self._last_native_corners_by_tag = tracked_packets

    def process_bgr_frame(
        self,
        image_bgr: np.ndarray,
        *,
        frame_packet: IsaacCameraFramePacket,
        tag_size_by_id: dict[int, float] | None = None,
    ) -> FrameMeasurementPack:
        tag_size_by_id = {} if tag_size_by_id is None else {int(key): float(value) for key, value in tag_size_by_id.items()}
        gray_image = image_bgr if image_bgr.ndim == 2 else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect_image(
            image_bgr,
            frame_index=int(frame_packet.frame_index),
            timestamp_s=float(frame_packet.timestamp_s),
            candidate_tag_ids=tuple(sorted(tag_size_by_id)),
        )
        detection_list = self._recover_with_roi(
            gray_image=gray_image,
            frame_index=int(frame_packet.frame_index),
            timestamp_s=float(frame_packet.timestamp_s),
            detections=list(detections.detections),
            candidate_tag_ids=tuple(sorted(tag_size_by_id)),
        )
        return self._build_measurement_pack(
            gray_image=gray_image,
            frame_packet=frame_packet,
            detection_list=detection_list,
            tag_size_by_id=tag_size_by_id,
        )

    def _build_measurement_pack(
        self,
        *,
        gray_image: np.ndarray,
        frame_packet: IsaacCameraFramePacket,
        detection_list: list[TagDetection],
        tag_size_by_id: dict[int, float],
    ) -> FrameMeasurementPack:
        intrinsics = frame_packet.intrinsics_snapshot
        detection_list = self._augment_with_temporal_tracks(
            gray_image=gray_image,
            frame_index=int(frame_packet.frame_index),
            detections=detection_list,
        )
        fx = float(intrinsics.get("fx_px", 0.0))
        fy = float(intrinsics.get("fy_px", 0.0))
        cx = float(intrinsics.get("cx_px", 0.0))
        cy = float(intrinsics.get("cy_px", 0.0))
        camera_matrix = None
        dist_coeffs = None
        if bool(self.solve_tag_pose):
            camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
            dist_coeffs = np.asarray(
                frame_packet.intrinsics_snapshot.get("distortion_coefficients", [0.0, 0.0, 0.0, 0.0, 0.0]),
                dtype=np.float64,
            )
        packets: list[IsaacTagDetectionPacket] = []
        anchor_visible = False
        temporal_detection_count = 0
        for detection in detection_list:
            tag_size_m = float(tag_size_by_id.get(int(detection.tag_id), 0.0))
            detector_backend = str(detection.quality.get("detector_backend", ""))
            measurement_source = _measurement_source_kind(detector_backend)
            retry_backend = measurement_source == "retry_detected"
            roi_recovered = measurement_source == "roi_recovered"
            temporal_backend = measurement_source == "temporal_tracked"
            temporal_gap_frames = int(detection.quality.get("temporal_chain_gap_frames", 0))
            anchor_temporal_degraded = bool(detection.quality.get("anchor_temporal_degraded", False))
            temporal_detection_count += int("temporal_track" in detector_backend)
            pnp_tag_size_m = None
            local_points: tuple[tuple[float, float, float], ...] = ()
            pose_camera_rvec = None
            pose_camera_tvec_m = None
            if bool(self.solve_tag_pose) and tag_size_m > 0.0:
                pnp_tag_size_m = (
                    float(apriltag_marker_side_m(tag_size_m)) if _measurement_pose_source(detector_backend) else None
                )
                local_points = () if pnp_tag_size_m is None else tuple(
                    tuple(float(value) for value in row)
                    for row in rendered_apriltag_local_corners_m(float(tag_size_m)).tolist()
                )
            if pnp_tag_size_m is not None and camera_matrix is not None and dist_coeffs is not None and fx > 0.0 and fy > 0.0:
                success, rvec, tvec = cv2.solvePnP(
                    objectPoints=np.asarray(local_points, dtype=np.float64),
                    imagePoints=np.asarray(detection.corners_xy_clockwise, dtype=np.float64),
                    cameraMatrix=np.asarray(camera_matrix, dtype=np.float64),
                    distCoeffs=np.asarray(dist_coeffs, dtype=np.float64),
                    flags=getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE),
                )
                if success:
                    pose_camera_rvec = tuple(float(value) for value in np.asarray(rvec, dtype=np.float64).reshape(3))
                    pose_camera_tvec_m = tuple(float(value) for value in np.asarray(tvec, dtype=np.float64).reshape(3))
            if int(detection.tag_id) == int(self.anchor_tag_id) and anchor_temporal_degraded:
                pose_camera_rvec = None
                pose_camera_tvec_m = None
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
                score=float(detection.quality.get("decision_margin", 0.0)),
                is_anchor=int(detection.tag_id) == int(self.anchor_tag_id),
                pose_camera_rvec=pose_camera_rvec,
                pose_camera_tvec_m=pose_camera_tvec_m,
                detector_backend=detector_backend,
                measurement_source=str(measurement_source),
                temporal_gap_frames=int(temporal_gap_frames),
                visibility_flags={
                    "detected": True,
                    "known_size": int(detection.tag_id) in tag_size_by_id,
                    "native_backend": measurement_source == "native_detected",
                    "retry_backend": retry_backend,
                    "roi_recovery_backend": roi_recovered,
                    "temporal_backend": temporal_backend,
                    "pose_ready": pose_camera_tvec_m is not None,
                    "anchor_temporal_degraded": bool(anchor_temporal_degraded),
                },
            )
            anchor_visible = anchor_visible or bool(packet.is_anchor)
            packets.append(packet)
        self._update_temporal_state(
            gray_image=gray_image,
            frame_index=int(frame_packet.frame_index),
            packets=packets,
        )
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
                "temporal_detections_per_frame": int(temporal_detection_count),
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
