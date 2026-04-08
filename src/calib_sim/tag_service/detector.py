"""AprilTag 36h11 detector based on OpenCV.

This module is the most "real" part of the scaffold: it can already detect tags
from images or videos in environments where OpenCV's aruco module is available.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

import cv2
import numpy as np

from calib_sim.common.models import FrameDetections, TagDetection


def _camera_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Build a standard 3x3 camera matrix."""
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _canonical_apriltag_object_points_m(tag_size_m: float) -> np.ndarray:
    half = float(tag_size_m) * 0.5
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def _quad_diagonal_intersection(points_xy: np.ndarray) -> np.ndarray:
    """Projective center of a quadrilateral from the diagonal intersection."""
    p0, p1, p2, p3 = np.asarray(points_xy, dtype=np.float64).reshape(4, 2)
    system = np.column_stack((p2 - p0, -(p3 - p1)))
    rhs = p1 - p0
    if abs(float(np.linalg.det(system))) < 1e-9:
        return np.mean(points_xy, axis=0)
    params = np.linalg.solve(system, rhs)
    return p0 + params[0] * (p2 - p0)


def _render_marker_template(tag_id: int, *, side_px: int, quiet_zone_px: int = 0) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker_side_px = max(int(side_px) - 2 * int(quiet_zone_px), 8)
    marker = cv2.aruco.generateImageMarker(dictionary, int(tag_id), marker_side_px)
    if quiet_zone_px <= 0:
        return np.asarray(marker, dtype=np.uint8)
    return cv2.copyMakeBorder(
        marker,
        int(quiet_zone_px),
        int(quiet_zone_px),
        int(quiet_zone_px),
        int(quiet_zone_px),
        cv2.BORDER_CONSTANT,
        value=255,
    )


def _warp_candidate(gray_image: np.ndarray, points_xy: np.ndarray, *, side_px: int) -> np.ndarray:
    source = np.asarray(points_xy, dtype=np.float32).reshape(4, 2)
    destination = np.array(
        [[0.0, 0.0], [float(side_px - 1), 0.0], [float(side_px - 1), float(side_px - 1)], [0.0, float(side_px - 1)]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(gray_image, homography, (int(side_px), int(side_px)))


def _order_quad_points_clockwise_top_left_first(points_xy: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float64)
    sums = np.sum(points, axis=1)
    diffs = points[:, 1] - points[:, 0]
    ordered[0] = points[int(np.argmin(sums))]  # top-left
    ordered[2] = points[int(np.argmax(sums))]  # bottom-right
    ordered[1] = points[int(np.argmin(diffs))]  # top-right
    ordered[3] = points[int(np.argmax(diffs))]  # bottom-left
    return ordered


def _candidate_edge_ratio(points_xy: np.ndarray) -> float:
    points = np.asarray(points_xy, dtype=np.float64).reshape(4, 2)
    edges = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    smallest = float(np.min(edges))
    if smallest <= 1e-9:
        return float("inf")
    return float(np.max(edges) / smallest)


def _find_bright_quad_candidates(gray_image: np.ndarray) -> list[tuple[np.ndarray, float, float]]:
    _, thresholded = cv2.threshold(np.asarray(gray_image, dtype=np.uint8), 245, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[np.ndarray, float, float]] = []
    for contour in contours:
        area_px2 = float(cv2.contourArea(contour))
        if area_px2 < 1200.0:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        approximation = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        if len(approximation) != 4:
            continue
        points = np.asarray(approximation, dtype=np.float64).reshape(4, 2)
        edge_ratio = _candidate_edge_ratio(points)
        if not np.isfinite(edge_ratio) or edge_ratio > 2.5:
            continue
        candidates.append((points, area_px2, edge_ratio))
    return candidates


def _match_rejected_candidate(
    warped_gray: np.ndarray,
    *,
    candidate_tag_ids: Sequence[int],
    template_side_px: int,
) -> tuple[int | None, float, int]:
    if warped_gray.size == 0:
        return None, 0.0, 0
    blurred = cv2.GaussianBlur(np.asarray(warped_gray, dtype=np.uint8), (5, 5), 0)
    _, warped_binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    best_tag_id: int | None = None
    best_score = 0.0
    best_rotation_ccw = 0
    quiet_zone_px = max(int(round(0.11 * float(template_side_px))), 4)
    for candidate_tag_id in candidate_tag_ids:
        for include_quiet_zone in (False, True):
            template = _render_marker_template(
                int(candidate_tag_id),
                side_px=int(template_side_px),
                quiet_zone_px=quiet_zone_px if include_quiet_zone else 0,
            )
            if template.shape != warped_binary.shape:
                template = cv2.resize(template, warped_binary.shape[::-1], interpolation=cv2.INTER_NEAREST)
            for rotation_ccw, rotated in enumerate((template, np.rot90(template, 1), np.rot90(template, 2), np.rot90(template, 3))):
                reference = np.asarray(rotated, dtype=np.uint8)
                agreement = float(np.mean(warped_binary == reference))
                if agreement > best_score:
                    best_score = agreement
                    best_tag_id = int(candidate_tag_id)
                    best_rotation_ccw = int(rotation_ccw)
    return best_tag_id, best_score, best_rotation_ccw


class AprilTag36h11Detector:
    """Detector for the OpenCV AprilTag 36h11 dictionary.

    Notes:
    - We preserve OpenCV's returned corner order (clockwise) in the public
      result object.
    - The "5 points" requested by the user are implemented as 4 corners + center.
    - Optional pose estimation is provided when intrinsics and tag size are given.
    """

    family_name = "36h11"

    def __init__(self) -> None:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        self._detector = cv2.aruco.ArucoDetector(dictionary, params)

    def detect_image(
        self,
        image_bgr: np.ndarray,
        frame_index: int = 0,
        timestamp_s: float | None = None,
        *,
        fx: float | None = None,
        fy: float | None = None,
        cx: float | None = None,
        cy: float | None = None,
        dist_coeffs: Sequence[float] | None = None,
        tag_size_m: float | None = None,
        candidate_tag_ids: Sequence[int] | None = None,
    ) -> FrameDetections:
        """Detect tags in a single BGR image.

        Pose is estimated only if intrinsics and `tag_size_m` are provided.
        """
        if image_bgr.ndim == 2:
            gray = image_bgr
        else:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected = self._detector.detectMarkers(gray)
        detections: List[TagDetection] = []

        direct_detections: list[tuple[int, np.ndarray, dict[str, Any]]] = []
        if ids is not None and len(ids) > 0:
            ids = ids.flatten().tolist()
            for tag_id, tag_corners in zip(ids, corners):
                direct_detections.append(
                    (
                        int(tag_id),
                        np.asarray(tag_corners, dtype=np.float64).reshape(4, 2),
                        {"detector_backend": "opencv_aruco_apriltag36h11"},
                    )
                )

        recovered_detections: list[tuple[int, np.ndarray, dict[str, Any]]] = []
        if candidate_tag_ids:
            claimed_tag_ids = {tag_id for tag_id, _points, _quality in direct_detections}
            best_rejected_by_tag_id: dict[int, tuple[float, np.ndarray, dict[str, Any]]] = {}
            for rejected_corners in rejected:
                rejected_points = _order_quad_points_clockwise_top_left_first(
                    np.asarray(rejected_corners, dtype=np.float64).reshape(4, 2)
                )
                area_px2 = cv2.contourArea(rejected_points.astype(np.float32))
                if float(area_px2) < 250.0:
                    continue
                edge_ratio = _candidate_edge_ratio(rejected_points)
                if not np.isfinite(edge_ratio) or edge_ratio > 3.0:
                    continue
                warped_gray = _warp_candidate(gray, rejected_points, side_px=128)
                matched_tag_id, match_score, rotation_ccw = _match_rejected_candidate(
                    warped_gray,
                    candidate_tag_ids=[int(tag_id) for tag_id in candidate_tag_ids if int(tag_id) not in claimed_tag_ids],
                    template_side_px=128,
                )
                shape_weight = 1.0 / max(edge_ratio, 1.0)
                adjusted_score = float(match_score) * float(shape_weight)
                if matched_tag_id is None or match_score < 0.55 or adjusted_score < 0.35:
                    continue
                canonical_points = np.roll(rejected_points, shift=int(rotation_ccw), axis=0)
                quality = {
                    "detector_backend": "opencv_aruco_apriltag36h11_rejected_candidate_match",
                    "template_agreement": float(match_score),
                    "adjusted_template_agreement": float(adjusted_score),
                    "candidate_area_px2": float(area_px2),
                    "candidate_edge_ratio": float(edge_ratio),
                    "template_rotation_ccw": int(rotation_ccw),
                }
                previous = best_rejected_by_tag_id.get(int(matched_tag_id))
                if previous is None or float(adjusted_score) > float(previous[0]):
                    best_rejected_by_tag_id[int(matched_tag_id)] = (float(adjusted_score), canonical_points, quality)
            for candidate_points, area_px2, edge_ratio in _find_bright_quad_candidates(gray):
                ordered_candidate_points = _order_quad_points_clockwise_top_left_first(candidate_points)
                warped_gray = _warp_candidate(gray, ordered_candidate_points, side_px=256)
                matched_tag_id, match_score, rotation_ccw = _match_rejected_candidate(
                    warped_gray,
                    candidate_tag_ids=[int(tag_id) for tag_id in candidate_tag_ids if int(tag_id) not in claimed_tag_ids],
                    template_side_px=256,
                )
                shape_weight = 1.0 / max(edge_ratio, 1.0)
                adjusted_score = float(match_score) * float(shape_weight)
                if matched_tag_id is None or match_score < 0.60 or adjusted_score < 0.40:
                    continue
                canonical_points = np.roll(ordered_candidate_points, shift=int(rotation_ccw), axis=0)
                quality = {
                    "detector_backend": "opencv_aruco_apriltag36h11_bright_quad_match",
                    "template_agreement": float(match_score),
                    "adjusted_template_agreement": float(adjusted_score),
                    "candidate_area_px2": float(area_px2),
                    "candidate_edge_ratio": float(edge_ratio),
                    "template_rotation_ccw": int(rotation_ccw),
                }
                previous = best_rejected_by_tag_id.get(int(matched_tag_id))
                if previous is None or float(adjusted_score) > float(previous[0]):
                    best_rejected_by_tag_id[int(matched_tag_id)] = (float(adjusted_score), canonical_points, quality)
            for matched_tag_id, (_score, matched_points, quality) in sorted(best_rejected_by_tag_id.items()):
                recovered_detections.append((int(matched_tag_id), matched_points, quality))

        all_detections = direct_detections + recovered_detections
        if not all_detections:
            return FrameDetections(frame_index=frame_index, timestamp_s=timestamp_s, detections=[])

        for tag_id, pts, quality in all_detections:
            pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
            center = _quad_diagonal_intersection(pts)
            points5 = np.vstack([pts, center])

            rvec_tuple = None
            tvec_tuple = None

            # Pose estimation is optional because many consumers only need the 2D
            # correspondence points and tag id.
            if None not in (fx, fy, cx, cy) and tag_size_m is not None:
                obj = _canonical_apriltag_object_points_m(float(tag_size_m))
                K = _camera_matrix(float(fx), float(fy), float(cx), float(cy))
                D = np.zeros((5, 1), dtype=np.float64) if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64)
                success, rvec, tvec = cv2.solvePnP(
                    objectPoints=obj,
                    imagePoints=pts,
                    cameraMatrix=K,
                    distCoeffs=D,
                    flags=getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE),
                )
                if success:
                    rvec_tuple = tuple(float(v) for v in rvec.reshape(3))
                    tvec_tuple = tuple(float(v) for v in tvec.reshape(3))

            detections.append(
                TagDetection(
                    family=self.family_name,
                    tag_id=int(tag_id),
                    corners_xy_clockwise=[tuple(map(float, row)) for row in pts.tolist()],
                    center_xy=tuple(map(float, center.tolist())),
                    points5_xy=[tuple(map(float, row)) for row in points5.tolist()],
                    pose_camera_rvec=rvec_tuple,
                    pose_camera_tvec=tvec_tuple,
                    quality=dict(quality),
                )
            )

        return FrameDetections(frame_index=frame_index, timestamp_s=timestamp_s, detections=detections)

    def detect_video(self, video_path: str | Path, *, every_n_frames: int = 1) -> List[FrameDetections]:
        """Run detection over a video file.

        The function returns a full in-memory list because that is easy for
        testing. For large datasets, convert this into a generator or stream.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        results: List[FrameDetections] = []
        frame_index = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if every_n_frames > 1 and frame_index % every_n_frames != 0:
                    frame_index += 1
                    continue

                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                timestamp_s = float(timestamp_ms) / 1000.0 if timestamp_ms > 0 else None
                results.append(self.detect_image(frame, frame_index=frame_index, timestamp_s=timestamp_s))
                frame_index += 1
        finally:
            cap.release()

        return results

    @staticmethod
    def to_jsonable(frame_detections: FrameDetections) -> dict:
        """Convert dataclass output to a JSON-friendly dictionary."""
        return {
            "frame_index": frame_detections.frame_index,
            "timestamp_s": frame_detections.timestamp_s,
            "detections": [asdict(det) for det in frame_detections.detections],
        }
