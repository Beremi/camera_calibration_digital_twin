"""AprilTag 36h11 detector based on OpenCV.

This module is the most "real" part of the scaffold: it can already detect tags
from images or videos in environments where OpenCV's aruco module is available.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import cv2
import numpy as np

from calib_sim.common.models import FrameDetections, TagDetection


def _camera_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Build a standard 3x3 camera matrix."""
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


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
    ) -> FrameDetections:
        """Detect tags in a single BGR image.

        Pose is estimated only if intrinsics and `tag_size_m` are provided.
        """
        if image_bgr.ndim == 2:
            gray = image_bgr
        else:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        corners, ids, _rejected = self._detector.detectMarkers(gray)
        detections: List[TagDetection] = []

        if ids is None or len(ids) == 0:
            return FrameDetections(frame_index=frame_index, timestamp_s=timestamp_s, detections=[])

        ids = ids.flatten().tolist()
        for tag_id, tag_corners in zip(ids, corners):
            pts = np.asarray(tag_corners, dtype=np.float64).reshape(4, 2)
            center = pts.mean(axis=0)
            points5 = np.vstack([pts, center])

            rvec_tuple = None
            tvec_tuple = None

            # Pose estimation is optional because many consumers only need the 2D
            # correspondence points and tag id.
            if None not in (fx, fy, cx, cy) and tag_size_m is not None:
                half = float(tag_size_m) / 2.0
                obj = np.array(
                    [
                        [-half, -half, 0.0],
                        [half, -half, 0.0],
                        [half, half, 0.0],
                        [-half, half, 0.0],
                    ],
                    dtype=np.float64,
                )
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
                    quality={"detector_backend": "opencv_aruco_apriltag36h11"},
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
