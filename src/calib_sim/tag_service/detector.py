"""AprilTag 36h11 detector backed by pupil_apriltags only."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, List, Sequence

import cv2
import numpy as np

from calib_sim.common.models import FrameDetections, TagDetection


_APRILTAG_TEXTURE_SIZE_PX = 624
_APRILTAG_MARKER_SIZE_PX = 512
_APRILTAG_QUIET_ZONE_PX = 56
APRILTAG_BOARD_THICKNESS_M = 0.003
APRILTAG_FACE_LOCAL_Z_M = 0.5 * APRILTAG_BOARD_THICKNESS_M + 1.0e-3


@lru_cache(maxsize=1)
def _reference_apriltag_texture() -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(dictionary, 0, _APRILTAG_MARKER_SIZE_PX)
    return cv2.copyMakeBorder(
        marker,
        _APRILTAG_QUIET_ZONE_PX,
        _APRILTAG_QUIET_ZONE_PX,
        _APRILTAG_QUIET_ZONE_PX,
        _APRILTAG_QUIET_ZONE_PX,
        cv2.BORDER_CONSTANT,
        value=255,
    )


@lru_cache(maxsize=1)
def rendered_apriltag_marker_bounds_px() -> tuple[int, int, int, int]:
    texture = _reference_apriltag_texture()
    ys, xs = np.where(texture < 250)
    if xs.size == 0 or ys.size == 0:
        raise RuntimeError("Could not measure AprilTag marker bounds from the generated reference texture.")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


@lru_cache(maxsize=1)
def rendered_apriltag_marker_bounds_uv() -> tuple[float, float, float, float]:
    left_px, top_px, right_px, bottom_px = rendered_apriltag_marker_bounds_px()
    texture = _reference_apriltag_texture()
    width_px = float(texture.shape[1])
    height_px = float(texture.shape[0])
    return (
        float(left_px / width_px),
        float(top_px / height_px),
        float(right_px / width_px),
        float(bottom_px / height_px),
    )


def rendered_apriltag_geometry_metadata() -> dict[str, Any]:
    texture = _reference_apriltag_texture()
    left_u, top_u, right_u, bottom_u = rendered_apriltag_marker_bounds_uv()
    left_px, top_px, right_px, bottom_px = rendered_apriltag_marker_bounds_px()
    return {
        "texture_size_px": int(texture.shape[1]),
        "marker_bounds_px": {
            "left": int(left_px),
            "top": int(top_px),
            "right": int(right_px),
            "bottom": int(bottom_px),
        },
        "marker_bounds_uv": {
            "left": float(left_u),
            "top": float(top_u),
            "right": float(right_u),
            "bottom": float(bottom_u),
        },
        "marker_side_to_printed_tag_side": float(right_u - left_u),
        "board_thickness_m": float(APRILTAG_BOARD_THICKNESS_M),
        "face_local_z_m": float(APRILTAG_FACE_LOCAL_Z_M),
    }


def apriltag_marker_side_m(printed_tag_side_m: float) -> float:
    geometry = rendered_apriltag_geometry_metadata()
    return float(printed_tag_side_m) * float(geometry["marker_side_to_printed_tag_side"])


def rendered_apriltag_local_corners_m(
    printed_tag_side_m: float,
    *,
    local_z_m: float = 0.0,
) -> np.ndarray:
    left_u, top_u, right_u, bottom_u = rendered_apriltag_marker_bounds_uv()
    side_m = float(printed_tag_side_m)
    left_x_m = (float(left_u) - 0.5) * side_m
    right_x_m = (float(right_u) - 0.5) * side_m
    top_y_m = (0.5 - float(top_u)) * side_m
    bottom_y_m = (0.5 - float(bottom_u)) * side_m
    z_m = float(local_z_m)
    return np.array(
        [
            [left_x_m, top_y_m, z_m],
            [right_x_m, top_y_m, z_m],
            [right_x_m, bottom_y_m, z_m],
            [left_x_m, bottom_y_m, z_m],
        ],
        dtype=np.float64,
    )


def rendered_apriltag_face_corners_m(printed_tag_side_m: float) -> np.ndarray:
    return rendered_apriltag_local_corners_m(
        printed_tag_side_m,
        local_z_m=float(APRILTAG_FACE_LOCAL_Z_M),
    )


def _camera_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _resolve_resize_interpolation(value: str | int) -> int:
    if isinstance(value, int):
        return int(value)
    lowered = str(value).strip().lower()
    mapping = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "lanczos4": cv2.INTER_LANCZOS4,
        "area": cv2.INTER_AREA,
    }
    return int(mapping.get(lowered, cv2.INTER_CUBIC))


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
    p0, p1, p2, p3 = np.asarray(points_xy, dtype=np.float64).reshape(4, 2)
    system = np.column_stack((p2 - p0, -(p3 - p1)))
    rhs = p1 - p0
    if abs(float(np.linalg.det(system))) < 1e-9:
        return np.mean(points_xy, axis=0)
    params = np.linalg.solve(system, rhs)
    return p0 + params[0] * (p2 - p0)


class AprilTag36h11Detector:
    """Pupil-backed detector for the AprilTag 36h11 family."""

    family_name = "36h11"

    def __init__(
        self,
        *,
        detector_backend: str = "pupil_apriltags",
        allow_rejected_candidate_match: bool = False,
        allow_bright_quad_match: bool = False,
        direct_detection_retry_scale: float = 1.0,
        direct_detection_retry_scales: Sequence[float] | None = None,
        detector_parameter_overrides: dict[str, Any] | None = None,
        retry_resize_interpolation: str | int = "cubic",
        retry_gaussian_blur_ksize: int = 3,
        retry_gaussian_blur_sigma: float = 0.0,
        pupil_nthreads: int = 1,
        pupil_quad_decimate: float = 1.0,
        pupil_quad_sigma: float = 0.0,
        pupil_refine_edges: bool = True,
        pupil_decode_sharpening: float = 0.25,
    ) -> None:
        if str(detector_backend).strip().lower() not in {"", "pupil_apriltags"}:
            raise ValueError("Only detector_backend='pupil_apriltags' is supported in this branch.")
        if allow_rejected_candidate_match or allow_bright_quad_match:
            raise ValueError("Heuristic OpenCV fallback matching was removed from this branch.")
        if detector_parameter_overrides:
            raise ValueError("detector_parameter_overrides are no longer supported in this branch.")
        self.detector_backend = "pupil_apriltags"
        self.direct_detection_retry_scale = max(float(direct_detection_retry_scale), 1.0)
        self.pupil_nthreads = max(int(pupil_nthreads), 1)
        self.pupil_quad_decimate = max(float(pupil_quad_decimate), 0.1)
        self.pupil_quad_sigma = float(pupil_quad_sigma)
        self.pupil_refine_edges = bool(pupil_refine_edges)
        self.pupil_decode_sharpening = float(pupil_decode_sharpening)
        self.retry_resize_interpolation = _resolve_resize_interpolation(retry_resize_interpolation)
        self.retry_gaussian_blur_ksize = max(int(retry_gaussian_blur_ksize), 0)
        if self.retry_gaussian_blur_ksize > 0 and self.retry_gaussian_blur_ksize % 2 == 0:
            self.retry_gaussian_blur_ksize += 1
        self.retry_gaussian_blur_sigma = float(max(retry_gaussian_blur_sigma, 0.0))
        retry_scales: list[float] = []
        if self.direct_detection_retry_scale > 1.0:
            retry_scales.append(float(self.direct_detection_retry_scale))
        if direct_detection_retry_scales is not None:
            retry_scales.extend(float(value) for value in direct_detection_retry_scales if float(value) > 1.0)
        self.direct_detection_retry_scales = tuple(
            scale
            for scale in sorted({round(float(value), 6) for value in retry_scales})
            if scale > 1.0
        )
        self._build_pupil_detector()

    def _build_pupil_detector(self) -> None:
        try:
            from pupil_apriltags import Detector as PupilDetector
        except ImportError as exc:  # pragma: no cover - exercised via targeted tests.
            raise ImportError(
                "The 'pupil_apriltags' package is required when detector_backend='pupil_apriltags'."
            ) from exc
        self._detector = PupilDetector(
            families="tag36h11",
            nthreads=int(self.pupil_nthreads),
            quad_decimate=float(self.pupil_quad_decimate),
            quad_sigma=float(self.pupil_quad_sigma),
            refine_edges=1 if self.pupil_refine_edges else 0,
            decode_sharpening=float(self.pupil_decode_sharpening),
        )
        self.applied_detector_parameters = {
            "families": "tag36h11",
            "nthreads": int(self.pupil_nthreads),
            "quad_decimate": float(self.pupil_quad_decimate),
            "quad_sigma": float(self.pupil_quad_sigma),
            "refine_edges": bool(self.pupil_refine_edges),
            "decode_sharpening": float(self.pupil_decode_sharpening),
        }

    def _base_backend_name(self) -> str:
        return "pupil_apriltags_apriltag36h11"

    def _detect_direct_markers(
        self,
        gray_image: np.ndarray,
        *,
        backend_name: str,
        scale_back: float = 1.0,
    ) -> list[tuple[int, np.ndarray, dict[str, Any]]]:
        scale = max(float(scale_back), 1e-9)
        results = self._detector.detect(np.asarray(gray_image, dtype=np.uint8), estimate_tag_pose=False)
        packets: list[tuple[int, np.ndarray, dict[str, Any]]] = []
        for result in results:
            points = np.asarray(result.corners, dtype=np.float64).reshape(4, 2) / scale
            packets.append(
                (
                    int(result.tag_id),
                    points,
                    {
                        "detector_backend": str(backend_name),
                        "decision_margin": float(getattr(result, "decision_margin", 0.0)),
                        "hamming": int(getattr(result, "hamming", 0)),
                    },
                )
            )
        return packets

    def _materialize_frame_detections(
        self,
        all_detections: list[tuple[int, np.ndarray, dict[str, Any]]],
        *,
        frame_index: int,
        timestamp_s: float | None,
        fx: float | None = None,
        fy: float | None = None,
        cx: float | None = None,
        cy: float | None = None,
        dist_coeffs: Sequence[float] | None = None,
        tag_size_m: float | None = None,
    ) -> FrameDetections:
        if not all_detections:
            return FrameDetections(frame_index=frame_index, timestamp_s=timestamp_s, detections=[])
        detections: List[TagDetection] = []
        for tag_id, pts, quality in all_detections:
            pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
            center = _quad_diagonal_intersection(pts)
            points5 = np.vstack([pts, center])
            rvec_tuple = None
            tvec_tuple = None
            if None not in (fx, fy, cx, cy) and tag_size_m is not None:
                obj = rendered_apriltag_local_corners_m(float(tag_size_m))
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
        gray = image_bgr if image_bgr.ndim == 2 else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        claimed_tag_ids: set[int] = set()
        direct_detections = self._detect_direct_markers(gray, backend_name=self._base_backend_name())
        claimed_tag_ids.update(tag_id for tag_id, _pts, _quality in direct_detections)
        for retry_scale in self.direct_detection_retry_scales:
            scaled_gray = cv2.resize(
                gray,
                dsize=None,
                fx=float(retry_scale),
                fy=float(retry_scale),
                interpolation=int(self.retry_resize_interpolation),
            )
            retry_gray = scaled_gray
            if self.retry_gaussian_blur_ksize > 0:
                retry_gray = cv2.GaussianBlur(
                    retry_gray,
                    (int(self.retry_gaussian_blur_ksize), int(self.retry_gaussian_blur_ksize)),
                    float(self.retry_gaussian_blur_sigma),
                )
            for tag_id, points_xy, quality in self._detect_direct_markers(
                retry_gray,
                backend_name=f"{self._base_backend_name()}_retry_upsampled_{retry_scale:.2f}x",
                scale_back=float(retry_scale),
            ):
                if int(tag_id) in claimed_tag_ids:
                    continue
                direct_detections.append((int(tag_id), points_xy, dict(quality)))
                claimed_tag_ids.add(int(tag_id))
        if candidate_tag_ids:
            allowed = {int(tag_id) for tag_id in candidate_tag_ids}
            direct_detections = [item for item in direct_detections if int(item[0]) in allowed]
        return self._materialize_frame_detections(
            direct_detections,
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            dist_coeffs=dist_coeffs,
            tag_size_m=tag_size_m,
        )

    def detect_video(self, video_path: str | Path, *, every_n_frames: int = 1) -> List[FrameDetections]:
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


__all__ = [
    "APRILTAG_BOARD_THICKNESS_M",
    "APRILTAG_FACE_LOCAL_Z_M",
    "AprilTag36h11Detector",
    "apriltag_marker_side_m",
    "rendered_apriltag_face_corners_m",
    "rendered_apriltag_geometry_metadata",
    "rendered_apriltag_local_corners_m",
    "rendered_apriltag_marker_bounds_px",
    "rendered_apriltag_marker_bounds_uv",
]
