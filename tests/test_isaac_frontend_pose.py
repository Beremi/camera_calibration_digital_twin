"""Isaac AprilTag frontend checks for the slim pupil-only tabletop workflow."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from calib_sim.common.models import FrameDetections, TagDetection
from calib_sim.isaac.frontend.apriltag_frontend import IsaacAprilTagFrontend
from calib_sim.isaac.logging.schemas import IsaacCameraFramePacket


def _frame_packet(*, frame_index: int = 0, timestamp_s: float = 0.0) -> IsaacCameraFramePacket:
    return IsaacCameraFramePacket(
        frame_index=int(frame_index),
        timestamp_s=float(timestamp_s),
        sim_time_s=float(timestamp_s),
        sensor_time_s=float(timestamp_s),
        host_time_s=float(timestamp_s),
        rgb_path="",
        intrinsics_snapshot={
            "fx_px": 862.1142578125,
            "fy_px": 862.1142578125,
            "cx_px": 640.0,
            "cy_px": 360.0,
            "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
        extrinsics_snapshot={},
        image_width_px=1280,
        image_height_px=720,
    )


def _project_corners(tag_side_m: float, *, z_m: float) -> list[tuple[float, float]]:
    half = 0.5 * float(tag_side_m)
    object_points = np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )
    camera_matrix = np.array(
        [
            [862.1142578125, 0.0, 640.0],
            [0.0, 862.1142578125, 360.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    image_points, _ = cv2.projectPoints(
        object_points,
        np.zeros((3, 1), dtype=np.float64),
        np.array([[0.0], [0.0], [float(z_m)]], dtype=np.float64),
        camera_matrix,
        np.zeros((5, 1), dtype=np.float64),
    )
    return [tuple(float(value) for value in point) for point in image_points.reshape(-1, 2)]


def _render_tag_frame(corners_xy: list[tuple[float, float]], *, shift_xy_px: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    shifted = np.asarray(corners_xy, dtype=np.float32).reshape(4, 2) + np.asarray(shift_xy_px, dtype=np.float32).reshape(1, 2)
    cv2.fillConvexPoly(image, shifted.astype(np.int32), (255, 255, 255))
    cv2.polylines(image, [shifted.astype(np.int32)], isClosed=True, color=(0, 0, 0), thickness=6)
    center_xy = np.mean(shifted, axis=0).astype(np.int32)
    cv2.circle(image, tuple(center_xy.tolist()), 8, (0, 0, 0), thickness=-1)
    return image


@dataclass(slots=True)
class _FakeDetector:
    detections: list[TagDetection]

    def detect_image(self, *_args, **_kwargs) -> FrameDetections:
        return FrameDetections(frame_index=0, timestamp_s=0.0, detections=list(self.detections))


@dataclass(slots=True)
class _SequenceFakeDetector:
    sequences: list[list[TagDetection]]
    _index: int = field(default=0, init=False)

    def detect_image(self, *_args, **_kwargs) -> FrameDetections:
        bounded_index = min(self._index, len(self.sequences) - 1)
        detections = list(self.sequences[bounded_index])
        self._index += 1
        return FrameDetections(frame_index=bounded_index, timestamp_s=float(bounded_index), detections=detections)


@dataclass(slots=True)
class _RoiRecoveryDetector:
    first_frame_corners: list[tuple[float, float]]
    _full_frame_calls: int = field(default=0, init=False)

    def detect_image(self, image, *_args, **_kwargs) -> FrameDetections:  # noqa: ANN001
        height, width = image.shape[:2]
        if width >= 1000:
            self._full_frame_calls += 1
            if self._full_frame_calls == 1:
                return FrameDetections(
                    frame_index=0,
                    timestamp_s=0.0,
                    detections=[
                        TagDetection(
                            family="36h11",
                            tag_id=201,
                            corners_xy_clockwise=self.first_frame_corners,
                            center_xy=tuple(np.mean(np.asarray(self.first_frame_corners, dtype=np.float64), axis=0).tolist()),
                            points5_xy=[],
                            quality={"detector_backend": "pupil_apriltags_apriltag36h11", "decision_margin": 1.0},
                        )
                    ],
                )
            return FrameDetections(frame_index=0, timestamp_s=0.0, detections=[])
        roi_corners = np.array(
            [
                [48.0, 48.0],
                [148.0, 48.0],
                [148.0, 148.0],
                [48.0, 148.0],
            ],
            dtype=np.float64,
        )
        return FrameDetections(
            frame_index=0,
            timestamp_s=0.0,
            detections=[
                TagDetection(
                    family="36h11",
                    tag_id=201,
                    corners_xy_clockwise=[tuple(float(value) for value in row) for row in roi_corners.tolist()],
                    center_xy=(98.0, 98.0),
                    points5_xy=[],
                    quality={"detector_backend": "pupil_apriltags_apriltag36h11", "decision_margin": 1.0},
                )
            ],
        )


def test_frontend_default_detector_uses_pupil_and_tracking() -> None:
    frontend = IsaacAprilTagFrontend(anchor_tag_id=0)

    assert frontend.detector is not None
    assert frontend.detector.detector_backend == "pupil_apriltags"
    assert frontend.roi_recovery_enabled is True
    assert frontend.temporal_tracking_enabled is True


def test_frontend_uses_coded_tag_side_for_native_pose_recovery() -> None:
    coded_tag_side_m = 0.2 * (512.0 / 624.0)
    frontend = IsaacAprilTagFrontend(
        anchor_tag_id=0,
        detector=_FakeDetector(
            detections=[
                TagDetection(
                    family="36h11",
                    tag_id=0,
                    corners_xy_clockwise=_project_corners(coded_tag_side_m, z_m=1.0),
                    center_xy=(640.0, 360.0),
                    points5_xy=[],
                    quality={"detector_backend": "pupil_apriltags_apriltag36h11", "decision_margin": 1.0},
                )
            ]
        ),
    )

    pack = frontend.process_bgr_frame(
        np.zeros((720, 1280, 3), dtype=np.uint8),
        frame_packet=_frame_packet(),
        tag_size_by_id={0: 0.2},
    )

    detection = pack.detections[0]
    assert detection.pnp_tag_size_m is not None
    assert abs(detection.pnp_tag_size_m - coded_tag_side_m) < 1e-9
    assert detection.pose_camera_tvec_m is not None
    assert np.allclose(np.asarray(detection.pose_camera_tvec_m, dtype=np.float64), [0.0, 0.0, 1.0], atol=1e-4)


def test_frontend_can_skip_native_pose_recovery() -> None:
    coded_tag_side_m = 0.2 * (512.0 / 624.0)
    frontend = IsaacAprilTagFrontend(
        anchor_tag_id=0,
        solve_tag_pose=False,
        detector=_FakeDetector(
            detections=[
                TagDetection(
                    family="36h11",
                    tag_id=0,
                    corners_xy_clockwise=_project_corners(coded_tag_side_m, z_m=1.0),
                    center_xy=(640.0, 360.0),
                    points5_xy=[],
                    quality={"detector_backend": "pupil_apriltags_apriltag36h11", "decision_margin": 1.0},
                )
            ]
        ),
    )

    pack = frontend.process_bgr_frame(
        np.zeros((720, 1280, 3), dtype=np.uint8),
        frame_packet=_frame_packet(),
        tag_size_by_id={0: 0.2},
    )

    detection = pack.detections[0]
    assert detection.corners_xy
    assert detection.pnp_tag_size_m is None
    assert detection.local_tag_points_m == ()
    assert detection.pose_camera_rvec is None
    assert detection.pose_camera_tvec_m is None
    assert detection.visibility_flags["pose_ready"] is False


def test_frontend_temporal_tracks_recent_tag_when_detector_misses() -> None:
    coded_tag_side_m = 0.2 * (512.0 / 624.0)
    base_corners = _project_corners(coded_tag_side_m, z_m=1.0)
    frontend = IsaacAprilTagFrontend(
        anchor_tag_id=0,
        detector=_SequenceFakeDetector(
            sequences=[
                [
                    TagDetection(
                        family="36h11",
                        tag_id=0,
                        corners_xy_clockwise=base_corners,
                        center_xy=(640.0, 360.0),
                        points5_xy=[],
                        quality={"detector_backend": "pupil_apriltags_apriltag36h11", "decision_margin": 1.0},
                    )
                ],
                [],
            ]
        ),
    )

    pack0 = frontend.process_bgr_frame(
        _render_tag_frame(base_corners),
        frame_packet=_frame_packet(frame_index=0, timestamp_s=0.0),
        tag_size_by_id={0: 0.2},
    )
    pack1 = frontend.process_bgr_frame(
        _render_tag_frame(base_corners, shift_xy_px=(3.0, 2.0)),
        frame_packet=_frame_packet(frame_index=1, timestamp_s=1.0 / 60.0),
        tag_size_by_id={0: 0.2},
    )

    assert pack0.detections[0].measurement_source == "native_detected"
    assert len(pack1.detections) == 1
    assert pack1.detections[0].measurement_source == "temporal_tracked"
    assert pack1.metadata["temporal_detections_per_frame"] == 1


def test_frontend_roi_recovery_recovers_missing_tag() -> None:
    initial_corners = [(520.0, 240.0), (620.0, 240.0), (620.0, 340.0), (520.0, 340.0)]
    frontend = IsaacAprilTagFrontend(
        anchor_tag_id=201,
        detector=_RoiRecoveryDetector(first_frame_corners=initial_corners),
        roi_padding_px=32,
    )

    frame0 = _render_tag_frame(initial_corners)
    frame1 = _render_tag_frame(initial_corners, shift_xy_px=(50.0, 50.0))

    pack0 = frontend.process_bgr_frame(
        frame0,
        frame_packet=_frame_packet(frame_index=0, timestamp_s=0.0),
        tag_size_by_id={201: 0.2},
    )
    pack1 = frontend.process_bgr_frame(
        frame1,
        frame_packet=_frame_packet(frame_index=1, timestamp_s=1.0 / 60.0),
        tag_size_by_id={201: 0.2},
    )

    assert len(pack0.detections) == 1
    assert len(pack1.detections) == 1
    assert pack1.detections[0].measurement_source in {"roi_recovered", "temporal_tracked"}
    assert pack1.detections[0].tag_id == 201
