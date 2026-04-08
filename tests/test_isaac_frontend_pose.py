"""Isaac AprilTag frontend pose-contract checks."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from calib_sim.common.models import FrameDetections, TagDetection
from calib_sim.isaac.frontend.apriltag_frontend import IsaacAprilTagFrontend
from calib_sim.isaac.logging.schemas import IsaacCameraFramePacket


def _frame_packet() -> IsaacCameraFramePacket:
    return IsaacCameraFramePacket(
        frame_index=0,
        timestamp_s=0.0,
        sim_time_s=0.0,
        sensor_time_s=0.0,
        host_time_s=0.0,
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


@dataclass(slots=True)
class _FakeDetector:
    detections: list[TagDetection]

    def detect_image(self, *_args, **_kwargs) -> FrameDetections:
        return FrameDetections(frame_index=0, timestamp_s=0.0, detections=list(self.detections))


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
                    quality={"detector_backend": "opencv_aruco_apriltag36h11"},
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


def test_frontend_does_not_assign_pose_to_fallback_matched_quads() -> None:
    frontend = IsaacAprilTagFrontend(
        anchor_tag_id=0,
        detector=_FakeDetector(
            detections=[
                TagDetection(
                    family="36h11",
                    tag_id=42,
                    corners_xy_clockwise=_project_corners(0.2 * (512.0 / 624.0), z_m=1.0),
                    center_xy=(640.0, 360.0),
                    points5_xy=[],
                    quality={"detector_backend": "opencv_aruco_apriltag36h11_bright_quad_match"},
                )
            ]
        ),
    )

    pack = frontend.process_bgr_frame(
        np.zeros((720, 1280, 3), dtype=np.uint8),
        frame_packet=_frame_packet(),
        tag_size_by_id={42: 0.2},
    )

    detection = pack.detections[0]
    assert detection.pose_camera_rvec is None
    assert detection.pose_camera_tvec_m is None
    assert detection.visibility_flags["native_backend"] is False
    assert detection.visibility_flags["pose_ready"] is False
    assert len(pack.anchor_detections) == 0
    assert len(pack.anchor_pose_detections) == 0


def test_frontend_distinguishes_anchor_visibility_from_pose_ready_anchor_updates() -> None:
    frontend = IsaacAprilTagFrontend(
        anchor_tag_id=42,
        detector=_FakeDetector(
            detections=[
                TagDetection(
                    family="36h11",
                    tag_id=42,
                    corners_xy_clockwise=_project_corners(0.2 * (512.0 / 624.0), z_m=1.0),
                    center_xy=(640.0, 360.0),
                    points5_xy=[],
                    quality={"detector_backend": "opencv_aruco_apriltag36h11_bright_quad_match"},
                )
            ]
        ),
    )

    pack = frontend.process_bgr_frame(
        np.zeros((720, 1280, 3), dtype=np.uint8),
        frame_packet=_frame_packet(),
        tag_size_by_id={42: 0.2},
    )

    assert pack.metadata["anchor_visible"] is True
    assert len(pack.anchor_detections) == 1
    assert len(pack.anchor_pose_detections) == 0
