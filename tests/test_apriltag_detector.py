"""Detector smoke test using a marker rendered on a white canvas."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

# Allow tests to run without installing the package into the interpreter.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from calib_sim.tag_service.detector import AprilTag36h11Detector  # noqa: E402


def test_generated_marker_is_detected() -> None:
    """A generated marker on a white background should be detectable."""
    image_path = PROJECT_ROOT / "assets" / "apriltag36h11_id0_canvas.png"
    image = cv2.imread(str(image_path))
    detector = AprilTag36h11Detector()
    result = detector.detect_image(image)
    assert len(result.detections) >= 1
    assert result.detections[0].family == "36h11"
    assert result.detections[0].tag_id == 0
