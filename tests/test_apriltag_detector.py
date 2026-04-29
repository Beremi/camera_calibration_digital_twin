"""Detector smoke tests for the slim pupil-only AprilTag path."""

from __future__ import annotations

import builtins
import sys
from pathlib import Path

import cv2
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from calib_sim.tag_service.detector import AprilTag36h11Detector  # noqa: E402


def test_generated_marker_is_detected_with_default_backend() -> None:
    image_path = PROJECT_ROOT / "assets" / "apriltag36h11_id0_canvas.png"
    image = cv2.imread(str(image_path))
    detector = AprilTag36h11Detector()
    result = detector.detect_image(image)
    assert detector.detector_backend == "pupil_apriltags"
    assert len(result.detections) >= 1
    assert result.detections[0].family == "36h11"
    assert result.detections[0].tag_id == 0


def test_pupil_backend_detects_generated_marker_with_expected_schema() -> None:
    image_path = PROJECT_ROOT / "assets" / "apriltag36h11_id0_canvas.png"
    image = cv2.imread(str(image_path))
    detector = AprilTag36h11Detector(detector_backend="pupil_apriltags", pupil_nthreads=2)
    result = detector.detect_image(image)
    assert len(result.detections) >= 1
    detection = result.detections[0]
    assert detection.family == "36h11"
    assert detection.tag_id == 0
    assert "pupil_apriltags_apriltag36h11" in detection.quality["detector_backend"]
    assert "decision_margin" in detection.quality


def test_detector_rejects_removed_branch_options() -> None:
    with pytest.raises(ValueError):
        AprilTag36h11Detector(detector_backend="opencv")
    with pytest.raises(ValueError):
        AprilTag36h11Detector(allow_rejected_candidate_match=True)


def test_missing_pupil_dependency_fails_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: ANN001
        if name == "pupil_apriltags":
            raise ImportError("blocked for test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)

    with pytest.raises(ImportError):
        AprilTag36h11Detector()
