"""Tests for Isaac timebase and replay scaffolding."""

from __future__ import annotations

from calib_sim.isaac.clocks import FixedRateClock
from calib_sim.isaac.runtime.replay import load_estimator_input_bundle

from tests._isaac_test_helpers import make_minimal_isaac_run


def test_fixed_rate_clock_emits_expected_due_times() -> None:
    clock = FixedRateClock(rate_hz=100.0, start_time_s=0.0)
    assert clock.advance_to(0.009) == []
    due = clock.advance_to(0.031)
    assert [round(item.sim_time_s, 3) for item in due] == [0.010, 0.020, 0.030]
    assert [item.sample_index for item in due] == [0, 1, 2]


def test_estimator_input_bundle_ignores_ground_truth(tmp_path) -> None:
    run_dir = make_minimal_isaac_run(tmp_path)
    bundle = load_estimator_input_bundle(run_dir)

    assert bundle.gt is None
    assert bundle.manifest.run_id == "test_run"
    assert len(bundle.raw["camera_frames"]) == 1
    assert len(bundle.raw["detections"]) == 1
    assert len(bundle.raw["imu"]) == 2
    assert len(bundle.raw["estimator_input"]) == 1
