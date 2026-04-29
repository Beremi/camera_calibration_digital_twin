"""Tests for the no-ground-truth inference boundary."""

from __future__ import annotations

from calib_sim.isaac.runtime.replay import load_estimator_input_bundle, load_replay_bundle

from tests._isaac_test_helpers import make_minimal_isaac_run


def test_estimator_loader_does_not_consume_gt_files(tmp_path) -> None:
    run_dir = make_minimal_isaac_run(tmp_path)
    (run_dir / "gt" / "camera_gt.csv").write_text(
        "timestamp_s,sim_time_s,px,py,pz\n0.10,0.10,999,999,999\n",
        encoding="utf-8",
    )

    bundle = load_estimator_input_bundle(run_dir)

    assert bundle.gt is None
    assert len(bundle.raw["camera_frames"]) == 1
    assert bundle.raw["camera_frames"][0]["frame_index"] == 0


def test_full_replay_can_load_gt_when_explicitly_requested(tmp_path) -> None:
    run_dir = make_minimal_isaac_run(tmp_path)
    bundle = load_replay_bundle(run_dir, include_gt=True)
    assert bundle.gt is not None
    assert len(bundle.gt["camera_gt"]) == 1
    assert bundle.gt["tag_gt"]["anchor_tag_id"] == 0
