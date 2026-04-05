"""Tests for the browser-friendly interactive simulator."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

from calib_sim.interactive.analysis import _compiled_objective_functions, _estimate_pose_newton, _state_diagnostics
from calib_sim.interactive.sim import InteractiveCalibrationSim, available_robot_arm_presets, available_scene_presets


def test_interactive_snapshot_contains_live_detections() -> None:
    """The synthetic phone camera should yield real AprilTag detections."""
    sim = InteractiveCalibrationSim("config/interactive/browser_game_demo.yaml")
    sim.step(0.1)
    snapshot = sim.render_snapshot()

    assert snapshot["config"]["name"] == "browser_game_demo"
    assert len(snapshot["observer_views"]) == 2
    assert snapshot["config"]["robot_arm"]["preset_id"] == "compact_bench"
    assert snapshot["catalog"]["scene_presets"]
    assert snapshot["catalog"]["robot_arm_presets"]
    assert snapshot["phone_view"]["image_data_url"].startswith("data:image/jpeg;base64,")
    assert {item["tag_id"] for item in snapshot["phone_view"]["detections"]} == {0, 42}
    assert snapshot["phone_view"]["camera_model"]["name"] == "pixel_9a_main"
    assert snapshot["phone_view"]["camera_model"]["projection_model"] == "pixel_processed_video_default"
    assert snapshot["phone_view"]["camera_model"]["apply_lens_distortion_in_render"] is False
    assert len(snapshot["phone_view"]["camera_pose_estimation"]) == 2
    assert len(snapshot["phone_view"]["camera_pose_estimation"][0]["measurement_vector_mm"]) == 10
    assert snapshot["automation"]["auto_demo_enabled"] is True
    assert snapshot["phone_view"]["ground_truth"]["tags"][0]["rendered_image_points_px"]

    rotation_cw = np.asarray(snapshot["phone_view"]["ground_truth"]["camera_world_pose"]["rotation_cw"], dtype=np.float64).reshape(3, 3)
    assert np.isclose(np.linalg.det(rotation_cw), 1.0, atol=1e-6)

    tag_truth = snapshot["phone_view"]["ground_truth"]["tags"][0]
    rotation_tc = np.asarray(tag_truth["camera_rotation_tc"], dtype=np.float64).reshape(3, 3)
    assert np.isclose(np.linalg.det(rotation_tc), 1.0, atol=1e-6)

    camera_position_tag = np.asarray(tag_truth["camera_position_tag_m"], dtype=np.float64)
    rotation_ct = rotation_tc.T
    translation_ct = -(rotation_ct @ camera_position_tag)
    rvec_ct, _ = cv2.Rodrigues(rotation_ct)
    state = np.concatenate([translation_ct, rvec_ct.reshape(3)])
    observation = np.asarray(tag_truth["image_plane_points_m"], dtype=np.float64)
    camera_model = sim.primary_camera_model
    assert camera_model is not None
    diagnostics = _state_diagnostics(
        state,
        observed_points_image_plane_m=observation,
        focal_length_m=float(camera_model.focal_length_mm) / 1000.0,
        pattern_half_extent_m=float(tag_truth["size_m"]) * 0.5,
    )
    assert diagnostics["reprojection_rmse_image_plane_m"] < 1e-8


def test_minimization_forward_model_reproduces_ground_truth_measurements() -> None:
    """The exact minimization model should reproduce synthetic truth at the true pose."""
    sim = InteractiveCalibrationSim("config/interactive/browser_game_demo.yaml")
    sim.step(0.1)
    snapshot = sim.render_snapshot()
    camera_model = sim.primary_camera_model
    assert camera_model is not None

    for tag_truth in snapshot["phone_view"]["ground_truth"]["tags"]:
        if not bool(tag_truth["visible"]):
            continue

        rotation_tc = np.asarray(tag_truth["camera_rotation_tc"], dtype=np.float64).reshape(3, 3)
        camera_position_tag = np.asarray(tag_truth["camera_position_tag_m"], dtype=np.float64)
        rotation_ct = rotation_tc.T
        translation_ct = -(rotation_ct @ camera_position_tag)
        rvec_ct, _ = cv2.Rodrigues(rotation_ct)
        state = np.concatenate([translation_ct, rvec_ct.reshape(3)])
        observation = np.asarray(tag_truth["image_plane_points_m"], dtype=np.float64)
        pattern_half_extent_m = float(tag_truth["size_m"]) * 0.5
        focal_length_m = float(camera_model.focal_length_mm) / 1000.0

        diagnostics = _state_diagnostics(
            state,
            observed_points_image_plane_m=observation,
            focal_length_m=focal_length_m,
            pattern_half_extent_m=pattern_half_extent_m,
        )
        assert diagnostics["reprojection_rmse_image_plane_m"] < 1e-8

        loss_fn, _, _ = _compiled_objective_functions(focal_length_m, pattern_half_extent_m)
        loss_value = float(loss_fn(state, observation))
        assert loss_value < 1e-16


def test_rendered_tag_geometry_matches_detected_corners() -> None:
    """Detected corners on the synthetic frame should align with projected tag geometry."""
    sim = InteractiveCalibrationSim("config/interactive/browser_game_demo.yaml")
    sim.step(0.1)
    snapshot = sim.render_snapshot()

    truth_by_tag = {int(item["tag_id"]): item for item in snapshot["phone_view"]["ground_truth"]["tags"]}
    for detection in snapshot["phone_view"]["detections"]:
        tag_id = int(detection["tag_id"])
        truth = truth_by_tag[tag_id]
        observed_pixels = np.asarray(
            [
                detection["center_xy"],
                detection["corners_xy_clockwise"][1],
                detection["corners_xy_clockwise"][2],
                detection["corners_xy_clockwise"][3],
                detection["corners_xy_clockwise"][0],
            ],
            dtype=np.float64,
        )
        ground_truth_pixels = np.asarray(truth["rendered_image_points_px"], dtype=np.float64)
        errors_px = np.linalg.norm(observed_pixels - ground_truth_pixels, axis=1)
        assert float(np.mean(errors_px)) < 1.0
        assert float(np.max(errors_px)) < 1.2


def test_scene_and_robot_arm_presets_can_be_switched() -> None:
    """Scene and robot-arm selectors should expose multiple choices and reload cleanly."""
    assert len(available_scene_presets()) >= 2
    assert len(available_robot_arm_presets()) >= 3

    sim = InteractiveCalibrationSim("config/interactive/browser_game_demo.yaml")
    sim.reload_config("config/interactive/tabletop_grab_challenge.yaml")
    sim.set_robot_arm_preset("config/interactive/robot_arms/franka_tabletop.yaml")
    sim.step(0.1)
    snapshot = sim.render_snapshot()

    assert snapshot["config"]["name"] == "tabletop_grab_challenge"
    assert snapshot["config"]["robot_arm"]["preset_id"] == "franka_tabletop"
    assert len(snapshot["config"]["tags"]) == 8
    assert any(tag["mount"] == "top" for tag in snapshot["config"]["tags"])
    assert len(snapshot["config"]["scene"]["boxes"]) >= 3
    assert len(snapshot["observer_views"]) == 3


def test_previous_frame_seed_path_still_reports_camera_obscura_reference() -> None:
    """Warm-started solves should work without requiring obscura to be a live candidate."""
    sim = InteractiveCalibrationSim("config/interactive/browser_game_demo.yaml")
    sim.step(0.1)
    snapshot = sim.render_snapshot()
    camera_model = sim.primary_camera_model
    assert camera_model is not None

    tag_truth = next(item for item in snapshot["phone_view"]["ground_truth"]["tags"] if int(item["tag_id"]) == 42)
    observation = np.asarray(tag_truth["image_plane_points_m"], dtype=np.float64)
    ground_truth_position = np.asarray(tag_truth["camera_position_tag_m"], dtype=np.float64)
    ground_truth_rotation_tc = np.asarray(tag_truth["camera_rotation_tc"], dtype=np.float64).reshape(3, 3)
    ground_truth_rotation_ct = ground_truth_rotation_tc.T
    ground_truth_translation_ct = -(ground_truth_rotation_ct @ ground_truth_position)
    ground_truth_rvec_ct, _ = cv2.Rodrigues(ground_truth_rotation_ct)
    previous_state = np.concatenate([ground_truth_translation_ct, ground_truth_rvec_ct.reshape(3)])

    ordered_pixels = np.asarray(tag_truth["rendered_image_points_px"], dtype=np.float64)
    estimate = _estimate_pose_newton(
        observation,
        ordered_pixels_px=ordered_pixels,
        camera_model=camera_model,
        focal_length_m=float(camera_model.focal_length_mm) / 1000.0,
        pattern_half_extent_m=float(tag_truth["size_m"]) * 0.5,
        candidate_seed_specs=[
            {
                "seed_name": "previous_frame",
                "state": [float(value) for value in previous_state.tolist()],
                "source_recording_frame_index": 12,
            }
        ],
        use_diagonal_damping=False,
        initial_damping=0.0,
    )

    assert estimate["selected_seed_name"] == "previous_frame"
    assert np.isfinite(float(estimate["camera_obscura_seed_reprojection_rmse_image_plane_m"]))
    assert estimate["candidate_seed_summaries"][0]["seed_name"] == "previous_frame"
    assert len(estimate["candidate_seed_summaries"][0]["seed_state"]) == 6


def test_interactive_recording_dump_writes_files(tmp_path: Path) -> None:
    """Recording should write raw video and structured ground-truth logs."""
    template_path = Path("config/interactive/browser_game_demo.yaml")
    config = yaml.safe_load(template_path.read_text())
    config["output_dir"] = str(tmp_path / "runs")
    config["analysis"] = {"auto_run_on_stop": False}

    config_path = tmp_path / "interactive_test.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    sim = InteractiveCalibrationSim(config_path)
    sim.set_recording(True)
    sim.step(0.1)
    sim.render_snapshot()

    assert sim.active_run_dir is not None
    assert (sim.active_run_dir / "metadata.json").exists()
    assert (sim.active_run_dir / "samples.jsonl").exists()
    assert (sim.active_run_dir / "phone_raw.mp4").exists()
    assert not list(sim.active_run_dir.glob("*.finalizing.mp4"))
    assert (sim.active_run_dir / "imu.csv").exists()
    assert (sim.active_run_dir / "camera_gt.csv").exists()
    samples = (sim.active_run_dir / "samples.jsonl").read_text(encoding="utf-8")
    assert '"camera_pose_estimation"' in samples
    assert '"ground_truth"' in samples


def test_interactive_recording_stop_runs_analysis(tmp_path: Path) -> None:
    """Stopping a recording should produce offline analysis artifacts."""
    template_path = Path("config/interactive/browser_game_demo.yaml")
    config = yaml.safe_load(template_path.read_text())
    config["output_dir"] = str(tmp_path / "runs")

    config_path = tmp_path / "interactive_analysis_test.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    sim = InteractiveCalibrationSim(config_path)
    sim.set_recording(True)
    for _ in range(4):
        sim.step(0.1)
        sim.render_snapshot()
    active_run_dir = sim.active_run_dir
    sim.set_recording(False)

    assert active_run_dir is not None
    assert sim.analysis_status["state"] == "completed"
    assert (active_run_dir / "analysis" / "report.md").exists()
    assert (active_run_dir / "analysis" / "summary.json").exists()
    assert (active_run_dir / "analysis" / "joint_world_estimates.jsonl").exists()
    assert (active_run_dir / "analysis" / "per_tag_summary.json").exists()
    assert (active_run_dir / "analysis" / "per_tag").is_dir()
    assert (active_run_dir / "analysis" / "single_pattern_report.md").exists()
    assert (active_run_dir / "analysis" / "single_pattern_overlay.png").exists()
    assert (active_run_dir / "analysis" / "worst_single_pattern_report.md").exists()
    assert (active_run_dir / "analysis" / "worst_single_pattern_overlay.png").exists()
    assert (active_run_dir / "analysis" / "real_measurement_known_solution_report.md").exists()
    assert (active_run_dir / "analysis" / "synthetic_single_pattern_report.md").exists()
    assert (active_run_dir / "analysis" / "synthetic_single_pattern_overlay.png").exists()
    assert (active_run_dir / "analysis" / "representative_fit.png").exists()
    assert (active_run_dir / "analysis" / "representative_pose_comparison.png").exists()
    assert (active_run_dir / "analysis" / "representative_observation.json").exists()
    assert (active_run_dir / "analysis" / "position_error_timeline.png").exists()
    assert (active_run_dir / "analysis" / "reprojection_timeline.png").exists()
    assert (active_run_dir / "analysis" / "optimizer_loss_trace.png").exists()
    assert (active_run_dir / "analysis" / "world_pose_by_tag" / "camera_world_x_m.png").exists()
    assert (active_run_dir / "analysis" / "world_pose_by_tag" / "camera_world_y_m.png").exists()
    assert (active_run_dir / "analysis" / "world_pose_by_tag" / "camera_world_z_m.png").exists()
    assert (active_run_dir / "analysis" / "world_pose_by_tag" / "camera_world_rx_deg.png").exists()
    assert (active_run_dir / "analysis" / "world_pose_by_tag" / "camera_world_ry_deg.png").exists()
    assert (active_run_dir / "analysis" / "world_pose_by_tag" / "camera_world_rz_deg.png").exists()
    report = (active_run_dir / "analysis" / "report.md").read_text(encoding="utf-8")
    per_tag_reports = list((active_run_dir / "analysis" / "per_tag").glob("tag_*/report.md"))
    single_pattern_report = (active_run_dir / "analysis" / "single_pattern_report.md").read_text(encoding="utf-8")
    worst_single_pattern_report = (active_run_dir / "analysis" / "worst_single_pattern_report.md").read_text(encoding="utf-8")
    real_measurement_known_solution_report = (
        active_run_dir / "analysis" / "real_measurement_known_solution_report.md"
    ).read_text(encoding="utf-8")
    synthetic_single_pattern_report = (active_run_dir / "analysis" / "synthetic_single_pattern_report.md").read_text(
        encoding="utf-8"
    )
    assert "## Representative Single-Pattern Solve" in report
    assert "## Per-Pattern Breakdown" in report
    assert "## World-Pose Components By Tag" in report
    assert "joint solve over all visible tag points" in report
    assert "### Units And Scale" in report
    assert "### Pose Comparison Against Ground Truth" in report
    assert "### Newton Convergence From Camera Obscura Seed" in report
    assert "## Image Comparison" in single_pattern_report
    assert "## Point Table" in single_pattern_report
    assert "## Optimization Result" in single_pattern_report
    assert "## Newton Convergence From Camera Obscura Seed" in worst_single_pattern_report
    assert "## Point Table" in worst_single_pattern_report
    assert "## Damped vs Undamped Summary" in real_measurement_known_solution_report
    assert "## Newton From Ground Truth On Real Measurement: Undamped" in real_measurement_known_solution_report
    assert "| gt - cam pos [m] |" in real_measurement_known_solution_report
    assert "## Key Result" in synthetic_single_pattern_report
    assert "## Damped vs Undamped Summary" in synthetic_single_pattern_report
    assert "## Newton Convergence On Perfect Synthetic Points" in synthetic_single_pattern_report
    assert "| alpha |" in synthetic_single_pattern_report
    assert "| gt - cam pos [m] |" in synthetic_single_pattern_report
    assert per_tag_reports
    assert "Separate Estimation Report" in per_tag_reports[0].read_text(encoding="utf-8")
    assert "||grad||" not in report
    assert "||step||" not in report
    assert not list((active_run_dir / "analysis").glob("*.finalizing.mp4"))
