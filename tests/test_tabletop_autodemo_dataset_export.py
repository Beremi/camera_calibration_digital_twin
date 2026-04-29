from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import subprocess

import cv2

from calib_sim.reporting.tabletop_autodemo_dataset import (
    _compute_gt_tag_image_projections,
    _capture_clean_master_run_subprocess,
    _detector_metrics,
    _extract_json_object_from_text,
    _noisy_bgr_image,
    apply_phone_physical_variant,
    _materialize_backend_replay_run,
    derive_noisy_dataset_variant,
    export_tabletop_autodemo_dataset,
    write_detection_overlay_video,
    write_phone_capture_mirror,
    write_dataset_convenience_exports,
    write_tabletop_autodemo_pixel_precision_report,
    write_tabletop_autodemo_dataset_report,
)
from tests._isaac_test_helpers import make_estimator_quality_isaac_run


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _pupil_summary(*, variant: str) -> dict[str, object]:
    return {
        "backend": "new_pupil",
        "variant": variant,
        "detection_metrics": {
            "frames_total": 10,
            "frames_with_detections": 8,
            "frames_with_gt_visible_tags": 7,
            "detections_total": 30,
            "detections_per_frame": 3.0,
            "gt_visible_hit_rate": 0.9,
            "pixel_metrics": {
                "matched_detection_count": 20,
                "corner_rmse_mean_px": 0.6,
                "corner_rmse_p95_px": 1.4,
                "center_error_mean_px": 0.3,
                "center_error_p95_px": 0.8,
            },
            "per_tag_coverage": {
                "0": {"visible_frames": 10, "detected_frames": 10, "visible_hit_rate": 1.0},
                "42": {"visible_frames": 6, "detected_frames": 5, "visible_hit_rate": 5.0 / 6.0},
            },
        },
        "measurement": {
            "anchor_only": {
                "mean_position_error_m": 0.01 if variant == "clean" else 0.02,
                "p95_position_error_m": 0.02 if variant == "clean" else 0.03,
                "mean_rotation_error_deg": 0.5,
                "p95_rotation_error_deg": 0.8,
                "sample_count": 10,
            },
            "multitag": {
                "mean_position_error_m": 0.008 if variant == "clean" else 0.015,
                "p95_position_error_m": 0.015 if variant == "clean" else 0.025,
                "mean_rotation_error_deg": 0.4,
                "p95_rotation_error_deg": 0.7,
                "sample_count": 10,
            },
        },
        "paths": {},
    }


def _pupil_corner_summary(*, variant: str) -> dict[str, object]:
    summary = _pupil_summary(variant=variant)
    summary.pop("measurement", None)
    summary["mode"] = "corners_only"
    return summary


def test_extract_json_object_from_mixed_stdout() -> None:
    payload = {"run_id": "capture_only_test", "capture_only": True}
    mixed_text = "isaac startup log\n{bad json\n" + json.dumps(payload, indent=2) + "\nmore logs after json\n"
    assert _extract_json_object_from_text(mixed_text) == payload


def test_capture_subprocess_recovers_when_isaac_stdout_has_no_json(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "capture_source"
    (output_dir / "raw").mkdir(parents=True)
    (output_dir / "manifest.json").write_text(json.dumps({"run_id": "fresh_capture"}), encoding="utf-8")
    (output_dir / "raw" / "camera_frames.jsonl").write_text("{}\n", encoding="utf-8")

    class Completed:
        returncode = 0
        stdout = "isaac startup logs without payload"
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: Completed())

    payload = _capture_clean_master_run_subprocess(
        output_dir,
        profile_key="tabletop_replica",
        seed=7,
        corners_only=True,
        skip_phone_export=False,
        phone_profile_path="config/camera/pixel_9a_main.toml",
    )

    assert payload["run_id"] == "fresh_capture"
    assert payload["run_dir"] == str(output_dir.resolve())
    assert payload["summary"]["source"] == "recovered_from_capture_directory_without_stdout_json"


def test_convenience_exports_and_noisy_variant_preserve_timestamps(tmp_path: Path) -> None:
    source_run = make_estimator_quality_isaac_run(tmp_path)
    dataset_root = tmp_path / "tabletop_autodemo_dataset"
    clean_dir = dataset_root / "clean"
    noisy_dir = dataset_root / "noisy"
    shutil.copytree(source_run, clean_dir)

    write_dataset_convenience_exports(clean_dir)
    assert (clean_dir / "mounted_video.mp4").exists()
    assert (clean_dir / "mounted_video_timestamps.csv").exists()
    assert (clean_dir / "gt_path.csv").exists()

    derive_noisy_dataset_variant(clean_dir, noisy_dir, seed=123)

    clean_frames = [
        json.loads(line)
        for line in (clean_dir / "raw" / "camera_frames.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    noisy_frames = [
        json.loads(line)
        for line in (noisy_dir / "raw" / "camera_frames.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["timestamp_s"] for row in clean_frames] == [row["timestamp_s"] for row in noisy_frames]

    clean_imu = _read_csv_rows(clean_dir / "raw" / "imu.csv")
    noisy_imu = _read_csv_rows(noisy_dir / "raw" / "imu.csv")
    assert [row["timestamp_s"] for row in clean_imu] == [row["timestamp_s"] for row in noisy_imu]
    assert [row["packet_index"] for row in clean_imu] == [row["packet_index"] for row in noisy_imu]
    assert noisy_imu[0]["noise_preset"] == "imu_nominal_phone"
    assert noisy_imu[0]["ax"] != clean_imu[0]["ax"] or noisy_imu[0]["wx"] != clean_imu[0]["wx"]

    assert (noisy_dir / "gt" / "camera_gt.csv").read_text(encoding="utf-8") == (
        clean_dir / "gt" / "camera_gt.csv"
    ).read_text(encoding="utf-8")

    clean_image_bytes = (clean_dir / "raw" / "rgb" / "frame_000000.png").read_bytes()
    noisy_image_bytes = (noisy_dir / "raw" / "rgb" / "frame_000000.png").read_bytes()
    assert clean_image_bytes != noisy_image_bytes


def test_phone_physical_variant_exports_sidecar_capture(tmp_path: Path) -> None:
    source_run = make_estimator_quality_isaac_run(tmp_path)
    dataset_root = tmp_path / "tabletop_autodemo_dataset"
    clean_dir = dataset_root / "clean"
    phone_clean_dir = dataset_root / "phone_clean"
    shutil.copytree(source_run, clean_dir)

    apply_phone_physical_variant(clean_dir, phone_clean_dir, seed=123)
    capture_manifest = write_phone_capture_mirror(phone_clean_dir)

    camera_snapshot = json.loads((phone_clean_dir / "config_snapshot" / "camera.json").read_text(encoding="utf-8"))
    frame_rows = [
        json.loads(line)
        for line in (phone_clean_dir / "raw" / "camera_frames.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert camera_snapshot["width_px"] == 1280
    assert camera_snapshot["height_px"] == 720
    assert camera_snapshot["distortion_model"] == "opencv_pinhole"
    assert not any(abs(float(value)) > 1e-12 for value in camera_snapshot["distortion_coefficients"])
    assert frame_rows[0]["image_width_px"] == 1280
    assert frame_rows[0]["image_height_px"] == 720
    assert not any(abs(float(value)) > 1e-12 for value in frame_rows[0]["intrinsics_snapshot"]["distortion_coefficients"])
    assert (phone_clean_dir / "gt" / "tag_image_projections.jsonl").exists()

    with (phone_clean_dir / "phone_capture" / "frames.csv").open("r", encoding="utf-8", newline="") as handle:
        frame_reader = csv.DictReader(handle)
        assert list(frame_reader.fieldnames or []) == [
            "frame_index",
            "camera_timestamp_nanos",
            "relative_session_nanos",
            "width",
            "height",
            "rotation_degrees",
        ]
        phone_frame = next(frame_reader)
        assert phone_frame["width"] == "640"
        assert phone_frame["height"] == "480"
        assert phone_frame["rotation_degrees"] == "0"
    with (phone_clean_dir / "phone_capture" / "camera_results.csv").open("r", encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle).fieldnames or []) == [
            "frame_number",
            "sensor_timestamp_nanos",
            "relative_session_nanos",
            "exposure_time_nanos",
            "sensitivity_iso",
            "frame_duration_nanos",
            "rolling_shutter_skew_nanos",
            "lens_focus_distance_diopters",
            "lens_state",
            "af_mode",
            "af_state",
            "ae_state",
            "awb_state",
            "video_stabilization_mode",
            "optical_stabilization_mode",
            "zoom_ratio",
            "crop_left",
            "crop_top",
            "crop_right",
            "crop_bottom",
            "active_physical_camera_id",
        ]
    with (phone_clean_dir / "phone_capture" / "imu.csv").open("r", encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle).fieldnames or []) == [
            "elapsed_realtime_nanos",
            "sensor_type",
            "x",
            "y",
            "z",
            "accuracy",
            "bias_x",
            "bias_y",
            "bias_z",
        ]

    session = json.loads((phone_clean_dir / "phone_capture" / "session.json").read_text(encoding="utf-8"))
    assert session["device"]["model"] == "Pixel 9a"
    assert session["frameCount"] == len(frame_rows)
    assert session["cameraResultCount"] == len(frame_rows)
    assert session["sampleCounts"]["accelerometer"] > 0
    assert capture_manifest["files"]["video_mp4"] == "phone_capture/video.mp4"

    if shutil.which("ffprobe") and not capture_manifest["video"]["fallback"]:
        completed = subprocess.run(
            [
                "ffprobe",
                "-hide_banner",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,pix_fmt,width,height,has_b_frames,color_space",
                "-of",
                "json",
                str(phone_clean_dir / "phone_capture" / "video.mp4"),
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        stream = json.loads(completed.stdout)["streams"][0]
        assert stream["codec_name"] == "h264"
        assert stream["pix_fmt"] == "yuv420p"
        assert stream["width"] == 1280
        assert stream["height"] == 720
        assert int(stream["has_b_frames"]) == 0
        assert stream["color_space"] == "bt709"


def test_phone_low_light_noise_is_deterministic_and_calibrated() -> None:
    import cv2
    import numpy as np

    clean = np.full((128, 96, 3), 127, dtype=np.uint8)
    noisy_a = _noisy_bgr_image(clean, frame_index=5, seed=123)
    noisy_b = _noisy_bgr_image(clean, frame_index=5, seed=123)
    noisy_c = _noisy_bgr_image(clean, frame_index=6, seed=123)

    assert np.array_equal(noisy_a, noisy_b)
    assert not np.array_equal(noisy_a, noisy_c)

    ycrcb = cv2.cvtColor(noisy_a, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    luma_residual = ycrcb[:, :, 0] - cv2.GaussianBlur(ycrcb[:, :, 0], (0, 0), sigmaX=1.2)
    chroma_residual = ycrcb[:, :, 1] - cv2.GaussianBlur(ycrcb[:, :, 1], (0, 0), sigmaX=1.2)
    assert 0.35 <= float(np.std(luma_residual)) <= 2.2
    assert 0.05 <= float(np.std(chroma_residual)) <= 0.8


def test_detection_overlay_video_is_written_for_visual_check(tmp_path: Path) -> None:
    variant_dir = make_estimator_quality_isaac_run(tmp_path)

    manifest = write_detection_overlay_video(
        variant_dir,
        backend="new_pupil",
        detections_path=variant_dir / "raw" / "detections.jsonl",
    )

    overlay_path = variant_dir / "localization" / "new_pupil" / "detections_overlay.mp4"
    assert overlay_path.exists()
    assert manifest["relative_path"] == "localization/new_pupil/detections_overlay.mp4"
    assert manifest["frame_count"] == 2
    assert manifest["detection_count"] > 0

    capture = cv2.VideoCapture(str(overlay_path))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 2
        assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == int(manifest["width_px"])
        assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == int(manifest["height_px"])
        ok, _frame = capture.read()
        assert ok
    finally:
        capture.release()


def test_gt_tag_image_projections_export_and_frame_visibility(tmp_path: Path) -> None:
    variant_dir = make_estimator_quality_isaac_run(tmp_path)

    rows = _compute_gt_tag_image_projections(variant_dir)
    assert rows
    assert (variant_dir / "gt" / "tag_image_projections.jsonl").exists()
    assert (variant_dir / "gt" / "tag_render_geometry.json").exists()
    assert (variant_dir / "gt" / "camera_projection_pose_trace.jsonl").exists()
    assert (variant_dir / "gt" / "camera_projection_gt.csv").exists()

    camera_frames = [
        json.loads(line)
        for line in (variant_dir / "raw" / "camera_frames.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    fully_visible_by_frame: dict[int, set[int]] = {}
    for row in rows:
        if bool(row["visibility_flags"]["fully_visible"]):
            fully_visible_by_frame.setdefault(int(row["frame_index"]), set()).add(int(row["tag_id"]))
    assert set(camera_frames[0]["visible_gt_tag_ids"]) == fully_visible_by_frame.get(0, set())
    assert set(camera_frames[1]["visible_gt_tag_ids"]) == fully_visible_by_frame.get(1, set())

    frame_zero_rows = [row for row in rows if row["frame_index"] == 0]
    assert frame_zero_rows
    assert all(len(row["corners_xy"]) == 4 for row in frame_zero_rows)
    assert all(len(row["center_xy"]) == 2 for row in frame_zero_rows)
    assert all(
        set(row["visibility_flags"]) == {
            "corners_in_front",
            "center_in_front",
            "any_corner_in_frame",
            "all_corners_in_frame",
            "fully_visible",
            "partially_visible",
        }
        for row in frame_zero_rows
    )
    assert all("projection_pose_frame_index" in row for row in frame_zero_rows)
    assert all("projection_pose_timestamp_s" in row for row in frame_zero_rows)


def test_dataset_report_summarizes_clean_and_noisy_variants(tmp_path: Path) -> None:
    dataset_root = tmp_path / "tabletop_autodemo_dataset"
    for variant in ("clean", "noisy"):
        summary_dir = dataset_root / variant / "localization" / "new_pupil"
        summary_dir.mkdir(parents=True, exist_ok=True)
        (summary_dir / "backend_summary.json").write_text(
            json.dumps(_pupil_summary(variant=variant)),
            encoding="utf-8",
        )

    report_path = write_tabletop_autodemo_dataset_report(dataset_root)
    report_text = report_path.read_text(encoding="utf-8")

    assert report_path.exists()
    assert "clean" in report_text
    assert "noisy" in report_text
    assert "new_pupil" in report_text
    assert "Supported detector on clean data" in report_text
    assert "Localization Summary" in report_text
    assert "GT-Visible Hit Rate" in report_text
    assert "gt/tag_image_projections.jsonl" in report_text
    assert "GT Basis" in report_text
    assert "frame-index matching" in report_text


def test_pixel_precision_report_exports_variant_curves(tmp_path: Path) -> None:
    source_run = make_estimator_quality_isaac_run(tmp_path)
    dataset_root = tmp_path / "tabletop_autodemo_dataset"
    clean_dir = dataset_root / "clean"
    noisy_dir = dataset_root / "noisy"
    shutil.copytree(source_run, clean_dir)
    write_dataset_convenience_exports(clean_dir)
    derive_noisy_dataset_variant(clean_dir, noisy_dir, seed=123)

    for variant_dir in (clean_dir, noisy_dir):
        gt_rows = _compute_gt_tag_image_projections(variant_dir)
        detection_rows = [
            json.loads(line)
            for line in (variant_dir / "raw" / "detections.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        frame_rows = [
            json.loads(line)
            for line in (variant_dir / "raw" / "camera_frames.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        backend_dir = variant_dir / "localization" / "new_pupil"
        backend_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(variant_dir / "raw" / "detections.jsonl", backend_dir / "detections.jsonl")
        summary = _pupil_summary(variant=variant_dir.name)
        summary["detection_metrics"] = _detector_metrics(frame_rows, detection_rows, gt_rows)
        (backend_dir / "backend_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    report_path = write_tabletop_autodemo_pixel_precision_report(dataset_root)
    report_text = report_path.read_text(encoding="utf-8")

    assert report_path.exists()
    assert (dataset_root / "analysis" / "clean_tag_corner_rmse_by_frame.csv").exists()
    assert (dataset_root / "analysis" / "noisy_tag_corner_rmse_by_frame.csv").exists()
    assert (dataset_root / "analysis" / "clean_tag_corner_rmse_by_frame.png").exists()
    assert (dataset_root / "analysis" / "noisy_tag_corner_rmse_by_frame.png").exists()
    assert "Corner RMSE Mean [px]" in report_text
    assert "Per-Tag Precision" in report_text
    assert "analysis/clean_tag_corner_rmse_by_frame.png" in report_text
    assert "analysis/noisy_tag_corner_rmse_by_frame.png" in report_text


def test_materialized_replay_run_uses_resolvable_inputs(tmp_path: Path) -> None:
    variant_dir = make_estimator_quality_isaac_run(tmp_path)
    backend_dir = tmp_path / "backend_localization"
    backend_dir.mkdir(parents=True, exist_ok=True)
    detections_path = backend_dir / "detections.jsonl"
    detections_path.write_text("", encoding="utf-8")

    replay_dir = _materialize_backend_replay_run(variant_dir, backend_dir, detections_path)

    assert (replay_dir / "manifest.json").exists()
    assert (replay_dir / "raw" / "camera_frames.jsonl").exists()
    assert (replay_dir / "raw" / "detections.jsonl").exists()


def test_export_tabletop_autodemo_dataset_uses_existing_source_run(tmp_path: Path, monkeypatch) -> None:
    source_run = make_estimator_quality_isaac_run(tmp_path)
    dataset_root = tmp_path / "tabletop_autodemo_dataset"

    def fake_generate_backend_localization(variant_dir: Path, *, backend: str) -> dict[str, object]:
        assert backend == "new_pupil"
        backend_dir = variant_dir / "localization" / backend
        backend_dir.mkdir(parents=True, exist_ok=True)
        summary = _pupil_summary(variant=variant_dir.name)
        (backend_dir / "backend_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return summary

    monkeypatch.setattr(
        "calib_sim.reporting.tabletop_autodemo_dataset._generate_backend_localization",
        fake_generate_backend_localization,
    )

    payload = export_tabletop_autodemo_dataset(output_root=dataset_root, source_run_dir=source_run)

    assert Path(payload["clean_dir"]).exists()
    assert Path(payload["phone_clean_dir"]).exists()
    assert Path(payload["noisy_dir"]).exists()
    assert payload["variants"] == ["clean", "phone_clean", "noisy"]
    assert (dataset_root / "clean" / "dataset_manifest.json").exists()
    assert (dataset_root / "phone_clean" / "dataset_manifest.json").exists()
    assert (dataset_root / "noisy" / "dataset_manifest.json").exists()
    assert (dataset_root / "clean" / "phone_capture" / "session.json").exists()
    assert (dataset_root / "phone_clean" / "phone_capture" / "session.json").exists()
    assert (dataset_root / "noisy" / "phone_capture" / "session.json").exists()
    clean_manifest = json.loads((dataset_root / "clean" / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert clean_manifest["phone_capture"]["video"]["width_px"] == 1280
    assert clean_manifest["phone_capture"]["video"]["height_px"] == 720
    assert (dataset_root / "clean" / "gt" / "tag_image_projections.jsonl").exists()
    assert (dataset_root / "phone_clean" / "gt" / "tag_image_projections.jsonl").exists()
    assert (dataset_root / "noisy" / "gt" / "tag_image_projections.jsonl").exists()
    assert (dataset_root / "clean" / "gt" / "camera_projection_gt.csv").exists()
    assert (dataset_root / "phone_clean" / "gt" / "camera_projection_gt.csv").exists()
    assert (dataset_root / "noisy" / "gt" / "camera_projection_gt.csv").exists()
    assert (dataset_root / "report.md").exists()


def test_export_tabletop_autodemo_dataset_stages_source_inside_output_root(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "tabletop_autodemo_dataset"
    source_run = dataset_root / "_source_clean"
    source_run.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(make_estimator_quality_isaac_run(tmp_path), source_run)

    def fake_generate_backend_localization(variant_dir: Path, *, backend: str) -> dict[str, object]:
        assert backend == "new_pupil"
        backend_dir = variant_dir / "localization" / backend
        backend_dir.mkdir(parents=True, exist_ok=True)
        summary = _pupil_summary(variant=variant_dir.name)
        (backend_dir / "backend_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return summary

    monkeypatch.setattr(
        "calib_sim.reporting.tabletop_autodemo_dataset._generate_backend_localization",
        fake_generate_backend_localization,
    )

    payload = export_tabletop_autodemo_dataset(output_root=dataset_root, source_run_dir=source_run)

    assert Path(payload["clean_dir"]).exists()
    assert Path(payload["noisy_dir"]).exists()
    assert not (dataset_root / "_source_clean").exists()
    assert (dataset_root / "clean" / "manifest.json").exists()
    assert (dataset_root / "report.md").exists()


def test_export_tabletop_autodemo_dataset_capture_only_finalizes_clean_variant(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "capture_only_variant"
    source_run = make_estimator_quality_isaac_run(tmp_path)

    def fake_capture(output_dir: Path, *, profile_key: str, seed: int) -> dict[str, object]:
        shutil.copytree(source_run, output_dir, dirs_exist_ok=True)
        return {
            "run_id": "capture_only_test",
            "run_dir": str(output_dir),
            "duration_s": 60.0,
            "summary": {"complete": True},
        }

    monkeypatch.setattr(
        "calib_sim.reporting.tabletop_autodemo_dataset._capture_clean_master_run",
        fake_capture,
    )

    payload = export_tabletop_autodemo_dataset(output_root=dataset_root, capture_only=True)

    assert payload["capture_only"] is True
    assert (dataset_root / "dataset_manifest.json").exists()
    assert (dataset_root / "dataset_export_manifest.json").exists()
    assert (dataset_root / "mounted_video.mp4").exists()
    assert (dataset_root / "gt_path.csv").exists()


def test_export_tabletop_autodemo_dataset_corners_only_skips_full_analysis(tmp_path: Path, monkeypatch) -> None:
    source_run = make_estimator_quality_isaac_run(tmp_path)
    dataset_root = tmp_path / "tabletop_autodemo_dataset"

    def fake_generate_backend_corner_summary(variant_dir: Path, *, backend: str) -> dict[str, object]:
        assert backend == "new_pupil"
        backend_dir = variant_dir / "localization" / backend
        backend_dir.mkdir(parents=True, exist_ok=True)
        summary = _pupil_corner_summary(variant=variant_dir.name)
        (backend_dir / "backend_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (backend_dir / "detections.jsonl").write_text("", encoding="utf-8")
        (variant_dir / "raw" / "detections.jsonl").write_text("", encoding="utf-8")
        return summary

    def fail_generate_backend_localization(*_args, **_kwargs) -> dict[str, object]:
        raise AssertionError("full localization should not run in corners-only export mode")

    monkeypatch.setattr(
        "calib_sim.reporting.tabletop_autodemo_dataset._generate_backend_corner_summary",
        fake_generate_backend_corner_summary,
    )
    monkeypatch.setattr(
        "calib_sim.reporting.tabletop_autodemo_dataset._generate_backend_localization",
        fail_generate_backend_localization,
    )

    payload = export_tabletop_autodemo_dataset(
        output_root=dataset_root,
        source_run_dir=source_run,
        corners_only=True,
    )

    clean_manifest = json.loads((dataset_root / "clean" / "dataset_manifest.json").read_text(encoding="utf-8"))
    phone_clean_manifest = json.loads((dataset_root / "phone_clean" / "dataset_manifest.json").read_text(encoding="utf-8"))
    noisy_manifest = json.loads((dataset_root / "noisy" / "dataset_manifest.json").read_text(encoding="utf-8"))
    clean_scene = json.loads((dataset_root / "clean" / "config_snapshot" / "scene.json").read_text(encoding="utf-8"))
    phone_clean_scene = json.loads((dataset_root / "phone_clean" / "config_snapshot" / "scene.json").read_text(encoding="utf-8"))
    noisy_scene = json.loads((dataset_root / "noisy" / "config_snapshot" / "scene.json").read_text(encoding="utf-8"))

    assert payload["export_mode"] == "corners_only"
    assert payload["variants"] == ["clean", "phone_clean", "noisy"]
    assert (dataset_root / "report.md").exists()
    assert not (dataset_root / "pixel_precision_report.md").exists()
    assert not (dataset_root / "clean" / "mounted_video.mp4").exists()
    assert not (dataset_root / "phone_clean" / "mounted_video.mp4").exists()
    assert not (dataset_root / "noisy" / "mounted_video.mp4").exists()
    assert not (dataset_root / "clean" / "gt_path.csv").exists()
    assert not (dataset_root / "phone_clean" / "gt_path.csv").exists()
    assert not (dataset_root / "noisy" / "gt_path.csv").exists()
    assert (dataset_root / "clean" / "phone_capture" / "video.mp4").exists()
    assert (dataset_root / "phone_clean" / "phone_capture" / "video.mp4").exists()
    assert (dataset_root / "noisy" / "phone_capture" / "video.mp4").exists()
    assert (dataset_root / "clean" / "gt" / "tag_image_projections.jsonl").exists()
    assert (dataset_root / "phone_clean" / "gt" / "tag_image_projections.jsonl").exists()
    assert (dataset_root / "noisy" / "gt" / "tag_image_projections.jsonl").exists()
    assert (dataset_root / "clean" / "raw" / "detections.jsonl").exists()
    assert (dataset_root / "phone_clean" / "raw" / "detections.jsonl").exists()
    assert (dataset_root / "noisy" / "raw" / "detections.jsonl").exists()
    assert clean_manifest["export_mode"] == "corners_only"
    assert phone_clean_manifest["export_mode"] == "corners_only"
    assert noisy_manifest["export_mode"] == "corners_only"
    assert "convenience_exports" not in clean_manifest
    assert "convenience_exports" not in phone_clean_manifest
    assert "convenience_exports" not in noisy_manifest
    assert "phone_capture" in clean_manifest
    assert "phone_capture" in phone_clean_manifest
    assert "phone_capture" in noisy_manifest
    assert clean_scene["frontend"]["solve_tag_pose"] is False
    assert phone_clean_scene["frontend"]["solve_tag_pose"] is False
    assert noisy_scene["frontend"]["solve_tag_pose"] is False


def test_export_tabletop_autodemo_dataset_skip_phone_export_preserves_legacy_variants(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_run = make_estimator_quality_isaac_run(tmp_path)
    dataset_root = tmp_path / "tabletop_autodemo_dataset"

    def fake_generate_backend_corner_summary(variant_dir: Path, *, backend: str) -> dict[str, object]:
        assert backend == "new_pupil"
        backend_dir = variant_dir / "localization" / backend
        backend_dir.mkdir(parents=True, exist_ok=True)
        summary = _pupil_corner_summary(variant=variant_dir.name)
        (backend_dir / "backend_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (backend_dir / "detections.jsonl").write_text("", encoding="utf-8")
        (variant_dir / "raw" / "detections.jsonl").write_text("", encoding="utf-8")
        return summary

    monkeypatch.setattr(
        "calib_sim.reporting.tabletop_autodemo_dataset._generate_backend_corner_summary",
        fake_generate_backend_corner_summary,
    )

    payload = export_tabletop_autodemo_dataset(
        output_root=dataset_root,
        source_run_dir=source_run,
        corners_only=True,
        skip_phone_export=True,
        phone_profile_path=tmp_path / "does_not_exist.toml",
    )

    assert payload["phone_export_enabled"] is False
    assert payload["variants"] == ["clean", "noisy"]
    assert payload["phone_clean_dir"] is None
    assert not (dataset_root / "phone_clean").exists()
    assert not (dataset_root / "clean" / "phone_capture").exists()
    assert not (dataset_root / "noisy" / "phone_capture").exists()
    assert (dataset_root / "clean" / "dataset_manifest.json").exists()
    assert (dataset_root / "noisy" / "dataset_manifest.json").exists()
