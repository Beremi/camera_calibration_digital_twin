"""Aggregate second-pass Isaac draft runs into publication-ready artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
from typing import Any

import cv2
import numpy as np

from calib_sim.reporting import (
    compute_isaac_estimator_quality,
    generate_isaac_report_artifacts,
    write_isaac_estimator_quality_figures,
)


DEFAULT_SECOND_PASS_DRAFT_LOCK = Path("docs/second_pass_draft_lock.json")
DEFAULT_SECOND_PASS_DRAFT_SEEDS = (7, 11, 17)
DEFAULT_SECOND_PASS_CONDITIONS = (
    "nominal_full_anchor",
    "intermittent_anchor",
    "servo_stress",
)
DEFAULT_SECOND_PASS_FUSED_FILTER_OVERRIDES = {
    "allow_anchor_reacquisition_after_first_lock": True,
    "disable_imu_prediction_while_anchor_suppressed": False,
    "suppression_propagation_mode": "full_imu",
    "suppression_imu_specific_force_gate_mps2": None,
    "dropout_post_reacquisition_covariance_scale": None,
}
_CONDITION_LABELS = {
    "nominal_full_anchor": "Nominal / full anchor",
    "intermittent_anchor": "Intermittent anchor",
    "servo_stress": "Servo stress",
}


def second_pass_draft_run_id(condition: str, estimator_mode: str, seed: int) -> str:
    return f"second_pass_draft_{estimator_mode}_{condition}_seed_{int(seed):03d}"


def second_pass_fused_filter_overrides(lock_payload: dict[str, Any]) -> dict[str, Any]:
    draft_selection = dict(lock_payload.get("draft_selection", {}))
    configured = dict(draft_selection.get("fused_nominal_filter_overrides", {}))
    merged = dict(DEFAULT_SECOND_PASS_FUSED_FILTER_OVERRIDES)
    merged.update(configured)
    return {
        "allow_anchor_reacquisition_after_first_lock": bool(
            merged.get("allow_anchor_reacquisition_after_first_lock", True)
        ),
        "disable_imu_prediction_while_anchor_suppressed": bool(
            merged.get("disable_imu_prediction_while_anchor_suppressed", False)
        ),
        "suppression_propagation_mode": str(merged.get("suppression_propagation_mode", "full_imu")),
        "suppression_imu_specific_force_gate_mps2": _float_or_none(
            merged.get("suppression_imu_specific_force_gate_mps2")
        ),
        "dropout_post_reacquisition_covariance_scale": _float_or_none(
            merged.get("dropout_post_reacquisition_covariance_scale")
        ),
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _float_or_none(value: Any) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def _mean_std(values: list[float | None]) -> tuple[float | None, float | None]:
    usable = [float(value) for value in values if value is not None]
    if not usable:
        return None, None
    array = np.asarray(usable, dtype=np.float64)
    return float(np.mean(array)), float(np.std(array))


def _tex_value(value: Any, *, digits: int = 3, percent: bool = False) -> str:
    if value in ("", None):
        return r"\ArtifactPending{}"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value.replace("_", r"\_")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if percent:
            return f"{numeric * 100.0:.1f}"
        return f"{numeric:.{digits}f}"
    return str(value)


def _format_mean_std(mean: float | None, std: float | None, *, percent: bool = False) -> str:
    if mean is None or std is None:
        return r"\ArtifactPending{}"
    if percent:
        return rf"{mean * 100.0:.1f} $\pm$ {std * 100.0:.1f}"
    return rf"{mean:.3f} $\pm$ {std:.3f}"


def _macro_definition(name: str, rows: list[str]) -> str:
    return "\n".join(
        [
            rf"\providecommand{{\{name}}}{{}}",
            rf"\renewcommand{{\{name}}}{{%",
            *rows,
            "}",
        ]
    )


def _save_text_figure(path: Path, *, title: str, lines: list[str]) -> None:
    canvas = np.full((420, 1280, 3), 252, dtype=np.uint8)
    cv2.putText(canvas, title, (28, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (22, 26, 32), 2, cv2.LINE_AA)
    y = 92
    for line in lines:
        cv2.putText(canvas, line[:128], (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (62, 66, 72), 1, cv2.LINE_AA)
        y += 34
    cv2.imwrite(str(path), canvas)


def _compose_two_panel_figure(
    output_path: Path,
    *,
    title: str,
    left_label: str,
    left_path: Path,
    right_label: str,
    right_path: Path,
) -> None:
    if not left_path.exists() or not right_path.exists():
        _save_text_figure(
            output_path,
            title=title,
            lines=[
                f"Missing source figure: {left_path if not left_path.exists() else right_path}",
            ],
        )
        return
    left = cv2.imread(str(left_path))
    right = cv2.imread(str(right_path))
    if left is None or right is None:
        _save_text_figure(output_path, title=title, lines=["Could not decode one of the source figures."])
        return
    height = max(left.shape[0], right.shape[0]) + 70
    width = left.shape[1] + right.shape[1] + 60
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    cv2.putText(canvas, title, (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (22, 26, 32), 2, cv2.LINE_AA)
    left_y = 52
    right_y = 52
    canvas[left_y : left_y + left.shape[0], 20 : 20 + left.shape[1]] = left
    right_x = 40 + left.shape[1]
    canvas[right_y : right_y + right.shape[0], right_x : right_x + right.shape[1]] = right
    cv2.putText(canvas, left_label, (24, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (53, 102, 188), 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        right_label,
        (right_x, height - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (188, 92, 60),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output_path), canvas)


def _save_grouped_summary_figure(
    output_path: Path,
    *,
    title: str,
    rows: list[dict[str, Any]],
    left_metric_key: str,
    right_metric_key: str,
    left_label: str,
    right_label: str,
) -> None:
    if not rows:
        _save_text_figure(output_path, title=title, lines=["No rows available for this summary figure."])
        return
    lines = [f"{row['condition_label']} / {row['estimator_mode']}: {left_label}={row.get(left_metric_key)!r}, {right_label}={row.get(right_metric_key)!r}" for row in rows]
    _save_text_figure(output_path, title=title, lines=lines)


def _copy_or_render_tuning_heatmap(output_path: Path, tuning_dir: Path) -> None:
    summary_path = tuning_dir / "summary.json"
    if not summary_path.exists():
        _save_text_figure(
            output_path,
            title="Fused Tuning Heatmap",
            lines=["No tuning summary was available at latest_second_pass_tuning/summary.json."],
        )
        return
    summary = _load_json(summary_path)
    rows = list(summary.get("rows", []))
    if not rows:
        _save_text_figure(output_path, title="Fused Tuning Heatmap", lines=["The tuning summary is empty."])
        return
    lines = []
    for row in rows[:12]:
        lines.append(
            " / ".join(
                [
                    f"backend={row.get('smoother_backend', 'n/a')}",
                    f"imu={row.get('imu_process_covariance_scale', 'n/a')}",
                    f"vision={row.get('vision_covariance_scale', 'n/a')}",
                    f"post={row.get('post_relocalization_covariance_scale', 'n/a')}",
                    f"pos={row.get('mean_position_error_m', 'n/a')}",
                    f"way={row.get('mean_waypoint_error_m', 'n/a')}",
                ]
            )
        )
    _save_text_figure(output_path, title="Fused Tuning Heatmap", lines=lines)


def _load_lock(lock_path: Path) -> dict[str, Any]:
    return _load_json(lock_path)


def _infer_suite_run_ids(lock_payload: dict[str, Any]) -> list[str]:
    configured = list(lock_payload.get("draft_selection", {}).get("suite_run_ids", []))
    if configured:
        return [str(run_id) for run_id in configured]
    return [
        second_pass_draft_run_id(condition, estimator_mode, seed)
        for condition in DEFAULT_SECOND_PASS_CONDITIONS
        for estimator_mode in ("visual", "fused")
        for seed in DEFAULT_SECOND_PASS_DRAFT_SEEDS
    ]


def _ensure_run_artifacts(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics_path = run_dir / "analysis" / "metrics.json"
    if not metrics_path.exists():
        generate_isaac_report_artifacts(run_dir, allow_incomplete=False, promote_links=False)
    metrics = _load_json(run_dir / "analysis" / "metrics.json")

    quality_path = run_dir / "analysis" / "estimator_quality.json"
    if quality_path.exists():
        quality = _load_json(quality_path)
    else:
        quality = compute_isaac_estimator_quality(run_dir)
        figure_paths = write_isaac_estimator_quality_figures(run_dir, quality)
        payload = {**quality, "figure_paths": figure_paths}
        _write_json(quality_path, payload)
        quality = payload
    return metrics, quality


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _mean_nis(run_dir: Path, key: str) -> float | None:
    rows = _load_jsonl(run_dir / "estimates" / "uncertainty.jsonl")
    values = []
    for row in rows:
        diagnostics = row.get("diagnostics", {})
        value = _float_or_none(diagnostics.get(key))
        if value is not None:
            values.append(value)
    return None if not values else float(np.mean(np.asarray(values, dtype=np.float64)))


def _condition_from_metrics(metrics: dict[str, Any]) -> str:
    actuation_preset = str(metrics.get("config", {}).get("actuation_preset", "") or "")
    visibility_preset = str(metrics.get("config", {}).get("visibility_preset", "") or "")
    if actuation_preset == "servo_stress":
        return "servo_stress"
    if visibility_preset == "anchor_dropout_nominal":
        return "intermittent_anchor"
    return "nominal_full_anchor"


def _condition_label(condition: str) -> str:
    return _CONDITION_LABELS.get(condition, condition.replace("_", " "))


def _seed_from_run_id(run_id: str) -> int | None:
    token = str(run_id).rsplit("_seed_", maxsplit=1)
    if len(token) != 2:
        return None
    try:
        return int(token[1])
    except ValueError:
        return None


def _record_from_run(run_dir: Path) -> dict[str, Any]:
    metrics, quality = _ensure_run_artifacts(run_dir)
    run_id = str(metrics.get("run_id", run_dir.name))
    condition = _condition_from_metrics(metrics)
    summary = dict(quality.get("summary", {}))
    return {
        "run_id": run_id,
        "run_dir": str(run_dir.resolve()),
        "condition": condition,
        "condition_label": _condition_label(condition),
        "seed": _seed_from_run_id(run_id),
        "estimator_mode": str(metrics.get("config", {}).get("estimator_mode", "")),
        "controller_mode": str(metrics.get("config", {}).get("controller_mode", "")),
        "actuation_preset": str(metrics.get("config", {}).get("actuation_preset", "")),
        "visibility_preset": str(metrics.get("config", {}).get("visibility_preset", "") or ""),
        "duration_s": _float_or_none(metrics.get("timing", {}).get("duration_s")),
        "camera_frames": int(metrics.get("counts", {}).get("camera_frames", 0) or 0),
        "imu_packets": int(metrics.get("counts", {}).get("imu_packets", 0) or 0),
        "commands": int(metrics.get("counts", {}).get("commands", 0) or 0),
        "mean_position_error_m": _float_or_none(metrics.get("trajectory", {}).get("mean_position_error_m")),
        "mean_waypoint_error_m": _float_or_none(metrics.get("control", {}).get("mean_waypoint_error_m")),
        "completion_fraction": _float_or_none(metrics.get("control", {}).get("completion_fraction")),
        "ik_failure_fraction": _float_or_none(metrics.get("control", {}).get("ik_failure_fraction")),
        "mean_actuator_tracking_error": _float_or_none(metrics.get("control", {}).get("mean_actuator_tracking_error")),
        "anchor_rmse_px": _float_or_none(summary.get("anchor_mean_reprojection_rmse_px")),
        "aux_rmse_px": _float_or_none(summary.get("auxiliary_mean_reprojection_rmse_px")),
        "mean_position_radius_95_m": _float_or_none(metrics.get("uncertainty_calibration", {}).get("mean_position_radius_95_m")),
        "empirical_95_coverage_percent": _float_or_none(metrics.get("uncertainty_calibration", {}).get("empirical_95_coverage_percent")),
        "pose_nees": _float_or_none(metrics.get("uncertainty_calibration", {}).get("pose_nees")),
        "sigma_error_correlation": _float_or_none(metrics.get("uncertainty_calibration", {}).get("sigma_error_correlation")),
        "anchor_nis": _mean_nis(run_dir, "last_anchor_nis"),
        "aux_nis": _mean_nis(run_dir, "last_auxiliary_nis"),
        "mean_auxiliary_tag_position_error_m": _float_or_none(metrics.get("map_quality", {}).get("mean_auxiliary_tag_position_error_m")),
        "p95_auxiliary_tag_position_error_m": _float_or_none(metrics.get("map_quality", {}).get("p95_auxiliary_tag_position_error_m")),
        "anchor_visible_raw_fraction": _float_or_none(metrics.get("estimation", {}).get("anchor_visible_raw_fraction")),
        "anchor_visible_effective_fraction": _float_or_none(metrics.get("estimation", {}).get("anchor_visible_effective_fraction")),
        "anchor_update_suppressed_fraction": _float_or_none(metrics.get("estimation", {}).get("anchor_update_suppressed_fraction")),
        "mean_smoother_feedback_norm_m": _float_or_none(summary.get("mean_smoother_correction_norm_m")),
        "smoother_backend": str(_load_json(run_dir / "config_snapshot" / "estimation.json").get("smoother", {}).get("backend", "lightweight")),
    }


def _aggregate_rows(records: list[dict[str, Any]], *, condition: str | None = None) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        if condition is not None and str(record["condition"]) != str(condition):
            continue
        grouped.setdefault((str(record["condition"]), str(record["estimator_mode"])), []).append(record)
    rows: list[dict[str, Any]] = []
    for (group_condition, estimator_mode), values in sorted(grouped.items()):
        row: dict[str, Any] = {
            "condition": group_condition,
            "condition_label": _condition_label(group_condition),
            "estimator_mode": estimator_mode,
            "sample_count": len(values),
            "run_ids": [str(value["run_id"]) for value in values],
        }
        for key in (
            "duration_s",
            "camera_frames",
            "imu_packets",
            "commands",
            "mean_position_error_m",
            "mean_waypoint_error_m",
            "completion_fraction",
            "ik_failure_fraction",
            "mean_actuator_tracking_error",
            "anchor_rmse_px",
            "aux_rmse_px",
            "mean_position_radius_95_m",
            "empirical_95_coverage_percent",
            "pose_nees",
            "sigma_error_correlation",
            "anchor_nis",
            "aux_nis",
            "mean_auxiliary_tag_position_error_m",
            "p95_auxiliary_tag_position_error_m",
            "anchor_visible_raw_fraction",
            "anchor_visible_effective_fraction",
            "anchor_update_suppressed_fraction",
            "mean_smoother_feedback_norm_m",
        ):
            mean, std = _mean_std([_float_or_none(value.get(key)) for value in values])
            row[key] = mean
            row[f"{key}_std"] = std
        row["smoother_backend"] = str(values[0].get("smoother_backend", "lightweight"))
        rows.append(row)
    return rows


def _required_media_assets_exist(record: dict[str, Any]) -> bool:
    run_dir = Path(record["run_dir"])
    required_paths = (
        run_dir / "analysis" / "report_data" / "trajectory_path.png",
        run_dir / "analysis" / "anchor_vs_aux_residuals.png",
        run_dir / "analysis" / "smoother_correction_timeline.png",
    )
    if not all(path.exists() for path in required_paths):
        return False
    rgb_dir = run_dir / "raw" / "rgb"
    return rgb_dir.exists() and any(rgb_dir.glob("*.png"))


def _median(values: list[float | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    if not usable:
        return None
    return float(np.median(np.asarray(usable, dtype=np.float64)))


def _is_clear_outlier(value: float | None, values: list[float | None]) -> bool:
    usable = [float(item) for item in values if item is not None]
    if value is None or len(usable) < 3:
        return False
    median = float(np.median(np.asarray(usable, dtype=np.float64)))
    deviations = sorted(abs(item - median) for item in usable)
    candidate_deviation = abs(float(value) - median)
    if not deviations or candidate_deviation < deviations[-1] - 1e-12:
        return False
    if len(deviations) == 1:
        return False
    second_largest = deviations[-2]
    return candidate_deviation > max(1.5 * second_largest, second_largest + 1e-9)


def _representative_runs(
    records: list[dict[str, Any]],
    *,
    lock_payload: dict[str, Any] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    locked = dict(lock_payload.get("draft_selection", {}).get("representative_run_ids", {})) if lock_payload else {}
    by_condition: dict[str, dict[str, dict[str, Any]]] = {}
    for condition in DEFAULT_SECOND_PASS_CONDITIONS:
        by_condition[condition] = {}
        for estimator_mode in ("visual", "fused"):
            candidates = [
                record
                for record in records
                if str(record["condition"]) == condition and str(record["estimator_mode"]) == estimator_mode
            ]
            if not candidates:
                continue
            locked_run_id = str(locked.get(condition, {}).get(estimator_mode, "") or "")
            if locked_run_id:
                locked_record = next((record for record in candidates if str(record["run_id"]) == locked_run_id), None)
                if locked_record is not None:
                    by_condition[condition][estimator_mode] = locked_record
                    continue

            mean_position, _ = _mean_std([_float_or_none(candidate.get("mean_position_error_m")) for candidate in candidates])
            median_waypoint = _median([_float_or_none(candidate.get("mean_waypoint_error_m")) for candidate in candidates])
            median_nees = _median([_float_or_none(candidate.get("pose_nees")) for candidate in candidates])
            waypoint_values = [_float_or_none(candidate.get("mean_waypoint_error_m")) for candidate in candidates]
            nees_values = [_float_or_none(candidate.get("pose_nees")) for candidate in candidates]

            ranked = sorted(
                candidates,
                key=lambda candidate: (
                    1 if not _required_media_assets_exist(candidate) else 0,
                    1 if _is_clear_outlier(_float_or_none(candidate.get("mean_waypoint_error_m")), waypoint_values) else 0,
                    1 if _is_clear_outlier(_float_or_none(candidate.get("pose_nees")), nees_values) else 0,
                    abs(_float_or_none(candidate.get("mean_position_error_m")) - mean_position)
                    if _float_or_none(candidate.get("mean_position_error_m")) is not None and mean_position is not None
                    else 1e9,
                    abs(_float_or_none(candidate.get("mean_waypoint_error_m")) - median_waypoint)
                    if _float_or_none(candidate.get("mean_waypoint_error_m")) is not None and median_waypoint is not None
                    else 1e9,
                    abs(_float_or_none(candidate.get("pose_nees")) - median_nees)
                    if _float_or_none(candidate.get("pose_nees")) is not None and median_nees is not None
                    else 1e9,
                    abs(int(candidate.get("seed") or 999) - 7),
                    int(candidate.get("seed") or 999),
                ),
            )
            by_condition[condition][estimator_mode] = ranked[0]
    return by_condition


def _copy_run_figure(run_record: dict[str, Any], relative_path: str, output_path: Path, *, title: str) -> None:
    source = Path(run_record["run_dir"]) / "analysis" / relative_path
    if source.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, output_path)
        return
    _save_text_figure(output_path, title=title, lines=[f"Missing source figure: {source}"])


def generate_second_pass_suite_artifacts(
    output_root: str | Path,
    *,
    lock_path: str | Path = DEFAULT_SECOND_PASS_DRAFT_LOCK,
    regenerate_run_artifacts: bool = False,
    artifact_source: str = "latest_second_pass_suite",
) -> dict[str, Any]:
    root = Path(output_root).resolve()
    suite_dir = root / "latest_second_pass_suite"
    analysis_dir = suite_dir / "analysis"
    report_data_dir = analysis_dir / "report_data"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    report_data_dir.mkdir(parents=True, exist_ok=True)

    lock_payload = _load_lock(Path(lock_path).resolve())
    run_ids = _infer_suite_run_ids(lock_payload)
    records: list[dict[str, Any]] = []
    for run_id in run_ids:
        run_dir = root / run_id
        if not run_dir.exists():
            raise ValueError(f"Required second-pass suite run is missing: {run_dir}")
        if regenerate_run_artifacts:
            generate_isaac_report_artifacts(run_dir, allow_incomplete=False, promote_links=False)
        records.append(_record_from_run(run_dir))

    runtime_rows = _aggregate_rows(records)
    nominal_rows = _aggregate_rows(records, condition="nominal_full_anchor")
    dropout_rows = _aggregate_rows(records, condition="intermittent_anchor")
    stress_rows = _aggregate_rows(records, condition="servo_stress")
    uncertainty_rows = [
        {
            "condition": row["condition"],
            "condition_label": row["condition_label"],
            "estimator_mode": row["estimator_mode"],
            "sample_count": row["sample_count"],
            "mean_position_radius_95_m": row["mean_position_radius_95_m"],
            "empirical_95_coverage_percent": row["empirical_95_coverage_percent"],
            "pose_nees": row["pose_nees"],
            "sigma_error_correlation": row["sigma_error_correlation"],
            "anchor_nis": row["anchor_nis"],
            "aux_nis": row["aux_nis"],
        }
        for row in runtime_rows
    ]
    map_quality_rows = [
        {
            "condition": row["condition"],
            "condition_label": row["condition_label"],
            "estimator_mode": row["estimator_mode"],
            "sample_count": row["sample_count"],
            "mean_auxiliary_tag_position_error_m": row["mean_auxiliary_tag_position_error_m"],
            "p95_auxiliary_tag_position_error_m": row["p95_auxiliary_tag_position_error_m"],
            "anchor_visible_raw_fraction": row["anchor_visible_raw_fraction"],
            "anchor_visible_effective_fraction": row["anchor_visible_effective_fraction"],
            "anchor_update_suppressed_fraction": row["anchor_update_suppressed_fraction"],
        }
        for row in runtime_rows
    ]

    _write_csv(analysis_dir / "draft_runtime_table.csv", runtime_rows)
    _write_csv(analysis_dir / "draft_nominal_table.csv", nominal_rows)
    _write_csv(analysis_dir / "draft_dropout_table.csv", dropout_rows)
    _write_csv(analysis_dir / "draft_actuation_stress_table.csv", stress_rows)
    _write_csv(analysis_dir / "draft_uncertainty_table.csv", uncertainty_rows)
    _write_csv(analysis_dir / "draft_map_quality_table.csv", map_quality_rows)

    runtime_tex_rows = [
        rf"{row['condition_label']} / {row['estimator_mode']} & {_tex_value(row['sample_count'])} & {_tex_value(row['duration_s'])} & {_tex_value(row['camera_frames'])} & {_tex_value(row['imu_packets'])} & {_tex_value(row['commands'])} \\"
        for row in runtime_rows
    ]
    nominal_tex_rows = [
        rf"{row['estimator_mode']} & {_format_mean_std(row['mean_position_error_m'], row['mean_position_error_m_std'])} & {_format_mean_std(row['mean_waypoint_error_m'], row['mean_waypoint_error_m_std'])} & {_format_mean_std(row['completion_fraction'], row['completion_fraction_std'], percent=True)} & {_format_mean_std(row['anchor_rmse_px'], row['anchor_rmse_px_std'])} & {_format_mean_std(row['aux_rmse_px'], row['aux_rmse_px_std'])} \\"
        for row in nominal_rows
    ]
    dropout_tex_rows = [
        rf"{row['estimator_mode']} & {_format_mean_std(row['mean_position_error_m'], row['mean_position_error_m_std'])} & {_format_mean_std(row['mean_waypoint_error_m'], row['mean_waypoint_error_m_std'])} & {_format_mean_std(row['completion_fraction'], row['completion_fraction_std'], percent=True)} & {_format_mean_std(row['anchor_visible_effective_fraction'], row['anchor_visible_effective_fraction_std'], percent=True)} & {_format_mean_std(row['anchor_update_suppressed_fraction'], row['anchor_update_suppressed_fraction_std'], percent=True)} \\"
        for row in dropout_rows
    ]
    stress_tex_rows = [
        rf"{row['estimator_mode']} & {_format_mean_std(row['mean_position_error_m'], row['mean_position_error_m_std'])} & {_format_mean_std(row['mean_waypoint_error_m'], row['mean_waypoint_error_m_std'])} & {_format_mean_std(row['completion_fraction'], row['completion_fraction_std'], percent=True)} & {_format_mean_std(row['mean_actuator_tracking_error'], row['mean_actuator_tracking_error_std'])} & {_format_mean_std(row['ik_failure_fraction'], row['ik_failure_fraction_std'], percent=True)} \\"
        for row in stress_rows
    ]
    uncertainty_tex_rows = [
        rf"{row['condition_label']} / {row['estimator_mode']} & {_tex_value(row['mean_position_radius_95_m'])} & {_tex_value(row['empirical_95_coverage_percent'], percent=True)} & {_tex_value(row['pose_nees'])} & {_tex_value(row['sigma_error_correlation'])} & {_tex_value(row['anchor_nis'])} & {_tex_value(row['aux_nis'])} \\"
        for row in uncertainty_rows
    ]
    map_quality_tex_rows = [
        rf"{row['condition_label']} / {row['estimator_mode']} & {_tex_value(row['mean_auxiliary_tag_position_error_m'])} & {_tex_value(row['p95_auxiliary_tag_position_error_m'])} & {_tex_value(row['anchor_visible_raw_fraction'], percent=True)} & {_tex_value(row['anchor_visible_effective_fraction'], percent=True)} & {_tex_value(row['anchor_update_suppressed_fraction'], percent=True)} \\"
        for row in map_quality_rows
    ]

    report_artifacts_path = report_data_dir / "second_pass_paper_artifacts.tex"
    report_artifacts_path.write_text(
        "\n\n".join(
            [
                _macro_definition("IsaacSecondPassRuntimeRows", runtime_tex_rows),
                _macro_definition("IsaacSecondPassNominalRows", nominal_tex_rows),
                _macro_definition("IsaacSecondPassDropoutRows", dropout_tex_rows),
                _macro_definition("IsaacSecondPassActuationStressRows", stress_tex_rows),
                _macro_definition("IsaacSecondPassUncertaintyRows", uncertainty_tex_rows),
                _macro_definition("IsaacSecondPassMapQualityRows", map_quality_tex_rows),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    representative = _representative_runs(records, lock_payload=lock_payload)
    nominal_visual = representative.get("nominal_full_anchor", {}).get("visual")
    nominal_fused = representative.get("nominal_full_anchor", {}).get("fused")
    dropout_visual = representative.get("intermittent_anchor", {}).get("visual")
    dropout_fused = representative.get("intermittent_anchor", {}).get("fused")
    stress_visual = representative.get("servo_stress", {}).get("visual")
    stress_fused = representative.get("servo_stress", {}).get("fused")

    if nominal_visual and nominal_fused:
        _compose_two_panel_figure(
            analysis_dir / "nominal_trajectory_compare.png",
            title="Nominal Trajectory Comparison",
            left_label="visual",
            left_path=Path(nominal_visual["run_dir"]) / "analysis" / "report_data" / "trajectory_path.png",
            right_label="fused",
            right_path=Path(nominal_fused["run_dir"]) / "analysis" / "report_data" / "trajectory_path.png",
        )
        _compose_two_panel_figure(
            analysis_dir / "anchor_vs_aux_residuals_nominal.png",
            title="Nominal Residual Quality",
            left_label="visual",
            left_path=Path(nominal_visual["run_dir"]) / "analysis" / "anchor_vs_aux_residuals.png",
            right_label="fused",
            right_path=Path(nominal_fused["run_dir"]) / "analysis" / "anchor_vs_aux_residuals.png",
        )
        _compose_two_panel_figure(
            analysis_dir / "smoother_feedback_compare.png",
            title="Smoother Feedback Comparison",
            left_label="visual",
            left_path=Path(nominal_visual["run_dir"]) / "analysis" / "smoother_correction_timeline.png",
            right_label="fused",
            right_path=Path(nominal_fused["run_dir"]) / "analysis" / "smoother_correction_timeline.png",
        )
    if dropout_visual and dropout_fused:
        _compose_two_panel_figure(
            analysis_dir / "dropout_trajectory_compare.png",
            title="Intermittent-Anchor Trajectory Comparison",
            left_label="visual",
            left_path=Path(dropout_visual["run_dir"]) / "analysis" / "report_data" / "trajectory_path.png",
            right_label="fused",
            right_path=Path(dropout_fused["run_dir"]) / "analysis" / "report_data" / "trajectory_path.png",
        )
        _compose_two_panel_figure(
            analysis_dir / "anchor_vs_aux_residuals_dropout.png",
            title="Intermittent-Anchor Residual Quality",
            left_label="visual",
            left_path=Path(dropout_visual["run_dir"]) / "analysis" / "anchor_vs_aux_residuals.png",
            right_label="fused",
            right_path=Path(dropout_fused["run_dir"]) / "analysis" / "anchor_vs_aux_residuals.png",
        )
    if stress_visual and stress_fused:
        _compose_two_panel_figure(
            analysis_dir / "stress_trajectory_compare.png",
            title="Servo-Stress Trajectory Comparison",
            left_label="visual",
            left_path=Path(stress_visual["run_dir"]) / "analysis" / "report_data" / "trajectory_path.png",
            right_label="fused",
            right_path=Path(stress_fused["run_dir"]) / "analysis" / "report_data" / "trajectory_path.png",
        )
    _save_grouped_summary_figure(
        analysis_dir / "coverage_nees_compare.png",
        title="Coverage and NEES Summary",
        rows=uncertainty_rows,
        left_metric_key="empirical_95_coverage_percent",
        right_metric_key="pose_nees",
        left_label="coverage",
        right_label="NEES",
    )
    _copy_or_render_tuning_heatmap(analysis_dir / "tuning_heatmap_fused.png", root / "latest_second_pass_tuning")

    summary_payload = {
        "artifact_source": artifact_source,
        "suite_dir": str(suite_dir.resolve()),
        "lock_path": str(Path(lock_path).resolve()),
        "run_ids": [str(record["run_id"]) for record in records],
        "draft_selection": dict(lock_payload.get("draft_selection", {})),
        "fused_filter_overrides": second_pass_fused_filter_overrides(lock_payload),
        "runtime_rows": runtime_rows,
        "nominal_rows": nominal_rows,
        "dropout_rows": dropout_rows,
        "stress_rows": stress_rows,
        "uncertainty_rows": uncertainty_rows,
        "map_quality_rows": map_quality_rows,
        "representative_runs": {
            condition: {
                estimator_mode: value["run_id"]
                for estimator_mode, value in estimators.items()
            }
            for condition, estimators in representative.items()
        },
        "paper_artifacts_tex": str(report_artifacts_path.resolve()),
    }
    summary_json = analysis_dir / "suite_summary.json"
    _write_json(summary_json, summary_payload)
    return {
        "suite_dir": str(suite_dir.resolve()),
        "summary_json": str(summary_json.resolve()),
        "paper_artifacts_tex": str(report_artifacts_path.resolve()),
        "artifact_source": artifact_source,
        "run_ids": [str(record["run_id"]) for record in records],
    }


__all__ = [
    "DEFAULT_SECOND_PASS_CONDITIONS",
    "DEFAULT_SECOND_PASS_DRAFT_LOCK",
    "DEFAULT_SECOND_PASS_DRAFT_SEEDS",
    "DEFAULT_SECOND_PASS_FUSED_FILTER_OVERRIDES",
    "generate_second_pass_suite_artifacts",
    "second_pass_fused_filter_overrides",
    "second_pass_draft_run_id",
]
