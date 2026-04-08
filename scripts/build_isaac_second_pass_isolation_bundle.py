#!/usr/bin/env python3
"""Aggregate the canonical second-pass isolation runs into one diagnostic bundle."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_RUN_IDS = [
    "second_pass_visual_closed_loop_anchor_only_seed_007",
    "second_pass_fused_closed_loop_anchor_only_seed_007",
    "second_pass_visual_closed_loop_aux_estimation_only_seed_007",
    "second_pass_fused_closed_loop_aux_estimation_only_seed_007",
    "second_pass_visual_closed_loop_aux_for_control_seed_007",
    "second_pass_fused_closed_loop_aux_for_control_seed_007",
]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _float_or_none(value: Any) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def _load_run_summary(run_dir: Path) -> dict[str, Any]:
    quality = _load_json(run_dir / "analysis" / "estimator_quality.json")
    metrics = _load_json(run_dir / "analysis" / "metrics.json")
    estimation_config = _load_json(run_dir / "config_snapshot" / "estimation.json")
    control_config = _load_json(run_dir / "config_snapshot" / "control.json")
    summary = dict(quality.get("summary", {}))
    detection_rows = list(quality.get("detection_residuals", []))
    native_backend_count = sum(1 for row in detection_rows if bool(row.get("native_backend", False)))
    native_pnp_fraction = None
    if detection_rows:
        native_pnp_fraction = float(native_backend_count / len(detection_rows))
    accepted_aux_updates = int(summary.get("accepted_auxiliary_updates", 0) or 0)
    rejected_aux_updates = int(summary.get("rejected_auxiliary_updates", 0) or 0)
    aux_total = accepted_aux_updates + rejected_aux_updates
    row = {
        "run_id": str(metrics.get("run_id", run_dir.name)),
        "estimator_mode": str(metrics.get("config", {}).get("estimator_mode", "")),
        "controller_mode": str(metrics.get("config", {}).get("controller_mode", "")),
        "use_aux_tags_in_filter": bool(estimation_config.get("filter", {}).get("use_aux_tags_in_filter", True)),
        "use_aux_tags_in_smoother": bool(estimation_config.get("smoother", {}).get("use_aux_tags_in_smoother", True)),
        "use_aux_map_for_control": bool(control_config.get("use_aux_map_for_control", True)),
        "mean_position_error_m": _float_or_none(metrics.get("trajectory", {}).get("mean_position_error_m")),
        "mean_waypoint_error_m": _float_or_none(metrics.get("control", {}).get("mean_waypoint_error_m")),
        "anchor_rmse_px": _float_or_none(summary.get("anchor_mean_reprojection_rmse_px")),
        "aux_rmse_px": _float_or_none(summary.get("auxiliary_mean_reprojection_rmse_px")),
        "native_pnp_fraction": native_pnp_fraction,
        "fallback_only_fraction": _float_or_none(metrics.get("estimation", {}).get("fallback_only_frame_fraction")),
        "aux_update_accept_fraction": None if aux_total <= 0 else float(accepted_aux_updates / aux_total),
        "mean_smoother_feedback_norm_m": _float_or_none(summary.get("mean_smoother_correction_norm_m")),
        "empirical_95_coverage_percent": _float_or_none(quality.get("uncertainty", {}).get("empirical_95_coverage_percent")),
        "pose_nees": _float_or_none(quality.get("uncertainty", {}).get("pose_nees")),
        "mean_auxiliary_tag_position_error_m": _float_or_none(
            quality.get("map_quality", {}).get("mean_auxiliary_tag_position_error_m")
        ),
        "anchor_pnp_success_fraction": _float_or_none(metrics.get("estimation", {}).get("anchor_pnp_success_fraction")),
        "mean_anchor_innovation_norm": _float_or_none(metrics.get("estimation", {}).get("mean_anchor_innovation_norm")),
        "smoother_backend": str(estimation_config.get("smoother", {}).get("backend", "lightweight")),
    }
    return {
        "row": row,
        "quality": quality,
        "metrics": metrics,
    }


def _anchor_only(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if not bool(row["use_aux_tags_in_filter"])
        and not bool(row["use_aux_tags_in_smoother"])
        and not bool(row["use_aux_map_for_control"])
    ]


def _aux_enabled(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if bool(row["use_aux_tags_in_filter"]) and bool(row["use_aux_tags_in_smoother"])]


def _classify_root_cause(rows: list[dict[str, Any]]) -> dict[str, Any]:
    anchor_only_rows = _anchor_only(rows)
    if anchor_only_rows and all(
        _float_or_none(row.get("anchor_rmse_px")) is not None
        and float(row["anchor_rmse_px"]) > 100.0
        and _float_or_none(row.get("anchor_pnp_success_fraction")) is not None
        and float(row["anchor_pnp_success_fraction"]) >= 0.99
        and _float_or_none(row.get("mean_anchor_innovation_norm")) is not None
        and float(row["mean_anchor_innovation_norm"]) < 1.0
        for row in anchor_only_rows
    ):
        return {
            "classification": "metric_projection_bug",
            "reason": "Anchor-only runs still report extreme reprojection RMSE despite near-perfect anchor PnP and modest anchor innovation.",
        }

    aux_rows = _aux_enabled(rows)
    if anchor_only_rows and aux_rows:
        anchor_aux_errors = [
            _float_or_none(row.get("mean_auxiliary_tag_position_error_m"))
            for row in anchor_only_rows
            if _float_or_none(row.get("mean_auxiliary_tag_position_error_m")) is not None
        ]
        enabled_aux_errors = [
            _float_or_none(row.get("mean_auxiliary_tag_position_error_m"))
            for row in aux_rows
            if _float_or_none(row.get("mean_auxiliary_tag_position_error_m")) is not None
        ]
        anchor_aux_rmse = [
            _float_or_none(row.get("aux_rmse_px"))
            for row in anchor_only_rows
            if _float_or_none(row.get("aux_rmse_px")) is not None
        ]
        enabled_aux_rmse = [
            _float_or_none(row.get("aux_rmse_px"))
            for row in aux_rows
            if _float_or_none(row.get("aux_rmse_px")) is not None
        ]
        if anchor_aux_errors and enabled_aux_errors:
            anchor_error = min(anchor_aux_errors)
            enabled_error = max(enabled_aux_errors)
            if anchor_error > 0.0 and enabled_error / anchor_error > 5.0:
                return {
                    "classification": "auxiliary_tag_path_bug",
                    "reason": "Aux-enabled runs inflate auxiliary map error by more than 5x relative to anchor-only runs.",
                }
        if anchor_aux_rmse and enabled_aux_rmse:
            anchor_rmse = min(anchor_aux_rmse)
            enabled_rmse = max(enabled_aux_rmse)
            if anchor_rmse > 0.0 and enabled_rmse / anchor_rmse > 5.0:
                return {
                    "classification": "auxiliary_tag_path_bug",
                    "reason": "Aux-enabled runs inflate auxiliary reprojection RMSE by more than 5x relative to anchor-only runs.",
                }

    visual_anchor_only = next((row for row in anchor_only_rows if str(row["estimator_mode"]) == "visual"), None)
    fused_anchor_only = next((row for row in anchor_only_rows if str(row["estimator_mode"]) == "fused"), None)
    if visual_anchor_only is not None and fused_anchor_only is not None:
        visual_pos = _float_or_none(visual_anchor_only.get("mean_position_error_m"))
        fused_pos = _float_or_none(fused_anchor_only.get("mean_position_error_m"))
        visual_nees = _float_or_none(visual_anchor_only.get("pose_nees"))
        fused_nees = _float_or_none(fused_anchor_only.get("pose_nees"))
        if visual_pos not in (None, 0.0) and fused_pos is not None and fused_pos > visual_pos * 1.10:
            return {
                "classification": "fused_mechanization_or_weighting_bug",
                "reason": "Fused anchor-only is more than 10% worse than visual anchor-only on mean position error.",
            }
        if visual_nees not in (None, 0.0) and fused_nees is not None and fused_nees > visual_nees * 2.0:
            return {
                "classification": "fused_mechanization_or_weighting_bug",
                "reason": "Fused anchor-only is more than 2x worse than visual anchor-only on pose NEES.",
            }

    if any(
        _float_or_none(row.get("mean_smoother_feedback_norm_m")) is not None
        and float(row["mean_smoother_feedback_norm_m"]) > 0.15
        for row in rows
    ):
        return {
            "classification": "smoother_feedback_corruption",
            "reason": "Mean smoother feedback correction exceeds the 0.15 m trust threshold.",
        }

    return {
        "classification": "inconclusive",
        "reason": "The six-run isolation bundle did not trigger one of the explicit second-pass root-cause rules.",
    }


def _markdown_summary(rows: list[dict[str, Any]], classification: dict[str, Any]) -> str:
    lines = [
        "# Isaac Second-Pass Isolation Summary",
        "",
        "This note is generated from the six canonical `second_pass_*` isolation runs.",
        "",
        f"- Root-cause classification: `{classification['classification']}`",
        f"- Decision note: {classification['reason']}",
        "",
        "| Run ID | Estimator | Aux Filter | Aux Smoother | Aux Control | Pos Err [m] | Waypoint Err [m] | Anchor RMSE [px] | Aux RMSE [px] | Coverage [%] | NEES | Aux Map Err [m] |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["run_id"]),
                    str(row["estimator_mode"]),
                    "on" if bool(row["use_aux_tags_in_filter"]) else "off",
                    "on" if bool(row["use_aux_tags_in_smoother"]) else "off",
                    "on" if bool(row["use_aux_map_for_control"]) else "off",
                    f"{float(row['mean_position_error_m']):.4f}" if row["mean_position_error_m"] is not None else "n/a",
                    f"{float(row['mean_waypoint_error_m']):.4f}" if row["mean_waypoint_error_m"] is not None else "n/a",
                    f"{float(row['anchor_rmse_px']):.2f}" if row["anchor_rmse_px"] is not None else "n/a",
                    f"{float(row['aux_rmse_px']):.2f}" if row["aux_rmse_px"] is not None else "n/a",
                    f"{float(row['empirical_95_coverage_percent']):.2f}"
                    if row["empirical_95_coverage_percent"] is not None
                    else "n/a",
                    f"{float(row['pose_nees']):.2f}" if row["pose_nees"] is not None else "n/a",
                    f"{float(row['mean_auxiliary_tag_position_error_m']):.4f}"
                    if row["mean_auxiliary_tag_position_error_m"] is not None
                    else "n/a",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "The frozen first-pass publication bundle under `output/isaac_runs/latest_first_pass_suite` was not modified by this second-pass isolation workflow.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate the canonical second-pass isolation runs.")
    parser.add_argument("--output-root", default="output/isaac_runs")
    parser.add_argument("--docs-path", default="docs/isaac_second_pass_isolation_summary.md")
    parser.add_argument("--run-ids", nargs="*", default=DEFAULT_RUN_IDS)
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    docs_path = Path(args.docs_path).resolve()
    bundle_dir = output_root / "latest_second_pass_isolation"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for run_id in args.run_ids:
        run_dir = output_root / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Missing second-pass run directory: {run_dir}")
        payload = _load_run_summary(run_dir)
        rows.append(payload["row"])

    classification = _classify_root_cause(rows)
    summary_payload = {
        "run_ids": [str(row["run_id"]) for row in rows],
        "classification": classification,
        "rows": rows,
    }
    (bundle_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(bundle_dir / "summary.csv", rows)
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(_markdown_summary(rows, classification) + "\n", encoding="utf-8")

    print(json.dumps({"bundle_dir": str(bundle_dir), **summary_payload}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
