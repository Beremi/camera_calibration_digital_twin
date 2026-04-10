#!/usr/bin/env python3
"""Build a static HTML dashboard for the local second-pass media bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="output/isaac_runs")
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_dashboard(output_root: Path) -> Path:
    suite_dir = output_root / "latest_second_pass_suite"
    analysis_dir = suite_dir / "analysis"
    presentation_dir = suite_dir / "presentation"
    suite_summary = _load_json(analysis_dir / "suite_summary.json")
    presentation_manifest = _load_json(presentation_dir / "presentation_manifest.json")

    figures = [
        "nominal_trajectory_compare.png",
        "dropout_trajectory_compare.png",
        "stress_trajectory_compare.png",
        "coverage_nees_compare.png",
        "anchor_vs_aux_residuals_nominal.png",
        "anchor_vs_aux_residuals_dropout.png",
        "smoother_feedback_compare.png",
        "tuning_heatmap_fused.png",
        "representative_nominal_first_frame.png",
        "representative_nominal_sim_waypoints.png",
        "representative_nominal_position_estimation.png",
        "representative_nominal_pattern_world_positions.png",
        "representative_nominal_imu_measurements.png",
        "representative_nominal_pattern_relative_camera_positions.png",
    ]
    videos = dict(presentation_manifest.get("videos", {}))
    captions = dict(presentation_manifest.get("captions", {}))
    stills = dict(presentation_manifest.get("stills", {}))
    runtime_rows = list(suite_summary.get("runtime_rows", []))
    control_success_rows = list(suite_summary.get("control_success_rows", []))
    representative_runs = dict(suite_summary.get("representative_runs", {}))

    def condition_row(condition: str, estimator: str) -> dict:
        return next(
            (
                row
                for row in runtime_rows
                if str(row.get("condition")) == condition and str(row.get("estimator_mode")) == estimator
            ),
            {},
        )

    def finding_box(condition: str, title: str) -> str:
        visual = condition_row(condition, "visual")
        fused = condition_row(condition, "fused")
        if not visual or not fused:
            return f"<section class='finding'><h3>{title}</h3><p>Missing suite rows for this condition.</p></section>"
        visual_waypoint = visual.get("mean_waypoint_error_m")
        fused_waypoint = fused.get("mean_waypoint_error_m")
        visual_position = visual.get("mean_position_error_m")
        fused_position = fused.get("mean_position_error_m")
        visual_coverage = visual.get("empirical_95_coverage_percent")
        fused_coverage = fused.get("empirical_95_coverage_percent")
        if condition == "nominal_full_anchor":
            takeaway = "Fused is competitive on mean position error, while visual remains the stronger waypoint baseline."
        elif condition == "intermittent_anchor":
            takeaway = "Fused no longer diverges catastrophically under scheduled anchor suppression, but visual still leads on the current error metrics."
        elif condition == "servo_stress":
            takeaway = "Fused remains stable and well calibrated under servo stress, but visual still retains the waypoint advantage."
        elif fused_waypoint is not None and visual_waypoint is not None and fused_waypoint < visual_waypoint:
            takeaway = "Fused improves waypoint tracking on this condition."
        elif fused_position is not None and visual_position is not None and fused_position < visual_position:
            takeaway = "Fused improves state accuracy on this condition."
        else:
            takeaway = "Visual remains competitive or better on this condition."
        detail = (
            f"visual pos={visual_position:.3f} m, fused pos={fused_position:.3f} m; "
            f"visual way={visual_waypoint:.3f} m, fused way={fused_waypoint:.3f} m; "
            f"visual cov={visual_coverage:.1f}%, fused cov={fused_coverage:.1f}%"
            if None not in (visual_position, fused_position, visual_waypoint, fused_waypoint, visual_coverage, fused_coverage)
            else "One or more summary fields are missing."
        )
        reps = representative_runs.get(condition, {})
        rep_line = ", ".join(f"{mode}: {run_id}" for mode, run_id in reps.items())
        return (
            f"<section class='finding'><h3>{title}</h3>"
            f"<p><strong>{takeaway}</strong> {detail}</p>"
            f"<p class='muted'>Representative runs: {rep_line}</p></section>"
        )

    remaining_weakness = "<p>The second-pass suite now supports a stabilization story rather than a broad fused win. The remaining weakness is practical: visual still leads on waypoint error, and intermittent-anchor fused, while no longer catastrophic, does not yet outperform visual on the current error metrics.</p>"

    control_rows_html = "\n".join(
        [
            "<tr>"
            + "".join(
                f"<td>{value}</td>"
                for value in (
                    row.get("condition_label", ""),
                    row.get("estimator_mode", ""),
                    f"{float(row['waypoint_success_fraction_1cm']) * 100.0:.1f}" if row.get("waypoint_success_fraction_1cm") is not None else "",
                    f"{float(row['waypoint_success_fraction_2cm']) * 100.0:.1f}" if row.get("waypoint_success_fraction_2cm") is not None else "",
                    f"{float(row['waypoint_success_fraction_5cm']) * 100.0:.1f}" if row.get("waypoint_success_fraction_5cm") is not None else "",
                    f"{float(row['mean_commanded_realized_path_deviation_m']):.3f}" if row.get("mean_commanded_realized_path_deviation_m") is not None else "",
                    f"{float(row['dropped_command_duration_s']):.3f}" if row.get("dropped_command_duration_s") is not None else "",
                )
            )
            + "</tr>"
            for row in control_success_rows
        ]
    )

    rows_html = "\n".join(
        [
            "<tr>"
            + "".join(
                f"<td>{row.get(key, '')}</td>"
                for key in (
                    "condition_label",
                    "estimator_mode",
                    "sample_count",
                    "mean_position_error_m",
                    "mean_waypoint_error_m",
                    "empirical_95_coverage_percent",
                    "pose_nees",
                )
            )
            + "</tr>"
            for row in runtime_rows
        ]
    )
    figures_html = "\n".join(
        [
            f'<figure><img src="../analysis/{name}" alt="{name}" style="max-width: 100%; border: 1px solid #ddd;"><figcaption>{name}</figcaption></figure>'
            for name in figures
            if (analysis_dir / name).exists()
        ]
    )
    videos_html = "\n".join(
        [
            f"""
            <section class="video-card">
              <h3>{name}</h3>
              <p class="muted">{captions.get(name, '')}</p>
              <video controls preload="metadata" style="max-width: 100%;">
                <source src="{Path(path).name}" type="video/mp4">
              </video>
            </section>
            """
            for name, path in videos.items()
            if Path(path).exists()
        ]
    )
    stills_html = "\n".join(
        [
            f"""
            <figure class="still-card">
              <img src="{Path(path).name}" alt="{name}" style="max-width: 100%; border: 1px solid #d4d9df;">
              <figcaption>{name.replace('_', ' ')}</figcaption>
            </figure>
            """
            for name, path in stills.items()
            if Path(path).exists()
        ]
    )
    table_links = "\n".join(
        [
            f'<li><a href="../analysis/{name}">{name}</a></li>'
            for name in (
                "draft_runtime_table.csv",
                "draft_nominal_table.csv",
                "draft_dropout_table.csv",
                "draft_actuation_stress_table.csv",
                "draft_uncertainty_table.csv",
                "draft_map_quality_table.csv",
                "control_success_summary.csv",
                "suite_summary.json",
            )
            if (analysis_dir / name).exists()
        ]
    )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Isaac Second-Pass Dashboard</title>
  <style>
    :root {{ color-scheme: light; --ink:#18212c; --muted:#58636f; --line:#dbe1e8; --panel:#f7f8fa; --accent:#154c79; --accent2:#8f4c21; }}
    body {{ font-family: Georgia, serif; margin: 2rem auto 4rem; max-width: 1180px; color: var(--ink); background: linear-gradient(180deg, #fbfcfd, #f3f6f9); }}
    h1, h2, h3 {{ font-family: 'Trebuchet MS', sans-serif; }}
    h1 {{ margin-bottom: 0.4rem; }}
    p {{ line-height: 1.5; }}
    .muted {{ color: var(--muted); }}
    .hero {{ background: white; border: 1px solid var(--line); padding: 1.25rem 1.5rem; border-radius: 14px; box-shadow: 0 8px 24px rgba(0,0,0,0.04); }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1rem; }}
    .grid-2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1rem; }}
    .finding, .video-card, .still-card, .link-panel {{ background: white; border: 1px solid var(--line); border-radius: 12px; padding: 1rem; box-shadow: 0 6px 18px rgba(0,0,0,0.03); }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; background: white; border: 1px solid var(--line); }}
    th, td {{ border: 1px solid var(--line); padding: 0.55rem; text-align: left; }}
    th {{ background: #eef3f7; }}
    ul {{ margin-top: 0.5rem; }}
    figure {{ margin: 0; }}
    img {{ display: block; }}
  </style>
</head>
<body>
  <section class="hero">
    <h1>Isaac Second-Pass Presentation Dashboard</h1>
    <p>This dashboard is generated from <code>output/isaac_runs/latest_second_pass_suite</code> on <code>feature/isaac-estimator-second-pass</code>. The first-pass publication bundle remains frozen and separate. This page is intended to make the second-pass draft readable without opening raw JSON or stitching figures by hand.</p>
  </section>

  <h2>Key Findings</h2>
  <div class="grid-2">
    {finding_box("nominal_full_anchor", "Nominal / full anchor")}
    {finding_box("intermittent_anchor", "Intermittent anchor")}
    {finding_box("servo_stress", "Servo stress")}
    <section class="finding"><h3>Remaining weakness</h3>{remaining_weakness}</section>
  </div>

  <h2>Control Success</h2>
  <p class="muted">These control-facing metrics complement the estimator tables by asking whether the camera actually reaches the zig-zag waypoints within tighter spatial tolerances and how closely the realized end-effector path follows the commanded anchored path.</p>
  <table>
    <thead>
      <tr>
        <th>Condition</th>
        <th>Estimator</th>
        <th>1 cm success [%]</th>
        <th>2 cm success [%]</th>
        <th>5 cm success [%]</th>
        <th>Mean EE path dev [m]</th>
        <th>Dropped cmd dur [s]</th>
      </tr>
    </thead>
    <tbody>
      {control_rows_html}
    </tbody>
  </table>

  <h2>Hero Stills</h2>
  <div class="grid">
    {stills_html}
  </div>

  <h2>Suite Summary</h2>
  <table>
    <thead>
      <tr>
        <th>Condition</th>
        <th>Estimator</th>
        <th>Samples</th>
        <th>Mean position error [m]</th>
        <th>Mean waypoint error [m]</th>
        <th>Coverage [%]</th>
        <th>NEES</th>
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>

  <div class="grid-2">
    <section class="link-panel">
      <h3>Artifact Tables</h3>
      <ul>
        {table_links}
      </ul>
    </section>
    <section class="link-panel">
      <h3>What To Look For</h3>
      <p class="muted">Read the nominal condition first, where fused is competitive on mean position error and well calibrated. Then use the intermittent-anchor and servo-stress rows to see that the second-pass result is a repaired, defensible fused baseline rather than a universal fused advantage.</p>
    </section>
  </div>

  <h2>Videos</h2>
  <div class="grid">
    {videos_html}
  </div>

  <h2>Figures</h2>
  <div class="grid">
    {figures_html}
  </div>
</body>
</html>
"""
    dashboard_path = presentation_dir / "index.html"
    dashboard_path.write_text(html, encoding="utf-8")
    return dashboard_path


def main() -> int:
    args = parse_args()
    dashboard_path = build_dashboard(Path(args.output_root).resolve())
    print(json.dumps({"dashboard_path": str(dashboard_path.resolve())}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
