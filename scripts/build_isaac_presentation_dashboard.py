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
    ]
    videos = dict(presentation_manifest.get("videos", {}))
    runtime_rows = list(suite_summary.get("runtime_rows", []))

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
            <section>
              <h3>{name}</h3>
              <video controls preload="metadata" style="max-width: 100%;">
                <source src="{Path(path).name}" type="video/mp4">
              </video>
            </section>
            """
            for name, path in videos.items()
            if Path(path).exists()
        ]
    )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Isaac Second-Pass Dashboard</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 2rem auto; max-width: 1100px; color: #222; }}
    h1, h2, h3 {{ font-family: 'Trebuchet MS', sans-serif; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }}
    th, td {{ border: 1px solid #ddd; padding: 0.5rem; text-align: left; }}
    th {{ background: #f5f5f5; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1rem; }}
  </style>
</head>
<body>
  <h1>Isaac Second-Pass Presentation Dashboard</h1>
  <p>This dashboard is generated from <code>output/isaac_runs/latest_second_pass_suite</code>. The frozen first-pass publication bundle is intentionally separate.</p>

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

  <h2>Videos</h2>
  {videos_html}

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
