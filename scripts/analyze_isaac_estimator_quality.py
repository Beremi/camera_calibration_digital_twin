#!/usr/bin/env python3
"""Analyze estimator-quality diagnostics for one Isaac run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from calib_sim.reporting import compute_isaac_estimator_quality, write_isaac_estimator_quality_figures


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the second-pass estimator-quality bundle for an Isaac run."
    )
    parser.add_argument("run_dir")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    quality = compute_isaac_estimator_quality(run_dir)
    figure_paths = write_isaac_estimator_quality_figures(run_dir, quality)

    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    summary_row = {
        "run_id": quality["run_id"],
        **dict(quality.get("summary", {})),
        "mean_auxiliary_tag_position_error_m": quality.get("map_quality", {}).get("mean_auxiliary_tag_position_error_m"),
        "p95_auxiliary_tag_position_error_m": quality.get("map_quality", {}).get("p95_auxiliary_tag_position_error_m"),
        "empirical_95_coverage_percent": quality.get("uncertainty", {}).get("empirical_95_coverage_percent"),
        "pose_nees": quality.get("uncertainty", {}).get("pose_nees"),
    }
    payload = {
        **quality,
        "figure_paths": figure_paths,
    }
    (analysis_dir / "estimator_quality.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(analysis_dir / "estimator_quality.csv", [summary_row])
    _write_csv(analysis_dir / "tag_update_breakdown.csv", list(quality.get("tag_update_breakdown", [])))

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
