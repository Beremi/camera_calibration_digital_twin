"""Isaac reporting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from calib_sim.reporting.isaac_report_figures import write_isaac_estimator_quality_figures, write_isaac_report_figures
from calib_sim.reporting.isaac_report_metrics import compute_isaac_estimator_quality, compute_isaac_run_metrics
from calib_sim.reporting.isaac_report_tables import write_isaac_report_tables


_COMPLETENESS_THRESHOLDS = {
    "duration_s": 3.0,
    "camera_frames": 60,
    "imu_packets": 400,
    "commands": 30,
    "controller_diagnostics": 30,
    "filter_states": 60,
    "smoother_states": 10,
    "uncertainty_states": 60,
}
_MIN_COMPLETE_REPORT_FIGURES = 6
_DURATION_TOLERANCE_S = 0.01


def _promote_run_link(run_dir: Path, link_name: str) -> str | None:
    if run_dir.parent.name != "isaac_runs":
        return None
    link_path = run_dir.parent / link_name
    try:
        if link_path.is_symlink() or link_path.exists():
            if link_path.resolve() == run_dir:
                return str(link_path)
            if link_path.is_dir() and not link_path.is_symlink():
                return None
            link_path.unlink()
        link_path.symlink_to(run_dir.name)
        return str(link_path)
    except OSError:
        return None


def _is_complete_run(metrics: dict[str, Any], *, figure_count: int | None = None) -> bool:
    counts = metrics.get("counts", {})
    timing = metrics.get("timing", {})
    trajectory = metrics.get("trajectory", {})
    control = metrics.get("control", {})
    if float(timing.get("duration_s", 0.0) or 0.0) + _DURATION_TOLERANCE_S < _COMPLETENESS_THRESHOLDS["duration_s"]:
        return False
    for key, threshold in _COMPLETENESS_THRESHOLDS.items():
        if key == "duration_s":
            continue
        if int(counts.get(key, 0) or 0) < int(threshold):
            return False
    if figure_count is not None and int(figure_count) < int(_MIN_COMPLETE_REPORT_FIGURES):
        return False
    if trajectory.get("mean_position_error_m") is None:
        return False
    if control.get("mean_waypoint_error_m") is None:
        return False
    return True


def generate_isaac_report_artifacts(run_dir: str | Path, *, allow_incomplete: bool = False) -> dict[str, Any]:
    """Generate report-ready metrics, tables, and figures for one Isaac run."""

    resolved = Path(run_dir).resolve()
    metrics = compute_isaac_run_metrics(resolved)
    latest_any_link = _promote_run_link(resolved, "latest_any")
    latest_link = _promote_run_link(resolved, "latest")
    tables = write_isaac_report_tables(resolved, metrics)
    figures = write_isaac_report_figures(resolved, metrics)
    generated_figure_count = sum(1 for path in figures.values() if Path(path).exists())
    complete = _is_complete_run(metrics, figure_count=generated_figure_count)
    if not complete and not allow_incomplete:
        raise ValueError(
            "Isaac run is incomplete and cannot be used for publication artifacts. "
            "Use --allow-incomplete for debugging, or rerun until the completeness thresholds are met."
        )
    latest_complete_link = _promote_run_link(resolved, "latest_complete") if complete else None
    return {
        "metrics": metrics,
        "tables": tables,
        "figures": figures,
        "generated_figure_count": int(generated_figure_count),
        "complete": complete,
        "latest_any_link": latest_any_link,
        "latest_complete_link": latest_complete_link,
        "latest_run_link": latest_link or latest_any_link,
    }


is_complete_run = _is_complete_run


__all__ = [
    "compute_isaac_run_metrics",
    "compute_isaac_estimator_quality",
    "generate_isaac_report_artifacts",
    "is_complete_run",
    "write_isaac_estimator_quality_figures",
    "write_isaac_report_figures",
    "write_isaac_report_tables",
]
