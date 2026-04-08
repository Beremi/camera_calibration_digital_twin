"""Isaac reporting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from calib_sim.reporting.isaac_report_figures import write_isaac_report_figures
from calib_sim.reporting.isaac_report_metrics import compute_isaac_run_metrics
from calib_sim.reporting.isaac_report_tables import write_isaac_report_tables


def _promote_run_as_latest(run_dir: Path) -> str | None:
    if run_dir.parent.name != "isaac_runs":
        return None
    latest_link = run_dir.parent / "latest"
    try:
        if latest_link.is_symlink() or latest_link.exists():
            if latest_link.resolve() == run_dir:
                return str(latest_link)
            if latest_link.is_dir() and not latest_link.is_symlink():
                return None
            latest_link.unlink()
        latest_link.symlink_to(run_dir.name)
        return str(latest_link)
    except OSError:
        return None


def generate_isaac_report_artifacts(run_dir: str | Path) -> dict[str, Any]:
    """Generate report-ready metrics, tables, and figures for one Isaac run."""

    resolved = Path(run_dir).resolve()
    metrics = compute_isaac_run_metrics(resolved)
    tables = write_isaac_report_tables(resolved, metrics)
    figures = write_isaac_report_figures(resolved, metrics)
    return {
        "metrics": metrics,
        "tables": tables,
        "figures": figures,
        "latest_run_link": _promote_run_as_latest(resolved),
    }


__all__ = [
    "compute_isaac_run_metrics",
    "generate_isaac_report_artifacts",
    "write_isaac_report_figures",
    "write_isaac_report_tables",
]
