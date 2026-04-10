"""Control-success metric regressions for second-pass reporting."""

from __future__ import annotations

import pytest

from calib_sim.reporting.isaac_report_metrics import _control_success_summary


def test_control_success_summary_captures_waypoint_hits_dwell_and_dropouts() -> None:
    control_config = {
        "waypoints": [
            {"position_world_m": [0.0, 0.0, 0.0], "tolerance_m": 0.05},
            {"position_world_m": [0.2, 0.0, 0.0], "tolerance_m": 0.05},
        ]
    }
    camera_gt_rows = [
        {"timestamp_s": 0.0, "px": 0.0, "py": 0.0, "pz": 0.0},
        {"timestamp_s": 0.1, "px": 0.01, "py": 0.0, "pz": 0.0},
        {"timestamp_s": 0.2, "px": 0.2, "py": 0.0, "pz": 0.0},
        {"timestamp_s": 0.3, "px": 0.21, "py": 0.0, "pz": 0.0},
    ]
    controller_rows = [
        {
            "timestamp_s": 0.0,
            "waypoint_index": 0,
            "desired_position_world_m": "0.0|0.0|0.0",
            "dropped_command": "False",
        },
        {
            "timestamp_s": 0.1,
            "waypoint_index": 0,
            "desired_position_world_m": "0.0|0.0|0.0",
            "dropped_command": "True",
        },
        {
            "timestamp_s": 0.2,
            "waypoint_index": 1,
            "desired_position_world_m": "0.2|0.0|0.0",
            "dropped_command": "False",
        },
        {
            "timestamp_s": 0.3,
            "waypoint_index": 1,
            "desired_position_world_m": "0.2|0.0|0.0",
            "dropped_command": "False",
        },
    ]
    realized_rows = [
        {"timestamp_s": 0.0, "end_effector_position_world_m": "0.0|0.0|0.0"},
        {"timestamp_s": 0.1, "end_effector_position_world_m": "0.03|0.0|0.0"},
        {"timestamp_s": 0.2, "end_effector_position_world_m": "0.22|0.0|0.0"},
        {"timestamp_s": 0.3, "end_effector_position_world_m": "0.21|0.0|0.0"},
    ]

    summary = _control_success_summary(
        control_config,
        camera_gt_rows,
        controller_rows,
        realized_rows,
        start_time_s=0.0,
        end_time_s=0.3,
    )

    assert summary["waypoint_success_fraction_1cm"] == pytest.approx(0.5)
    assert summary["waypoint_success_fraction_2cm"] == pytest.approx(0.5)
    assert summary["waypoint_success_fraction_5cm"] == pytest.approx(0.5)
    assert summary["mean_waypoint_dwell_time_s"] == pytest.approx(0.05)
    assert summary["p95_waypoint_dwell_time_s"] == pytest.approx(0.095)
    assert summary["final_completion_time_s"] == pytest.approx(0.2)
    assert summary["mean_commanded_realized_path_deviation_m"] == pytest.approx(0.015)
    assert summary["p95_commanded_realized_path_deviation_m"] == pytest.approx(0.0285)
    assert summary["time_integrated_commanded_realized_path_deviation_m_s"] == pytest.approx(0.005)
    assert summary["dropped_command_event_count"] == pytest.approx(1.0)
    assert summary["dropped_command_duration_s"] == pytest.approx(0.1)
