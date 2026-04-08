"""Smoke test for Isaac lifecycle on an Isaac-enabled machine."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(importlib.util.find_spec("isaacsim") is None, reason="Isaac Sim is not installed in this environment.")


def test_standalone_runtime_can_start_and_step_once(tmp_path: Path) -> None:
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

    from calib_sim.isaac.app import IsaacAppBootstrapConfig, create_runtime

    runtime = create_runtime(
        IsaacAppBootstrapConfig(
            scene_config_path="config/isaac/scene/anchor_room.yaml",
            robot_config_path="config/isaac/robot/franka_phone_head.yaml",
            camera_config_path="config/isaac/camera/phone_main.yaml",
            imu_config_path="config/isaac/imu/phone_nominal.yaml",
            actuation_config_path="config/isaac/actuation/servo_nominal.yaml",
            estimation_config_path="config/isaac/estimation/anchored_vio.yaml",
            control_config_path="config/isaac/control/path_tracking.yaml",
            run_id="test_runtime_smoke",
            run_dir=str((tmp_path / "isaac_runs" / "test_runtime_smoke").resolve()),
            headless=True,
            duration_s=0.1,
            max_steps=5,
        )
    )
    runtime.start()
    try:
        runtime.step(1)
    finally:
        runtime.shutdown()

    run_dir = tmp_path / "isaac_runs" / "test_runtime_smoke"
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "raw").exists()
    assert any(path.is_file() for path in (run_dir / "raw").rglob("*"))
