"""Smoke test for Isaac lifecycle on an Isaac-enabled machine."""

from __future__ import annotations

import importlib.util

import pytest


pytestmark = pytest.mark.skipif(importlib.util.find_spec("isaacsim") is None, reason="Isaac Sim is not installed in this environment.")


def test_standalone_runtime_can_start_and_step_once() -> None:
    from calib_sim.isaac.runtime.main_loop import IsaacRuntimeConfig, IsaacStandaloneRuntime

    runtime = IsaacStandaloneRuntime(
        IsaacRuntimeConfig(
            stage_path="assets/isaac/anchor_room.usd",
            robot_preset="ur5e_phone_head",
            anchor_tag_id=0,
            headless=True,
        )
    )
    runtime.start()
    try:
        runtime.step(1)
    finally:
        runtime.shutdown()
