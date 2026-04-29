"""Isaac-oriented anchored visual-inertial runtime scaffolding.

This package intentionally avoids importing the Isaac runtime at module import
time so replay, reporting, and schema tools remain usable in environments where
Isaac Sim is installed but should not be bootstrapped for a simple data read.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from calib_sim.isaac.app import IsaacAppBootstrapConfig, create_runtime, load_isaac_yaml
    from calib_sim.isaac.estimation.fixed_lag_smoother import FixedLagSmoother
    from calib_sim.isaac.estimation.online_filter import AnchoredOnlineFilter
    from calib_sim.isaac.logging.run_manifest import IsaacRunManifest, build_run_manifest
    from calib_sim.isaac.logging.schemas import (
        IsaacCameraFramePacket,
        IsaacImuPacket,
        IsaacJointCommandPacket,
        IsaacRealizedJointPacket,
        IsaacTagDetectionPacket,
    )
    from calib_sim.isaac.logging.writer import IsaacRunWriter
    from calib_sim.isaac.runtime.main_loop import IsaacRuntimeConfig, IsaacStandaloneRuntime
    from calib_sim.isaac.runtime.replay import IsaacReplayBundle, load_estimator_input_bundle, load_replay_bundle


_EXPORTS = {
    "AnchoredOnlineFilter": ("calib_sim.isaac.estimation.online_filter", "AnchoredOnlineFilter"),
    "FixedLagSmoother": ("calib_sim.isaac.estimation.fixed_lag_smoother", "FixedLagSmoother"),
    "IsaacAppBootstrapConfig": ("calib_sim.isaac.app", "IsaacAppBootstrapConfig"),
    "IsaacCameraFramePacket": ("calib_sim.isaac.logging.schemas", "IsaacCameraFramePacket"),
    "IsaacImuPacket": ("calib_sim.isaac.logging.schemas", "IsaacImuPacket"),
    "IsaacJointCommandPacket": ("calib_sim.isaac.logging.schemas", "IsaacJointCommandPacket"),
    "IsaacRealizedJointPacket": ("calib_sim.isaac.logging.schemas", "IsaacRealizedJointPacket"),
    "IsaacReplayBundle": ("calib_sim.isaac.runtime.replay", "IsaacReplayBundle"),
    "IsaacRunManifest": ("calib_sim.isaac.logging.run_manifest", "IsaacRunManifest"),
    "IsaacRunWriter": ("calib_sim.isaac.logging.writer", "IsaacRunWriter"),
    "IsaacRuntimeConfig": ("calib_sim.isaac.runtime.main_loop", "IsaacRuntimeConfig"),
    "IsaacStandaloneRuntime": ("calib_sim.isaac.runtime.main_loop", "IsaacStandaloneRuntime"),
    "IsaacTagDetectionPacket": ("calib_sim.isaac.logging.schemas", "IsaacTagDetectionPacket"),
    "build_run_manifest": ("calib_sim.isaac.logging.run_manifest", "build_run_manifest"),
    "create_runtime": ("calib_sim.isaac.app", "create_runtime"),
    "load_estimator_input_bundle": ("calib_sim.isaac.runtime.replay", "load_estimator_input_bundle"),
    "load_isaac_yaml": ("calib_sim.isaac.app", "load_isaac_yaml"),
    "load_replay_bundle": ("calib_sim.isaac.runtime.replay", "load_replay_bundle"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = __import__(module_name, fromlist=[attr_name])
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORTS)
