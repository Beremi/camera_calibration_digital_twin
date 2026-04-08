"""Runtime orchestration helpers with lazy imports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from calib_sim.isaac.runtime.main_loop import IsaacRuntimeConfig, IsaacStandaloneRuntime
    from calib_sim.isaac.runtime.replay import IsaacReplayBundle, load_estimator_input_bundle, load_replay_bundle


_EXPORTS = {
    "IsaacReplayBundle": ("calib_sim.isaac.runtime.replay", "IsaacReplayBundle"),
    "IsaacRuntimeConfig": ("calib_sim.isaac.runtime.main_loop", "IsaacRuntimeConfig"),
    "IsaacStandaloneRuntime": ("calib_sim.isaac.runtime.main_loop", "IsaacStandaloneRuntime"),
    "load_estimator_input_bundle": ("calib_sim.isaac.runtime.replay", "load_estimator_input_bundle"),
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
