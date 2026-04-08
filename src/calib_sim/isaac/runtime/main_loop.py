"""Standalone Isaac runtime orchestration.

The class below is intentionally light: it provides a deterministic shell for
future Isaac integration while remaining importable and testable without Isaac.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from calib_sim.sim.runtime import RuntimeConfig, SimulationRuntime


@dataclass(slots=True)
class IsaacRuntimeConfig:
    stage_path: str
    robot_preset: str
    anchor_tag_id: int
    headless: bool = True
    width: int = 1280
    height: int = 720
    use_ros2_bridge: bool = False
    seed: int = 7
    config_paths: dict[str, str] = field(default_factory=dict)

    def runtime_config(self) -> RuntimeConfig:
        return RuntimeConfig(headless=self.headless, width=self.width, height=self.height)

    def summary(self) -> dict[str, Any]:
        return {
            "stage_path": self.stage_path,
            "robot_preset": self.robot_preset,
            "anchor_tag_id": int(self.anchor_tag_id),
            "headless": bool(self.headless),
            "width": int(self.width),
            "height": int(self.height),
            "use_ros2_bridge": bool(self.use_ros2_bridge),
            "seed": int(self.seed),
            "config_paths": dict(self.config_paths),
        }


class IsaacStandaloneRuntime:
    """Minimal lifecycle wrapper around the soft-import SimulationRuntime."""

    def __init__(self, config: IsaacRuntimeConfig) -> None:
        self.config = config
        self._runtime = SimulationRuntime(config.runtime_config())
        self._step_hooks: list[Callable[[int], None]] = []
        self.started = False
        self.steps_executed = 0

    def register_step_hook(self, callback: Callable[[int], None]) -> None:
        self._step_hooks.append(callback)

    def start(self) -> None:
        self._runtime.start()
        self.started = True

    def step(self, steps: int = 1) -> None:
        if not self.started:
            raise RuntimeError("Isaac runtime has not been started.")
        for _ in range(max(int(steps), 0)):
            self._runtime.step()
            for callback in self._step_hooks:
                callback(self.steps_executed)
            self.steps_executed += 1

    def shutdown(self) -> None:
        self._runtime.shutdown()
        self.started = False

    def validate_stage_path(self) -> Path:
        if not self.config.stage_path:
            raise ValueError("stage_path is required.")
        return Path(self.config.stage_path)
