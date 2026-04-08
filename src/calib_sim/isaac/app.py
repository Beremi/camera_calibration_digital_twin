"""Bootstrap helpers for the Isaac runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from calib_sim.isaac.runtime.main_loop import IsaacRuntimeConfig, IsaacStandaloneRuntime


def load_isaac_yaml(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML mapping in {resolved}")
    return raw


@dataclass(slots=True)
class IsaacAppBootstrapConfig:
    """High-level runtime bootstrap options."""

    scene_config_path: str
    robot_config_path: str
    camera_config_path: str
    imu_config_path: str
    actuation_config_path: str
    estimation_config_path: str
    control_config_path: str
    headless: bool = True
    width: int = 1280
    height: int = 720
    use_ros2_bridge: bool = False
    seed: int = 7

    def as_runtime_config(self) -> IsaacRuntimeConfig:
        return IsaacRuntimeConfig(
            stage_path=str(load_isaac_yaml(self.scene_config_path).get("stage_path", "")),
            robot_preset=str(load_isaac_yaml(self.robot_config_path).get("name", Path(self.robot_config_path).stem)),
            anchor_tag_id=int(load_isaac_yaml(self.scene_config_path).get("anchor_tag_id", 0)),
            headless=bool(self.headless),
            width=int(self.width),
            height=int(self.height),
            use_ros2_bridge=bool(self.use_ros2_bridge),
            seed=int(self.seed),
            config_paths={
                "scene": self.scene_config_path,
                "robot": self.robot_config_path,
                "camera": self.camera_config_path,
                "imu": self.imu_config_path,
                "actuation": self.actuation_config_path,
                "estimation": self.estimation_config_path,
                "control": self.control_config_path,
            },
        )


def create_runtime(config: IsaacAppBootstrapConfig) -> IsaacStandaloneRuntime:
    """Create a standalone runtime from config paths without starting Isaac."""

    return IsaacStandaloneRuntime(config.as_runtime_config())
