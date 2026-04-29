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
    run_id: str
    run_dir: str
    headless: bool = True
    width: int = 1280
    height: int = 720
    streaming_backend: str = "webrtc"
    webrtc_signal_port: int = 49110
    webrtc_stream_port: int = 47990
    webrtc_target_fps: int = 60
    webrtc_allow_dynamic_resize: bool = True
    webrtc_public_ip: str = ""
    limit_cpu_threads: int | None = None
    disable_viewport_updates: bool | None = None
    seed: int = 7
    duration_s: float = 10.0
    max_steps: int | None = None
    estimator_mode: str = "fused"
    controller_mode: str = "closed-loop"
    bootstrap_control_policy: str = "hold_until_first_detection"
    mode: str | None = None
    promote_latest_complete: bool = False
    allow_gt_debug_control: bool = False

    def as_runtime_config(self) -> IsaacRuntimeConfig:
        config_paths = {
            "scene": self.scene_config_path,
            "robot": self.robot_config_path,
            "camera": self.camera_config_path,
            "imu": self.imu_config_path,
            "actuation": self.actuation_config_path,
            "estimation": self.estimation_config_path,
            "control": self.control_config_path,
        }
        config_payloads = {name: load_isaac_yaml(path) for name, path in config_paths.items()}
        scene_payload = config_payloads["scene"]
        return IsaacRuntimeConfig(
            stage_path=str(scene_payload.get("stage_path", "")),
            robot_preset=str(config_payloads["robot"].get("name", Path(self.robot_config_path).stem)),
            anchor_tag_id=int(scene_payload.get("anchor_tag_id", 0)),
            run_id=self.run_id,
            run_dir=self.run_dir,
            headless=bool(self.headless),
            width=int(self.width),
            height=int(self.height),
            streaming_backend=str(self.streaming_backend),
            webrtc_signal_port=int(self.webrtc_signal_port),
            webrtc_stream_port=int(self.webrtc_stream_port),
            webrtc_target_fps=int(self.webrtc_target_fps),
            webrtc_allow_dynamic_resize=bool(self.webrtc_allow_dynamic_resize),
            webrtc_public_ip=str(self.webrtc_public_ip),
            limit_cpu_threads=None if self.limit_cpu_threads is None else int(self.limit_cpu_threads),
            disable_viewport_updates=None
            if self.disable_viewport_updates is None
            else bool(self.disable_viewport_updates),
            seed=int(self.seed),
            duration_s=float(self.duration_s),
            max_steps=None if self.max_steps is None else int(self.max_steps),
            estimator_mode=str(self.estimator_mode),
            controller_mode=str(self.controller_mode),
            bootstrap_control_policy=str(self.bootstrap_control_policy),
            mode=None if self.mode is None else str(self.mode),
            promote_latest_complete=bool(self.promote_latest_complete),
            allow_gt_debug_control=bool(self.allow_gt_debug_control),
            config_paths=config_paths,
            config_payloads=config_payloads,
        )


def create_runtime(config: IsaacAppBootstrapConfig) -> IsaacStandaloneRuntime:
    """Create a standalone runtime from config paths without starting Isaac."""

    return IsaacStandaloneRuntime(config.as_runtime_config())


__all__ = ["IsaacAppBootstrapConfig", "create_runtime", "load_isaac_yaml"]
