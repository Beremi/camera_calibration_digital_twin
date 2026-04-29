"""Isaac demo profile catalog for the slim tabletop-only branch."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]


def _repo_path(path_like: str | Path) -> Path:
    candidate = Path(path_like)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def _load_yaml_payload(path_like: str | Path) -> dict[str, Any]:
    resolved = _repo_path(path_like)
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {resolved}")
    return payload


@dataclass(frozen=True, slots=True)
class IsaacDemoProfile:
    key: str
    label: str
    description: str
    scene_config_path: str
    robot_config_path: str
    camera_config_path: str
    imu_config_path: str
    actuation_config_path: str
    estimation_config_path: str
    control_config_path: str
    default_control_mode: str
    auto_demo_behavior: str
    allow_auto_path: bool = False
    allow_manual_task: bool = True
    default_estimator_mode: str = "fused"
    default_runtime_controller_mode: str = "closed-loop"
    bootstrap_control_policy: str = "hold_until_first_detection"
    workspace_bounds_world_m: dict[str, tuple[float, float, float]] | None = None
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "description": self.description,
            "scene_config_path": self.scene_config_path,
            "robot_config_path": self.robot_config_path,
            "camera_config_path": self.camera_config_path,
            "imu_config_path": self.imu_config_path,
            "actuation_config_path": self.actuation_config_path,
            "estimation_config_path": self.estimation_config_path,
            "control_config_path": self.control_config_path,
            "default_control_mode": self.default_control_mode,
            "auto_demo_behavior": self.auto_demo_behavior,
            "allow_auto_path": bool(self.allow_auto_path),
            "allow_manual_task": bool(self.allow_manual_task),
            "default_estimator_mode": self.default_estimator_mode,
            "default_runtime_controller_mode": self.default_runtime_controller_mode,
            "bootstrap_control_policy": self.bootstrap_control_policy,
            "workspace_bounds_world_m": None
            if self.workspace_bounds_world_m is None
            else {
                "min": [float(value) for value in self.workspace_bounds_world_m["min"]],
                "max": [float(value) for value in self.workspace_bounds_world_m["max"]],
            },
            "extra_metadata": dict(self.extra_metadata),
        }


def _workspace_bounds_from_control(control_path: str) -> dict[str, tuple[float, float, float]] | None:
    payload = _load_yaml_payload(control_path)
    raw = payload.get("workspace_aabb_world_m")
    if not isinstance(raw, dict):
        return None
    lower = raw.get("min")
    upper = raw.get("max")
    if not isinstance(lower, (list, tuple)) or not isinstance(upper, (list, tuple)):
        return None
    if len(lower) < 3 or len(upper) < 3:
        return None
    return {
        "min": (float(lower[0]), float(lower[1]), float(lower[2])),
        "max": (float(upper[0]), float(upper[1]), float(upper[2])),
    }


_TABLETOP_CONTROL = "config/isaac/control/tabletop_demo.yaml"
_TABLETOP_PROFILE = IsaacDemoProfile(
    key="tabletop_replica",
    label="Tabletop Replica",
    description="The supported tabletop auto-demo scene with mounted and observer camera views.",
    scene_config_path="config/isaac/scene/tabletop_grab_challenge_demo.yaml",
    robot_config_path="config/isaac/robot/franka_tabletop_grab_challenge.yaml",
    camera_config_path="config/isaac/camera/phone_tabletop_demo.yaml",
    imu_config_path="config/isaac/imu/phone_nominal.yaml",
    actuation_config_path="config/isaac/actuation/servo_nominal.yaml",
    estimation_config_path="config/isaac/estimation/tabletop_demo_vio.yaml",
    control_config_path=_TABLETOP_CONTROL,
    default_control_mode="auto_demo",
    auto_demo_behavior="motion_program",
    allow_auto_path=False,
    workspace_bounds_world_m=_workspace_bounds_from_control(_TABLETOP_CONTROL),
    extra_metadata={"scene_family": "tabletop_replica"},
)

DEMO_PROFILES = {_TABLETOP_PROFILE.key: _TABLETOP_PROFILE}
DEFAULT_PROFILE_KEY = _TABLETOP_PROFILE.key


def list_demo_profiles() -> list[dict[str, Any]]:
    return [_TABLETOP_PROFILE.summary()]


def get_demo_profile(profile_key: str | None) -> IsaacDemoProfile:
    key = DEFAULT_PROFILE_KEY if profile_key in (None, "") else str(profile_key)
    if key != DEFAULT_PROFILE_KEY:
        raise KeyError(f"Unknown Isaac demo profile: {key}")
    return _TABLETOP_PROFILE


__all__ = [
    "DEFAULT_PROFILE_KEY",
    "DEMO_PROFILES",
    "IsaacDemoProfile",
    "get_demo_profile",
    "list_demo_profiles",
]
