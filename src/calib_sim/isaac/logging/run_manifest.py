"""Run-manifest helpers for Isaac outputs."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import subprocess
from typing import Any


def _git_sha(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


@dataclass(slots=True)
class IsaacRunManifest:
    run_id: str
    git_commit_sha: str
    isaac_sim_version: str
    stage_usd_path: str
    robot_preset: str
    anchor_tag_id: int
    estimator_mode: str
    controller_mode: str
    bootstrap_control_policy: str
    noise_presets: dict[str, str]
    random_seed: int
    controller_config: dict[str, Any]
    estimator_config: dict[str, Any]
    ros2_bridge_used: bool
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "git_commit_sha": self.git_commit_sha,
            "isaac_sim_version": self.isaac_sim_version,
            "stage_usd_path": self.stage_usd_path,
            "robot_preset": self.robot_preset,
            "anchor_tag_id": int(self.anchor_tag_id),
            "estimator_mode": self.estimator_mode,
            "controller_mode": self.controller_mode,
            "bootstrap_control_policy": self.bootstrap_control_policy,
            "noise_presets": copy.deepcopy(self.noise_presets),
            "random_seed": int(self.random_seed),
            "controller_config": copy.deepcopy(self.controller_config),
            "estimator_config": copy.deepcopy(self.estimator_config),
            "ros2_bridge_used": bool(self.ros2_bridge_used),
            "created_at_utc": self.created_at_utc,
        }


def build_run_manifest(
    *,
    repo_root: str | Path,
    run_id: str,
    isaac_sim_version: str,
    stage_usd_path: str,
    robot_preset: str,
    anchor_tag_id: int,
    estimator_mode: str,
    controller_mode: str,
    bootstrap_control_policy: str,
    noise_presets: dict[str, str],
    random_seed: int,
    controller_config: dict[str, Any],
    estimator_config: dict[str, Any],
    ros2_bridge_used: bool = False,
) -> IsaacRunManifest:
    resolved_root = Path(repo_root).resolve()
    return IsaacRunManifest(
        run_id=run_id,
        git_commit_sha=_git_sha(resolved_root),
        isaac_sim_version=isaac_sim_version,
        stage_usd_path=stage_usd_path,
        robot_preset=robot_preset,
        anchor_tag_id=int(anchor_tag_id),
        estimator_mode=str(estimator_mode),
        controller_mode=str(controller_mode),
        bootstrap_control_policy=str(bootstrap_control_policy),
        noise_presets=copy.deepcopy(noise_presets),
        random_seed=int(random_seed),
        controller_config=copy.deepcopy(controller_config),
        estimator_config=copy.deepcopy(estimator_config),
        ros2_bridge_used=bool(ros2_bridge_used),
    )
