"""Replay loaders for Isaac runs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from calib_sim.isaac.logging.run_manifest import IsaacRunManifest


class IsaacReplayError(ValueError):
    """Raised when an Isaac replay directory is incomplete."""


@dataclass(slots=True)
class IsaacReplayBundle:
    run_dir: Path
    manifest: IsaacRunManifest
    raw: dict[str, Any]
    estimates: dict[str, Any]
    gt: dict[str, Any] | None = None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _manifest_from_summary(payload: dict[str, Any]) -> IsaacRunManifest:
    return IsaacRunManifest(
        run_id=str(payload["run_id"]),
        git_commit_sha=str(payload["git_commit_sha"]),
        isaac_sim_version=str(payload["isaac_sim_version"]),
        stage_usd_path=str(payload["stage_usd_path"]),
        robot_preset=str(payload["robot_preset"]),
        anchor_tag_id=int(payload["anchor_tag_id"]),
        noise_presets={str(key): str(value) for key, value in dict(payload["noise_presets"]).items()},
        random_seed=int(payload["random_seed"]),
        controller_config=dict(payload["controller_config"]),
        estimator_config=dict(payload["estimator_config"]),
        ros2_bridge_used=bool(payload["ros2_bridge_used"]),
        created_at_utc=str(payload.get("created_at_utc", "")),
    )


def load_estimator_input_bundle(run_dir: str | Path) -> IsaacReplayBundle:
    """Load raw inputs only and intentionally ignore ground-truth artifacts."""

    resolved = Path(run_dir).resolve()
    manifest_path = resolved / "manifest.json"
    if not manifest_path.exists():
        raise IsaacReplayError(f"Missing manifest.json in {resolved}")
    manifest = _manifest_from_summary(_load_json(manifest_path))
    raw = {
        "camera_frames": _load_jsonl(resolved / "raw" / "camera_frames.jsonl"),
        "detections": _load_jsonl(resolved / "raw" / "detections.jsonl"),
        "estimator_input": _load_jsonl(resolved / "raw" / "estimator_input.jsonl"),
        "imu": _load_csv(resolved / "raw" / "imu.csv"),
        "commands": _load_csv(resolved / "raw" / "commands.csv"),
        "realized_joints": _load_csv(resolved / "raw" / "realized_joints.csv"),
    }
    estimates = {
        "filter_state": _load_jsonl(resolved / "estimates" / "filter_state.jsonl"),
        "smoother_state": _load_jsonl(resolved / "estimates" / "smoother_state.jsonl"),
        "uncertainty": _load_jsonl(resolved / "estimates" / "uncertainty.jsonl"),
    }
    return IsaacReplayBundle(run_dir=resolved, manifest=manifest, raw=raw, estimates=estimates, gt=None)


def load_replay_bundle(run_dir: str | Path, *, include_gt: bool = True) -> IsaacReplayBundle:
    bundle = load_estimator_input_bundle(run_dir)
    gt: dict[str, Any] | None = None
    if include_gt:
        resolved = bundle.run_dir
        gt = {
            "camera_gt": _load_csv(resolved / "gt" / "camera_gt.csv"),
            "imu_gt": _load_csv(resolved / "gt" / "imu_gt.csv"),
            "joint_gt": _load_csv(resolved / "gt" / "joint_gt.csv"),
            "tag_gt": _load_json(resolved / "gt" / "tag_gt.json") if (resolved / "gt" / "tag_gt.json").exists() else {},
        }
    return IsaacReplayBundle(run_dir=bundle.run_dir, manifest=bundle.manifest, raw=bundle.raw, estimates=bundle.estimates, gt=gt)
