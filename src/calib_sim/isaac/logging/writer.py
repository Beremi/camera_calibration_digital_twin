"""Artifact writer for Isaac runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from calib_sim.isaac.estimation.state_defs import FilterStateSnapshot, SmootherStateSnapshot, UncertaintySnapshot
from calib_sim.isaac.logging.run_manifest import IsaacRunManifest
from calib_sim.isaac.logging.schemas import (
    IsaacCameraFramePacket,
    IsaacImuPacket,
    IsaacJointCommandPacket,
    IsaacRealizedJointPacket,
    IsaacTagDetectionPacket,
)


class IsaacRunWriter:
    """Write the mandated Isaac run layout in a deterministic way."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.config_snapshot_dir = self.run_dir / "config_snapshot"
        self.raw_dir = self.run_dir / "raw"
        self.gt_dir = self.run_dir / "gt"
        self.estimates_dir = self.run_dir / "estimates"
        self.analysis_dir = self.run_dir / "analysis"
        self.report_data_dir = self.analysis_dir / "report_data"
        for path in [
            self.run_dir,
            self.config_snapshot_dir,
            self.raw_dir / "rgb",
            self.gt_dir,
            self.estimates_dir,
            self.analysis_dir,
            self.report_data_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def _append_csv_row(self, path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
        write_header = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _append_dynamic_csv_row(self, path: Path, row: dict[str, Any]) -> None:
        fieldnames = list(row.keys())
        self._append_csv_row(path, fieldnames, row)

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def write_config_snapshot(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.config_snapshot_dir / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def append_camera_frame(self, packet: IsaacCameraFramePacket) -> None:
        self._append_jsonl(self.raw_dir / "camera_frames.jsonl", packet.as_json())

    def append_detection(self, packet: IsaacTagDetectionPacket) -> None:
        self._append_jsonl(self.raw_dir / "detections.jsonl", packet.as_json())

    def append_estimator_input(self, payload: dict[str, Any]) -> None:
        self._append_jsonl(self.raw_dir / "estimator_input.jsonl", dict(payload))

    def append_imu(self, packet: IsaacImuPacket) -> None:
        self._append_csv_row(self.raw_dir / "imu.csv", packet.csv_fieldnames(), packet.to_csv_row())

    def append_command(self, packet: IsaacJointCommandPacket) -> None:
        self._append_csv_row(self.raw_dir / "commands.csv", packet.csv_fieldnames(), packet.to_csv_row())

    def append_realized_joint(self, packet: IsaacRealizedJointPacket) -> None:
        self._append_csv_row(self.raw_dir / "realized_joints.csv", packet.csv_fieldnames(), packet.to_csv_row())

    def append_gt_camera(self, payload: dict[str, Any]) -> None:
        self._append_dynamic_csv_row(self.gt_dir / "camera_gt.csv", dict(payload))

    def append_gt_imu(self, payload: dict[str, Any]) -> None:
        self._append_dynamic_csv_row(self.gt_dir / "imu_gt.csv", dict(payload))

    def append_gt_joint(self, payload: dict[str, Any]) -> None:
        self._append_dynamic_csv_row(self.gt_dir / "joint_gt.csv", dict(payload))

    def write_tag_gt(self, payload: dict[str, Any]) -> None:
        (self.gt_dir / "tag_gt.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def append_filter_state(self, snapshot: FilterStateSnapshot) -> None:
        self._append_jsonl(self.estimates_dir / "filter_state.jsonl", snapshot.as_json())

    def append_smoother_state(self, snapshot: SmootherStateSnapshot) -> None:
        self._append_jsonl(self.estimates_dir / "smoother_state.jsonl", snapshot.as_json())

    def append_uncertainty(self, snapshot: UncertaintySnapshot) -> None:
        self._append_jsonl(self.estimates_dir / "uncertainty.jsonl", snapshot.as_json())

    def write_manifest(self, manifest: IsaacRunManifest) -> Path:
        path = self.run_dir / "manifest.json"
        path.write_text(json.dumps(manifest.summary(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path
