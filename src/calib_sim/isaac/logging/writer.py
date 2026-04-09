"""Artifact writer for Isaac runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from calib_sim.isaac.estimation.state_defs import FilterStateSnapshot, SmootherStateSnapshot, UncertaintySnapshot
from calib_sim.isaac.logging.run_manifest import IsaacRunManifest
from calib_sim.isaac.logging.schemas import (
    IsaacCameraFramePacket,
    IsaacControllerDiagnosticPacket,
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
        self.raw_rgb_dir = self.raw_dir / "rgb"
        self.gt_dir = self.run_dir / "gt"
        self.estimates_dir = self.run_dir / "estimates"
        self.analysis_dir = self.run_dir / "analysis"
        self.report_data_dir = self.analysis_dir / "report_data"
        for path in [
            self.run_dir,
            self.config_snapshot_dir,
            self.raw_dir,
            self.raw_rgb_dir,
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
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def write_config_snapshot(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.config_snapshot_dir / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def save_rgb_image(self, *, frame_index: int, image_rgb: np.ndarray) -> str:
        path = self.raw_rgb_dir / f"frame_{int(frame_index):06d}.png"
        image_bgr = cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), image_bgr)
        return str(path.relative_to(self.run_dir))

    def write_camera_frame(self, packet: IsaacCameraFramePacket, image_rgb: np.ndarray | None = None) -> None:
        if image_rgb is not None and not packet.rgb_path:
            packet = IsaacCameraFramePacket(
                frame_index=packet.frame_index,
                timestamp_s=packet.timestamp_s,
                sim_time_s=packet.sim_time_s,
                sensor_time_s=packet.sensor_time_s,
                host_time_s=packet.host_time_s,
                rgb_path=self.save_rgb_image(frame_index=packet.frame_index, image_rgb=image_rgb),
                intrinsics_snapshot=packet.intrinsics_snapshot,
                extrinsics_snapshot=packet.extrinsics_snapshot,
                image_width_px=packet.image_width_px,
                image_height_px=packet.image_height_px,
                visible_gt_tag_ids=packet.visible_gt_tag_ids,
            )
        self._append_jsonl(self.raw_dir / "camera_frames.jsonl", packet.as_json())

    def write_detection(self, packet: IsaacTagDetectionPacket) -> None:
        self._append_jsonl(self.raw_dir / "detections.jsonl", packet.as_json())

    def write_estimator_input(self, payload: dict[str, Any]) -> None:
        self._append_jsonl(self.raw_dir / "estimator_input.jsonl", dict(payload))

    def write_dropout_debug_frame(self, payload: dict[str, Any]) -> None:
        self._append_csv_row(
            self.raw_dir / "dropout_debug_frames.csv",
            [
                "timestamp_s",
                "frame_index",
                "anchor_visible_raw",
                "anchor_visible_effective",
                "suppression_active",
                "imu_prediction_disabled",
                "imu_packets_since_last_frame",
                "propagation_dt_s",
                "position_error_norm_m",
                "velocity_norm_mps",
                "gyro_bias_norm_rps",
                "accel_bias_norm_mps2",
                "covariance_trace",
                "covariance_min_eigenvalue",
                "covariance_max_eigenvalue",
            ],
            payload,
        )

    def write_dropout_debug_event(self, payload: dict[str, Any]) -> None:
        self._append_csv_row(
            self.raw_dir / "dropout_debug_events.csv",
            [
                "timestamp_s",
                "frame_index",
                "event_kind",
                "anchor_update_attempted",
                "accepted",
                "reason",
                "is_reacquisition",
                "pose_innovation_norm_m",
                "orientation_innovation_norm_deg",
                "velocity_innovation_norm_mps",
                "relocalization_correction_norm_m",
                "post_update_covariance_trace",
            ],
            payload,
        )

    def write_imu_packet(self, packet: IsaacImuPacket) -> None:
        self._append_csv_row(self.raw_dir / "imu.csv", packet.csv_fieldnames(), packet.to_csv_row())

    def write_command(self, packet: IsaacJointCommandPacket) -> None:
        self._append_csv_row(self.raw_dir / "commands.csv", packet.csv_fieldnames(), packet.to_csv_row())

    def write_controller_diagnostic(self, packet: IsaacControllerDiagnosticPacket) -> None:
        self._append_csv_row(
            self.raw_dir / "controller_diagnostics.csv",
            packet.csv_fieldnames(),
            packet.to_csv_row(),
        )

    def write_realized_state(self, packet: IsaacRealizedJointPacket) -> None:
        self._append_csv_row(self.raw_dir / "realized_joints.csv", packet.csv_fieldnames(), packet.to_csv_row())

    def write_gt(self, *, namespace: str, payload: dict[str, Any]) -> None:
        filename = {
            "camera": "camera_gt.csv",
            "imu": "imu_gt.csv",
            "joint": "joint_gt.csv",
        }.get(namespace)
        if filename is None:
            raise ValueError(f"Unsupported GT namespace: {namespace}")
        self._append_dynamic_csv_row(self.gt_dir / filename, dict(payload))

    def write_tag_gt(self, payload: dict[str, Any]) -> None:
        (self.gt_dir / "tag_gt.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def write_filter_state(self, snapshot: FilterStateSnapshot) -> None:
        self._append_jsonl(self.estimates_dir / "filter_state.jsonl", snapshot.as_json())

    def write_smoother_state(self, snapshot: SmootherStateSnapshot) -> None:
        self._append_jsonl(self.estimates_dir / "smoother_state.jsonl", snapshot.as_json())

    def write_uncertainty(self, snapshot: UncertaintySnapshot) -> None:
        self._append_jsonl(self.estimates_dir / "uncertainty.jsonl", snapshot.as_json())

    def write_manifest(self, manifest: IsaacRunManifest) -> Path:
        path = self.run_dir / "manifest.json"
        path.write_text(json.dumps(manifest.summary(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def finalize_summary(self, payload: dict[str, Any]) -> Path:
        path = self.analysis_dir / "run_summary.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    # Backward-compatible aliases kept for scaffold tests and existing imports.
    append_camera_frame = write_camera_frame
    append_detection = write_detection
    append_estimator_input = write_estimator_input
    append_imu = write_imu_packet
    append_command = write_command
    append_controller_diagnostic = write_controller_diagnostic
    append_realized_joint = write_realized_state
    append_gt_camera = lambda self, payload: self.write_gt(namespace="camera", payload=payload)
    append_gt_imu = lambda self, payload: self.write_gt(namespace="imu", payload=payload)
    append_gt_joint = lambda self, payload: self.write_gt(namespace="joint", payload=payload)
    append_filter_state = write_filter_state
    append_smoother_state = write_smoother_state
    append_uncertainty = write_uncertainty
