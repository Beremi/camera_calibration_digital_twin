"""Low-latency anchored visual-inertial filter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from calib_sim.estimation._geometry import rotation_matrix_from_rvec
from calib_sim.isaac.estimation.imu_preintegration import preintegrate_imu_packets
from calib_sim.isaac.estimation.state_defs import FilterStateSnapshot, MotionState, UncertaintySnapshot
from calib_sim.isaac.estimation.uncertainty import radius_95_from_covariance, sanitize_covariance
from calib_sim.isaac.logging.schemas import IsaacImuPacket, IsaacTagDetectionPacket
from calib_sim.isaac.tag_builder import TagPoseSpec


def _camera_pose_from_detection(tag_detection: IsaacTagDetectionPacket, tag_pose: TagPoseSpec) -> tuple[np.ndarray, np.ndarray]:
    if tag_detection.pose_camera_rvec is None or tag_detection.pose_camera_tvec_m is None:
        raise ValueError("Tag detection is missing solvePnP pose for anchored update.")
    rotation_ct = rotation_matrix_from_rvec(np.asarray(tag_detection.pose_camera_rvec, dtype=np.float64))
    rotation_tc = rotation_ct.T
    translation_ct = np.asarray(tag_detection.pose_camera_tvec_m, dtype=np.float64).reshape(3)
    rotation_wt = np.asarray(tag_pose.rotation_wt, dtype=np.float64).reshape(3, 3)
    position_wt = np.asarray(tag_pose.position_world_m, dtype=np.float64).reshape(3)
    position_world = position_wt + rotation_wt @ (-rotation_tc @ translation_ct)
    rotation_wi = rotation_wt @ rotation_tc
    return position_world.astype(np.float64), rotation_wi.astype(np.float64)


@dataclass(slots=True)
class AnchoredOnlineFilter:
    state: MotionState
    covariance: np.ndarray
    estimator_mode: str = "fused"
    use_imu_prediction: bool = True
    use_auxiliary_tag_updates: bool = True
    last_timestamp_s: float = 0.0
    last_innovation_norm: float = 0.0
    rejected_updates_count: int = 0
    anchor_relocalizations: int = 0
    anchor_update_count: int = 0
    auxiliary_update_count: int = 0
    auxiliary_rejection_count: int = 0
    last_anchor_position_world_m: np.ndarray | None = None
    last_anchor_timestamp_s: float | None = None

    @classmethod
    def identity_initialized(cls, *, estimator_mode: str = "fused") -> "AnchoredOnlineFilter":
        normalized_mode = str(estimator_mode).strip().lower()
        if normalized_mode not in {"visual", "fused"}:
            normalized_mode = {
                "visual_anchor_only": "visual",
                "visual_anchor_plus_aux_tags": "visual",
                "visual_inertial_anchor_plus_aux_tags": "fused",
            }.get(normalized_mode, "fused")
        return cls(
            state=MotionState(
                rotation_wi=np.eye(3, dtype=np.float64),
                position_world_m=np.zeros(3, dtype=np.float64),
                velocity_world_mps=np.zeros(3, dtype=np.float64),
                gyro_bias_rps=np.zeros(3, dtype=np.float64),
                accel_bias_mps2=np.zeros(3, dtype=np.float64),
            ),
            covariance=np.eye(15, dtype=np.float64) * 1e-3,
            estimator_mode=normalized_mode,
            use_imu_prediction=normalized_mode == "fused",
            use_auxiliary_tag_updates=True,
            last_timestamp_s=0.0,
        )

    def predict(self, imu_packets: tuple[IsaacImuPacket, ...]) -> None:
        if not self.use_imu_prediction:
            if imu_packets:
                self.last_timestamp_s = float(imu_packets[-1].timestamp_s)
            return
        delta = preintegrate_imu_packets(
            packets=imu_packets,
            start_rotation_wi=self.state.rotation_wi,
            start_position_world_m=self.state.position_world_m,
            start_velocity_world_mps=self.state.velocity_world_mps,
            accel_bias_mps2=self.state.accel_bias_mps2,
            gyro_bias_rps=self.state.gyro_bias_rps,
        )
        self.state.rotation_wi = delta.final_rotation_wi
        self.state.velocity_world_mps = delta.final_velocity_world_mps
        self.state.position_world_m = delta.final_position_world_m
        if imu_packets:
            self.last_timestamp_s = float(imu_packets[-1].timestamp_s)
        process_noise = np.eye(self.covariance.shape[0], dtype=np.float64) * max(delta.delta_time_s, 1e-6) * 1e-4
        self.covariance = sanitize_covariance(self.covariance + process_noise)

    def propagate(self, packets: tuple[IsaacImuPacket, ...]) -> None:
        self.predict(packets)

    def update_anchor(
        self,
        *,
        tag_detection: IsaacTagDetectionPacket,
        tag_pose: TagPoseSpec,
        measurement_std_m: float = 0.01,
        position_gain: float = 1.0,
        rotation_gain: float = 1.0,
        velocity_gain: float = 1.0,
        relocalization_threshold_m: float = 0.15,
    ) -> None:
        measured_position_world_m, measured_rotation_wi = _camera_pose_from_detection(tag_detection, tag_pose)
        innovation = measured_position_world_m - self.state.position_world_m
        innovation_norm = float(np.linalg.norm(innovation))
        if innovation_norm > float(relocalization_threshold_m):
            self.anchor_relocalizations += 1
        self.anchor_update_count += 1
        measured_velocity_world_mps = self.state.velocity_world_mps.copy()
        if self.last_anchor_position_world_m is not None and self.last_anchor_timestamp_s is not None:
            dt_s = float(tag_detection.timestamp_s) - float(self.last_anchor_timestamp_s)
            if dt_s > 1e-6:
                measured_velocity_world_mps = (
                    np.asarray(measured_position_world_m, dtype=np.float64) - self.last_anchor_position_world_m
                ) / dt_s
        position_gain = float(np.clip(position_gain, 0.0, 1.0))
        velocity_gain = float(np.clip(velocity_gain, 0.0, 1.0))
        self.state.position_world_m = self.state.position_world_m + position_gain * innovation
        self.state.velocity_world_mps = (
            (1.0 - velocity_gain) * self.state.velocity_world_mps + velocity_gain * measured_velocity_world_mps
        )
        self.state.rotation_wi = measured_rotation_wi.copy()
        self.last_innovation_norm = innovation_norm
        self.covariance = sanitize_covariance(self.covariance * max(1.0 - 0.4 * position_gain, 1e-6))
        self.covariance[:3, :3] = np.eye(3, dtype=np.float64) * max(measurement_std_m**2, 1e-9)
        self.covariance[3:6, 3:6] = np.eye(3, dtype=np.float64) * max(rotation_gain * measurement_std_m**2, 1e-9)
        self.covariance[6:9, 6:9] = np.eye(3, dtype=np.float64) * max((2.0 * measurement_std_m) ** 2, 1e-9)
        self.last_timestamp_s = float(tag_detection.timestamp_s)
        self.last_anchor_position_world_m = np.asarray(measured_position_world_m, dtype=np.float64).reshape(3)
        self.last_anchor_timestamp_s = float(tag_detection.timestamp_s)

    def anchor_visual_update(
        self,
        *,
        measured_position_world_m: np.ndarray,
        measured_rotation_wi: np.ndarray,
        measurement_std_m: float = 0.01,
        rotation_gain: float = 0.5,
    ) -> None:
        innovation = np.asarray(measured_position_world_m, dtype=np.float64).reshape(3) - self.state.position_world_m
        kalman_gain = 1.0 / max(1.0 + measurement_std_m * 100.0, 1e-6)
        self.state.position_world_m = self.state.position_world_m + kalman_gain * innovation
        self.state.rotation_wi = np.asarray(measured_rotation_wi, dtype=np.float64).reshape(3, 3)
        self.last_innovation_norm = float(np.linalg.norm(innovation))
        self.covariance = sanitize_covariance(self.covariance * (1.0 - 0.25 * kalman_gain))
        self.covariance[:3, :3] = np.eye(3, dtype=np.float64) * max(measurement_std_m**2, 1e-9)
        self.covariance[3:6, 3:6] = np.eye(3, dtype=np.float64) * max(rotation_gain * measurement_std_m**2, 1e-9)

    def update_aux_tags(
        self,
        *,
        tag_detections: tuple[IsaacTagDetectionPacket, ...],
        mapped_tag_poses: dict[int, TagPoseSpec],
        trust: float = 0.15,
    ) -> None:
        if not self.use_auxiliary_tag_updates:
            return
        candidate_positions: list[np.ndarray] = []
        candidate_rotations: list[np.ndarray] = []
        for detection in tag_detections:
            if int(detection.tag_id) not in mapped_tag_poses:
                continue
            if detection.pose_camera_rvec is None or detection.pose_camera_tvec_m is None:
                self.rejected_updates_count += 1
                self.auxiliary_rejection_count += 1
                continue
            try:
                position_world, rotation_wi = _camera_pose_from_detection(detection, mapped_tag_poses[int(detection.tag_id)])
            except ValueError:
                self.rejected_updates_count += 1
                self.auxiliary_rejection_count += 1
                continue
            candidate_positions.append(position_world)
            candidate_rotations.append(rotation_wi)
        if not candidate_positions:
            return
        self.auxiliary_update_count += len(candidate_positions)
        mean_position = np.mean(np.stack(candidate_positions, axis=0), axis=0)
        relative_position_delta_m = mean_position - self.state.position_world_m
        self.state.position_world_m = self.state.position_world_m + float(trust) * relative_position_delta_m
        self.state.rotation_wi = candidate_rotations[0]
        self.last_innovation_norm = float(np.linalg.norm(relative_position_delta_m))
        self.covariance = sanitize_covariance(self.covariance * max(1.0 - float(trust) * 0.1, 1e-6))
        self.last_timestamp_s = float(tag_detections[0].timestamp_s)

    def auxiliary_visual_update(self, *, relative_position_delta_m: np.ndarray, trust: float = 0.15) -> None:
        relative_position_delta_m = np.asarray(relative_position_delta_m, dtype=np.float64).reshape(3)
        self.state.position_world_m = self.state.position_world_m + float(trust) * relative_position_delta_m
        self.last_innovation_norm = float(np.linalg.norm(relative_position_delta_m))
        self.covariance = sanitize_covariance(self.covariance * max(1.0 - float(trust) * 0.1, 1e-6))

    def current_state(self) -> MotionState:
        return MotionState(
            rotation_wi=self.state.rotation_wi.copy(),
            position_world_m=self.state.position_world_m.copy(),
            velocity_world_mps=self.state.velocity_world_mps.copy(),
            gyro_bias_rps=self.state.gyro_bias_rps.copy(),
            accel_bias_mps2=self.state.accel_bias_mps2.copy(),
        )

    def current_uncertainty_summary(self, *, sim_time_s: float) -> UncertaintySnapshot:
        pose_covariance = sanitize_covariance(self.covariance[:3, :3])
        velocity_covariance = sanitize_covariance(self.covariance[6:9, 6:9])
        bias_covariance = sanitize_covariance(self.covariance[9:15, 9:15])
        return UncertaintySnapshot(
            timestamp_s=float(self.last_timestamp_s),
            sim_time_s=float(sim_time_s),
            pose_covariance=pose_covariance,
            velocity_covariance=velocity_covariance,
            bias_covariance=bias_covariance,
            position_radius_95_m=float(radius_95_from_covariance(pose_covariance)),
            velocity_radius_95_mps=float(radius_95_from_covariance(velocity_covariance)),
            diagnostics={
                "last_innovation_norm": float(self.last_innovation_norm),
                "rejected_updates_count": float(self.rejected_updates_count),
                "anchor_relocalizations": float(self.anchor_relocalizations),
                "anchor_update_count": float(self.anchor_update_count),
                "auxiliary_update_count": float(self.auxiliary_update_count),
                "auxiliary_rejection_count": float(self.auxiliary_rejection_count),
            },
        )

    def export_snapshot(self, *, sim_time_s: float) -> FilterStateSnapshot:
        return FilterStateSnapshot(
            timestamp_s=float(self.last_timestamp_s),
            sim_time_s=float(sim_time_s),
            rotation_wi=self.state.rotation_wi.copy(),
            position_world_m=self.state.position_world_m.copy(),
            velocity_world_mps=self.state.velocity_world_mps.copy(),
            gyro_bias_rps=self.state.gyro_bias_rps.copy(),
            accel_bias_mps2=self.state.accel_bias_mps2.copy(),
            covariance=self.covariance.copy(),
            mode=self.estimator_mode,
            innovation_diagnostics={
                "last_innovation_norm": float(self.last_innovation_norm),
                "position_radius_95_m": float(radius_95_from_covariance(self.covariance[:3, :3])),
                "rejected_updates_count": float(self.rejected_updates_count),
                "anchor_relocalizations": float(self.anchor_relocalizations),
                "anchor_update_count": float(self.anchor_update_count),
                "auxiliary_update_count": float(self.auxiliary_update_count),
                "auxiliary_rejection_count": float(self.auxiliary_rejection_count),
            },
        )


__all__ = ["AnchoredOnlineFilter"]
