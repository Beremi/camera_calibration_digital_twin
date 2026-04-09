"""Low-latency anchored visual-inertial filter."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from calib_sim.estimation._geometry import rotation_matrix_from_rvec
from calib_sim.isaac.estimation.imu_preintegration import preintegrate_imu_packets
from calib_sim.isaac.estimation.state_defs import FilterStateSnapshot, MotionState, UncertaintySnapshot
from calib_sim.isaac.estimation.uncertainty import radius_95_from_covariance, sanitize_covariance
from calib_sim.isaac.logging.schemas import IsaacImuPacket, IsaacTagDetectionPacket
from calib_sim.isaac.tag_builder import TagPoseSpec


_AUXILIARY_REJECTION_REASON_CODES = (
    "fallback_only",
    "corner_margin_fail",
    "reproj_fail",
    "innovation_fail",
    "visibility_streak_fail",
    "covariance_fail",
)


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


def _camera_matrix_and_distortion(
    intrinsics_snapshot: dict[str, object] | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not isinstance(intrinsics_snapshot, dict):
        return None, None
    fx = intrinsics_snapshot.get("fx_px")
    fy = intrinsics_snapshot.get("fy_px")
    cx = intrinsics_snapshot.get("cx_px")
    cy = intrinsics_snapshot.get("cy_px")
    if any(value in (None, "") for value in (fx, fy, cx, cy)):
        return None, None
    camera_matrix = np.array(
        [
            [float(fx), 0.0, float(cx)],
            [0.0, float(fy), float(cy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = intrinsics_snapshot.get("distortion_coefficients", [0.0, 0.0, 0.0, 0.0, 0.0])
    dist_coeffs = np.asarray(distortion, dtype=np.float64).reshape(-1, 1)
    return camera_matrix, dist_coeffs


def _world_points_for_tag(detection: IsaacTagDetectionPacket, tag_pose: TagPoseSpec) -> np.ndarray | None:
    local_points = np.asarray(detection.local_tag_points_m, dtype=np.float64)
    if local_points.shape != (4, 3):
        return None
    rotation_wt = np.asarray(tag_pose.rotation_wt, dtype=np.float64).reshape(3, 3)
    position_wt = np.asarray(tag_pose.position_world_m, dtype=np.float64).reshape(3)
    return (rotation_wt @ local_points.T).T + position_wt.reshape(1, 3)


def _reprojection_rmse_px(
    *,
    position_world_m: np.ndarray,
    rotation_wi: np.ndarray,
    detection: IsaacTagDetectionPacket,
    tag_pose: TagPoseSpec,
    intrinsics_snapshot: dict[str, object] | None,
) -> float | None:
    camera_matrix, dist_coeffs = _camera_matrix_and_distortion(intrinsics_snapshot)
    if camera_matrix is None or dist_coeffs is None:
        return None
    world_points = _world_points_for_tag(detection, tag_pose)
    observed_corners = np.asarray(detection.corners_xy, dtype=np.float64)
    if world_points is None or observed_corners.shape != (4, 2):
        return None
    rotation_wc = np.asarray(rotation_wi, dtype=np.float64).reshape(3, 3)
    rotation_cw = rotation_wc.T
    translation_cw = -rotation_cw @ np.asarray(position_world_m, dtype=np.float64).reshape(3)
    rvec_cw, _ = cv2.Rodrigues(rotation_cw)
    projected_points, _ = cv2.projectPoints(
        world_points,
        rvec_cw,
        translation_cw.reshape(3, 1),
        camera_matrix,
        dist_coeffs,
    )
    projected_corners = np.asarray(projected_points, dtype=np.float64).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum((projected_corners - observed_corners) ** 2, axis=1))))


def _corner_margin_px(
    corners_xy: tuple[tuple[float, float], ...],
    *,
    image_width_px: int,
    image_height_px: int,
) -> float:
    corners = np.asarray(corners_xy, dtype=np.float64).reshape(-1, 2)
    if corners.size == 0:
        return 0.0
    margins = np.column_stack(
        (
            corners[:, 0],
            corners[:, 1],
            float(max(image_width_px - 1, 0)) - corners[:, 0],
            float(max(image_height_px - 1, 0)) - corners[:, 1],
        )
    )
    return float(np.min(margins))


def _native_pose_ready(detection: IsaacTagDetectionPacket) -> bool:
    flags = dict(detection.visibility_flags)
    native_backend = bool(flags.get("native_backend", False))
    pose_ready = bool(flags.get("pose_ready", False))
    return native_backend and pose_ready and detection.pose_camera_tvec_m is not None and detection.pose_camera_rvec is not None


def _position_mahalanobis_score(position_delta_world_m: np.ndarray, covariance: np.ndarray) -> float | None:
    candidate = np.asarray(position_delta_world_m, dtype=np.float64).reshape(3)
    if covariance.shape[0] < 6 or covariance.shape[1] < 6:
        return None
    try:
        return float(candidate.T @ np.linalg.solve(np.asarray(covariance[3:6, 3:6], dtype=np.float64), candidate))
    except np.linalg.LinAlgError:
        return None


@dataclass(slots=True)
class AuxiliaryUpdateDecision:
    tag_id: int
    accepted: bool
    reason: str
    innovation_norm: float | None = None
    reprojection_rmse_px: float | None = None
    visibility_streak: int = 0
    native_pnp: bool = False
    corner_margin_px: float | None = None
    mahalanobis_score: float | None = None

    def as_json(self) -> dict[str, object]:
        return {
            "tag_id": int(self.tag_id),
            "accepted": bool(self.accepted),
            "reason": str(self.reason),
            "innovation_norm": None if self.innovation_norm is None else float(self.innovation_norm),
            "reprojection_rmse_px": None
            if self.reprojection_rmse_px is None
            else float(self.reprojection_rmse_px),
            "visibility_streak": int(self.visibility_streak),
            "native_pnp": bool(self.native_pnp),
            "corner_margin_px": None if self.corner_margin_px is None else float(self.corner_margin_px),
            "mahalanobis_score": None if self.mahalanobis_score is None else float(self.mahalanobis_score),
        }


@dataclass(slots=True)
class AuxiliaryUpdateSummary:
    visible_count: int = 0
    native_pose_ready_count: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    accepted_tag_ids: tuple[int, ...] = ()
    rejected_tag_ids: tuple[int, ...] = ()
    rejection_reason_counts: dict[str, int] = field(default_factory=dict)
    decisions: tuple[AuxiliaryUpdateDecision, ...] = ()

    def as_json(self) -> dict[str, object]:
        return {
            "visible_count": int(self.visible_count),
            "native_pose_ready_count": int(self.native_pose_ready_count),
            "accepted_count": int(self.accepted_count),
            "rejected_count": int(self.rejected_count),
            "accepted_tag_ids": [int(tag_id) for tag_id in self.accepted_tag_ids],
            "rejected_tag_ids": [int(tag_id) for tag_id in self.rejected_tag_ids],
            "rejection_reason_counts": {str(key): int(value) for key, value in self.rejection_reason_counts.items()},
            "decisions": [decision.as_json() for decision in self.decisions],
        }


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
    vision_covariance_scale: float = 1.0
    anchor_vision_covariance_scale: float | None = None
    aux_vision_covariance_scale: float | None = None
    imu_process_covariance_scale: float = 1.0
    gyro_process_covariance_scale: float | None = None
    accel_process_covariance_scale: float | None = None
    post_relocalization_covariance_scale: float = 1.0
    last_anchor_nis: float | None = None
    last_auxiliary_nis: float | None = None
    last_anchor_position_world_m: np.ndarray | None = None
    last_anchor_timestamp_s: float | None = None
    auxiliary_rejection_reason_counts: dict[str, int] = field(
        default_factory=lambda: {reason: 0 for reason in _AUXILIARY_REJECTION_REASON_CODES}
    )
    auxiliary_visibility_streaks: dict[int, int] = field(default_factory=dict)
    last_auxiliary_summary: AuxiliaryUpdateSummary = field(default_factory=AuxiliaryUpdateSummary)

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

    def _anchor_covariance_scale(self) -> float:
        if self.anchor_vision_covariance_scale is not None:
            return max(float(self.anchor_vision_covariance_scale), 1e-6)
        return max(float(self.vision_covariance_scale), 1e-6)

    def _aux_covariance_scale(self) -> float:
        if self.aux_vision_covariance_scale is not None:
            return max(float(self.aux_vision_covariance_scale), 1e-6)
        return max(float(self.vision_covariance_scale), 1e-6)

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
        global_scale = max(float(self.imu_process_covariance_scale), 1e-6)
        gyro_scale = global_scale if self.gyro_process_covariance_scale is None else max(float(self.gyro_process_covariance_scale), 1e-6)
        accel_scale = global_scale if self.accel_process_covariance_scale is None else max(float(self.accel_process_covariance_scale), 1e-6)
        dt_s = max(delta.delta_time_s, 1e-6)
        process_noise = np.eye(self.covariance.shape[0], dtype=np.float64) * dt_s * 1e-4
        process_noise[0:3, 0:3] *= gyro_scale
        process_noise[3:6, 3:6] *= accel_scale
        process_noise[6:9, 6:9] *= accel_scale
        process_noise[9:12, 9:12] *= gyro_scale
        process_noise[12:15, 12:15] *= accel_scale
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
        relocalized = innovation_norm > float(relocalization_threshold_m)
        if relocalized:
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
        measurement_variance = max(measurement_std_m**2 * self._anchor_covariance_scale(), 1e-9)
        self.last_anchor_nis = float(innovation.T @ innovation / measurement_variance)
        self.covariance[0:3, 0:3] = np.eye(3, dtype=np.float64) * max(rotation_gain * measurement_variance, 1e-9)
        self.covariance[3:6, 3:6] = np.eye(3, dtype=np.float64) * measurement_variance
        self.covariance[6:9, 6:9] = np.eye(3, dtype=np.float64) * max((2.0 * measurement_std_m) ** 2, 1e-9)
        if relocalized:
            self.covariance = sanitize_covariance(
                self.covariance * max(float(self.post_relocalization_covariance_scale), 1.0)
            )
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
        self.covariance[0:3, 0:3] = np.eye(3, dtype=np.float64) * max(rotation_gain * measurement_std_m**2, 1e-9)
        self.covariance[3:6, 3:6] = np.eye(3, dtype=np.float64) * max(measurement_std_m**2, 1e-9)

    def update_aux_tags(
        self,
        *,
        tag_detections: tuple[IsaacTagDetectionPacket, ...],
        mapped_tag_poses: dict[int, TagPoseSpec],
        trust: float = 0.15,
        intrinsics_snapshot: dict[str, object] | None = None,
        image_width_px: int = 0,
        image_height_px: int = 0,
        min_corner_margin_px: float = 0.0,
        min_visibility_streak: int = 1,
        max_reprojection_error_px: float | None = None,
        max_innovation_norm: float | None = None,
        max_mahalanobis_score: float | None = None,
        max_feedback_correction_norm_m: float | None = None,
        tag_feedback_diagnostics: dict[int, dict[str, object]] | None = None,
    ) -> AuxiliaryUpdateSummary:
        current_tag_ids = {int(detection.tag_id) for detection in tag_detections if not detection.is_anchor}
        for tag_id in list(self.auxiliary_visibility_streaks):
            if int(tag_id) in current_tag_ids:
                continue
            self.auxiliary_visibility_streaks[int(tag_id)] = 0
        for tag_id in current_tag_ids:
            self.auxiliary_visibility_streaks[int(tag_id)] = self.auxiliary_visibility_streaks.get(int(tag_id), 0) + 1
        native_pose_ready_count = sum(1 for detection in tag_detections if _native_pose_ready(detection))
        if not self.use_auxiliary_tag_updates:
            self.last_auxiliary_summary = AuxiliaryUpdateSummary(
                visible_count=len(tag_detections),
                native_pose_ready_count=native_pose_ready_count,
            )
            return self.last_auxiliary_summary
        candidate_positions: list[np.ndarray] = []
        candidate_rotations: list[np.ndarray] = []
        accepted_tag_ids: list[int] = []
        rejected_tag_ids: list[int] = []
        decisions: list[AuxiliaryUpdateDecision] = []
        per_frame_reason_counts = {reason: 0 for reason in _AUXILIARY_REJECTION_REASON_CODES}
        tag_feedback_diagnostics = {} if tag_feedback_diagnostics is None else dict(tag_feedback_diagnostics)
        for detection in tag_detections:
            tag_id = int(detection.tag_id)
            if tag_id not in mapped_tag_poses:
                continue
            native_pose_ready = _native_pose_ready(detection)
            if not native_pose_ready:
                self.rejected_updates_count += 1
                self.auxiliary_rejection_count += 1
                self.auxiliary_rejection_reason_counts["fallback_only"] += 1
                per_frame_reason_counts["fallback_only"] += 1
                rejected_tag_ids.append(tag_id)
                decisions.append(
                    AuxiliaryUpdateDecision(
                        tag_id=tag_id,
                        accepted=False,
                        reason="fallback_only",
                        visibility_streak=self.auxiliary_visibility_streaks.get(tag_id, 0),
                        native_pnp=False,
                    )
                )
                continue
            corner_margin_px = _corner_margin_px(
                detection.corners_xy,
                image_width_px=max(int(image_width_px), 0),
                image_height_px=max(int(image_height_px), 0),
            )
            if corner_margin_px < float(min_corner_margin_px):
                self.rejected_updates_count += 1
                self.auxiliary_rejection_count += 1
                self.auxiliary_rejection_reason_counts["corner_margin_fail"] += 1
                per_frame_reason_counts["corner_margin_fail"] += 1
                rejected_tag_ids.append(tag_id)
                decisions.append(
                    AuxiliaryUpdateDecision(
                        tag_id=tag_id,
                        accepted=False,
                        reason="corner_margin_fail",
                        visibility_streak=self.auxiliary_visibility_streaks.get(tag_id, 0),
                        native_pnp=True,
                        corner_margin_px=corner_margin_px,
                    )
                )
                continue
            visibility_streak = self.auxiliary_visibility_streaks.get(tag_id, 0)
            if visibility_streak < int(min_visibility_streak):
                self.rejected_updates_count += 1
                self.auxiliary_rejection_count += 1
                self.auxiliary_rejection_reason_counts["visibility_streak_fail"] += 1
                per_frame_reason_counts["visibility_streak_fail"] += 1
                rejected_tag_ids.append(tag_id)
                decisions.append(
                    AuxiliaryUpdateDecision(
                        tag_id=tag_id,
                        accepted=False,
                        reason="visibility_streak_fail",
                        visibility_streak=visibility_streak,
                        native_pnp=True,
                        corner_margin_px=corner_margin_px,
                    )
                )
                continue
            try:
                position_world, rotation_wi = _camera_pose_from_detection(detection, mapped_tag_poses[tag_id])
            except ValueError:
                self.rejected_updates_count += 1
                self.auxiliary_rejection_count += 1
                self.auxiliary_rejection_reason_counts["fallback_only"] += 1
                per_frame_reason_counts["fallback_only"] += 1
                rejected_tag_ids.append(tag_id)
                decisions.append(
                    AuxiliaryUpdateDecision(
                        tag_id=tag_id,
                        accepted=False,
                        reason="fallback_only",
                        visibility_streak=visibility_streak,
                        native_pnp=False,
                        corner_margin_px=corner_margin_px,
                    )
                )
                continue
            reprojection_rmse_px = _reprojection_rmse_px(
                position_world_m=position_world,
                rotation_wi=rotation_wi,
                detection=detection,
                tag_pose=mapped_tag_poses[tag_id],
                intrinsics_snapshot=intrinsics_snapshot,
            )
            if max_reprojection_error_px is not None and reprojection_rmse_px is not None:
                if float(reprojection_rmse_px) > float(max_reprojection_error_px):
                    self.rejected_updates_count += 1
                    self.auxiliary_rejection_count += 1
                    self.auxiliary_rejection_reason_counts["reproj_fail"] += 1
                    per_frame_reason_counts["reproj_fail"] += 1
                    rejected_tag_ids.append(tag_id)
                    decisions.append(
                        AuxiliaryUpdateDecision(
                            tag_id=tag_id,
                            accepted=False,
                            reason="reproj_fail",
                            reprojection_rmse_px=reprojection_rmse_px,
                            visibility_streak=visibility_streak,
                            native_pnp=True,
                            corner_margin_px=corner_margin_px,
                        )
                    )
                    continue
            innovation_world_m = position_world - self.state.position_world_m
            innovation_norm = float(np.linalg.norm(innovation_world_m))
            mahalanobis_score = _position_mahalanobis_score(innovation_world_m, self.covariance)
            if (
                (max_innovation_norm is not None and innovation_norm > float(max_innovation_norm))
                or (
                    max_mahalanobis_score is not None
                    and mahalanobis_score is not None
                    and mahalanobis_score > float(max_mahalanobis_score)
                )
            ):
                self.rejected_updates_count += 1
                self.auxiliary_rejection_count += 1
                self.auxiliary_rejection_reason_counts["innovation_fail"] += 1
                per_frame_reason_counts["innovation_fail"] += 1
                rejected_tag_ids.append(tag_id)
                decisions.append(
                    AuxiliaryUpdateDecision(
                        tag_id=tag_id,
                        accepted=False,
                        reason="innovation_fail",
                        innovation_norm=innovation_norm,
                        reprojection_rmse_px=reprojection_rmse_px,
                        visibility_streak=visibility_streak,
                        native_pnp=True,
                        corner_margin_px=corner_margin_px,
                        mahalanobis_score=mahalanobis_score,
                    )
                )
                continue
            feedback_diagnostics = tag_feedback_diagnostics.get(tag_id, {})
            correction_norm_m = feedback_diagnostics.get("feedback_correction_norm_m")
            trusted = bool(feedback_diagnostics.get("trusted", True))
            if (
                not trusted
                or (
                    max_feedback_correction_norm_m is not None
                    and correction_norm_m not in (None, "")
                    and float(correction_norm_m) > float(max_feedback_correction_norm_m)
                )
            ):
                self.rejected_updates_count += 1
                self.auxiliary_rejection_count += 1
                self.auxiliary_rejection_reason_counts["covariance_fail"] += 1
                per_frame_reason_counts["covariance_fail"] += 1
                rejected_tag_ids.append(tag_id)
                decisions.append(
                    AuxiliaryUpdateDecision(
                        tag_id=tag_id,
                        accepted=False,
                        reason="covariance_fail",
                        innovation_norm=innovation_norm,
                        reprojection_rmse_px=reprojection_rmse_px,
                        visibility_streak=visibility_streak,
                        native_pnp=True,
                        corner_margin_px=corner_margin_px,
                        mahalanobis_score=mahalanobis_score,
                    )
                )
                continue
            candidate_positions.append(position_world)
            candidate_rotations.append(rotation_wi)
            accepted_tag_ids.append(tag_id)
            decisions.append(
                AuxiliaryUpdateDecision(
                    tag_id=tag_id,
                    accepted=True,
                    reason="accepted",
                    innovation_norm=innovation_norm,
                    reprojection_rmse_px=reprojection_rmse_px,
                    visibility_streak=visibility_streak,
                    native_pnp=True,
                    corner_margin_px=corner_margin_px,
                    mahalanobis_score=mahalanobis_score,
                )
            )
        if not candidate_positions:
            self.last_auxiliary_summary = AuxiliaryUpdateSummary(
                visible_count=len(tag_detections),
                native_pose_ready_count=native_pose_ready_count,
                accepted_count=0,
                rejected_count=len(rejected_tag_ids),
                accepted_tag_ids=(),
                rejected_tag_ids=tuple(rejected_tag_ids),
                rejection_reason_counts={key: value for key, value in per_frame_reason_counts.items() if value > 0},
                decisions=tuple(decisions),
            )
            return self.last_auxiliary_summary
        self.auxiliary_update_count += len(candidate_positions)
        mean_position = np.mean(np.stack(candidate_positions, axis=0), axis=0)
        relative_position_delta_m = mean_position - self.state.position_world_m
        accepted_decisions = [decision for decision in decisions if bool(decision.accepted)]
        accepted_scores = [
            float(decision.mahalanobis_score)
            for decision in accepted_decisions
            if decision.mahalanobis_score is not None
        ]
        self.last_auxiliary_nis = None if not accepted_scores else float(np.mean(accepted_scores))
        effective_trust = float(trust) / max(self._aux_covariance_scale(), 1.0)
        self.state.position_world_m = self.state.position_world_m + effective_trust * relative_position_delta_m
        self.state.rotation_wi = candidate_rotations[0]
        self.last_innovation_norm = float(np.linalg.norm(relative_position_delta_m))
        self.covariance = sanitize_covariance(self.covariance * max(1.0 - effective_trust * 0.1, 1e-6))
        self.last_timestamp_s = float(tag_detections[0].timestamp_s)
        self.last_auxiliary_summary = AuxiliaryUpdateSummary(
            visible_count=len(tag_detections),
            native_pose_ready_count=native_pose_ready_count,
            accepted_count=len(accepted_tag_ids),
            rejected_count=len(rejected_tag_ids),
            accepted_tag_ids=tuple(accepted_tag_ids),
            rejected_tag_ids=tuple(rejected_tag_ids),
            rejection_reason_counts={key: value for key, value in per_frame_reason_counts.items() if value > 0},
            decisions=tuple(decisions),
        )
        return self.last_auxiliary_summary

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
        pose_covariance = sanitize_covariance(self.covariance[3:6, 3:6])
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
                "vision_covariance_scale": float(self.vision_covariance_scale),
                "anchor_vision_covariance_scale": None
                if self.anchor_vision_covariance_scale is None
                else float(self.anchor_vision_covariance_scale),
                "aux_vision_covariance_scale": None
                if self.aux_vision_covariance_scale is None
                else float(self.aux_vision_covariance_scale),
                "imu_process_covariance_scale": float(self.imu_process_covariance_scale),
                "gyro_process_covariance_scale": None
                if self.gyro_process_covariance_scale is None
                else float(self.gyro_process_covariance_scale),
                "accel_process_covariance_scale": None
                if self.accel_process_covariance_scale is None
                else float(self.accel_process_covariance_scale),
                "post_relocalization_covariance_scale": float(self.post_relocalization_covariance_scale),
                "last_anchor_nis": None if self.last_anchor_nis is None else float(self.last_anchor_nis),
                "last_auxiliary_nis": None if self.last_auxiliary_nis is None else float(self.last_auxiliary_nis),
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
                "position_radius_95_m": float(radius_95_from_covariance(self.covariance[3:6, 3:6])),
                "rejected_updates_count": float(self.rejected_updates_count),
                "anchor_relocalizations": float(self.anchor_relocalizations),
                "anchor_update_count": float(self.anchor_update_count),
                "auxiliary_update_count": float(self.auxiliary_update_count),
                "auxiliary_rejection_count": float(self.auxiliary_rejection_count),
                "last_auxiliary_visible_count": float(self.last_auxiliary_summary.visible_count),
                "last_auxiliary_native_pose_ready_count": float(self.last_auxiliary_summary.native_pose_ready_count),
                "last_auxiliary_accepted_count": float(self.last_auxiliary_summary.accepted_count),
                "last_auxiliary_rejected_count": float(self.last_auxiliary_summary.rejected_count),
                "vision_covariance_scale": float(self.vision_covariance_scale),
                "anchor_vision_covariance_scale": None
                if self.anchor_vision_covariance_scale is None
                else float(self.anchor_vision_covariance_scale),
                "aux_vision_covariance_scale": None
                if self.aux_vision_covariance_scale is None
                else float(self.aux_vision_covariance_scale),
                "imu_process_covariance_scale": float(self.imu_process_covariance_scale),
                "gyro_process_covariance_scale": None
                if self.gyro_process_covariance_scale is None
                else float(self.gyro_process_covariance_scale),
                "accel_process_covariance_scale": None
                if self.accel_process_covariance_scale is None
                else float(self.accel_process_covariance_scale),
                "post_relocalization_covariance_scale": float(self.post_relocalization_covariance_scale),
                "last_anchor_nis": None if self.last_anchor_nis is None else float(self.last_anchor_nis),
                "last_auxiliary_nis": None if self.last_auxiliary_nis is None else float(self.last_auxiliary_nis),
            },
            auxiliary_rejection_reason_counts=dict(self.auxiliary_rejection_reason_counts),
        )


__all__ = [
    "AnchoredOnlineFilter",
    "AuxiliaryUpdateDecision",
    "AuxiliaryUpdateSummary",
]
