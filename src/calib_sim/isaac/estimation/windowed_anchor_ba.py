"""Anchored fixed-lag bundle adjustment backend for second-pass smoothing."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

try:
    from scipy.optimize import least_squares
except Exception:  # pragma: no cover - SciPy is expected in both local envs, but keep runtime-safe.
    least_squares = None

from calib_sim.isaac.estimation.fixed_lag_smoother import TagEstimate
from calib_sim.isaac.estimation.imu_preintegration import preintegrate_imu_packets
from calib_sim.isaac.estimation.state_defs import FilterStateSnapshot, SmootherStateSnapshot
from calib_sim.isaac.estimation.uncertainty import sanitize_covariance
from calib_sim.isaac.logging.schemas import IsaacImuPacket, IsaacTagDetectionPacket
from calib_sim.isaac.tag_builder import TagPoseSpec


def _pose_matrix(position_world_m: np.ndarray, rotation_wt: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(rotation_wt, dtype=np.float64).reshape(3, 3)
    matrix[:3, 3] = np.asarray(position_world_m, dtype=np.float64).reshape(3)
    return matrix


def _rotation_to_rotvec(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    rotvec, _ = cv2.Rodrigues(rotation)
    return np.asarray(rotvec, dtype=np.float64).reshape(3)


def _rotvec_to_rotation(rotvec: np.ndarray) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(np.asarray(rotvec, dtype=np.float64).reshape(3, 1))
    return np.asarray(rotation, dtype=np.float64).reshape(3, 3)


def _rotation_error_rotvec(rotation_a: np.ndarray, rotation_b: np.ndarray) -> np.ndarray:
    return _rotation_to_rotvec(np.asarray(rotation_a, dtype=np.float64).T @ np.asarray(rotation_b, dtype=np.float64))


def _camera_matrix_and_distortion(intrinsics_snapshot: dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None]:
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
    return camera_matrix, np.asarray(distortion, dtype=np.float64).reshape(-1, 1)


def _world_points(local_tag_points_m: np.ndarray, tag_pose_wt: np.ndarray) -> np.ndarray:
    rotation_wt = np.asarray(tag_pose_wt[:3, :3], dtype=np.float64).reshape(3, 3)
    position_wt = np.asarray(tag_pose_wt[:3, 3], dtype=np.float64).reshape(3)
    return (rotation_wt @ local_tag_points_m.T).T + position_wt.reshape(1, 3)


def _project_corners(
    *,
    position_world_m: np.ndarray,
    rotation_wi: np.ndarray,
    tag_pose_wt: np.ndarray,
    local_tag_points_m: np.ndarray,
    intrinsics_snapshot: dict[str, Any],
) -> np.ndarray | None:
    camera_matrix, dist_coeffs = _camera_matrix_and_distortion(intrinsics_snapshot)
    if camera_matrix is None or dist_coeffs is None:
        return None
    world_points = _world_points(np.asarray(local_tag_points_m, dtype=np.float64).reshape(4, 3), tag_pose_wt)
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
    return np.asarray(projected_points, dtype=np.float64).reshape(-1, 2)


@dataclass(slots=True)
class _WindowObservation:
    tag_id: int
    is_anchor: bool
    corners_xy: np.ndarray
    local_tag_points_m: np.ndarray
    intrinsics_snapshot: dict[str, Any]
    native_backend: bool


@dataclass(slots=True)
class WindowedAnchorBASmoother:
    lag_size: int = 30
    use_auxiliary_tags: bool = True
    backend_name: str = "windowed_ba"
    position_prior_std_m: float = 0.20
    rotation_prior_std_rad: float = 0.35
    first_state_position_prior_std_m: float = 0.05
    first_state_rotation_prior_std_rad: float = 0.10
    velocity_prior_std_mps: float = 0.25
    bias_prior_std: float = 0.05
    tag_position_prior_std_m: float = 0.20
    tag_rotation_prior_std_rad: float = 0.40
    visual_residual_scale_px: float = 8.0
    imu_position_std_m: float = 0.08
    imu_velocity_std_mps: float = 0.12
    imu_rotation_std_rad: float = 0.08
    solver_max_nfev: int = 40
    _snapshots: deque[FilterStateSnapshot] = field(default_factory=deque)
    _observations: deque[list[_WindowObservation]] = field(default_factory=deque)
    _imu_intervals: deque[tuple[IsaacImuPacket, ...]] = field(default_factory=deque)
    _tag_pose_estimates: dict[int, TagEstimate] = field(default_factory=dict)
    _latest_tag_pose_map: dict[int, TagPoseSpec] = field(default_factory=dict)
    _cost_trace: deque[float] = field(default_factory=lambda: deque(maxlen=64))
    _last_feedback_diagnostics: dict[str, float] = field(default_factory=dict)

    def push_snapshot(self, snapshot: FilterStateSnapshot) -> None:
        self._snapshots.append(snapshot)
        self._observations.append([])
        self._imu_intervals.append(())
        while len(self._snapshots) > int(self.lag_size):
            self._snapshots.popleft()
            self._observations.popleft()
            self._imu_intervals.popleft()

    def observe_imu_interval(self, *, imu_packets: tuple[IsaacImuPacket, ...]) -> None:
        if not self._imu_intervals:
            return
        self._imu_intervals[-1] = tuple(imu_packets)

    def observe_auxiliary_detections(
        self,
        *,
        detections: tuple[IsaacTagDetectionPacket, ...],
        current_state: FilterStateSnapshot,
        anchor_pose_map: dict[int, TagPoseSpec],
        intrinsics_snapshot: dict[str, Any] | None = None,
    ) -> None:
        del current_state
        self._latest_tag_pose_map = {int(tag_id): tag_pose for tag_id, tag_pose in anchor_pose_map.items()}
        for tag_id, tag_pose in self._latest_tag_pose_map.items():
            if int(tag_id) not in self._tag_pose_estimates:
                self._tag_pose_estimates[int(tag_id)] = TagEstimate(
                    pose_wt=_pose_matrix(
                        np.asarray(tag_pose.position_world_m, dtype=np.float64),
                        np.asarray(tag_pose.rotation_wt, dtype=np.float64),
                    ),
                    size_m=float(tag_pose.size_m),
                    is_anchor=bool(tag_pose.is_anchor),
                    observation_count=0,
                )
        if not self._observations or not isinstance(intrinsics_snapshot, dict):
            return
        entries: list[_WindowObservation] = []
        for detection in detections:
            if detection.is_anchor is False and not self.use_auxiliary_tags:
                continue
            local_tag_points = np.asarray(detection.local_tag_points_m, dtype=np.float64)
            corners_xy = np.asarray(detection.corners_xy, dtype=np.float64)
            if local_tag_points.shape != (4, 3) or corners_xy.shape != (4, 2):
                continue
            entries.append(
                _WindowObservation(
                    tag_id=int(detection.tag_id),
                    is_anchor=bool(detection.is_anchor),
                    corners_xy=corners_xy,
                    local_tag_points_m=local_tag_points,
                    intrinsics_snapshot=dict(intrinsics_snapshot),
                    native_backend="fallback" not in str(detection.detector_backend or "").lower(),
                )
            )
        self._observations[-1].extend(entries)

    def _active_tag_ids(self) -> list[int]:
        tag_ids: set[int] = set()
        for observations in self._observations:
            for observation in observations:
                if observation.is_anchor:
                    continue
                tag_ids.add(int(observation.tag_id))
        return sorted(tag_ids)

    def _initial_tag_pose(self, tag_id: int) -> np.ndarray:
        estimate = self._tag_pose_estimates.get(int(tag_id))
        if estimate is not None:
            return estimate.pose_wt.copy()
        tag_pose = self._latest_tag_pose_map[int(tag_id)]
        return _pose_matrix(
            np.asarray(tag_pose.position_world_m, dtype=np.float64),
            np.asarray(tag_pose.rotation_wt, dtype=np.float64),
        )

    def active_tag_estimates(self, *, min_observation_count: int = 0) -> dict[int, TagEstimate]:
        return {
            int(tag_id): TagEstimate(
                pose_wt=estimate.pose_wt.copy(),
                size_m=float(estimate.size_m),
                is_anchor=bool(estimate.is_anchor),
                observation_count=int(estimate.observation_count),
            )
            for tag_id, estimate in self._tag_pose_estimates.items()
            if estimate.observation_count >= int(min_observation_count)
        }

    def active_tag_pose_specs(self, *, min_observation_count: int = 0) -> dict[int, TagPoseSpec]:
        output: dict[int, TagPoseSpec] = {}
        for tag_id, estimate in self.active_tag_estimates(min_observation_count=min_observation_count).items():
            output[int(tag_id)] = TagPoseSpec(
                tag_id=int(tag_id),
                size_m=float(estimate.size_m),
                position_world_m=tuple(float(value) for value in estimate.pose_wt[:3, 3]),
                rotation_wt=tuple(
                    tuple(float(entry) for entry in row)
                    for row in np.asarray(estimate.pose_wt[:3, :3], dtype=np.float64).tolist()
                ),
                is_anchor=bool(estimate.is_anchor),
            )
        return output

    def _empty_snapshot(self) -> SmootherStateSnapshot:
        covariance = np.eye(6, dtype=np.float64) * 1e-3
        return SmootherStateSnapshot(
            timestamp_s=0.0,
            sim_time_s=0.0,
            active_tag_poses={tag_id: estimate.pose_wt.copy() for tag_id, estimate in self._tag_pose_estimates.items()},
            cloned_positions_world_m=(),
            covariance=covariance,
            cost_trace=tuple(float(value) for value in self._cost_trace),
            diagnostics={
                "backend": self.backend_name,
                "lag_size": float(self.lag_size),
                "active_tag_count": float(len(self._tag_pose_estimates)),
                "refinable_tag_count": float(sum(estimate.observation_count >= 5 for estimate in self._tag_pose_estimates.values())),
                "anchored_window": False,
                "visual_factor_count": 0.0,
                "imu_factor_count": 0.0,
                "residual_rmse_before_px": None,
                "residual_rmse_after_px": None,
                "covariance_trace_before": float(np.trace(covariance)),
                "covariance_trace_after": float(np.trace(covariance)),
                **self._last_feedback_diagnostics,
            },
        )

    def solve(self) -> SmootherStateSnapshot:
        if not self._snapshots:
            return self._empty_snapshot()
        snapshots = list(self._snapshots)
        observations_by_state = list(self._observations)
        imu_intervals = list(self._imu_intervals)
        active_tag_ids = self._active_tag_ids()
        if least_squares is None or not any(observations_by_state):
            latest = snapshots[-1]
            covariance = sanitize_covariance(np.mean([snapshot.covariance[:6, :6] for snapshot in snapshots], axis=0))
            return SmootherStateSnapshot(
                timestamp_s=float(latest.timestamp_s),
                sim_time_s=float(latest.sim_time_s),
                active_tag_poses={tag_id: estimate.pose_wt.copy() for tag_id, estimate in self._tag_pose_estimates.items()},
                cloned_positions_world_m=tuple(snapshot.position_world_m.copy() for snapshot in snapshots),
                covariance=covariance,
                cost_trace=tuple(float(value) for value in self._cost_trace),
                diagnostics={
                    "backend": self.backend_name,
                    "lag_size": float(self.lag_size),
                    "active_tag_count": float(len(self._tag_pose_estimates)),
                    "refinable_tag_count": float(sum(estimate.observation_count >= 5 for estimate in self._tag_pose_estimates.values())),
                    "anchored_window": False,
                    "visual_factor_count": float(sum(len(rows) for rows in observations_by_state)),
                    "imu_factor_count": float(sum(1 for packets in imu_intervals if packets)),
                    "residual_rmse_before_px": None,
                    "residual_rmse_after_px": None,
                    "covariance_trace_before": float(np.trace(covariance)),
                    "covariance_trace_after": float(np.trace(covariance)),
                    **self._last_feedback_diagnostics,
                },
            )

        fused_mode = str(snapshots[-1].mode).strip().lower() == "fused"
        positions0 = [np.asarray(snapshot.position_world_m, dtype=np.float64).reshape(3) for snapshot in snapshots]
        rotations0 = [_rotation_to_rotvec(snapshot.rotation_wi) for snapshot in snapshots]
        velocities0 = [np.asarray(snapshot.velocity_world_mps, dtype=np.float64).reshape(3) for snapshot in snapshots]
        gyro_bias0 = np.mean([np.asarray(snapshot.gyro_bias_rps, dtype=np.float64).reshape(3) for snapshot in snapshots], axis=0)
        accel_bias0 = np.mean([np.asarray(snapshot.accel_bias_mps2, dtype=np.float64).reshape(3) for snapshot in snapshots], axis=0)
        tag_pose0 = {tag_id: self._initial_tag_pose(tag_id) for tag_id in active_tag_ids}
        tag_observation_counts: dict[int, int] = {int(tag_id): 0 for tag_id in active_tag_ids}
        for rows in observations_by_state:
            for observation in rows:
                if not observation.is_anchor:
                    tag_observation_counts[int(observation.tag_id)] = tag_observation_counts.get(int(observation.tag_id), 0) + 1

        x0_parts: list[np.ndarray] = []
        for position, rotation in zip(positions0, rotations0):
            x0_parts.append(position)
            x0_parts.append(rotation)
        if fused_mode:
            for velocity in velocities0:
                x0_parts.append(velocity)
            x0_parts.append(gyro_bias0)
            x0_parts.append(accel_bias0)
        for tag_id in active_tag_ids:
            x0_parts.append(np.asarray(tag_pose0[tag_id][:3, 3], dtype=np.float64).reshape(3))
            x0_parts.append(_rotation_to_rotvec(tag_pose0[tag_id][:3, :3]))
        x0 = np.concatenate(x0_parts) if x0_parts else np.zeros(0, dtype=np.float64)

        num_states = len(snapshots)
        cursor = 0
        state_position_slices: list[slice] = []
        state_rotation_slices: list[slice] = []
        for _ in range(num_states):
            state_position_slices.append(slice(cursor, cursor + 3))
            cursor += 3
            state_rotation_slices.append(slice(cursor, cursor + 3))
            cursor += 3
        state_velocity_slices: list[slice] = []
        if fused_mode:
            for _ in range(num_states):
                state_velocity_slices.append(slice(cursor, cursor + 3))
                cursor += 3
            gyro_bias_slice = slice(cursor, cursor + 3)
            cursor += 3
            accel_bias_slice = slice(cursor, cursor + 3)
            cursor += 3
        else:
            gyro_bias_slice = None
            accel_bias_slice = None
        tag_slices: dict[int, tuple[slice, slice]] = {}
        for tag_id in active_tag_ids:
            position_slice = slice(cursor, cursor + 3)
            cursor += 3
            rotation_slice = slice(cursor, cursor + 3)
            cursor += 3
            tag_slices[int(tag_id)] = (position_slice, rotation_slice)

        initial_covariance = sanitize_covariance(np.mean([snapshot.covariance[:6, :6] for snapshot in snapshots], axis=0))
        covariance_trace_before = float(np.trace(initial_covariance))

        def decode(vector: np.ndarray) -> dict[str, Any]:
            state_positions = [np.asarray(vector[slice_], dtype=np.float64).reshape(3) for slice_ in state_position_slices]
            state_rotations = [_rotvec_to_rotation(vector[slice_]) for slice_ in state_rotation_slices]
            state_velocities = (
                [np.asarray(vector[slice_], dtype=np.float64).reshape(3) for slice_ in state_velocity_slices]
                if fused_mode
                else [np.asarray(snapshot.velocity_world_mps, dtype=np.float64).reshape(3) for snapshot in snapshots]
            )
            gyro_bias = (
                np.asarray(vector[gyro_bias_slice], dtype=np.float64).reshape(3)
                if gyro_bias_slice is not None
                else np.zeros(3, dtype=np.float64)
            )
            accel_bias = (
                np.asarray(vector[accel_bias_slice], dtype=np.float64).reshape(3)
                if accel_bias_slice is not None
                else np.zeros(3, dtype=np.float64)
            )
            tag_poses = {}
            for tag_id in active_tag_ids:
                position_slice, rotation_slice = tag_slices[int(tag_id)]
                pose = np.eye(4, dtype=np.float64)
                pose[:3, 3] = np.asarray(vector[position_slice], dtype=np.float64).reshape(3)
                pose[:3, :3] = _rotvec_to_rotation(vector[rotation_slice])
                tag_poses[int(tag_id)] = pose
            return {
                "positions": state_positions,
                "rotations": state_rotations,
                "velocities": state_velocities,
                "gyro_bias": gyro_bias,
                "accel_bias": accel_bias,
                "tag_poses": tag_poses,
            }

        def visual_errors(vector: np.ndarray) -> list[float]:
            decoded = decode(vector)
            errors: list[float] = []
            for index, observations in enumerate(observations_by_state):
                position_world_m = decoded["positions"][index]
                rotation_wi = decoded["rotations"][index]
                for observation in observations:
                    if observation.is_anchor:
                        tag_spec = self._latest_tag_pose_map.get(int(observation.tag_id))
                        if tag_spec is None:
                            continue
                        tag_pose_wt = _pose_matrix(
                            np.asarray(tag_spec.position_world_m, dtype=np.float64),
                            np.asarray(tag_spec.rotation_wt, dtype=np.float64),
                        )
                    else:
                        tag_pose_wt = decoded["tag_poses"].get(int(observation.tag_id))
                        if tag_pose_wt is None:
                            continue
                    projected = _project_corners(
                        position_world_m=position_world_m,
                        rotation_wi=rotation_wi,
                        tag_pose_wt=tag_pose_wt,
                        local_tag_points_m=observation.local_tag_points_m,
                        intrinsics_snapshot=observation.intrinsics_snapshot,
                    )
                    if projected is None:
                        continue
                    per_corner_error = np.linalg.norm(projected - observation.corners_xy, axis=1)
                    errors.extend(float(value) for value in per_corner_error)
            return errors

        def residuals(vector: np.ndarray) -> np.ndarray:
            decoded = decode(vector)
            pieces: list[np.ndarray] = []
            for index, observations in enumerate(observations_by_state):
                position_world_m = decoded["positions"][index]
                rotation_wi = decoded["rotations"][index]
                initial_position = positions0[index]
                initial_rotation = _rotvec_to_rotation(rotations0[index])
                position_std = self.first_state_position_prior_std_m if index == 0 else self.position_prior_std_m
                rotation_std = self.first_state_rotation_prior_std_rad if index == 0 else self.rotation_prior_std_rad
                pieces.append((position_world_m - initial_position) / max(position_std, 1e-6))
                pieces.append(_rotation_error_rotvec(initial_rotation, rotation_wi) / max(rotation_std, 1e-6))
                if fused_mode:
                    pieces.append((decoded["velocities"][index] - velocities0[index]) / max(self.velocity_prior_std_mps, 1e-6))
                for observation in observations:
                    if observation.is_anchor:
                        tag_spec = self._latest_tag_pose_map.get(int(observation.tag_id))
                        if tag_spec is None:
                            continue
                        tag_pose_wt = _pose_matrix(
                            np.asarray(tag_spec.position_world_m, dtype=np.float64),
                            np.asarray(tag_spec.rotation_wt, dtype=np.float64),
                        )
                    else:
                        tag_pose_wt = decoded["tag_poses"].get(int(observation.tag_id))
                        if tag_pose_wt is None:
                            continue
                    projected = _project_corners(
                        position_world_m=position_world_m,
                        rotation_wi=rotation_wi,
                        tag_pose_wt=tag_pose_wt,
                        local_tag_points_m=observation.local_tag_points_m,
                        intrinsics_snapshot=observation.intrinsics_snapshot,
                    )
                    if projected is None:
                        continue
                    pieces.append(((projected - observation.corners_xy).reshape(-1)) / max(self.visual_residual_scale_px, 1e-6))
            if fused_mode:
                pieces.append((decoded["gyro_bias"] - gyro_bias0) / max(self.bias_prior_std, 1e-6))
                pieces.append((decoded["accel_bias"] - accel_bias0) / max(self.bias_prior_std, 1e-6))
                for index in range(1, num_states):
                    packets = imu_intervals[index]
                    if not packets:
                        continue
                    previous_rotation = decoded["rotations"][index - 1]
                    previous_position = decoded["positions"][index - 1]
                    previous_velocity = decoded["velocities"][index - 1]
                    delta = preintegrate_imu_packets(
                        packets=packets,
                        start_rotation_wi=previous_rotation,
                        start_position_world_m=previous_position,
                        start_velocity_world_mps=previous_velocity,
                        accel_bias_mps2=decoded["accel_bias"],
                        gyro_bias_rps=decoded["gyro_bias"],
                    )
                    pieces.append(
                        _rotation_error_rotvec(delta.final_rotation_wi, decoded["rotations"][index])
                        / max(self.imu_rotation_std_rad, 1e-6)
                    )
                    pieces.append(
                        (decoded["positions"][index] - delta.final_position_world_m) / max(self.imu_position_std_m, 1e-6)
                    )
                    pieces.append(
                        (decoded["velocities"][index] - delta.final_velocity_world_mps) / max(self.imu_velocity_std_mps, 1e-6)
                    )
            for tag_id in active_tag_ids:
                initial_pose = tag_pose0[int(tag_id)]
                tag_pose = decoded["tag_poses"][int(tag_id)]
                pieces.append(
                    (tag_pose[:3, 3] - initial_pose[:3, 3]) / max(self.tag_position_prior_std_m, 1e-6)
                )
                pieces.append(
                    _rotation_error_rotvec(initial_pose[:3, :3], tag_pose[:3, :3]) / max(self.tag_rotation_prior_std_rad, 1e-6)
                )
            if not pieces:
                return np.zeros(0, dtype=np.float64)
            return np.concatenate([np.asarray(piece, dtype=np.float64).reshape(-1) for piece in pieces])

        before_visual_errors = visual_errors(x0)
        before_rmse_px = None if not before_visual_errors else float(np.sqrt(np.mean(np.square(before_visual_errors))))
        result = least_squares(
            residuals,
            x0,
            method="trf",
            loss="huber",
            f_scale=max(self.visual_residual_scale_px, 1.0),
            max_nfev=int(self.solver_max_nfev),
        )
        solution = x0 if not result.success else np.asarray(result.x, dtype=np.float64)
        after_visual_errors = visual_errors(solution)
        after_rmse_px = None if not after_visual_errors else float(np.sqrt(np.mean(np.square(after_visual_errors))))
        self._cost_trace.append(float(result.cost if result.success else np.sum(np.square(residuals(x0)))))
        decoded_solution = decode(solution)
        for tag_id in active_tag_ids:
            pose_wt = decoded_solution["tag_poses"][int(tag_id)]
            previous = self._tag_pose_estimates.get(int(tag_id))
            tag_spec = self._latest_tag_pose_map.get(int(tag_id))
            size_m = float(tag_spec.size_m) if tag_spec is not None else (float(previous.size_m) if previous is not None else 0.0)
            is_anchor = bool(tag_spec.is_anchor) if tag_spec is not None else (bool(previous.is_anchor) if previous is not None else False)
            observation_count = int(tag_observation_counts.get(int(tag_id), 0))
            if previous is not None:
                observation_count = max(observation_count, int(previous.observation_count))
            self._tag_pose_estimates[int(tag_id)] = TagEstimate(
                pose_wt=pose_wt.copy(),
                size_m=size_m,
                is_anchor=is_anchor,
                observation_count=observation_count,
            )
        anchored_window = any(observation.is_anchor for rows in observations_by_state for observation in rows)
        improvement_scale = 1.0
        if before_rmse_px not in (None, 0.0) and after_rmse_px is not None:
            improvement_scale = float(np.clip(after_rmse_px / max(before_rmse_px, 1e-6), 0.2, 4.0))
        covariance = sanitize_covariance(initial_covariance * improvement_scale)
        covariance_trace_after = float(np.trace(covariance))
        latest = snapshots[-1]
        return SmootherStateSnapshot(
            timestamp_s=float(latest.timestamp_s),
            sim_time_s=float(latest.sim_time_s),
            active_tag_poses={tag_id: estimate.pose_wt.copy() for tag_id, estimate in self._tag_pose_estimates.items()},
            cloned_positions_world_m=tuple(position.copy() for position in decoded_solution["positions"]),
            covariance=covariance,
            cost_trace=tuple(float(value) for value in self._cost_trace),
            diagnostics={
                "backend": self.backend_name,
                "lag_size": float(self.lag_size),
                "active_tag_count": float(len(self._tag_pose_estimates)),
                "refinable_tag_count": float(sum(estimate.observation_count >= 5 for estimate in self._tag_pose_estimates.values())),
                "anchored_window": bool(anchored_window),
                "visual_factor_count": float(sum(len(rows) for rows in observations_by_state)),
                "imu_factor_count": float(sum(1 for packets in imu_intervals if packets)),
                "residual_rmse_before_px": before_rmse_px,
                "residual_rmse_after_px": after_rmse_px,
                "covariance_trace_before": covariance_trace_before,
                "covariance_trace_after": covariance_trace_after,
                **self._last_feedback_diagnostics,
            },
        )

    def record_feedback_diagnostics(
        self,
        *,
        feedback_correction_norm_m: float,
        max_feedback_correction_norm_m: float,
        trusted_feedback_count: int,
        total_feedback_candidates: int,
    ) -> None:
        self._last_feedback_diagnostics = {
            "feedback_correction_norm_m": float(feedback_correction_norm_m),
            "max_feedback_correction_norm_m": float(max_feedback_correction_norm_m),
            "trusted_feedback_count": float(trusted_feedback_count),
            "total_feedback_candidates": float(total_feedback_candidates),
        }


__all__ = ["WindowedAnchorBASmoother"]
