"""Fixed-lag smoother over recent filter snapshots and tag-map estimates."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from calib_sim.isaac.estimation.state_defs import FilterStateSnapshot, SmootherStateSnapshot
from calib_sim.isaac.estimation.uncertainty import sanitize_covariance
from calib_sim.isaac.logging.schemas import IsaacTagDetectionPacket
from calib_sim.isaac.tag_builder import TagPoseSpec


def _pose_matrix(position_world_m: np.ndarray, rotation_wt: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(rotation_wt, dtype=np.float64).reshape(3, 3)
    matrix[:3, 3] = np.asarray(position_world_m, dtype=np.float64).reshape(3)
    return matrix


@dataclass(slots=True)
class FixedLagSmoother:
    lag_size: int = 30
    _snapshots: deque[FilterStateSnapshot] = field(default_factory=deque)
    _tag_pose_estimates: dict[int, np.ndarray] = field(default_factory=dict)
    _tag_update_counts: dict[int, int] = field(default_factory=dict)
    _cost_trace: deque[float] = field(default_factory=lambda: deque(maxlen=64))

    def push_snapshot(self, snapshot: FilterStateSnapshot) -> None:
        self._snapshots.append(snapshot)
        while len(self._snapshots) > int(self.lag_size):
            self._snapshots.popleft()
        innovation = float(snapshot.innovation_diagnostics.get("last_innovation_norm", 0.0))
        self._cost_trace.append(innovation**2)

    def observe_auxiliary_detections(
        self,
        *,
        detections: tuple[IsaacTagDetectionPacket, ...],
        current_state: FilterStateSnapshot,
        anchor_pose_map: dict[int, TagPoseSpec],
    ) -> None:
        rotation_wi = np.asarray(current_state.rotation_wi, dtype=np.float64)
        position_world_m = np.asarray(current_state.position_world_m, dtype=np.float64)
        for detection in detections:
            if detection.is_anchor or detection.pose_camera_rvec is None or detection.pose_camera_tvec_m is None:
                continue
            tag_points = np.asarray(detection.local_tag_points_m, dtype=np.float64)
            tag_center_camera = np.asarray(detection.pose_camera_tvec_m, dtype=np.float64).reshape(3)
            tag_center_world = position_world_m + rotation_wi @ tag_center_camera
            pose_wt = _pose_matrix(tag_center_world, rotation_wi)
            self.refine_tag_pose(tag_id=int(detection.tag_id), pose_wt=pose_wt)
        for tag_id, tag_pose in anchor_pose_map.items():
            if tag_id not in self._tag_pose_estimates:
                self._tag_pose_estimates[int(tag_id)] = _pose_matrix(
                    np.asarray(tag_pose.position_world_m, dtype=np.float64),
                    np.asarray(tag_pose.rotation_wt, dtype=np.float64),
                )

    def refine_tag_pose(self, *, tag_id: int, pose_wt: np.ndarray) -> None:
        pose_wt = np.asarray(pose_wt, dtype=np.float64).reshape(4, 4)
        previous = self._tag_pose_estimates.get(int(tag_id))
        count = self._tag_update_counts.get(int(tag_id), 0)
        if previous is None:
            self._tag_pose_estimates[int(tag_id)] = pose_wt
            self._tag_update_counts[int(tag_id)] = 1
            return
        blend = 1.0 / float(count + 1)
        self._tag_pose_estimates[int(tag_id)] = (1.0 - blend) * previous + blend * pose_wt
        self._tag_update_counts[int(tag_id)] = count + 1

    def active_tag_pose_specs(self) -> dict[int, TagPoseSpec]:
        output: dict[int, TagPoseSpec] = {}
        for tag_id, pose in self._tag_pose_estimates.items():
            output[int(tag_id)] = TagPoseSpec(
                tag_id=int(tag_id),
                size_m=0.0,
                position_world_m=tuple(float(value) for value in pose[:3, 3]),
                rotation_wt=tuple(tuple(float(entry) for entry in row) for row in pose[:3, :3].tolist()),
                is_anchor=False,
            )
        return output

    def solve(self) -> SmootherStateSnapshot:
        if not self._snapshots:
            covariance = np.eye(6, dtype=np.float64) * 1e-3
            return SmootherStateSnapshot(
                timestamp_s=0.0,
                sim_time_s=0.0,
                active_tag_poses=dict(self._tag_pose_estimates),
                cloned_positions_world_m=(),
                covariance=covariance,
                cost_trace=tuple(float(value) for value in self._cost_trace),
                diagnostics={"lag_size": float(self.lag_size)},
            )
        latest = self._snapshots[-1]
        positions = tuple(snapshot.position_world_m.copy() for snapshot in self._snapshots)
        covariance = sanitize_covariance(np.mean([snapshot.covariance[:6, :6] for snapshot in self._snapshots], axis=0))
        return SmootherStateSnapshot(
            timestamp_s=float(latest.timestamp_s),
            sim_time_s=float(latest.sim_time_s),
            active_tag_poses={tag_id: pose.copy() for tag_id, pose in self._tag_pose_estimates.items()},
            cloned_positions_world_m=positions,
            covariance=covariance,
            cost_trace=tuple(float(value) for value in self._cost_trace),
            diagnostics={
                "lag_size": float(self.lag_size),
                "active_tag_count": float(len(self._tag_pose_estimates)),
            },
        )


__all__ = ["FixedLagSmoother"]
