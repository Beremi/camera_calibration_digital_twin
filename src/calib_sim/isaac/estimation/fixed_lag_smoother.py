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
class TagEstimate:
    pose_wt: np.ndarray
    size_m: float
    is_anchor: bool
    observation_count: int = 0


@dataclass(slots=True)
class FixedLagSmoother:
    lag_size: int = 30
    use_auxiliary_tags: bool = True
    _snapshots: deque[FilterStateSnapshot] = field(default_factory=deque)
    _tag_pose_estimates: dict[int, TagEstimate] = field(default_factory=dict)
    _tag_update_counts: dict[int, int] = field(default_factory=dict)
    _cost_trace: deque[float] = field(default_factory=lambda: deque(maxlen=64))
    _last_feedback_diagnostics: dict[str, float] = field(default_factory=dict)

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
        if not self.use_auxiliary_tags:
            for tag_id, tag_pose in anchor_pose_map.items():
                if tag_id not in self._tag_pose_estimates:
                    self._tag_pose_estimates[int(tag_id)] = TagEstimate(
                        pose_wt=_pose_matrix(
                            np.asarray(tag_pose.position_world_m, dtype=np.float64),
                            np.asarray(tag_pose.rotation_wt, dtype=np.float64),
                        ),
                        size_m=float(tag_pose.size_m),
                        is_anchor=bool(tag_pose.is_anchor),
                    )
            return
        rotation_wi = np.asarray(current_state.rotation_wi, dtype=np.float64)
        position_world_m = np.asarray(current_state.position_world_m, dtype=np.float64)
        for detection in detections:
            if (
                detection.is_anchor
                or detection.pose_camera_rvec is None
                or detection.pose_camera_tvec_m is None
                or not detection.detector_backend
                or "fallback" in detection.detector_backend
            ):
                continue
            tag_points = np.asarray(detection.local_tag_points_m, dtype=np.float64)
            tag_center_camera = np.asarray(detection.pose_camera_tvec_m, dtype=np.float64).reshape(3)
            tag_center_world = position_world_m + rotation_wi @ tag_center_camera
            pose_wt = _pose_matrix(tag_center_world, rotation_wi)
            tag_spec = anchor_pose_map.get(int(detection.tag_id))
            if tag_spec is None:
                continue
            self.refine_tag_pose(
                tag_id=int(detection.tag_id),
                pose_wt=pose_wt,
                size_m=float(tag_spec.size_m),
                is_anchor=bool(tag_spec.is_anchor),
            )
        for tag_id, tag_pose in anchor_pose_map.items():
            if tag_id not in self._tag_pose_estimates:
                self._tag_pose_estimates[int(tag_id)] = TagEstimate(
                    pose_wt=_pose_matrix(
                        np.asarray(tag_pose.position_world_m, dtype=np.float64),
                        np.asarray(tag_pose.rotation_wt, dtype=np.float64),
                    ),
                    size_m=float(tag_pose.size_m),
                    is_anchor=bool(tag_pose.is_anchor),
                )

    def refine_tag_pose(self, *, tag_id: int, pose_wt: np.ndarray, size_m: float, is_anchor: bool) -> None:
        pose_wt = np.asarray(pose_wt, dtype=np.float64).reshape(4, 4)
        previous = self._tag_pose_estimates.get(int(tag_id))
        count = self._tag_update_counts.get(int(tag_id), 0)
        if previous is None:
            self._tag_pose_estimates[int(tag_id)] = TagEstimate(
                pose_wt=pose_wt,
                size_m=float(size_m),
                is_anchor=bool(is_anchor),
                observation_count=1,
            )
            self._tag_update_counts[int(tag_id)] = 1
            return
        blend = 1.0 / float(count + 1)
        previous.pose_wt = (1.0 - blend) * previous.pose_wt + blend * pose_wt
        previous.size_m = float(size_m)
        previous.is_anchor = bool(is_anchor)
        previous.observation_count += 1
        self._tag_update_counts[int(tag_id)] = count + 1

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
            pose = estimate.pose_wt
            output[int(tag_id)] = TagPoseSpec(
                tag_id=int(tag_id),
                size_m=float(estimate.size_m),
                position_world_m=tuple(float(value) for value in pose[:3, 3]),
                rotation_wt=tuple(tuple(float(entry) for entry in row) for row in pose[:3, :3].tolist()),
                is_anchor=bool(estimate.is_anchor),
            )
        return output

    def solve(self) -> SmootherStateSnapshot:
        if not self._snapshots:
            covariance = np.eye(6, dtype=np.float64) * 1e-3
            return SmootherStateSnapshot(
                timestamp_s=0.0,
                sim_time_s=0.0,
                active_tag_poses={tag_id: estimate.pose_wt.copy() for tag_id, estimate in self._tag_pose_estimates.items()},
                cloned_positions_world_m=(),
                covariance=covariance,
                cost_trace=tuple(float(value) for value in self._cost_trace),
                diagnostics={"lag_size": float(self.lag_size), **self._last_feedback_diagnostics},
            )
        latest = self._snapshots[-1]
        positions = tuple(snapshot.position_world_m.copy() for snapshot in self._snapshots)
        covariance = sanitize_covariance(np.mean([snapshot.covariance[:6, :6] for snapshot in self._snapshots], axis=0))
        return SmootherStateSnapshot(
            timestamp_s=float(latest.timestamp_s),
            sim_time_s=float(latest.sim_time_s),
            active_tag_poses={tag_id: estimate.pose_wt.copy() for tag_id, estimate in self._tag_pose_estimates.items()},
            cloned_positions_world_m=positions,
            covariance=covariance,
            cost_trace=tuple(float(value) for value in self._cost_trace),
            diagnostics={
                "lag_size": float(self.lag_size),
                "active_tag_count": float(len(self._tag_pose_estimates)),
                "refinable_tag_count": float(sum(estimate.observation_count >= 5 for estimate in self._tag_pose_estimates.values())),
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


__all__ = ["FixedLagSmoother", "TagEstimate"]
