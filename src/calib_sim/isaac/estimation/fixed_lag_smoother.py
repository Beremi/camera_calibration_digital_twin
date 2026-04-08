"""Fixed-lag smoother over recent filter snapshots and tag-map estimates."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from calib_sim.isaac.estimation.state_defs import FilterStateSnapshot, SmootherStateSnapshot
from calib_sim.isaac.estimation.uncertainty import sanitize_covariance


@dataclass(slots=True)
class FixedLagSmoother:
    lag_size: int = 30
    _snapshots: deque[FilterStateSnapshot] = field(default_factory=deque)
    _tag_pose_estimates: dict[int, np.ndarray] = field(default_factory=dict)

    def push_snapshot(self, snapshot: FilterStateSnapshot) -> None:
        self._snapshots.append(snapshot)
        while len(self._snapshots) > int(self.lag_size):
            self._snapshots.popleft()

    def refine_tag_pose(self, *, tag_id: int, pose_wt: np.ndarray) -> None:
        pose_wt = np.asarray(pose_wt, dtype=np.float64).reshape(4, 4)
        previous = self._tag_pose_estimates.get(int(tag_id))
        if previous is None:
            self._tag_pose_estimates[int(tag_id)] = pose_wt
            return
        self._tag_pose_estimates[int(tag_id)] = 0.5 * (previous + pose_wt)

    def solve(self) -> SmootherStateSnapshot:
        if not self._snapshots:
            covariance = np.eye(6, dtype=np.float64) * 1e-3
            return SmootherStateSnapshot(
                timestamp_s=0.0,
                sim_time_s=0.0,
                active_tag_poses=dict(self._tag_pose_estimates),
                cloned_positions_world_m=(),
                covariance=covariance,
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
        )
