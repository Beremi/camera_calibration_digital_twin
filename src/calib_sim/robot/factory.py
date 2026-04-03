"""Robot asset abstraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from calib_sim.common.models import RobotConfig


@dataclass
class LoadedRobot:
    """Normalized loaded-robot handle.

    In the full simulator, this would also carry the articulation prim path and
    controller handles created by Isaac Sim.
    """

    config: RobotConfig
    prim_path: str


class RobotFactory:
    """Loads robot assets from configuration rather than from hard-coded logic."""

    def load(self, config: RobotConfig) -> LoadedRobot:
        """Load a robot into the scene.

        This scaffold returns a normalized handle only. Replace the body with:
        - USD reference loading,
        - URDF import path,
        - articulation-root discovery,
        - joint-order validation.
        """
        prim_path = f"/World/Robots/{config.name}"
        return LoadedRobot(config=config, prim_path=prim_path)
