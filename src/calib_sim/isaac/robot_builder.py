"""Robot-mount specifications and live bindings for the Isaac runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class RobotMountSpec:
    robot_name: str
    articulation_prim_path: str
    ee_frame: str
    camera_mount_prim_path: str
    imu_mount_prim_path: str
    joint_names: tuple[str, ...] = field(default_factory=tuple)
    control_mode: str = "position"
    asset_mode: str = "usd"
    asset_path: str = ""
    base_position_world_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_orientation_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    default_joint_positions_deg: tuple[float, ...] = ()
    ik_robot_name: str = "Franka"
    ik_frame_name: str = "panda_hand"


@dataclass(slots=True)
class LiveRobotBinding:
    robot: RobotMountSpec
    articulation: Any
    end_effector_prim: Any
    articulation_kinematics_solver: Any | None = None
    base_frame: str = "B"
    control_mode: str = "position"

    def summary(self) -> dict[str, object]:
        return {
            "robot_name": self.robot.robot_name,
            "articulation_prim_path": self.robot.articulation_prim_path,
            "ee_frame": self.robot.ee_frame,
            "camera_mount_prim_path": self.robot.camera_mount_prim_path,
            "imu_mount_prim_path": self.robot.imu_mount_prim_path,
            "joint_names": list(self.robot.joint_names),
            "base_frame": self.base_frame,
            "control_mode": self.control_mode,
            "has_articulation_ik": self.articulation_kinematics_solver is not None,
        }

    def set_default_joint_positions(self) -> None:
        if not self.robot.default_joint_positions_deg:
            return
        current = self.get_joint_positions()
        radians = np.radians(np.asarray(self.robot.default_joint_positions_deg, dtype=np.float64))
        target = current.copy()
        limit = min(len(target), len(radians))
        target[:limit] = radians[:limit]
        self.articulation.set_joint_positions(target)

    def get_joint_positions(self) -> np.ndarray:
        return np.asarray(self.articulation.get_joint_positions(), dtype=np.float64)

    def get_joint_velocities(self) -> np.ndarray:
        return np.asarray(self.articulation.get_joint_velocities(), dtype=np.float64)

    def joint_limit_margin(self, joint_positions: np.ndarray) -> float | None:
        positions = np.asarray(joint_positions, dtype=np.float64).reshape(-1)
        lower: np.ndarray | None = None
        upper: np.ndarray | None = None
        try:
            if hasattr(self.articulation, "dof_properties"):
                dof_properties = self.articulation.dof_properties
                if isinstance(dof_properties, dict):
                    if "lower" in dof_properties and "upper" in dof_properties:
                        lower = np.asarray(dof_properties["lower"], dtype=np.float64).reshape(-1)
                        upper = np.asarray(dof_properties["upper"], dtype=np.float64).reshape(-1)
            if (lower is None or upper is None) and hasattr(self.articulation, "get_dof_limits"):
                limits = np.asarray(self.articulation.get_dof_limits(), dtype=np.float64)
                if limits.ndim == 2 and limits.shape[-1] == 2:
                    lower = limits[:, 0].reshape(-1)
                    upper = limits[:, 1].reshape(-1)
            if (lower is None or upper is None) and hasattr(self.articulation, "get_joint_limits"):
                limits = np.asarray(self.articulation.get_joint_limits(), dtype=np.float64)
                if limits.ndim == 2 and limits.shape[-1] == 2:
                    lower = limits[:, 0].reshape(-1)
                    upper = limits[:, 1].reshape(-1)
        except Exception:
            lower = None
            upper = None
        if lower is None or upper is None:
            return None
        limit = min(len(positions), len(lower), len(upper))
        if limit <= 0:
            return None
        lower_margin = positions[:limit] - lower[:limit]
        upper_margin = upper[:limit] - positions[:limit]
        return float(np.min(np.minimum(lower_margin, upper_margin)))

    def get_end_effector_pose(self) -> tuple[np.ndarray, np.ndarray]:
        position, orientation = self.end_effector_prim.get_world_pose()
        return np.asarray(position, dtype=np.float64), np.asarray(orientation, dtype=np.float64)

    def compute_joint_targets_from_pose(
        self,
        *,
        target_position_world_m: np.ndarray,
        target_orientation_wxyz: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        if self.articulation_kinematics_solver is None:
            return self.get_joint_positions(), False
        base_position, base_orientation = self.articulation.get_world_pose()
        self.articulation_kinematics_solver.get_kinematics_solver().set_robot_base_pose(base_position, base_orientation)
        action, success = self.articulation_kinematics_solver.compute_inverse_kinematics(
            np.asarray(target_position_world_m, dtype=np.float64),
            np.asarray(target_orientation_wxyz, dtype=np.float64),
        )
        if not success:
            return self.get_joint_positions(), False
        joint_positions = getattr(action, "joint_positions", None)
        if joint_positions is None:
            return self.get_joint_positions(), False
        joint_positions = np.asarray(joint_positions, dtype=np.float64).reshape(-1)
        current = self.get_joint_positions()
        if joint_positions.size < current.size:
            padded = current.copy()
            padded[: joint_positions.size] = joint_positions
            joint_positions = padded
        elif joint_positions.size > current.size:
            joint_positions = joint_positions[: current.size]
        return joint_positions, True


def robot_mount_spec_from_config(robot_config: dict[str, Any]) -> RobotMountSpec:
    return RobotMountSpec(
        robot_name=str(robot_config.get("robot_name", robot_config.get("name", "robot"))),
        articulation_prim_path=str(robot_config["articulation_prim_path"]),
        ee_frame=str(robot_config["ee_frame"]),
        camera_mount_prim_path=str(robot_config["camera_mount_prim_path"]),
        imu_mount_prim_path=str(robot_config["imu_mount_prim_path"]),
        joint_names=tuple(str(value) for value in robot_config.get("joint_names", [])),
        control_mode=str(robot_config.get("control_mode", "position")),
        asset_mode=str(robot_config.get("asset_mode", robot_config.get("asset", {}).get("mode", "usd"))),
        asset_path=str(robot_config.get("asset_path", robot_config.get("asset", {}).get("path", ""))),
        base_position_world_m=tuple(float(value) for value in robot_config.get("base_position_world_m", (0.0, 0.0, 0.0))),
        base_orientation_rpy_deg=tuple(float(value) for value in robot_config.get("base_orientation_rpy_deg", (0.0, 0.0, 0.0))),
        default_joint_positions_deg=tuple(float(value) for value in robot_config.get("default_joint_positions_deg", ())),
        ik_robot_name=str(robot_config.get("ik_robot_name", "Franka")),
        ik_frame_name=str(robot_config.get("ik_frame_name", robot_config.get("ee_frame", "panda_hand"))),
    )


def build_robot_binding(*, world: Any, robot_config: dict[str, Any]) -> LiveRobotBinding:
    """Create or bind the first-pass Franka/UR articulation."""

    from isaacsim.core.api.robots import Robot
    from isaacsim.core.prims import SingleRigidPrim
    from isaacsim.core.utils.extensions import enable_extension
    from isaacsim.core.utils.prims import is_prim_path_valid
    from isaacsim.core.utils.rotations import euler_angles_to_quat
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver, interface_config_loader
    from isaacsim.storage.native import get_assets_root_path
    from pxr import UsdGeom

    spec = robot_mount_spec_from_config(robot_config)
    if spec.asset_mode != "usd":
        raise ValueError(f"First-pass live Isaac runtime only supports USD robot assets, got {spec.asset_mode!r}.")

    asset_path = spec.asset_path
    if asset_path.startswith("Isaac/"):
        assets_root = get_assets_root_path()
        if assets_root is None:
            raise RuntimeError("Could not resolve the Isaac Sim built-in assets root.")
        asset_path = assets_root.rstrip("/") + "/" + asset_path

    if not is_prim_path_valid(spec.articulation_prim_path):
        add_reference_to_stage(usd_path=asset_path, prim_path=spec.articulation_prim_path)

    robot = Robot(
        prim_path=spec.articulation_prim_path,
        name=spec.robot_name,
        position=np.asarray(spec.base_position_world_m, dtype=np.float64),
        orientation=euler_angles_to_quat(np.asarray(spec.base_orientation_rpy_deg, dtype=np.float64), degrees=True),
    )
    world.scene.add(robot)

    stage = world.stage
    ee_prim_path = f"{spec.articulation_prim_path}/{spec.ee_frame}"
    for mount_path in (spec.camera_mount_prim_path, spec.imu_mount_prim_path):
        parent_path = mount_path.rsplit("/", 1)[0]
        if parent_path and not is_prim_path_valid(parent_path):
            UsdGeom.Xform.Define(stage, parent_path)
        if mount_path and not is_prim_path_valid(mount_path):
            UsdGeom.Xform.Define(stage, mount_path)
    end_effector = SingleRigidPrim(prim_path=ee_prim_path, name=f"{spec.robot_name}_ee")

    articulation_ik = None
    try:
        enable_extension("isaacsim.robot_motion.motion_generation")
        ik_config = interface_config_loader.load_supported_lula_kinematics_solver_config(spec.ik_robot_name)
        if ik_config:
            lula_solver = LulaKinematicsSolver(**ik_config)
            articulation_ik = ArticulationKinematicsSolver(robot, lula_solver, spec.ik_frame_name)
    except Exception:
        articulation_ik = None

    return LiveRobotBinding(
        robot=spec,
        articulation=robot,
        end_effector_prim=end_effector,
        articulation_kinematics_solver=articulation_ik,
        control_mode=spec.control_mode,
    )


__all__ = [
    "LiveRobotBinding",
    "RobotMountSpec",
    "build_robot_binding",
    "robot_mount_spec_from_config",
]
