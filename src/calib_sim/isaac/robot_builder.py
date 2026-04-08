"""Robot-mount specifications for the Isaac runtime."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class RobotMountSpec:
    robot_name: str
    articulation_prim_path: str
    ee_frame: str
    camera_mount_prim_path: str
    imu_mount_prim_path: str
    joint_names: tuple[str, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class RobotStageBinding:
    robot: RobotMountSpec
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
        }
