"""Static stage description helpers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class StageTagSpec:
    tag_id: int
    prim_path: str
    size_m: float
    is_anchor: bool = False


@dataclass(slots=True)
class IsaacStageSpec:
    """Repository-level description of the intended Isaac stage."""

    stage_path: str
    robot_prim_path: str
    camera_prim_path: str
    imu_prim_path: str
    physics_rate_hz: float
    observer_camera_prim_paths: tuple[str, ...] = ()
    tags: tuple[StageTagSpec, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        if self.physics_rate_hz <= 0.0:
            raise ValueError("physics_rate_hz must be positive.")
        if not any(tag.is_anchor for tag in self.tags):
            raise ValueError("Exactly one anchor tag is required in the stage spec.")
        if sum(1 for tag in self.tags if tag.is_anchor) != 1:
            raise ValueError("Stage spec must contain exactly one anchor tag.")
