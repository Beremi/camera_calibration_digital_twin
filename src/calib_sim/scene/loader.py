"""Scene loading and room-asset management."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class SceneConfig:
    """Resolved room-scene configuration."""

    name: str
    asset_path: str
    metadata: Dict[str, Any]


class SceneLoader:
    """Loads furnished rooms and applies scene-level metadata.

    In a real Isaac Sim integration this class would:
    - open or reference a USD room asset,
    - apply lighting presets,
    - create named anchor frames for tags/cameras,
    - expose the loaded root prim path.
    """

    def __init__(self) -> None:
        self.current_scene: SceneConfig | None = None

    def load(self, config: SceneConfig) -> SceneConfig:
        """Record the desired scene as loaded.

        The method is deliberately lightweight here. The important design point is
        that the scene is a *config-driven object* rather than a hard-coded file.
        """
        if not config.asset_path:
            raise ValueError("Scene asset path must not be empty.")
        self.current_scene = config
        return config
