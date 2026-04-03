"""rosbag2 / MCAP recording helper."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List


@dataclass
class RecorderConfig:
    """Minimal rosbag recorder configuration."""

    output_dir: str
    topics: List[str] = field(default_factory=list)
    storage_id: str = "mcap"


class RosbagRecorder:
    """Builds the rosbag2 recording command.

    This file keeps the recording policy centralized so it can be reused by:
    - the API layer,
    - CLI tools,
    - automated tests.
    """

    def __init__(self, config: RecorderConfig) -> None:
        self.config = config

    def build_command(self) -> list[str]:
        """Return the ros2 bag CLI command as a tokenized list."""
        cmd = ["ros2", "bag", "record", "--storage", self.config.storage_id, "-o", self.config.output_dir]
        cmd.extend(self.config.topics)
        return cmd
