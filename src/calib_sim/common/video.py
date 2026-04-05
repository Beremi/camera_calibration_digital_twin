"""Helpers for writing MP4 files that preview cleanly in VS Code."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(slots=True)
class FinalizedVideo:
    """Summary of how an MP4 file was finalized on disk."""

    path: Path
    codec: str
    preview_compatible: bool


class ManagedMp4Writer:
    """Write frames with OpenCV, then finalize to a browser-friendly MP4."""

    def __init__(self, path: str | Path, *, fps: float, frame_size: tuple[int, int]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(
            str(self.path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(max(fps, 1.0)),
            (int(frame_size[0]), int(frame_size[1])),
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"Could not open MP4 writer for {self.path}")

    def write(self, frame_bgr: np.ndarray) -> None:
        if self._writer is None:
            raise RuntimeError(f"Cannot write frame after closing {self.path}")
        self._writer.write(frame_bgr)

    def close(self) -> FinalizedVideo:
        if self._writer is None:
            return FinalizedVideo(path=self.path, codec="unknown", preview_compatible=self.path.exists())

        self._writer.release()
        self._writer = None
        return finalize_mp4(self.path)


def finalize_mp4(path: str | Path) -> FinalizedVideo:
    """Finalize an OpenCV MP4 into H.264 when ffmpeg is available."""

    output = Path(path)
    if not output.exists():
        raise FileNotFoundError(f"Expected MP4 was not written: {output}")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        transcoded = output.with_name(f"{output.stem}.finalizing{output.suffix}")
        command = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(output),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-preset",
            "veryfast",
            str(transcoded),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode == 0 and transcoded.exists() and transcoded.stat().st_size > 0:
            transcoded.replace(output)
            return FinalizedVideo(path=output, codec="h264", preview_compatible=True)
        transcoded.unlink(missing_ok=True)

    return FinalizedVideo(path=output, codec="mpeg4", preview_compatible=False)
