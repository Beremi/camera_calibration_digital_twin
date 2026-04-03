"""Motion-preset library.

This module is intentionally functional and reusable. It generates named
trajectories independent of the simulator backend.
"""

from __future__ import annotations

import math
from typing import List

from calib_sim.common.models import MotionPreset, MotionSample


def _sample_count(duration_s: float, sample_rate_hz: float) -> int:
    return max(2, int(round(duration_s * sample_rate_hz)))


def build_snap_yaw(duration_s: float = 1.0, sample_rate_hz: float = 120.0) -> MotionPreset:
    """Short yaw sweep used for detector smoke tests."""
    samples: List[MotionSample] = []
    n = _sample_count(duration_s, sample_rate_hz)
    for i in range(n):
        t = i / sample_rate_hz
        phase = (i / max(1, n - 1)) * math.pi
        yaw = 12.0 * math.sin(phase)
        samples.append(MotionSample(t_s=t, xyz_m=(0.45, 0.0, 0.55), rpy_deg=(0.0, 0.0, yaw)))
    return MotionPreset("snap_yaw", duration_s, sample_rate_hz, samples)


def build_calib_sweep_basic(duration_s: float = 5.0, sample_rate_hz: float = 120.0) -> MotionPreset:
    """Medium-duration pose sweep that excites translation and rotation."""
    samples: List[MotionSample] = []
    n = _sample_count(duration_s, sample_rate_hz)
    for i in range(n):
        t = i / sample_rate_hz
        u = t / max(duration_s, 1e-6)
        x = 0.45 + 0.08 * math.sin(2 * math.pi * u)
        y = 0.05 * math.sin(4 * math.pi * u)
        z = 0.55 + 0.05 * math.cos(2 * math.pi * u)
        roll = 5.0 * math.sin(2 * math.pi * u)
        pitch = 12.0 * math.sin(2 * math.pi * u + math.pi / 3)
        yaw = 15.0 * math.sin(2 * math.pi * u + math.pi / 2)
        samples.append(MotionSample(t_s=t, xyz_m=(x, y, z), rpy_deg=(roll, pitch, yaw)))
    return MotionPreset("calib_sweep_basic", duration_s, sample_rate_hz, samples)


def build_vi_excitation(duration_s: float = 10.0, sample_rate_hz: float = 120.0) -> MotionPreset:
    """Main visual-inertial excitation preset.

    Uses Lissajous-like translation and multi-axis orientation oscillation. The
    exact numbers are intentionally conservative so they are easier to execute on
    a wide range of arms.
    """
    samples: List[MotionSample] = []
    n = _sample_count(duration_s, sample_rate_hz)
    for i in range(n):
        t = i / sample_rate_hz
        x = 0.45 + 0.10 * math.sin(2 * math.pi * 0.8 * t)
        y = 0.00 + 0.08 * math.sin(2 * math.pi * 1.1 * t + math.pi / 4)
        z = 0.55 + 0.06 * math.sin(2 * math.pi * 1.4 * t + math.pi / 2)
        roll = 10.0 * math.sin(2 * math.pi * 0.7 * t)
        pitch = 18.0 * math.sin(2 * math.pi * 1.0 * t + math.pi / 5)
        yaw = 22.0 * math.sin(2 * math.pi * 1.3 * t + math.pi / 7)
        samples.append(MotionSample(t_s=t, xyz_m=(x, y, z), rpy_deg=(roll, pitch, yaw)))
    return MotionPreset("vi_excitation", duration_s, sample_rate_hz, samples)


def build_preset(preset_id: str, duration_s: float, sample_rate_hz: float = 120.0) -> MotionPreset:
    """Factory function used by the API layer."""
    if preset_id == "snap_yaw":
        return build_snap_yaw(duration_s=duration_s, sample_rate_hz=sample_rate_hz)
    if preset_id == "calib_sweep_basic":
        return build_calib_sweep_basic(duration_s=duration_s, sample_rate_hz=sample_rate_hz)
    if preset_id == "vi_excitation":
        return build_vi_excitation(duration_s=duration_s, sample_rate_hz=sample_rate_hz)
    raise ValueError(f"Unknown preset_id={preset_id!r}")
