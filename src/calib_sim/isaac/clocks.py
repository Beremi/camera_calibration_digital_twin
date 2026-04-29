"""Clock helpers for deterministic Isaac logging and replay."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TimestampTriplet:
    sim_time_s: float
    sensor_time_s: float
    host_time_s: float

    def summary(self) -> dict[str, float]:
        return {
            "sim_time_s": float(self.sim_time_s),
            "sensor_time_s": float(self.sensor_time_s),
            "host_time_s": float(self.host_time_s),
        }


@dataclass(slots=True)
class ScheduledSensorTick:
    sample_index: int
    sim_time_s: float
    dt_s: float


class FixedRateClock:
    """Small fixed-rate scheduler used by tests and replay."""

    def __init__(self, *, rate_hz: float, start_time_s: float = 0.0) -> None:
        if rate_hz <= 0.0:
            raise ValueError("rate_hz must be positive.")
        self.rate_hz = float(rate_hz)
        self.period_s = 1.0 / self.rate_hz
        self.next_time_s = float(start_time_s) + self.period_s
        self.sample_index = 0

    def advance_to(self, sim_time_s: float) -> list[ScheduledSensorTick]:
        due: list[ScheduledSensorTick] = []
        while self.next_time_s <= float(sim_time_s) + 1e-12:
            due.append(
                ScheduledSensorTick(
                    sample_index=self.sample_index,
                    sim_time_s=float(self.next_time_s),
                    dt_s=float(self.period_s),
                )
            )
            self.sample_index += 1
            self.next_time_s += self.period_s
        return due
