"""In-process command buffering for the standalone runtime."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from calib_sim.isaac.logging.schemas import IsaacJointCommandPacket


@dataclass(slots=True)
class CommandBusStats:
    published: int = 0
    consumed: int = 0


class CommandBus:
    def __init__(self) -> None:
        self._queue: deque[IsaacJointCommandPacket] = deque()
        self.stats = CommandBusStats()

    def publish(self, packet: IsaacJointCommandPacket) -> None:
        self._queue.append(packet)
        self.stats.published += 1

    def drain(self) -> list[IsaacJointCommandPacket]:
        packets = list(self._queue)
        self._queue.clear()
        self.stats.consumed += len(packets)
        return packets
