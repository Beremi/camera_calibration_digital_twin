"""Bridge helpers.

ROS 2 mirroring is deliberately deferred; the only supported bridge in the
current checkpoint scaffold is the in-process queue.
"""

from calib_sim.isaac.bridge.inproc_queue import InprocQueue

__all__ = ["InprocQueue"]
