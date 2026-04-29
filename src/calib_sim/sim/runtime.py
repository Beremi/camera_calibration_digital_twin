"""Simulation runtime bootstrap.

This file intentionally keeps the Isaac Sim dependency behind a soft import so
the rest of the repository can be inspected or unit-tested without Isaac being
installed in the current environment.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

from calib_sim.sim.isaac_compat import prepare_isaac_runtime_environment


prepare_isaac_runtime_environment()

try:  # pragma: no cover - Isaac Sim is not expected in basic unit tests.
    from isaacsim import SimulationApp  # type: ignore
except Exception:  # pragma: no cover
    SimulationApp = None  # type: ignore


@dataclass
class RuntimeConfig:
    """Top-level runtime options."""

    headless: bool = True
    width: int = 1280
    height: int = 720
    streaming_backend: str = "webrtc"
    webrtc_signal_port: int = 49110
    webrtc_stream_port: int = 47990
    webrtc_target_fps: int = 60
    webrtc_allow_dynamic_resize: bool = True
    webrtc_public_ip: str = ""
    limit_cpu_threads: int = 32
    disable_viewport_updates: bool = False


class SimulationRuntime:
    """Thin wrapper around the simulator process.

    In production, this class would own:
    - SimulationApp startup/shutdown
    - physics timestep configuration
    - stage load/reset
    - the per-tick callback pipeline
    """

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.app = None
        self.started = False

    def start(self) -> None:
        """Start Isaac Sim.

        We fail loudly if Isaac is not installed, because silent degradation here
        would make debugging confusing.
        """
        if SimulationApp is None:
            raise RuntimeError(
                "Isaac Sim Python modules are not available in this environment. "
                "Install Isaac Sim 6.0 and run this module from its Python environment."
            )
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

        self.app = SimulationApp(
            {
                "headless": self.config.headless,
                "width": self.config.width,
                "height": self.config.height,
                "limit_cpu_threads": int(self.config.limit_cpu_threads),
                "disable_viewport_updates": bool(self.config.disable_viewport_updates),
            }
        )
        self._enable_standard_streaming_if_requested()
        self.started = True

    def _enable_standard_streaming_if_requested(self) -> None:
        backend = str(self.config.streaming_backend).strip().lower()
        if backend != "webrtc":
            return
        import carb.settings
        from isaacsim.core.utils.extensions import enable_extension

        settings = carb.settings.get_settings()
        settings.set_string("exts/omni.kit.livestream.app/primaryStream/streamType", "webrtc")
        settings.set_int("exts/omni.kit.livestream.app/primaryStream/signalPort", int(self.config.webrtc_signal_port))
        settings.set_int("exts/omni.kit.livestream.app/primaryStream/streamPort", int(self.config.webrtc_stream_port))
        settings.set_int("exts/omni.kit.livestream.app/primaryStream/targetFps", int(self.config.webrtc_target_fps))
        settings.set_bool(
            "exts/omni.kit.livestream.app/primaryStream/allowDynamicResize",
            bool(self.config.webrtc_allow_dynamic_resize),
        )
        settings.set_string("exts/omni.kit.livestream.app/primaryStream/publicIp", str(self.config.webrtc_public_ip))
        enable_extension("omni.kit.livestream.webrtc")
        enable_extension("omni.kit.livestream.app")

    def step(self) -> None:
        """Advance one simulator frame.

        Replace this placeholder with the actual world stepping / callback logic
        once the Isaac integration layer is connected.
        """
        if not self.started or self.app is None:
            raise RuntimeError("Simulation runtime has not been started yet.")
        self.app.update()

    def shutdown(self) -> None:
        """Cleanly terminate the simulator process."""
        if self.app is not None:
            self.app.close()
        self.started = False
