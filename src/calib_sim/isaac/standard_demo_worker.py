"""Worker orchestration for the slim tabletop-only standard Isaac demo."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timezone
import multiprocessing as mp
import os
from pathlib import Path
import queue
import threading
import time
from typing import Any

import numpy as np

from calib_sim.isaac.demo_profiles import DEFAULT_PROFILE_KEY, IsaacDemoProfile, get_demo_profile, list_demo_profiles
from calib_sim.isaac.demo_protocol import validate_control_mode, validate_noise_modes
from calib_sim.isaac.demo_worker import (
    OBSERVER_STREAM_JPEG_QUALITY,
    OBSERVER_STREAM_MAX_WIDTH_PX,
    PRIMARY_STREAM_JPEG_QUALITY,
    _NullIsaacRunWriter,
    _actuation_config_path_for_mode,
    _build_arm_joint_payload,
    _build_bootstrap,
    _build_pattern_locations,
    _cached_frame_data_url,
    _draw_detection_overlay,
    _imu_preset_name_for_mode,
    _initialize_recording,
    _load_yaml_payload,
    _overlay_entries,
    _resolve_session_state,
    _stop_recording,
)
from calib_sim.isaac.standard_demo_protocol import (
    empty_standard_snapshot,
    initial_standard_catalog,
    standard_snapshot_with_error,
)


STANDARD_PRIMARY_STREAM_MAX_WIDTH_PX = 1920
SNAPSHOT_PUBLISH_INTERVAL_S = 0.1
FORCED_SNAPSHOT_PUBLISH_INTERVAL_S = 0.1


def _align_standard_preview_sensor_rates(runtime: Any) -> None:
    scene_payload = runtime.config.config_payloads.get("scene", {})
    if not isinstance(scene_payload, dict):
        return
    observer_cameras = scene_payload.get("observer_cameras")
    if not isinstance(observer_cameras, list):
        return
    preview_rate_hz = 1.0 / max(float(SNAPSHOT_PUBLISH_INTERVAL_S), 1e-6)
    for item in observer_cameras:
        if not isinstance(item, dict):
            continue
        current_rate_hz = float(item.get("rate_hz", preview_rate_hz))
        item["rate_hz"] = min(current_rate_hz, preview_rate_hz)


def _build_standard_snapshot(
    runtime: Any,
    profile: IsaacDemoProfile,
    noise_modes: dict[str, str],
    recording_state: dict[str, Any],
    analysis_state: dict[str, Any],
    session_state: str,
    *,
    image_cache: dict[str, dict[str, Any]],
    session_error: str | None = None,
) -> dict[str, Any]:
    overlay_entries = _overlay_entries(runtime)
    display_image_rgb = runtime._latest_display_camera_image_rgb
    if display_image_rgb is None:
        display_image_rgb = runtime._latest_camera_image_rgb
    primary_overlay = _draw_detection_overlay(
        display_image_rgb,
        overlay_entries,
        control_mode=str(runtime._interactive_control_mode or profile.default_control_mode),
        noise_modes=noise_modes,
    )
    display_frame_packet = runtime._latest_display_camera_frame_packet
    if display_frame_packet is None:
        display_frame_packet = runtime._latest_camera_frame_packet
    observer_views = []
    for item in sorted(runtime._latest_observer_views.values(), key=lambda entry: str(entry.get("name", ""))):
        observer_name = str(item.get("name", "observer"))
        observer_frame_index = item.get("frame_index")
        observer_views.append(
            {
                "name": observer_name,
                "frame_index": observer_frame_index,
                "timestamp_s": item.get("timestamp_s"),
                "image_data_url": _cached_frame_data_url(
                    cache=image_cache,
                    stream_key=f"standard:observer:{observer_name}",
                    frame_index=observer_frame_index,
                    image_rgb=item.get("image_rgb"),
                    max_width_px=OBSERVER_STREAM_MAX_WIDTH_PX,
                    jpeg_quality=OBSERVER_STREAM_JPEG_QUALITY,
                ),
                "source": "observer camera",
            }
        )
    measured_imu = runtime._latest_imu_packet
    truth = runtime._latest_imu_truth or {}
    innovation_norm = None if runtime._filter is None else float(runtime._filter.last_innovation_norm)
    estimator_summary = "Waiting for filter state."
    if runtime._latest_filter_snapshot is not None:
        position = np.asarray(runtime._latest_filter_snapshot.position_world_m, dtype=np.float64).reshape(3)
        velocity = np.asarray(runtime._latest_filter_snapshot.velocity_world_mps, dtype=np.float64).reshape(3)
        estimator_summary = (
            f"p=[{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}] m | "
            f"v=[{velocity[0]:.3f}, {velocity[1]:.3f}, {velocity[2]:.3f}] m/s"
        )

    snapshot = empty_standard_snapshot(profile_key=profile.key)
    snapshot["session"] = {
        "state": session_state,
        "profile_key": profile.key,
        "profile_label": profile.label,
        "streaming_backend": "webrtc",
        "last_error": session_error,
    }
    snapshot["frames"] = {
        "primary": {
            "image_data_url": _cached_frame_data_url(
                cache=image_cache,
                stream_key="standard:primary",
                frame_index=None if display_frame_packet is None else int(display_frame_packet.frame_index),
                image_rgb=primary_overlay,
                max_width_px=STANDARD_PRIMARY_STREAM_MAX_WIDTH_PX,
                jpeg_quality=PRIMARY_STREAM_JPEG_QUALITY,
            ),
            "frame_index": None if display_frame_packet is None else int(display_frame_packet.frame_index),
            "timestamp_s": None if display_frame_packet is None else float(display_frame_packet.timestamp_s),
            "source": "annotated_mounted_preview",
        },
        "observers": observer_views,
    }
    snapshot["robot"] = {"arm_joints": _build_arm_joint_payload(runtime)}
    snapshot["control"] = {
        "control_mode": str(runtime._interactive_control_mode or profile.default_control_mode),
        "tracking_error_norm_m": None
        if runtime._latest_controller_diagnostic is None
        else float(runtime._latest_controller_diagnostic.tracking_error_norm_m),
        "safety_reason": None
        if runtime._latest_controller_diagnostic is None
        else str(runtime._latest_controller_diagnostic.safety_reason),
        "desired_position_world_m": None
        if runtime._latest_controller_diagnostic is None
        else [float(value) for value in runtime._latest_controller_diagnostic.desired_position_world_m],
    }
    snapshot["estimation"] = {
        "estimator_mode": str(runtime.config.estimator_mode),
        "anchor_visible": bool(runtime._last_anchor_visible),
        "position_radius_95_m": None
        if runtime._latest_uncertainty is None
        else float(runtime._latest_uncertainty.position_radius_95_m),
        "innovation_norm": innovation_norm,
        "summary": estimator_summary,
    }
    snapshot["detections"] = {
        "count": int(len(runtime._last_detections)),
        "pattern_locations": _build_pattern_locations(runtime),
        "backend": "new_pupil",
        "backend_state": "active",
        "backend_error": None,
        "input_transport": None,
        "bridge_latency_ms": None,
        "timeouts": 0,
        "fallback_active": False,
    }
    snapshot["imu"] = {
        "measured": {
            "accel_mps2": [0.0, 0.0, 0.0]
            if measured_imu is None
            else [float(measured_imu.ax), float(measured_imu.ay), float(measured_imu.az)],
            "gyro_rps": [0.0, 0.0, 0.0]
            if measured_imu is None
            else [float(measured_imu.wx), float(measured_imu.wy), float(measured_imu.wz)],
        },
        "truth": {
            "accel_mps2": [
                float(value) for value in np.asarray(truth.get("accel_mps2", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
            ],
            "gyro_rps": [
                float(value) for value in np.asarray(truth.get("gyro_rps", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
            ],
        },
    }
    snapshot["noise"] = dict(noise_modes)
    snapshot["recording"] = deepcopy(recording_state)
    snapshot["analysis"] = deepcopy(analysis_state)
    streaming = runtime.standard_streaming_status()
    streaming["control_path_active"] = session_state in {"starting", "live"}
    streaming["primary_camera_publishing"] = runtime._latest_camera_frame_packet is not None
    streaming["health"] = {
        "runtime_boot": bool(runtime.started),
        "control_path_active": bool(streaming["control_path_active"]),
        "transport_active": bool(streaming.get("transport_active", False)),
        "primary_camera_publishing": bool(streaming["primary_camera_publishing"]),
    }
    connection_info = dict(streaming.get("connection_info", {}) or {})
    if runtime._camera_binding is not None:
        connection_info.setdefault(
            "mounted_camera",
            {
                "resolution_px": [
                    int(runtime._camera_binding.spec.width_px),
                    int(runtime._camera_binding.spec.height_px),
                ],
                "rate_hz": float(runtime._camera_binding.spec.rate_hz),
                "frame_id": str(runtime._camera_binding.spec.frame_id),
            },
        )
    streaming["connection_info"] = connection_info
    snapshot["streaming"] = streaming
    return snapshot


def _publish_snapshot(snapshot_queue: mp.Queue, snapshot: dict[str, Any]) -> None:
    try:
        while True:
            snapshot_queue.get_nowait()
    except queue.Empty:
        pass
    try:
        snapshot_queue.put_nowait(snapshot)
    except queue.Full:
        pass


def standard_demo_worker_main(
    *,
    command_queue: mp.Queue,
    snapshot_queue: mp.Queue,
    stop_event: mp.Event,
    session_id: str,
    headless: bool,
    width: int,
    height: int,
) -> None:
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    profile = get_demo_profile(DEFAULT_PROFILE_KEY)
    noise_modes = {
        "imu_mode": "ideal",
        "vision_mode": "clean",
        "actuation_mode": "none",
        "detector_mode": "new_pupil",
    }
    recording_state = {"enabled": False, "active_run_dir": None, "last_run_dir": None, "output_root": "output/isaac_demo_runs"}
    analysis_state = {
        "state": "idle",
        "last_started_at_utc": None,
        "last_completed_at_utc": None,
        "last_run_dir": None,
        "summary": None,
        "report_path": None,
        "error": None,
    }
    null_writer = _NullIsaacRunWriter()
    image_cache: dict[str, dict[str, Any]] = {}
    runtime = None
    session_state = "starting"
    session_error = None
    try:
        from calib_sim.isaac.app import create_runtime

        browsing_run_dir = Path("/tmp") / "calib_sim_isaac_standard_demo" / f"session_{session_id}"
        browsing_run_dir.mkdir(parents=True, exist_ok=True)
        bootstrap = _build_bootstrap(
            profile,
            run_dir=browsing_run_dir,
            session_id=session_id,
            headless=headless,
            width=width,
            height=height,
            streaming_backend="webrtc",
        )
        runtime = create_runtime(bootstrap)
        runtime.set_writer(null_writer, reset_recording_baseline=True)
        _align_standard_preview_sensor_rates(runtime)
        runtime.start()
        runtime.set_interactive_control_mode(profile.default_control_mode)
        runtime.set_imu_noise_preset(_imu_preset_name_for_mode(noise_modes["imu_mode"]))
        runtime.set_vision_noise_mode(noise_modes["vision_mode"])
        runtime.set_actuation_config_payload(_load_yaml_payload(_actuation_config_path_for_mode(noise_modes["actuation_mode"])))
        runtime.set_tag_detection_mode("new_pupil")

        last_publish_at = 0.0
        last_forced_publish_at = 0.0
        last_published_frame_index: int | None = None
        last_published_session_state: str | None = None
        last_published_error: str | None = None
        last_published_stream_state: str | None = None

        while not stop_event.is_set():
            while True:
                try:
                    command = command_queue.get_nowait()
                except queue.Empty:
                    break
                if not isinstance(command, Mapping):
                    continue
                command_type = str(command.get("type", "")).strip().lower()
                if command_type == "stop":
                    stop_event.set()
                    break
                if command_type == "launch_session":
                    requested_mode = command.get("control_mode", profile.default_control_mode)
                    runtime.set_interactive_control_mode(
                        validate_control_mode(str(requested_mode), allow_auto_path=False)
                    )
                    session_state = "starting"
                    session_error = None
                elif command_type == "set_control_mode":
                    runtime.set_interactive_control_mode(
                        validate_control_mode(
                            str(command.get("control_mode", profile.default_control_mode)),
                            allow_auto_path=False,
                        )
                    )
                elif command_type == "set_joint_targets":
                    runtime.set_manual_joint_targets_deg(np.asarray(command.get("joint_targets_deg", []), dtype=np.float64))
                elif command_type == "nudge_task_target":
                    runtime.nudge_manual_task_target(
                        np.asarray(command.get("delta_world_m", [0.0, 0.0, 0.0]), dtype=np.float64)
                    )
                elif command_type == "set_noise_modes":
                    updates = validate_noise_modes(
                        imu_mode=command.get("imu_mode"),
                        vision_mode=command.get("vision_mode"),
                        actuation_mode=command.get("actuation_mode"),
                        detector_mode=command.get("detector_mode"),
                    )
                    if "imu_mode" in updates:
                        noise_modes["imu_mode"] = updates["imu_mode"]
                        runtime.set_imu_noise_preset(_imu_preset_name_for_mode(noise_modes["imu_mode"]))
                    if "vision_mode" in updates:
                        noise_modes["vision_mode"] = updates["vision_mode"]
                        runtime.set_vision_noise_mode(noise_modes["vision_mode"])
                    if "actuation_mode" in updates:
                        noise_modes["actuation_mode"] = updates["actuation_mode"]
                        runtime.set_actuation_config_payload(
                            _load_yaml_payload(_actuation_config_path_for_mode(noise_modes["actuation_mode"]))
                        )
                    runtime.set_tag_detection_mode("new_pupil")
                elif command_type == "set_recording":
                    if bool(command.get("enabled", False)):
                        if not recording_state["enabled"]:
                            _initialize_recording(runtime, profile, recording_state, noise_modes)
                    else:
                        _stop_recording(runtime, recording_state, null_writer)
                elif command_type == "run_analysis":
                    analysis_state["state"] = "disabled"
                    analysis_state["summary"] = "Offline report generation was removed from the slim branch."
                elif command_type == "home":
                    runtime.home()
                elif command_type == "refresh":
                    pass
            if stop_event.is_set():
                break
            runtime.step(1)
            session_state = _resolve_session_state(runtime, session_state)
            now = time.monotonic()
            if now - last_publish_at >= SNAPSHOT_PUBLISH_INTERVAL_S:
                frame_index = None if runtime._latest_camera_frame_packet is None else int(runtime._latest_camera_frame_packet.frame_index)
                stream_state = str(runtime.standard_streaming_status().get("state", "idle"))
                should_publish = (
                    frame_index != last_published_frame_index
                    or session_state != last_published_session_state
                    or session_error != last_published_error
                    or stream_state != last_published_stream_state
                    or now - last_forced_publish_at >= FORCED_SNAPSHOT_PUBLISH_INTERVAL_S
                )
                if should_publish:
                    _publish_snapshot(
                        snapshot_queue,
                        _build_standard_snapshot(
                            runtime,
                            profile,
                            noise_modes,
                            recording_state,
                            analysis_state,
                            session_state,
                            image_cache=image_cache,
                            session_error=session_error,
                        ),
                    )
                    last_published_frame_index = frame_index
                    last_published_session_state = session_state
                    last_published_error = session_error
                    last_published_stream_state = stream_state
                    last_forced_publish_at = now
                last_publish_at = now
    except Exception as exc:
        _publish_snapshot(snapshot_queue, standard_snapshot_with_error(str(exc), profile_key=profile.key))
    finally:
        if runtime is not None:
            if recording_state.get("enabled"):
                _stop_recording(runtime, recording_state, null_writer)
            runtime.shutdown()


class IsaacStandardDemoController:
    """Service-side controller for the tabletop-only standard worker."""

    def __init__(self) -> None:
        self._ctx = mp.get_context("spawn")
        self._lock = threading.Lock()
        self._process: mp.Process | None = None
        self._command_queue: mp.Queue | None = None
        self._snapshot_queue: mp.Queue | None = None
        self._stop_event: mp.Event | None = None
        self._reader_thread: threading.Thread | None = None
        self._session_id = "tabletop-live"
        self._headless = True
        self._width = 1600
        self._height = 900
        self._latest_snapshot = empty_standard_snapshot(profile_key=DEFAULT_PROFILE_KEY)

    def catalog(self) -> dict[str, Any]:
        return {
            "profiles": list_demo_profiles(),
            "default_profile_key": DEFAULT_PROFILE_KEY,
            "default_streaming_backend": "webrtc",
            "streaming_backends": initial_standard_catalog()["streaming_backends"],
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._latest_snapshot)

    def ensure_session(self) -> None:
        with self._lock:
            if self._process is not None and self._process.is_alive():
                return
        self.restart()

    def restart(self) -> None:
        self.stop()
        command_queue = self._ctx.Queue(maxsize=64)
        snapshot_queue = self._ctx.Queue(maxsize=2)
        stop_event = self._ctx.Event()
        session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        worker = self._ctx.Process(
            target=standard_demo_worker_main,
            kwargs={
                "command_queue": command_queue,
                "snapshot_queue": snapshot_queue,
                "stop_event": stop_event,
                "session_id": session_id,
                "headless": self._headless,
                "width": self._width,
                "height": self._height,
            },
            daemon=True,
        )
        worker.start()
        with self._lock:
            self._process = worker
            self._command_queue = command_queue
            self._snapshot_queue = snapshot_queue
            self._stop_event = stop_event
            self._session_id = session_id
            self._latest_snapshot = empty_standard_snapshot(profile_key=DEFAULT_PROFILE_KEY)
            self._latest_snapshot["session"]["state"] = "starting"
            self._latest_snapshot["estimation"]["summary"] = "Booting Isaac worker."
            self._latest_snapshot["streaming"]["state"] = "starting"
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

    def stop(self) -> None:
        process = None
        command_queue = None
        stop_event = None
        with self._lock:
            process = self._process
            command_queue = self._command_queue
            stop_event = self._stop_event
            self._process = None
            self._command_queue = None
            self._snapshot_queue = None
            self._stop_event = None
        if command_queue is not None:
            try:
                command_queue.put_nowait({"type": "stop"})
            except Exception:
                pass
        if stop_event is not None:
            stop_event.set()
        if process is not None and process.is_alive():
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)

    def send_command(self, payload: Mapping[str, Any]) -> None:
        if not payload:
            return
        command_type = str(payload.get("type", "")).strip().lower()
        if command_type == "reset_session":
            self.restart()
            return
        self.ensure_session()
        with self._lock:
            command_queue = self._command_queue
        if command_queue is None:
            return
        try:
            command_queue.put_nowait(dict(payload))
        except Exception:
            pass

    def _reader_loop(self) -> None:
        while True:
            with self._lock:
                process = self._process
                snapshot_queue = self._snapshot_queue
            if process is None or snapshot_queue is None:
                return
            try:
                snapshot = snapshot_queue.get(timeout=0.5)
            except queue.Empty:
                if not process.is_alive():
                    with self._lock:
                        current = deepcopy(self._latest_snapshot)
                        if current["session"]["state"] != "error":
                            self._latest_snapshot = standard_snapshot_with_error(
                                "Isaac worker process exited unexpectedly.",
                                profile_key=DEFAULT_PROFILE_KEY,
                            )
                    return
                continue
            if isinstance(snapshot, dict):
                with self._lock:
                    self._latest_snapshot = snapshot


__all__ = ["IsaacStandardDemoController", "standard_demo_worker_main"]
