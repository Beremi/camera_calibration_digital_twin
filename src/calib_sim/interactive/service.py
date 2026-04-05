"""FastAPI service exposing the interactive browser simulator."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from calib_sim.interactive.sim import InteractiveCalibrationSim, REPO_ROOT, available_robot_arm_presets, available_scene_presets
from calib_sim.interactive.ui import interactive_dashboard_html

DEFAULT_CONFIG_PATH = "config/interactive/browser_game_demo.yaml"

app = FastAPI(title="Interactive Calibration Sim", version="0.1.0")
sim = InteractiveCalibrationSim(DEFAULT_CONFIG_PATH)
sim_lock = asyncio.Lock()


@app.get("/")
def root() -> HTMLResponse:
    """Serve the browser dashboard."""
    return HTMLResponse(interactive_dashboard_html(sim.config.config_path))


@app.get("/health")
def health() -> dict[str, Any]:
    """Return process and config status."""
    return {
        "ok": True,
        "service": "interactive_calibration_sim",
        "repo_root": str(REPO_ROOT),
        "config": sim.config.summary(),
        "catalog": sim.catalog_summary(),
    }


@app.get("/v1/config")
def get_config() -> dict[str, Any]:
    """Return the currently loaded interactive config."""
    return {"config": sim.config.summary(), "catalog": sim.catalog_summary()}


@app.get("/v1/catalog")
def get_catalog() -> dict[str, Any]:
    """Return available scene and robot-arm presets."""
    return {
        "scene_presets": available_scene_presets(),
        "robot_arm_presets": available_robot_arm_presets(),
        "selected_scene_config_path": sim.selected_scene_config_path,
        "selected_robot_arm_preset_path": sim.selected_robot_arm_preset_path,
    }


def _handle_client_message(message: dict[str, Any]) -> None:
    message_type = message.get("type")
    if message_type == "set_servos":
        sim.set_servo_targets(message.get("targets_deg", [0.0, 0.0, 0.0]))
        return
    if message_type == "adjust_servos":
        sim.nudge_servo_targets(message.get("delta_deg", [0.0, 0.0, 0.0]))
        return
    if message_type == "set_recording":
        sim.set_recording(bool(message.get("enabled", False)))
        return
    if message_type == "set_auto_demo":
        sim.set_auto_demo(bool(message.get("enabled", False)))
        return
    if message_type == "run_analysis":
        try:
            sim.run_analysis_on_last_run()
        except Exception as exc:
            sim.analysis_status = {
                "state": "failed",
                "last_started_at_utc": sim.analysis_status.get("last_started_at_utc"),
                "last_completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "last_run_dir": sim.analysis_status.get("last_run_dir"),
                "summary": None,
                "report_path": None,
                "error": str(exc),
            }
        return
    if message_type == "reload_config":
        config_path = str(message.get("config_path") or DEFAULT_CONFIG_PATH)
        sim.reload_config(config_path)
        return
    if message_type == "set_scene_preset":
        config_path = str(message.get("config_path") or sim.selected_scene_config_path or DEFAULT_CONFIG_PATH)
        sim.reload_config(config_path)
        return
    if message_type == "set_robot_arm_preset":
        preset_path = str(message.get("preset_path") or sim.selected_robot_arm_preset_path)
        sim.set_robot_arm_preset(preset_path)
        return


@app.websocket("/ws/live")
async def live_stream(websocket: WebSocket) -> None:
    """Stream live sim snapshots and accept simple control messages."""
    await websocket.accept()
    frame_interval_s = max(1.0 / max(sim.config.stream_fps, 1.0), 0.05)
    last_frame_time = time.monotonic()

    async with sim_lock:
        await websocket.send_json(sim.render_snapshot())

    while True:
        try:
            timeout_s = max(0.0, frame_interval_s - (time.monotonic() - last_frame_time))
            message = await asyncio.wait_for(websocket.receive_json(), timeout=timeout_s)
            async with sim_lock:
                _handle_client_message(message)
        except asyncio.TimeoutError:
            now = time.monotonic()
            dt_s = max(now - last_frame_time, 1.0 / 120.0)
            async with sim_lock:
                sim.step(dt_s)
                snapshot = sim.render_snapshot()
            await websocket.send_json(snapshot)
            last_frame_time = now
        except WebSocketDisconnect:
            return
