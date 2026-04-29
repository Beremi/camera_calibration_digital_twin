"""FastAPI app for the slim tabletop standard Isaac demo."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from calib_sim.isaac.demo_profiles import DEFAULT_PROFILE_KEY, get_demo_profile
from calib_sim.isaac.standard_demo_ui import isaac_standard_dashboard_html
from calib_sim.isaac.standard_demo_worker import IsaacStandardDemoController


app = FastAPI(title="Tabletop Auto-Demo Deck", version="0.1.0")
controller = IsaacStandardDemoController()


@app.get("/")
def root() -> HTMLResponse:
    profile = get_demo_profile(DEFAULT_PROFILE_KEY)
    return HTMLResponse(isaac_standard_dashboard_html(profile.key, profile.label))


@app.get("/health")
def health() -> dict[str, Any]:
    snapshot = controller.snapshot()
    return {
        "ok": True,
        "service": "tabletop_auto_demo_deck",
        "profile_key": snapshot["session"]["profile_key"],
        "session_state": snapshot["session"]["state"],
        "streaming_backend": snapshot["session"].get("streaming_backend", "webrtc"),
        "transport_state": snapshot.get("streaming", {}).get("state", "idle"),
    }


@app.get("/v1/catalog")
def catalog() -> dict[str, Any]:
    return controller.catalog()


@app.get("/v1/session")
def session() -> dict[str, Any]:
    return {"catalog": controller.catalog(), "snapshot": controller.snapshot()}


@app.websocket("/ws/live")
async def live_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    controller.ensure_session()
    frame_interval_s = 0.1
    last_frame_time = time.monotonic()
    try:
        await websocket.send_json(controller.snapshot())
    except WebSocketDisconnect:
        return

    while True:
        try:
            timeout_s = max(0.0, frame_interval_s - (time.monotonic() - last_frame_time))
            message = await asyncio.wait_for(websocket.receive_json(), timeout=timeout_s)
            if isinstance(message, dict):
                controller.send_command(message)
        except asyncio.TimeoutError:
            try:
                await websocket.send_json(controller.snapshot())
            except WebSocketDisconnect:
                return
            last_frame_time = time.monotonic()
        except WebSocketDisconnect:
            return


__all__ = ["app", "controller"]
