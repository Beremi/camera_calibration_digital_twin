"""WebSocket stubs for simulator status streaming."""

from __future__ import annotations

from fastapi import APIRouter, WebSocket

router = APIRouter()


@router.websocket("/ws/v1/sessions/{session_id}/events")
async def session_events(websocket: WebSocket, session_id: str) -> None:
    """Minimal status channel.

    The goal here is to document the intended interface. A production
    implementation would publish simulator state transitions, recorder events,
    and health warnings from an event bus.
    """
    await websocket.accept()
    await websocket.send_json(
        {
            "event": "connected",
            "session_id": session_id,
            "message": "WebSocket scaffold connected.",
        }
    )
    await websocket.close()
