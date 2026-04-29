"""Minimal WebRTC streaming helpers for the tabletop standard app."""

from __future__ import annotations

from typing import Any


def validate_streaming_backend(backend: str) -> str:
    candidate = str(backend).strip().lower()
    if candidate != "webrtc":
        raise ValueError(f"Unsupported streaming backend: {backend}")
    return "webrtc"


def streaming_backend_availability(backend: str) -> dict[str, Any]:
    validate_streaming_backend(backend)
    return {
        "available": True,
        "reason": None,
        "experimental": False,
    }


def standard_streaming_backend_catalog() -> list[dict[str, Any]]:
    return [
        {
            "key": "webrtc",
            "label": "WebRTC",
            "available": True,
            "experimental": False,
            "reason": None,
        }
    ]


def default_streaming_status(backend: str) -> dict[str, Any]:
    return {
        "backend": validate_streaming_backend(backend),
        "state": "idle",
        "transport_active": False,
        "control_path_active": False,
        "primary_camera_publishing": False,
        "health": {
            "runtime_boot": False,
            "control_path_active": False,
            "transport_active": False,
            "primary_camera_publishing": False,
        },
        "connection_info": {},
        "notes": [],
        "error": None,
    }


def build_webrtc_connection_info(
    *,
    signal_port: int,
    stream_port: int,
    target_fps: int,
    allow_dynamic_resize: bool,
    public_ip: str = "",
) -> dict[str, Any]:
    return {
        "backend": "webrtc",
        "signal_port": int(signal_port),
        "stream_port": int(stream_port),
        "target_fps": int(target_fps),
        "allow_dynamic_resize": bool(allow_dynamic_resize),
        "public_ip": str(public_ip),
        "viewer": {
            "kind": "isaac_webrtc_streaming_client",
            "instructions": (
                "Open the Isaac Sim WebRTC Streaming Client or compatible browser viewer "
                "and connect to the signal port below."
            ),
        },
    }


__all__ = [
    "build_webrtc_connection_info",
    "default_streaming_status",
    "standard_streaming_backend_catalog",
    "streaming_backend_availability",
    "validate_streaming_backend",
]
