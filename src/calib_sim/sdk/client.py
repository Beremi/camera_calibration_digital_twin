"""Thin Python SDK for orchestration API calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import requests


@dataclass
class SessionHandle:
    session_id: str
    state: str


@dataclass
class RunHandle:
    run_id: str
    state: str


class CalibSimClient:
    """Very small convenience wrapper over the REST API."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def create_session(self, session_name: str, headless: bool = True) -> SessionHandle:
        response = requests.post(f"{self.base_url}/v1/sessions", json={"session_name": session_name, "headless": headless})
        response.raise_for_status()
        payload = response.json()
        return SessionHandle(session_id=payload["session_id"], state=payload["state"])

    def load_scene(self, session_id: str, config_path: str) -> Dict[str, Any]:
        response = requests.post(f"{self.base_url}/v1/sessions/{session_id}/scene", json={"config_path": config_path})
        response.raise_for_status()
        return response.json()

    def load_robot(self, session_id: str, config_path: str) -> Dict[str, Any]:
        response = requests.post(f"{self.base_url}/v1/sessions/{session_id}/robot", json={"config_path": config_path})
        response.raise_for_status()
        return response.json()

    def attach_phone_rig(self, session_id: str, config_path: str) -> Dict[str, Any]:
        response = requests.post(f"{self.base_url}/v1/sessions/{session_id}/phone_rig", json={"config_path": config_path})
        response.raise_for_status()
        return response.json()

    def spawn_tags(self, session_id: str, config_path: str) -> Dict[str, Any]:
        response = requests.post(f"{self.base_url}/v1/sessions/{session_id}/tags", json={"config_path": config_path})
        response.raise_for_status()
        return response.json()

    def create_run(self, session_id: str, run_name: str, record: bool = True) -> RunHandle:
        response = requests.post(
            f"{self.base_url}/v1/sessions/{session_id}/runs",
            json={"run_name": run_name, "record": record, "record_storage": "mcap"},
        )
        response.raise_for_status()
        payload = response.json()
        return RunHandle(run_id=payload["run_id"], state=payload["state"])

    def execute_preset(self, run_id: str, preset_id: str, duration_s: float) -> Dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/v1/runs/{run_id}/presets/execute",
            json={"preset_id": preset_id, "duration_s": duration_s},
        )
        response.raise_for_status()
        return response.json()
