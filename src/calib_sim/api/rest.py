"""FastAPI orchestration layer.

This service is intentionally lightweight. Its purpose is to define the external
contract and to be easy to extend. It is not the hard real-time control loop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from calib_sim.motions.library import build_preset

app = FastAPI(title="Calibration Simulator API", version="0.1.0")


class CreateSessionRequest(BaseModel):
    """Request body for creating a new simulator session."""

    session_name: str
    headless: bool = True


class LoadConfigRequest(BaseModel):
    """Generic request body that points to a config file."""

    config_path: str = Field(..., description="Relative or absolute path to a YAML config file.")


class CreateRunRequest(BaseModel):
    """Request body for opening a new run."""

    run_name: str
    record: bool = True
    record_storage: str = "mcap"


class ExecutePresetRequest(BaseModel):
    """Request body for motion execution."""

    preset_id: str
    duration_s: float = Field(..., gt=0.0)
    sample_rate_hz: float = Field(120.0, gt=0.0)
    repeat: int = Field(1, ge=1)


@dataclass
class SessionState:
    """In-memory session state placeholder."""

    session_id: str
    state: str = "IDLE"
    scene_config_path: Optional[str] = None
    robot_config_path: Optional[str] = None
    device_config_path: Optional[str] = None
    tag_config_path: Optional[str] = None


@dataclass
class RunState:
    """In-memory run state placeholder."""

    run_id: str
    session_id: str
    state: str = "CREATED"
    record: bool = True
    record_storage: str = "mcap"
    preset_id: Optional[str] = None


_SESSIONS: Dict[str, SessionState] = {}
_RUNS: Dict[str, RunState] = {}


@app.post("/v1/sessions")
def create_session(request: CreateSessionRequest) -> dict:
    """Create a session.

    In production, this would also boot or reserve a simulator runtime. Here we
    only create the metadata object so the API contract is explicit.
    """
    session_id = f"sess_{len(_SESSIONS) + 1:03d}"
    state = SessionState(session_id=session_id)
    _SESSIONS[session_id] = state
    return asdict(state)


@app.post("/v1/sessions/{session_id}/scene")
def load_scene(session_id: str, request: LoadConfigRequest) -> dict:
    """Attach a scene config to a session."""
    session = _SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session_id.")
    session.scene_config_path = request.config_path
    session.state = "SCENE_READY"
    return asdict(session)


@app.post("/v1/sessions/{session_id}/robot")
def load_robot(session_id: str, request: LoadConfigRequest) -> dict:
    """Attach a robot config to a session."""
    session = _SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session_id.")
    session.robot_config_path = request.config_path
    return asdict(session)


@app.post("/v1/sessions/{session_id}/phone_rig")
def attach_phone_rig(session_id: str, request: LoadConfigRequest) -> dict:
    """Attach the logical phone rig to a session."""
    session = _SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session_id.")
    session.device_config_path = request.config_path
    return asdict(session)


@app.post("/v1/sessions/{session_id}/tags")
def spawn_tags(session_id: str, request: LoadConfigRequest) -> dict:
    """Attach the tag-layout config to a session."""
    session = _SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session_id.")
    session.tag_config_path = request.config_path
    return asdict(session)


@app.post("/v1/sessions/{session_id}/runs")
def create_run(session_id: str, request: CreateRunRequest) -> dict:
    """Create a run under a session."""
    session = _SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session_id.")
    run_id = f"run_{len(_RUNS) + 1:03d}"
    run = RunState(run_id=run_id, session_id=session_id, record=request.record, record_storage=request.record_storage)
    _RUNS[run_id] = run
    return asdict(run)


@app.post("/v1/runs/{run_id}/presets/execute")
def execute_preset(run_id: str, request: ExecutePresetRequest) -> dict:
    """Queue or execute a motion preset.

    The current scaffold just materializes the preset and returns how many
    samples it contains. That is enough to prove the API design.
    """
    run = _RUNS.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Unknown run_id.")
    preset = build_preset(request.preset_id, duration_s=request.duration_s, sample_rate_hz=request.sample_rate_hz)
    run.preset_id = request.preset_id
    run.state = "QUEUED"
    return {
        "run_id": run_id,
        "accepted": True,
        "queued": True,
        "preset_id": request.preset_id,
        "sample_count": len(preset.samples),
    }


@app.get("/v1/runs/{run_id}")
def get_run(run_id: str) -> dict:
    """Return the run record."""
    run = _RUNS.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Unknown run_id.")
    return asdict(run)
