from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import WebSocketDisconnect

from calib_sim.isaac import demo_profiles
from calib_sim.isaac import standard_demo_protocol
from calib_sim.isaac import standard_demo_service
from calib_sim.isaac import standard_demo_worker
from calib_sim.isaac import standard_streaming
from calib_sim.isaac.app import IsaacAppBootstrapConfig


def test_standard_demo_dashboard_and_service_imports() -> None:
    response = standard_demo_service.root()
    assert response.status_code == 200
    body = response.body.decode("utf-8")
    assert "Tabletop Auto-Demo Deck" in body
    assert "/ws/live" in body
    assert "Live Previews" in body
    assert "Mounted Camera" in body
    assert "room_wide" in body
    assert "side_overwatch" in body
    assert "top_oblique" in body
    assert "IMU Noise" in body
    assert "Vision Noise" in body
    assert "Actuation Noise" in body
    assert "Launch Session" in body
    assert "Streaming Backend" not in body
    assert 'id="profile-select"' not in body
    assert 'id="detector-select"' not in body
    assert 'id="streaming-backend-select"' not in body
    assert "{{" not in body
    assert "}}" not in body

    health = standard_demo_service.health()
    assert health["ok"] is True
    assert health["service"] == "tabletop_auto_demo_deck"

    catalog = standard_demo_service.catalog()
    assert catalog["default_profile_key"] == demo_profiles.DEFAULT_PROFILE_KEY
    assert [item["key"] for item in catalog["profiles"]] == ["tabletop_replica"]
    assert catalog["default_streaming_backend"] == "webrtc"
    assert [item["key"] for item in catalog["streaming_backends"]] == ["webrtc"]


def test_empty_standard_snapshot_defaults() -> None:
    snapshot = standard_demo_protocol.empty_standard_snapshot()
    assert snapshot["session"]["profile_key"] == demo_profiles.DEFAULT_PROFILE_KEY
    assert snapshot["session"]["streaming_backend"] == "webrtc"
    assert snapshot["streaming"]["backend"] == "webrtc"
    assert snapshot["streaming"]["state"] == "idle"
    assert snapshot["noise"]["detector_mode"] == "new_pupil"
    assert snapshot["detections"]["backend"] == "new_pupil"
    assert snapshot["detections"]["backend_state"] == "active"


def test_standard_controller_ensure_session_starts_worker(monkeypatch) -> None:
    controller = standard_demo_worker.IsaacStandardDemoController()
    restarted: list[str] = []

    monkeypatch.setattr(controller, "restart", lambda: restarted.append("restart"))
    controller.ensure_session()

    assert restarted == ["restart"]


def test_standard_controller_reset_restarts_worker(monkeypatch) -> None:
    controller = standard_demo_worker.IsaacStandardDemoController()
    restarted: list[str] = []

    monkeypatch.setattr(controller, "restart", lambda: restarted.append("restart"))
    controller.send_command({"type": "reset_session"})

    assert restarted == ["restart"]


def test_standard_streaming_catalog_is_webrtc_only() -> None:
    assert standard_streaming.validate_streaming_backend("webrtc") == "webrtc"
    availability = standard_streaming.streaming_backend_availability("webrtc")
    assert availability == {"available": True, "reason": None, "experimental": False}


def test_standard_build_bootstrap_uses_webrtc_only() -> None:
    profile = demo_profiles.get_demo_profile("tabletop_replica")
    bootstrap = standard_demo_worker._build_bootstrap(
        profile,
        run_dir=Path("/tmp/demo"),
        session_id="abc",
        headless=True,
        width=1600,
        height=900,
        streaming_backend="webrtc",
        webrtc_signal_port=49234,
        webrtc_stream_port=48234,
    )
    runtime_config = bootstrap.as_runtime_config()
    sim_runtime_config = runtime_config.runtime_config()
    assert runtime_config.streaming_backend == "webrtc"
    assert sim_runtime_config.streaming_backend == "webrtc"
    assert sim_runtime_config.webrtc_signal_port == 49234
    assert sim_runtime_config.webrtc_stream_port == 48234


def test_standard_preview_alignment_downshifts_observer_rates_to_snapshot_cadence() -> None:
    runtime = type("RuntimeLike", (), {})()
    runtime.config = type("ConfigLike", (), {})()
    runtime.config.config_payloads = {
        "scene": {
            "observer_cameras": [
                {"name": "room_wide", "rate_hz": 20.0},
                {"name": "side_overwatch", "rate_hz": 8.0},
                {"name": "top_oblique", "rate_hz": 0.5},
            ]
        }
    }

    standard_demo_worker._align_standard_preview_sensor_rates(runtime)

    rates = [item["rate_hz"] for item in runtime.config.config_payloads["scene"]["observer_cameras"]]
    assert rates == [10.0, 8.0, 0.5]


def test_standard_websocket_auto_starts_session(monkeypatch) -> None:
    called: list[str] = []

    def _fake_ensure_session(*args, **kwargs) -> None:
        called.append("ensure")

    monkeypatch.setattr(standard_demo_service.controller, "ensure_session", _fake_ensure_session)

    class _FakeWebSocket:
        async def accept(self) -> None:
            return None

        async def send_json(self, payload) -> None:
            assert "session" in payload
            raise WebSocketDisconnect()

    asyncio.run(standard_demo_service.live_stream(_FakeWebSocket()))
    assert called == ["ensure"]


def test_app_bootstrap_runtime_config_carries_webrtc_ports() -> None:
    bootstrap = IsaacAppBootstrapConfig(
        scene_config_path="config/isaac/scene/tabletop_grab_challenge_demo.yaml",
        robot_config_path="config/isaac/robot/franka_tabletop_grab_challenge.yaml",
        camera_config_path="config/isaac/camera/phone_tabletop_demo.yaml",
        imu_config_path="config/isaac/imu/phone_nominal.yaml",
        actuation_config_path="config/isaac/actuation/servo_nominal.yaml",
        estimation_config_path="config/isaac/estimation/tabletop_demo_vio.yaml",
        control_config_path="config/isaac/control/tabletop_demo.yaml",
        run_id="demo",
        run_dir="/tmp/demo",
        streaming_backend="webrtc",
        webrtc_signal_port=49234,
        webrtc_stream_port=48234,
    )
    runtime_config = bootstrap.as_runtime_config()
    sim_runtime_config = runtime_config.runtime_config()
    assert runtime_config.streaming_backend == "webrtc"
    assert sim_runtime_config.streaming_backend == "webrtc"
    assert sim_runtime_config.webrtc_signal_port == 49234
    assert sim_runtime_config.webrtc_stream_port == 48234
