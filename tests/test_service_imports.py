"""Import smoke tests for the FastAPI services."""

from __future__ import annotations


def test_tag_service_app_imports() -> None:
    """The upload-based tag service should import without optional dependency errors."""
    from calib_sim.tag_service.service import app

    assert app.title == "AprilTag Detection Service"
    assert "/" in {route.path for route in app.routes}


def test_rest_api_app_imports() -> None:
    """The orchestration API should import successfully."""
    from calib_sim.api.rest import app

    assert app.title == "Calibration Simulator API"
    assert "/" in {route.path for route in app.routes}


def test_interactive_sim_app_imports() -> None:
    """The browser-facing interactive sim app should import successfully."""
    from calib_sim.interactive.service import app

    assert app.title == "Interactive Calibration Sim"
    assert "/" in {route.path for route in app.routes}


def test_websocket_runtime_dependency_imports() -> None:
    """Uvicorn needs a websocket transport package for the live dashboard."""
    import websockets

    assert getattr(websockets, "__version__", "")
