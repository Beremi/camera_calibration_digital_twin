# 10 — Repository tree and module purpose map

This file reflects the current checkpointed repo layout, not the earlier pure blueprint scaffold.

## Tree

```text
camera_calibration_digital_twin/
├── README.md
├── .gitignore
├── pyproject.toml
├── assets/
├── config/
│   ├── camera/
│   ├── device/
│   ├── interactive/
│   │   └── robot_arms/
│   ├── motions/
│   ├── robot/
│   ├── scene/
│   └── tags/
├── docs/
│   ├── README.md
│   ├── app.md
│   ├── checkpoint_01.md
│   ├── img/checkpoint_01/
│   ├── 00_system_goals.md
│   ├── ...
│   ├── 11_interactive_pose_estimation_pipeline.md
│   └── 12_pose_estimation_investigation.md
├── src/
│   └── calib_sim/
│       ├── api/
│       │   └── rest.py
│       ├── common/
│       │   ├── models.py
│       │   └── video.py
│       ├── interactive/
│       │   ├── analysis.py
│       │   ├── camera_model.py
│       │   ├── service.py
│       │   ├── sim.py
│       │   └── ui.py
│       ├── sim/
│       │   └── runtime.py
│       └── tag_service/
│           ├── detector.py
│           └── service.py
├── tests/
│   ├── test_camera_model.py
│   ├── test_interactive_sim.py
│   └── test_service_imports.py
├── tools/
│   ├── bootstrap_sim_env.sh
│   └── generate_apriltag.py
└── output/
    └── interactive_runs/   # generated, intentionally git-ignored
```

## Top-Level Files

### `README.md`

Current operator-facing entrypoint.
It describes the checkpoint state, prerequisites, bootstrap script, and how to launch the browser app.

### `.gitignore`

Keeps the large generated sim outputs and local caches out of Git.

### `pyproject.toml`

Python package metadata and runtime dependencies for the current browser sim, detector service, and analysis pipeline.

## Config Surface

### `config/camera/`

Phone camera-model presets, including the current processed-video Pixel 9a profile and alternate exact/raw-like variants.

### `config/device/`

Phone/device metadata presets.
The current browser demos use `pixel_9a_phone.yaml`.

### `config/interactive/`

Browser-sim scene entrypoints.
This is the main place to switch between demo scenes, tags, observers, motion, and robot-arm presets.

### `config/interactive/robot_arms/`

Swapable arm presets for the browser app.

## Documentation

### `docs/README.md`

Docs index and “what is current vs historical” guide.

### `docs/app.md`

Detailed browser app description: controls, outputs, configs, and current behavior.

### `docs/checkpoint_01.md`

Frozen checkpoint report copied from the current canonical run.
This is the current checked-in analysis snapshot.

### `docs/11_interactive_pose_estimation_pipeline.md`

Historical build-up of the browser pipeline.
Useful for development history, but superseded by `checkpoint_01.md` for current metrics.

### `docs/12_pose_estimation_investigation.md`

Historical debugging note for the estimator.
Still useful for understanding the failure modes that were fixed.

## Runtime Code

### `src/calib_sim/interactive/sim.py`

Main browser-sim runtime:

- robot kinematics
- primary phone camera render
- observer cameras
- scene props and tags
- recording lifecycle
- configuration reload

### `src/calib_sim/interactive/service.py`

FastAPI service that serves the browser UI and streams live sim snapshots over WebSocket.

### `src/calib_sim/interactive/ui.py`

HTML/JS dashboard generator for the browser app.

### `src/calib_sim/interactive/camera_model.py`

TOML-driven phone camera model used by rendering and analysis.

### `src/calib_sim/interactive/analysis.py`

Offline replay, re-detection, single-tag fitting, joint all-points fitting, and report generation.

### `src/calib_sim/tag_service/detector.py`

OpenCV AprilTag detector wrapper with 5-point output and optional pose estimation.

### `src/calib_sim/tag_service/service.py`

FastAPI wrapper for the tag detector service.

### `src/calib_sim/api/rest.py`

Small scaffold orchestration API that still exists alongside the browser app.

### `src/calib_sim/sim/runtime.py`

Isaac Sim runtime scaffold.
Important to keep, but it is not the current primary runnable path.

## Tests

### `tests/test_interactive_sim.py`

End-to-end coverage for the browser sim, recording flow, analysis artifacts, and current preset switching.

### `tests/test_camera_model.py`

Camera-model and measurement-shape checks.

### `tests/test_service_imports.py`

Import and service smoke tests for the FastAPI entrypoints.

## Tools

### `tools/bootstrap_sim_env.sh`

Bootstraps `.venv` and installs the current runtime/test dependencies.

### `tools/generate_apriltag.py`

Generates canonical marker images used by the detector tests and assets.

## Generated Output

### `output/interactive_runs/`

Recorded runs and analysis artifacts from the browser sim.
This directory is intentionally git-ignored because it can grow quickly; the repository keeps only the curated checkpoint copy in `docs/`.
