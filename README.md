# Camera Calibration Digital Twin

This repository is now in a hybrid state:

- a runnable browser-based calibration simulator is the primary working target
- a richer Isaac Sim based runtime is still scaffolded, not production-ready

The current frozen checkpoint is documented in [docs/checkpoint_01.md](docs/checkpoint_01.md). It corresponds to the analysis run `output/interactive_runs/run_20260405_141532` and represents the first checkpoint where the interactive app, recording pipeline, and pose fitting are all working together cleanly enough to preserve.

## Checkpoint 01 State

Checkpoint 01 represents this current repo state:

- browser app with live phone view, observer views, servo control, scene selection, and robot-arm selection
- smooth tabletop demo with the arm base at table height and cube-top calibration patterns
- raw recording of phone video, IMU-like telemetry, camera truth, and per-frame metadata
- offline AprilTag re-detection and pose fitting
- continuity-aware single-tag estimation plus accurate joint all-points trajectory fitting

Checkpoint 01 summary numbers from [docs/checkpoint_01.md](docs/checkpoint_01.md):

- single-tag mean position error: `0.00419 m`
- single-tag median position error: `0.00230 m`
- single-tag max position error: `0.12990 m`
- joint all-points mean position error: `0.00057 m`
- joint all-points max position error: `0.00630 m`

Those results are for the current processed-video Pixel 9a camera model and the tabletop challenge scene.

## Prerequisites

To run the current browser sim you need:

- Python `3.10+`
- a platform where `opencv-python`, `jax`, and `fastapi` install normally
- a modern browser

You do not need Isaac Sim for the current browser checkpoint.

## Bootstrap The Environment

Use the bootstrap script:

```bash
./tools/bootstrap_sim_env.sh
```

This creates `.venv` if needed and installs the project plus test/runtime dependencies.

If you want to activate the environment manually:

```bash
source .venv/bin/activate.fish
```

or in Bash/Zsh:

```bash
. .venv/bin/activate
```

You can also skip activation and run the venv Python directly.

## Run The Interactive Sim

```bash
./tools/bootstrap_sim_env.sh
.venv/bin/python -m uvicorn calib_sim.interactive.service:app --reload --port 8002
```

Then open:

```text
http://127.0.0.1:8002/
```

The current main challenge scene is:

- `config/interactive/tabletop_grab_challenge.yaml`

The browser app details are documented in [docs/app.md](docs/app.md).

## Run Tests

```bash
.venv/bin/python -m pytest -q
```

## Optional Services

### Tag Detection Service

```bash
.venv/bin/python -m uvicorn calib_sim.tag_service.service:app --reload --port 8000
```

Open:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`

### Scaffold Orchestration API

```bash
.venv/bin/python -m uvicorn calib_sim.api.rest:app --reload --port 8001
```

Open:

- `http://127.0.0.1:8001/docs`

## What The App Can Do

At a high level, the browser app gives you:

- a robot-mounted phone-camera view with realtime detections
- IMU-like accel and gyro readouts
- servo target control plus automatic demo motion
- multiple observer cameras
- scene and robot-arm swapping from the UI
- raw recording and offline analysis

The detailed behavior, configs, and artifacts are described in [docs/app.md](docs/app.md).

## Current Camera And Scene Presets

Main phone-camera presets:

- `config/camera/pixel_9a_main.toml`
- `config/camera/pixel_9a_raw_distorted.toml`
- `config/camera/pixel_9a_camera_pose_estimation_exact.toml`

Main device preset:

- `config/device/pixel_9a_phone.yaml`

Interactive scenes:

- `config/interactive/browser_game_demo.yaml`
- `config/interactive/tabletop_grab_challenge.yaml`

Robot-arm presets:

- `config/interactive/robot_arms/compact_bench.yaml`
- `config/interactive/robot_arms/long_reach_tabletop.yaml`
- `config/interactive/robot_arms/franka_tabletop.yaml`

## Repo Status

What is working now:

- browser sim runtime under `src/calib_sim/interactive/`
- FastAPI-based browser service
- AprilTag detector service
- recording and analysis pipeline
- checkpoint-quality documentation in `docs/`

What is still scaffold-level:

- Isaac Sim runtime integration in `src/calib_sim/sim/runtime.py`
- broader ROS/Isaac system from the original blueprint docs

## Documentation

Start with:

- [docs/README.md](docs/README.md)
- [docs/checkpoint_01.md](docs/checkpoint_01.md)
- [docs/app.md](docs/app.md)

Important notes:

- `docs/11_interactive_pose_estimation_pipeline.md` and `docs/12_pose_estimation_investigation.md` are historical development notes. They remain valuable, but their intermediate metrics are superseded by Checkpoint 01.
- generated simulator runs under `output/interactive_runs/` are intentionally git-ignored; the canonical checked-in snapshot is the docs checkpoint.
