# Camera Calibration Digital Twin

This repository is now in a hybrid state:

- the preserved browser-based checkpoints remain the regression baseline and historical path
- a first real Isaac Sim pass now exists locally, with live sensing, online estimation, closed-loop control, and artifact-driven reporting
- the Isaac path is usable for first-pass experiments, but it is not yet tuned or benchmark-complete

The current frozen checkpoint is documented in [docs/checkpoint_01.md](docs/checkpoint_01.md). It corresponds to the corrected analysis run `output/interactive_runs/run_20260406_083611` and represents the first checkpoint where the interactive app, recording pipeline, visual pose fitting, and IMU trajectory reconstruction are all working together cleanly enough to preserve.

The new batch-estimation milestones are documented in [docs/checkpoint_02_batch_estimation.md](docs/checkpoint_02_batch_estimation.md), [docs/checkpoint_03_scientific_report.md](docs/checkpoint_03_scientific_report.md), and [docs/estimation.md](docs/estimation.md).

As of April 8, 2026, the current local Isaac first-pass headline run in this workspace is:

- `output/isaac_runs/first_real_pass_nominal_v2`

The paper/report pipeline consumes:

- `output/isaac_runs/latest_complete`

The expert-oriented repo map for the current first-pass branch is:

- `docs/isaac_first_pass_expert_handoff.md`

## Checkpoint 01 State

Checkpoint 01 represents this current repo state:

- browser app with live phone view, observer views, servo control, scene selection, and robot-arm selection
- smooth tabletop demo with the arm base at table height and cube-top calibration patterns
- raw recording of phone video, IMU-like telemetry, camera truth, and per-frame metadata
- offline AprilTag re-detection and pose fitting
- continuity-aware single-tag estimation plus accurate joint all-points trajectory fitting
- corrected exact IMU logging plus IMU-only trajectory reconstruction anchored to ground truth start pose

Checkpoint 01 summary numbers from [docs/checkpoint_01.md](docs/checkpoint_01.md):

- single-tag mean position error: `0.00419 m`
- single-tag median position error: `0.00230 m`
- single-tag max position error: `0.12990 m`
- joint all-points mean position error: `0.00057 m`
- joint all-points max position error: `0.00630 m`
- IMU-only mean position error: `2.16e-07 m`
- IMU-only max position error: `7.15e-07 m`

Those results are for the current processed-video Pixel 9a camera model, the tabletop challenge scene, and the corrected IMU recording path.

## Prerequisites

To run the current browser sim you need:

- Python `3.10+`
- a platform where `opencv-python`, `jax`, and `fastapi` install normally
- a modern browser

You do not need Isaac Sim for the current browser checkpoint.

To run the Isaac first pass on this machine you need:

- the dedicated Isaac environment `.venv-isaac`
- Isaac Sim 6.x installed into that environment
- an RTX-capable Linux workstation

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

## Run Isaac First Pass

For the current mounted-camera Franka first pass:

```bash
source .venv-isaac/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
python scripts/run_isaac_anchor_vio.py --headless --duration-s 10.0 --promote-latest-complete
```

To regenerate report artifacts for a finished Isaac run:

```bash
source .venv/bin/activate
python scripts/generate_isaac_report_artifacts.py output/isaac_runs/first_real_pass_nominal_v2
```

To compile the paper against `latest_complete`:

```bash
source .venv/bin/activate
cd report_tex
latexmk -pdf -interaction=nonstopmode publication_report_template.tex
```

## Run Tests

```bash
.venv/bin/python -m pytest -q
```

Isaac-specific tests should be run from `.venv-isaac`.

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

What is now first-pass real:

- Isaac runtime under `src/calib_sim/isaac/`
- Franka-first live stage build with anchor and auxiliary tags
- live camera and IMU logging into `output/isaac_runs/<run_id>/`
- online AprilTag frontend, anchored filter, fixed-lag smoother, and closed-loop path tracking
- artifact-driven LaTeX report generation through `report_tex/`

What is still not final:

- camera placement and control tuning are first-pass, not polished
- headline quality metrics are real but not yet competitive or tuned
- multi-seed and stress-ablation campaigns remain future work
- ROS 2 mirroring and broader Isaac integration remain secondary to the standalone first-pass runtime

## Documentation

Start with:

- [docs/README.md](docs/README.md)
- [docs/isaac_first_pass_expert_handoff.md](docs/isaac_first_pass_expert_handoff.md)
- [docs/checkpoint_01.md](docs/checkpoint_01.md)
- [docs/checkpoint_02_batch_estimation.md](docs/checkpoint_02_batch_estimation.md)
- [docs/checkpoint_03_scientific_report.md](docs/checkpoint_03_scientific_report.md)
- [docs/checkpoint_03_scientific_core.md](docs/checkpoint_03_scientific_core.md)
- [docs/estimation.md](docs/estimation.md)
- [docs/intermediate_report/README.md](docs/intermediate_report/README.md)
- [docs/app.md](docs/app.md)

Important notes:

- `docs/11_interactive_pose_estimation_pipeline.md` and `docs/12_pose_estimation_investigation.md` are historical development notes. They remain valuable, but their intermediate metrics are superseded by Checkpoint 01.
- generated simulator runs under `output/interactive_runs/` are intentionally git-ignored; the canonical checked-in snapshot is the docs checkpoint.
