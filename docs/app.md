# Interactive App

## Purpose

The browser app is the primary runnable target in this repository right now. It provides a synthetic but end-to-end calibration workflow without requiring Isaac Sim to be installed.

Entrypoint:

- `src/calib_sim/interactive/service.py`

Main runtime modules:

- `src/calib_sim/interactive/sim.py`
- `src/calib_sim/interactive/analysis.py`
- `src/calib_sim/interactive/camera_model.py`
- `src/calib_sim/interactive/ui.py`

## What The App Does

The app combines:

- a robot-mounted phone camera view
- IMU-style accel and gyro readouts
- manual servo control plus automatic demo motion
- observer-camera side views
- live AprilTag detection overlays
- recording of raw phone video and simulator truth
- offline pose-fitting and report generation

## UI Features

### Session Controls

- scene preset selector
- robot-arm preset selector
- auto-demo toggle
- recording toggle
- analyze-last-run button

### Live Views

- phone camera view with detections
- observer cameras for external scene context
- side/top coverage of the arm and table scene

### Telemetry

- current servo positions and targets
- accelerometer and gyroscope readings
- current analysis state

## Scene And Robot Selection

The browser app can swap scene configs and robot-arm presets at runtime.

Current main tabletop scene:

- `config/interactive/tabletop_grab_challenge.yaml`

Current robot-arm presets:

- `config/interactive/robot_arms/compact_bench.yaml`
- `config/interactive/robot_arms/long_reach_tabletop.yaml`
- `config/interactive/robot_arms/franka_tabletop.yaml`

The tabletop challenge currently places the arm base at table height, keeps both cubes in the phone view, and includes top-mounted calibration patterns on the cubes.

## Camera Models

Main phone-camera presets:

- `config/camera/pixel_9a_main.toml` — processed-video Pixel 9a model used by the current checkpoint
- `config/camera/pixel_9a_raw_distorted.toml` — stronger raw-like distorted variant
- `config/camera/pixel_9a_camera_pose_estimation_exact.toml` — exact repo-style idealized model

Device metadata preset:

- `config/device/pixel_9a_phone.yaml`

## Recording Outputs

Each recorded run under `output/interactive_runs/run_*/` writes:

- `phone_raw.mp4`
- `imu.csv`
- `camera_gt.csv`
- `samples.jsonl`
- `analysis/report.md`
- `analysis/summary.json`
- `analysis/pose_estimates.jsonl`
- `analysis/joint_world_estimates.jsonl`

The raw `output/interactive_runs/` folder is intentionally git-ignored because it grows quickly. The canonical checked-in checkpoint copy lives in [checkpoint_01.md](checkpoint_01.md).

## Estimation Behavior

Single-tag estimation now follows this order:

1. Use `camera obscura` only for the first frame of a tag series.
2. Use `previous_frame` when the same tag stayed visible in the immediately previous frame.
3. Use `previous_world_frame` when the tag reappears after a gap but the previous frame still has a reliable joint world pose.
4. If a single-tag estimate is obviously inconsistent with the same-frame joint solution, rerun it from `same_frame_joint_world`.

This is the checkpoint behavior represented by `checkpoint_01.md`.

## Current Limitations

- The browser app is synthetic, not a real Isaac Sim-backed runtime yet.
- `src/calib_sim/sim/runtime.py` is still the Isaac-side scaffold.
- The joint all-points trajectory is already very accurate, but a small number of single-tag observations can still sit near the 10 cm threshold in hard viewing geometry.
