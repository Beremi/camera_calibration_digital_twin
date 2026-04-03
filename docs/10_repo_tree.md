# 10 — Repository tree and module purpose map

## Tree

```text
calib_sim_blueprint/
├── README.md
├── mkdocs.yml
├── pyproject.toml
├── assets/
│   ├── apriltag36h11_id0.png
│   ├── apriltag36h11_id0_canvas.png
│   ├── apriltag36h11_id42.png
│   └── apriltag36h11_id42_canvas.png
├── config/
│   ├── device/
│   │   └── phone_default.yaml
│   ├── motions/
│   │   └── presets.yaml
│   ├── robot/
│   │   ├── franka_panda.yaml
│   │   └── ur5e.yaml
│   ├── scene/
│   │   └── replicacad_room_01.yaml
│   └── tags/
│       └── apriltag36h11_single.yaml
├── docs/
│   ├── 00_system_goals.md
│   ├── 01_stack_selection.md
│   ├── 02_architecture.md
│   ├── 03_build_plan.md
│   ├── 04_scene_assets_and_tags.md
│   ├── 05_robot_arm_and_swapability.md
│   ├── 06_sensors_recording_and_time_sync.md
│   ├── 07_apis.md
│   ├── 08_motion_presets.md
│   ├── 09_tag_detection_service.md
│   └── 10_repo_tree.md
├── src/
│   └── calib_sim/
│       ├── api/
│       │   ├── rest.py
│       │   └── ws.py
│       ├── common/
│       │   └── models.py
│       ├── motions/
│       │   ├── executor.py
│       │   └── library.py
│       ├── recording/
│       │   └── rosbag_recorder.py
│       ├── robot/
│       │   ├── controller.py
│       │   └── factory.py
│       ├── scene/
│       │   ├── loader.py
│       │   └── tag_layout.py
│       ├── sdk/
│       │   └── client.py
│       ├── sensors/
│       │   ├── external_cameras.py
│       │   └── phone_rig.py
│       ├── sim/
│       │   └── runtime.py
│       └── tag_service/
│           ├── detector.py
│           └── service.py
├── tests/
│   └── test_apriltag_detector.py
└── tools/
    └── generate_apriltag.py
```

## File-by-file purpose

### `README.md`

Top-level orientation.  
Explains the chosen stack, architecture, and where to start.

### `mkdocs.yml`

Optional site navigation if you later want to publish the docs as a mini internal handbook.

### `config/*`

Resolved experiment configuration.  
These files are the main extensibility surface for non-core changes.

### `src/calib_sim/common/models.py`

Shared dataclasses and schemas used across modules.  
This file exists to stop your APIs from silently diverging.

### `src/calib_sim/sim/runtime.py`

Starts the simulator, owns the tick loop, and exposes hooks for loading scenes, stepping, and stopping.

### `src/calib_sim/scene/loader.py`

Loads room assets and applies scene-level metadata such as lighting and anchor frames.

### `src/calib_sim/scene/tag_layout.py`

Creates single tags or boards from config and preserves the tag truth model.

### `src/calib_sim/robot/factory.py`

Loads the robot from config and normalizes robot-specific asset details into a common interface.

### `src/calib_sim/robot/controller.py`

Wraps articulation control and exposes joint-space / task-space execution methods.

### `src/calib_sim/sensors/phone_rig.py`

Defines the phone-like RGB + IMU assembly and its topic namespace.

### `src/calib_sim/sensors/external_cameras.py`

Manages additional scene cameras and their publishers.

### `src/calib_sim/recording/rosbag_recorder.py`

Centralizes bag-recording configuration and run artifact layout.

### `src/calib_sim/motions/library.py`

Contains named motion generators such as `snap_yaw`, `calib_sweep_basic`, and `vi_excitation`.

### `src/calib_sim/motions/executor.py`

Takes a generated preset and feeds it into the robot controller while respecting timing.

### `src/calib_sim/api/rest.py`

Experiment-control API: sessions, runs, presets, artifact queries.

### `src/calib_sim/api/ws.py`

WebSocket status and preview metadata channel.

### `src/calib_sim/sdk/client.py`

Small Python client for notebooks, tests, and automation scripts.

### `src/calib_sim/tag_service/detector.py`

Actual AprilTag detection logic.  
This is the most directly reusable module on both simulated and real video.

### `src/calib_sim/tag_service/service.py`

FastAPI wrapper exposing frame/video detection endpoints.

### `tests/test_apriltag_detector.py`

Smoke test to prove the detector sees a generated tag.

### `tools/generate_apriltag.py`

Convenience script to generate marker images from the selected dictionary.

## Commenting policy

All code files in the scaffold are commented around:

- module purpose
- public interfaces
- simulator-specific assumptions
- extension points
- “replace this with real Isaac code” boundaries

That is deliberate.  
The goal is not just to create files, but to make the codebase handoff-friendly.
