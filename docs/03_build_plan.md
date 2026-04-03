# 03 — Detailed build plan

This file is the **implementation order** I would actually follow.

## Phase 0 — Environment bootstrap

### Objective

Get a reproducible developer environment before writing simulator logic.

### Actions

1. Install **Isaac Sim 5.1** on a Linux workstation with a supported RTX-class GPU and driver that matches the production docs.[isaac-51-quick][isaac-51-req]
2. Install **ROS 2 Humble or Jazzy** and verify Isaac Sim can see the ROS environment.[isaac-ros2]
3. Create a Python project for the orchestration API and tag-detection service.
4. Decide whether Isaac runs:
   - as a desktop app for iteration,
   - headless for CI,
   - or both.

### Output

- one developer machine can launch Isaac Sim
- one ROS 2 shell can talk to Isaac Sim
- one Python environment can run FastAPI and OpenCV

## Phase 1 — Create the simulator shell

### Objective

Boot Isaac Sim from Python and create a minimal world.

### Actions

1. Create `SimulationRuntime` that:
   - starts Isaac Sim,
   - loads a physics scene,
   - sets timestep,
   - supports headless / UI mode,
   - runs a tick loop.
2. Add a tiny world smoke test:
   - ground plane,
   - one cube,
   - one camera.

### Output

You can start/step/stop the simulation from Python.

## Phase 2 — Asset pipeline for room scenes

### Objective

Load furnished indoor rooms without hard-coding stage contents.

### Actions

1. Download one or more **ReplicaCAD** scenes.[replicacad]
2. Convert source formats to USD if needed using Isaac Sim’s asset converter support for formats such as `.gltf`, `.obj`, and `.fbx`.[isaac-formats]
3. Create a `SceneLoader` that reads scene config:
   - room asset path
   - asset scale
   - collision profile
   - lighting preset
   - list of tag anchors
4. Add a utility to optionally add Poly Haven props or HDRIs.[polyhaven]

### Output

A room can be selected by config and loaded deterministically.

## Phase 3 — Tag placement system

### Objective

Support single tags and boards without changing code.

### Actions

1. Create a `TagSpec` config with:
   - family
   - id
   - physical edge length in meters
   - asset/image path
   - world pose
   - optional board grouping metadata
2. Generate canonical markers using OpenCV’s predefined AprilTag dictionary.[opencv-aruco]
3. Implement a `TagSpawner`:
   - creates tag planes or decals,
   - assigns textures,
   - stores world pose metadata,
   - optionally groups tags into an AprilGrid.
4. Add a “target visibility check” helper so motion presets can reject trajectories that move the phone where the tag becomes invisible for too long.

### Output

Tags are fully data-driven.

## Phase 4 — Robot arm abstraction

### Objective

Load one arm now, keep the system open for others later.

### Actions

1. Define a `RobotConfig` schema with:
   - `usd_path` or `urdf_path`
   - articulation root
   - joint order
   - end-effector frame
   - mount frame for the phone
   - drive gains
   - velocity/effort limits
2. Implement `RobotFactory`.
3. Start with **Franka Panda** because Isaac Sim has a direct manipulator tutorial and example path for it.[isaac-franka]
4. Add a second config example for UR5e/UR10e-style import using the URDF workflow.[isaac-urdf]
5. Implement a wrapper for the articulation controller that can send:
   - joint position targets
   - joint velocity targets
   - effort targets
   - task-space pose targets through a higher-level controller

### Output

One robot is working; a second robot can be swapped in by config.

## Phase 5 — Phone rig and scene cameras

### Objective

Create the actual sensor payload.

### Actions

1. Build a `PhoneRig` rigid transform tree under the robot’s mount frame:
   - `/tool0/phone_mount`
   - `/tool0/phone_mount/camera_rgb`
   - `/tool0/phone_mount/imu`
2. Configure the camera:
   - width / height
   - fps
   - focal lengths / principal point
   - distortion coefficients
3. Configure the IMU:
   - rate
   - local transform
   - bias/noise model
4. Add one or more `SceneCameraRig` configs for fixed or moving external cameras.
5. Keep every rig under a namespace.

Isaac Sim’s camera docs include distortion model support, and its IMU docs describe sensor creation and placement on rigid bodies.[isaac-cam][isaac-imu]

### Output

You have a phone-like RGB+IMU rig plus optional observer cameras.

## Phase 6 — ROS 2 bridge wiring

### Objective

Expose the simulator to external tools.

### Actions

1. Publish:
   - `/clock`
   - `/tf` and `/tf_static`
   - `/joint_states`
   - `/phone/rgb/image_raw`
   - `/phone/rgb/camera_info`
   - `/phone/imu`
   - `/scene_cam/<name>/image_raw`
   - `/scene_cam/<name>/camera_info`
2. Verify topic rates match the configured sensor rates.
3. Add health checks:
   - camera frame count increasing
   - IMU rate within tolerance
   - joint states monotonic in time

### Output

External ROS 2 tools can consume the synthetic device as though it were real.

## Phase 7 — Recorder

### Objective

Make experiments reproducible.

### Actions

1. Start a `SessionRecorder` that launches `ros2 bag record --storage mcap` or uses the rosbag API.[ros2-bag][rosbag2-github]
2. Write sidecar run metadata:
   - scene config hash
   - robot config hash
   - motion preset id
   - start/end sim time
   - tag layout
   - git revision
3. Optionally export MP4 previews for quick review, but keep the bag as the primary artifact.
4. Save exact intrinsics/extrinsics/noise configs alongside the bag.

### Output

One command starts a recording, one command stops it, and the run can be replayed later.

## Phase 8 — Motion library

### Objective

Generate deterministic calibration motions.

### Actions

1. Create preset generators for:
   - 1 s motions
   - 2 s motions
   - 5 s motions
   - 10 s motions
2. Each preset should:
   - keep the tag in view most of the time
   - excite both translation and rotation
   - avoid immediate self-collision
3. Sample trajectories at a fixed control rate and send them through the robot controller.

### Output

A call like `execute_preset("vi_excitation", duration_s=10.0)` works.

## Phase 9 — Orchestration API

### Objective

Make the bench scriptable.

### Actions

1. Add REST endpoints for:
   - create session
   - load scene
   - load robot
   - attach phone rig
   - spawn tags
   - start/stop run
   - execute preset
   - fetch artifacts
2. Add a WebSocket channel for:
   - status events
   - progress
   - warnings
   - live preview metadata
3. Add a thin Python SDK.

### Output

A notebook or CI script can run experiments without opening the GUI.

## Phase 10 — AprilTag detection service

### Objective

Provide the analysis module you requested.

### Actions

1. Implement detector around OpenCV’s `DICT_APRILTAG_36h11` and AprilTag corner refinement.[opencv-aruco]
2. Accept:
   - live frames,
   - frame batches,
   - recorded video paths/uploads.
3. Return:
   - `family`
   - `id`
   - `corners`
   - `center`
   - `5 points`
   - optional pose if intrinsics and tag size are provided.
4. Publish detections back into ROS 2 or expose them only through REST/WebSocket, depending on how tightly you want the loop coupled.

### Output

The tag detector can be run independently of the simulator.

## Phase 11 — Validation harness

### Objective

Use the simulator to grade calibration software.

### Actions

1. Run a known motion preset.
2. Feed the bag to your calibration pipeline.
3. Compare estimated:
   - camera intrinsics
   - `T_phone_imu`
   - `T_base_phone`
   - temporal offsets (if estimated)
4. Compare with simulator ground truth.
5. Save metrics to JSON:
   - translational error
   - angular error
   - reprojection error
   - run duration
   - dropped frame counts

### Output

You now have a regression test bench, not just a simulator.

## Recommended first milestone

The first milestone should be modest:

- one ReplicaCAD room
- one Franka Panda
- one phone rig
- one static observer camera
- one 10 cm AprilTag
- one 5 s preset
- bag recording
- offline detector

That is enough to prove the architecture.

## Sources

[isaac-51-quick]: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/quick-install.html
[isaac-51-req]: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html
[isaac-ros2]: https://docs.isaacsim.omniverse.nvidia.com/6.0.0/ros2_tutorials/ros2_landing_page.html
[replicacad]: https://aihabitat.org/datasets/replica_cad/
[isaac-formats]: https://docs.isaacsim.omniverse.nvidia.com/4.5.0/assets/formats.html
[polyhaven]: https://polyhaven.com/
[opencv-aruco]: https://docs.opencv.org/4.x/de/d67/group__objdetect__aruco.html
[isaac-franka]: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/core_api_tutorials/tutorial_core_adding_manipulator.html
[isaac-urdf]: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/importer_exporter/import_urdf.html
[isaac-cam]: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/sensors/isaacsim_sensors_camera.html
[isaac-imu]: https://docs.isaacsim.omniverse.nvidia.com/4.5.0/sensors/isaacsim_sensors_physics_imu.html
[ros2-bag]: https://docs.ros.org/en/rolling/Tutorials/Beginner-CLI-Tools/Recording-And-Playing-Back-Data/Recording-And-Playing-Back-Data.html
[rosbag2-github]: https://github.com/ros2/rosbag2
