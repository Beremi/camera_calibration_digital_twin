# 00 — System goals, scope, and success criteria

## Goal

Build a **simulation-first calibration test bench** for a robot-mounted phone-like sensor rig.  
The simulator must let you:

1. place AprilTag 2 / `36h11` targets in realistic indoor scenes,
2. mount a phone-like camera + IMU to a robot arm,
3. execute reproducible motion presets,
4. record synchronized RGB / IMU / joint / TF streams,
5. stream sensor data to external calibration software in real time,
6. replay runs offline,
7. run a tag-detection module over live or recorded video.

The intended outcome is not just a pretty simulator. It is a **deterministic experiment harness** for calibration software.

## Hard requirements translated into engineering terms

| Requirement from prompt | Engineering translation |
|---|---|
| Open-source 3D rooms with furniture | Use importable furnished room datasets; start with ReplicaCAD and optionally enrich with Poly Haven assets. |
| AprilTag 2 / 36h11 patterns | Use OpenCV’s `DICT_APRILTAG_36h11` for marker generation/detection compatibility and keep marker metadata in config. |
| Robot arm with servos | Use an articulated manipulator with explicit joint drive config and a control layer that accepts joint or task-space commands. |
| Robot should be swapable | Make robot selection a pure config problem: `usd_path`/`urdf_path`, joint order, end-effector frame, drive gains, and mount frame. |
| Mobile attached to robot arm | Create a `PhoneRig` rigid assembly: RGB camera + IMU + extrinsic calibration parameters mounted under tool frame. |
| Other cameras possible | Add `SceneCameraRig` objects defined by config and published/recorded exactly like the phone camera. |
| API for streaming data + commanding robot | Use ROS 2 for the robotics data path; use REST/WebSocket for orchestration and experiment control. |
| Predefined motions | Create parameterized motion presets and expose them by preset id + duration + gain profile. |
| Tag detector service | Separate module that accepts live frames or recorded video and returns `{family, id, corners, center}` plus optional pose. |
| Human-readable API docs | Write markdown docs for topics, endpoints, schemas, versioning, and extension rules. |

## Non-goals for v1

The simulator does **not** need all of these on day one:

- exact smartphone ISP emulation
- photorealistic human hand motion
- perfect MEMS-grade IMU error matching
- GPU AprilTag detection
- multi-user cloud orchestration
- reinforcement learning integration

These are valid later upgrades, but they should not block the first useful system.

## Success criteria

A v1 simulator is successful when the following end-to-end workflow works without manual intervention:

1. Load a furnished room.
2. Load one robot arm from config.
3. Spawn a phone rig on the wrist.
4. Spawn one or more AprilTag targets from config.
5. Start a run.
6. Execute a motion preset.
7. Publish phone camera, CameraInfo, IMU, joint states, `/tf`, and scene camera topics.
8. Record the run to rosbag2/MCAP.
9. Run the detector service on the resulting phone video or live frames.
10. Recover tags with correct `id`, `4 corners`, and `center`.
11. Feed the bag or live stream into external calibration software.

If you can do those 11 steps reliably, the project is already useful.

## Design principles

### 1. One source of truth for time

Every sensor message, joint state, and detector result should be stamped from the **simulation clock**.  
Do not let the renderer, recorder, and API layer invent independent time bases.

### 2. Raw data first, convenience data second

Always record the raw channels:

- RGB image
- camera intrinsics / distortion
- IMU accel / gyro
- joint states
- robot transforms
- ground-truth tag poses (optional debug channel)

Derived products such as MP4, detector JSON, or calibration summaries should be **sidecars**, not the only output.

### 3. Configurable, not hard-coded

Anything likely to change between experiments should live in YAML or JSON config:

- robot type
- room asset
- tag ids and sizes
- phone intrinsics
- IMU noise
- motion presets
- recorder options

### 4. Separate orchestration from the simulation loop

HTTP is fine for “start run”, “load scene”, “execute preset”.  
It is **not** where you want your inner servo loop to live. Use an in-process controller or ROS 2 path for fast command delivery, and use REST/WebSocket for human- and tool-friendly orchestration.

## Why the baseline stack fits these goals

Isaac Sim exposes official support for Python scripting, camera sensors, IMU sensors, articulated robot control, robot assets, URDF import, and ROS 2 bridging, which together cover nearly the entire simulator core required here.[isaac-what][isaac-cam][isaac-imu][isaac-articulation][isaac-urdf][isaac-ros2]  
ReplicaCAD provides furnished indoor scenes intended for interactive simulation and distributed under CC BY 4.0.[replicacad]  
OpenCV exposes the `DICT_APRILTAG_36h11` dictionary and canonical marker generation functions, which makes the tag layer straightforward to implement in Python.[opencv-aruco]

## Recommended v1 deliverables

- `sim_core` that can launch Isaac Sim headless or with UI
- `scene_loader` for room assets and tag placement
- `robot_factory` for loading Franka/UR5e-style robots from config
- `phone_rig` and `scene_camera_rig`
- `session_recorder`
- `motion_library`
- `orchestrator_api`
- `apriltag_service`
- documentation + example configs

## Sources

[isaac-what]: https://docs.isaacsim.omniverse.nvidia.com/
[isaac-cam]: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/sensors/isaacsim_sensors_camera.html
[isaac-imu]: https://docs.isaacsim.omniverse.nvidia.com/4.5.0/sensors/isaacsim_sensors_physics_imu.html
[isaac-articulation]: https://docs.isaacsim.omniverse.nvidia.com/4.5.0/robot_simulation/articulation_controller.html
[isaac-urdf]: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/importer_exporter/import_urdf.html
[isaac-ros2]: https://docs.isaacsim.omniverse.nvidia.com/6.0.0/ros2_tutorials/ros2_landing_page.html
[replicacad]: https://aihabitat.org/datasets/replica_cad/
[opencv-aruco]: https://docs.opencv.org/4.x/de/d67/group__objdetect__aruco.html
