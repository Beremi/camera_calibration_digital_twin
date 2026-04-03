# Calibration Simulator Blueprint (Python + CUDA)

This repository is a **design blueprint + starter scaffold** for a simulation-first calibration bench aimed at:

- a **swapable robot arm** with servo-like joint drives
- a **phone-like recording rig** mounted to the tool flange that publishes **RGB + IMU**
- additional **scene cameras** with synchronized recording
- **AprilTag 2 / 36h11** fiducials placed in the environment
- a **real-time control path** for sending robot actions while streaming sensor data
- a separate **tag-detection service** that returns **4 corners + center + tag id**
- a recording/export path suitable for **offline calibration tooling** and regression tests

The blueprint is intentionally opinionated. It chooses a stack that minimizes glue code for robotics work instead of optimizing for a purely open-source simulator core.

## Recommended baseline stack

| Layer | Choice | Why |
|---|---|---|
| Simulator | **NVIDIA Isaac Sim 5.1** | Stable baseline with Python scripting, camera sensors, IMU sensors, ROS 2 bridge, robot assets, URDF/USD workflows, and GPU-first sensor simulation. |
| Open-source room assets | **ReplicaCAD** as the default room source, optionally enriched with **Poly Haven** props/materials/HDRIs | ReplicaCAD gives furnished indoor environments under CC BY 4.0; Poly Haven is CC0 and useful for visual enrichment. |
| Default robot arm | **Franka Panda** | First-class manipulator path in Isaac Sim docs; good default for a research scaffold. |
| Alternative robot arms | **UR5/UR10e or any URDF/USD articulation** | Isaac Sim supports URDF import and provides manipulator tutorials. |
| Phone-like sensor rig | One RGB camera + one IMU rigidly mounted under `tool0` | Directly matches your “mobile attached to robot arm” requirement. |
| Data plane | **ROS 2** | Native Isaac bridge for camera and IMU publishers; easy integration with robotics tooling. |
| Control/orchestration plane | **FastAPI + WebSockets** | Human-readable API for experiment control, run lifecycle, and UI integration. |
| Recording | **rosbag2 / MCAP** + sidecar MP4/JSON metadata | Good default for replay, regression tests, and interoperability. |
| Tag detector service | **OpenCV aruco/AprilTag path** using `DICT_APRILTAG_36h11` | Simple Python packaging, canonical marker generation, and AprilTag-family support. |

## Why this specific baseline

Isaac Sim is the most practical fit for this project because the official documentation already exposes the exact subsystems you need: Python scripting, articulated robot control, camera sensors, IMU sensors, ROS 2 publication, robot asset workflows, and URDF import.[isaac-what][isaac-cam][isaac-imu][isaac-ros2][isaac-articulation][isaac-urdf]  
I am explicitly recommending **Isaac Sim 5.1 as the build target** because the Isaac Sim 6.0 documentation currently marks 6.0 as an **Early Developer Release** with incomplete documentation and GA artifacts not yet available at that page.[isaac-6-edr]

ReplicaCAD is a strong default for room assets because it is a furnished indoor dataset intended for interactive simulation and released under **CC BY 4.0**.[replicacad]  
If you need extra props, materials, or lighting environments, Poly Haven is useful because its assets are published as **CC0**.[polyhaven]

For the fiducial layer, OpenCV exposes built-in AprilTag dictionaries including **`DICT_APRILTAG_36h11`**, can **generate canonical marker images**, and includes an AprilTag-based corner refinement mode.[opencv-aruco]  
Kalibr’s target guidance is also relevant to the simulator design: for calibration boards, it recommends **Aprilgrid** because partial visibility is acceptable and the pose is fully resolved without flips.[kalibr-targets]

## Architecture at a glance

```text
┌──────────────────────────────────────────────────────────────────────┐
│                     Isaac Sim 5.1 Runtime (Python)                  │
│                                                                      │
│  Room Scene  ─┐   AprilTags/Boards ─┐   Robot Arm + Tool Mount ─┐   │
│               │                     │                            │   │
│               └────────────┬────────┴───────────────┬────────────┘   │
│                            │                        │                │
│                    Phone Rig (RGB + IMU)      Scene Cameras         │
│                            │                        │                │
│                            └────────────┬───────────┘                │
│                                         │                            │
│                                 ROS 2 Bridge                         │
└───────────────────────────────┬─────────┬────────────────────────────┘
                                │         │
                     sensor topics         robot control topics/services
                                │         │
                      ┌─────────▼─────────▼─────────┐
                      │     FastAPI Orchestrator    │
                      │ sessions / runs / presets   │
                      │ WebSocket events / control  │
                      └─────────┬─────────┬─────────┘
                                │         │
                                │         └─────────────► Calibration client / SDK
                                │
                                └───────────────────────► rosbag2 (MCAP) recorder

                      ┌────────────────────────────────┐
                      │   AprilTag Detection Service   │
                      │  live frame / video endpoints  │
                      │ returns id + 4 corners + center│
                      └────────────────────────────────┘
```

## What this repo contains

- **Markdown documentation tree** describing how to build the simulator
- **Config examples** for robot, phone rig, scene, motions, and tags
- **Python scaffold** with comments/docstrings for the major modules
- **Implemented AprilTag detector module** using OpenCV’s AprilTag dictionary path
- **Example generated tag images** for `36h11`

## Documentation map

- [`docs/00_system_goals.md`](docs/00_system_goals.md) — requirements, scope, and success criteria
- [`docs/01_stack_selection.md`](docs/01_stack_selection.md) — why this stack was chosen
- [`docs/02_architecture.md`](docs/02_architecture.md) — system design and component boundaries
- [`docs/03_build_plan.md`](docs/03_build_plan.md) — detailed implementation sequence
- [`docs/04_scene_assets_and_tags.md`](docs/04_scene_assets_and_tags.md) — rooms, tags, asset pipeline
- [`docs/05_robot_arm_and_swapability.md`](docs/05_robot_arm_and_swapability.md) — robot abstraction and phone mount
- [`docs/06_sensors_recording_and_time_sync.md`](docs/06_sensors_recording_and_time_sync.md) — cameras, IMU, bags, metadata
- [`docs/07_apis.md`](docs/07_apis.md) — ROS 2 topics, REST API, WebSocket events, SDK contract
- [`docs/08_motion_presets.md`](docs/08_motion_presets.md) — predefined 1/2/5/10 s motion families
- [`docs/09_tag_detection_service.md`](docs/09_tag_detection_service.md) — live and offline AprilTag detection service
- [`docs/10_repo_tree.md`](docs/10_repo_tree.md) — file-by-file purpose map

## Build philosophy

This project should be developed in **layers**:

1. **Simulator shell**: room + robot + phone rig + one scene camera  
2. **Sensor truth path**: RGB, CameraInfo, IMU, TF, joint states  
3. **Recorder path**: bag + video sidecars + run metadata  
4. **Robot action path**: preset motions + direct command API  
5. **Tag detection service**: offline video first, then live stream  
6. **Validation**: compare estimated calibration results against simulator ground truth  

That order matters. If you try to build everything at once, debugging becomes expensive.

## Notes on the “small tag, visible from afar” requirement

For your stated use case, a **single large 10 cm AprilTag** is usually more useful than a dense calibration board when the camera may be far away and fine detail is wasted.  
However, for **camera–IMU calibration datasets** you will often still want the option to swap in a **sparse AprilGrid** because it gives multiple correspondences while keeping the target fully orientation-resolved.[kalibr-targets]  
The blueprint therefore supports **both**:

- **single-tag mode** for long-range, simple pose estimation
- **board mode** for richer calibration sequences and multi-camera testing

## Sources

[isaac-what]: https://docs.isaacsim.omniverse.nvidia.com/
[isaac-cam]: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/sensors/isaacsim_sensors_camera.html
[isaac-imu]: https://docs.isaacsim.omniverse.nvidia.com/4.5.0/sensors/isaacsim_sensors_physics_imu.html
[isaac-ros2]: https://docs.isaacsim.omniverse.nvidia.com/6.0.0/ros2_tutorials/ros2_landing_page.html
[isaac-articulation]: https://docs.isaacsim.omniverse.nvidia.com/4.5.0/robot_simulation/articulation_controller.html
[isaac-urdf]: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/importer_exporter/import_urdf.html
[isaac-6-edr]: https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/index.html
[replicacad]: https://aihabitat.org/datasets/replica_cad/
[polyhaven]: https://polyhaven.com/
[opencv-aruco]: https://docs.opencv.org/4.x/de/d67/group__objdetect__aruco.html
[kalibr-targets]: https://github.com/ethz-asl/kalibr/wiki/calibration-targets
