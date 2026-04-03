# 02 — System architecture

## Component breakdown

The system is easiest to maintain when split into **six modules**:

1. **Simulation runtime**
2. **Scene and asset layer**
3. **Robot and sensor rigs**
4. **Recorder**
5. **Experiment/orchestration API**
6. **AprilTag detection service**

```text
calib_sim/
├── sim/            # Isaac Sim bootstrap and runtime loop
├── scene/          # room loading, tag placement, world metadata
├── robot/          # robot factory, controllers, tool mount logic
├── sensors/        # phone rig, extra scene cameras
├── recording/      # rosbag2/MCAP + video sidecars + metadata
├── motions/        # preset generation and execution
├── api/            # REST + WebSocket orchestration
├── tag_service/    # frame/video detection API
└── sdk/            # thin Python client for tests and external tools
```

## Two-plane architecture

This project should use **two communication planes**:

### A. Robotics data plane (ROS 2)

Use this for time-sensitive or ecosystem-facing streams:

- `/clock`
- `/tf`
- `/joint_states`
- `/phone/rgb/image_raw`
- `/phone/rgb/camera_info`
- `/phone/imu`
- `/scene_cam/*`
- `/tags/detections` (optional)
- `/robot/command/*` or `/robot/joint_trajectory` (optional)

Isaac Sim’s official docs describe ROS 2 bridge connectivity and camera publication workflows, and it includes IMU publication nodes as part of the ROS bridge path.[isaac-ros2][isaac-ros2-cam][isaac-ros2-imu]

### B. Control/orchestration plane (FastAPI + WebSocket)

Use this for experiment lifecycle and UX-friendly commands:

- load scene
- switch robot config
- attach phone rig
- spawn or randomize tag layout
- choose motion preset
- start/stop recording
- query artifacts
- push operator status updates
- open a live preview channel

This plane is friendlier for dashboards, notebooks, and CI test harnesses.

## Runtime data flow

```text
        Scene config / Robot config / Motion preset
                        │
                        ▼
              FastAPI orchestrator receives request
                        │
                        ▼
               In-process command queue / state machine
                        │
                        ▼
                 Isaac Sim tick loop consumes action
                        │
          ┌─────────────┴──────────────┐
          ▼                            ▼
   Articulation controller       Sensor rig updates
          │                            │
          ▼                            ▼
   joint states / tf            RGB / CameraInfo / IMU
          └─────────────┬──────────────┘
                        ▼
                     ROS 2 bridge
                        │
             ┌──────────┴───────────┐
             ▼                      ▼
        rosbag2 recorder      external client / SDK
```

## Ground-truth model

A simulator used for calibration testing should publish not only synthetic sensor streams, but also **ground truth**.  
At minimum, make the following ground-truth channels available internally and optionally externally:

- `T_world_robot_base`
- `T_world_tool0`
- `T_tool_phone`
- `T_world_phone`
- `T_world_tag_i`
- `T_phone_tag_i`
- joint positions / velocities
- exact simulated intrinsics / distortion parameters
- exact IMU placement and noise model parameters

These channels let you answer the questions that matter during validation:

- Did the calibration algorithm converge to the right extrinsics?
- Did motion excitation help or hurt?
- Is the detector biasing the estimate?
- Is the bag writer dropping data?

## Time synchronization

Use **simulation time** as the single source of truth.  
That means:

- publish `/clock`
- stamp all messages from the same sim tick time
- write the same timestamps into sidecar JSON
- include `sim_frame_idx` in your internal metadata

Do not stamp video from wall-clock time while the IMU uses sim time.  
That will silently damage calibration experiments.

## Internal state machine

The orchestrator should maintain a simple explicit state machine:

- `IDLE`
- `SCENE_READY`
- `RUNNING`
- `RECORDING`
- `PAUSED`
- `STOPPING`
- `ERROR`

This matters because the calibration bench will eventually be run from scripts, and scriptability is much easier when state is explicit.

### Example transitions

- `IDLE -> SCENE_READY` after loading room + robot + sensors
- `SCENE_READY -> RUNNING` after physics starts
- `RUNNING -> RECORDING` when recorder arms and all publishers are healthy
- `RECORDING -> RUNNING` after recorder stops
- `RUNNING -> IDLE` when scene resets
- `* -> ERROR` on failed asset load, bad command, or timing fault

## Robot abstraction boundary

The robot module should expose an interface such as:

- `load_robot(config)`
- `get_joint_state()`
- `execute_joint_targets()`
- `execute_ee_pose_targets()`
- `attach_rigid_mount(child_name, transform)`
- `get_frame_pose(frame_name)`

Everything Isaac-specific should stay inside this module.  
The orchestration layer should not care whether the loaded arm is Franka, UR5e, or another URDF/USD articulation.

## Sensor abstraction boundary

The sensor layer should present **logical devices**, not raw simulator primitives:

- `PhoneRig`
- `SceneCameraRig`
- `RecorderInputs`

For example, the phone rig should own:

- RGB camera prim path
- IMU prim path
- intrinsics
- distortion
- rates
- mount transform
- topic namespace

This makes it easy to replace “generic phone” with “iPhone-like wide camera profile” later.

## Detector service boundary

The tag detector service must be separate from the simulator loop for two reasons:

1. You want to test offline video just as easily as live streams.
2. You do not want detector latency to stall the simulation loop.

So the detector gets:

- frame bytes or video path
- timestamp metadata
- optional camera intrinsics
- optional tag size

And it returns:

- `family`
- `id`
- `4 corners`
- `center`
- `5 points`
- optional pose estimate
- quality scores

## Extension points to plan now

Even if you do not implement them in v1, keep clear extension points for:

- rolling-shutter phone camera models
- multi-phone rigs
- sparse AprilGrid boards
- pose-noise injection
- detector comparison (OpenCV vs AprilRobotics library)
- motion blur stress tests
- calibration result scoring

## Why this architecture is consistent with official tooling

The architecture mirrors the capabilities already documented by Isaac Sim: articulated control, camera sensors, IMU sensors, ROS 2 publishing, and Python-driven workflows.[isaac-articulation][isaac-cam][isaac-imu][isaac-ros2]  
For offline recording, rosbag2 provides recording/replay and supports MCAP storage plugins.[ros2-bag][rosbag2-github]

## Sources

[isaac-articulation]: https://docs.isaacsim.omniverse.nvidia.com/4.5.0/robot_simulation/articulation_controller.html
[isaac-cam]: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/sensors/isaacsim_sensors_camera.html
[isaac-imu]: https://docs.isaacsim.omniverse.nvidia.com/4.5.0/sensors/isaacsim_sensors_physics_imu.html
[isaac-ros2]: https://docs.isaacsim.omniverse.nvidia.com/6.0.0/ros2_tutorials/ros2_landing_page.html
[isaac-ros2-cam]: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/ros2_tutorials/tutorial_ros2_camera.html
[isaac-ros2-imu]: https://docs.isaacsim.omniverse.nvidia.com/4.5.0/py/source/extensions/isaacsim.ros2.bridge/docs/ogn/OgnROS2PublishImu.html
[ros2-bag]: https://docs.ros.org/en/rolling/Tutorials/Beginner-CLI-Tools/Recording-And-Playing-Back-Data/Recording-And-Playing-Back-Data.html
[rosbag2-github]: https://github.com/ros2/rosbag2
