# 06 — Sensors, recording, and time synchronization

## Phone rig sensor model

Your “mobile attached to robot arm” should be represented as a **phone rig** with two mandatory sensors:

1. **RGB camera**
2. **IMU** (accelerometer + gyroscope)

Isaac Sim documents both camera and IMU sensors, including camera distortion models and IMU placement on rigid bodies.[isaac-cam][isaac-imu]

## Camera model

The phone camera config should include:

- image size
- fps
- `fx, fy, cx, cy`
- distortion model and coefficients
- near/far clip
- rolling-shutter flag (future option)
- exposure / motion blur policy (future option)

Isaac Sim’s camera docs expose APIs for calibration and lens distortion models, including OpenCV pinhole/fisheye style parameterization.[isaac-cam]

### Recommended v1 defaults

- `resolution`: `1920x1080`
- `fps`: `30`
- `model`: pinhole
- `distortion`: radial-tangential or zero-distortion for the first smoke test

Start with zero or mild distortion first.  
Once the full pipeline works, reintroduce realistic distortion.

## IMU model

The phone IMU config should include:

- sample rate
- local transform
- accel noise density
- gyro noise density
- accel bias random walk
- gyro bias random walk
- optional gravity read flag

The Isaac Sim IMU docs note that the sensor outputs simulated accelerometer and gyroscope readings and can be placed on rigid body prims.[isaac-imu]

### Recommended v1 defaults

- `imu_rate_hz`: `200`
- `bias`: start at zero
- `noise`: start low but non-zero
- `gravity`: enabled

The point of v1 is to validate the data path first; you can harden the inertial realism later.

## Scene cameras

Scene cameras should be first-class configurable devices, not ad hoc debug viewports.

Each external camera should have:

- name / namespace
- mount type (`fixed`, `rigid_to_robot`, `scripted`)
- pose or parent frame
- intrinsics
- fps
- record flag

Typical roles:

- wide room observer
- side observer
- calibration reference camera
- debugging camera

## Time model

This is one of the most important parts of the project.

### Rule

Every outgoing sample must be stamped with the **same simulation time base**.

That means the following messages should share a coherent clock:

- RGB frames
- CameraInfo
- IMU
- joint states
- `/tf`
- tag detection output
- recorder metadata

### Why

External calibration tools are extremely sensitive to time inconsistencies.  
If the phone video and IMU drift because one is stamped from wall-clock and the other from sim time, your calibration results become untrustworthy.

## Publish rates

A good v1 set is:

| Stream | Rate |
|---|---:|
| Physics | 240 Hz |
| Joint states | 120–240 Hz |
| Phone RGB | 30 Hz |
| Phone IMU | 200 Hz |
| Scene cameras | 10–30 Hz |
| Detector output | event-driven or frame-rate matched |

These numbers are design choices, not official constraints.

## ROS 2 topics to record

Record at least:

- `/clock`
- `/tf`
- `/tf_static`
- `/joint_states`
- `/phone/rgb/image_raw`
- `/phone/rgb/camera_info`
- `/phone/imu`
- `/scene_cam/*/image_raw`
- `/scene_cam/*/camera_info`

If you want reproducible debugging, also record:

- `/sim/ground_truth/tag_poses`
- `/sim/ground_truth/phone_pose`
- `/sim/ground_truth/tool_pose`

## Recording format

ROS 2 documentation presents `ros2 bag` as the standard tool for recording and replaying data.[ros2-bag]  
The rosbag2 project states that its storage plugins include `mcap` and `sqlite3`, and that the default is `mcap`.[rosbag2-github]

So the recommended layout for each run is:

```text
runs/<run_id>/
├── bag/                  # rosbag2 / MCAP
├── preview/              # optional MP4 or JPEG contact sheet
├── metadata/run.json     # run metadata
├── metadata/scene.json   # resolved scene config
├── metadata/robot.json   # resolved robot config
├── metadata/sensors.json # intrinsics, extrinsics, IMU params
└── detector/             # optional tag detections
```

## Metadata schema

Every run should save:

- `run_id`
- UTC wall-clock start/end
- sim start/end time
- git SHA
- scenario id
- robot config hash
- motion preset id
- tag layout config
- sensor configs
- dropped-frame counters
- mean message rates

The metadata file often saves more debugging time than the bag itself.

## Failure detection

Add health checks for:

- missing `CameraInfo`
- IMU rate mismatch
- zero-length bag
- joint states not updating
- NaN transforms
- unexpected detector silence when the tag should be visible

## Recommended first test

1. Static scene
2. Static robot
3. Static phone rig
4. One tag in clear view
5. Record 10 seconds
6. Verify:
   - topic counts
   - image timestamps monotonic
   - IMU timestamps monotonic
   - detector sees the tag in nearly every frame

Only after this should you add motion.

## Sources

[isaac-cam]: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/sensors/isaacsim_sensors_camera.html
[isaac-imu]: https://docs.isaacsim.omniverse.nvidia.com/4.5.0/sensors/isaacsim_sensors_physics_imu.html
[ros2-bag]: https://docs.ros.org/en/rolling/Tutorials/Beginner-CLI-Tools/Recording-And-Playing-Back-Data/Recording-And-Playing-Back-Data.html
[rosbag2-github]: https://github.com/ros2/rosbag2
