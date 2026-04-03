# 05 — Robot arm model, servo behavior, and swapability

## Default arm choice

Use **Franka Panda** as the default robot in the first implementation.

That is not because Franka is magically the best calibration arm in every setting.  
It is because Isaac Sim’s official manipulator tutorial directly uses Franka Panda, which lowers integration risk for the first working version.[isaac-franka]

## Alternative arms

Support at least one additional arm config from day one, even if you do not fully tune it yet.  
Good alternatives are:

- UR5 / UR5e
- UR10e
- any other URDF/USD articulation with a stable wrist frame

Isaac Sim’s robot workflows include URDF import and manipulator setup tutorials, so this is a realistic extension path.[isaac-urdf][isaac-ur10e]

## What “swapable by config” should mean

Swapping the robot should **not** require touching controller code.

A robot should be described by config fields such as:

```yaml
name: franka_panda
asset:
  mode: usd
  path: "Isaac/Robots/Franka/franka_alt_fingers.usd"
frames:
  base: "panda_link0"
  ee: "panda_hand"
  mount: "panda_hand"
joints:
  order:
    - panda_joint1
    - panda_joint2
    - panda_joint3
    - panda_joint4
    - panda_joint5
    - panda_joint6
    - panda_joint7
drive:
  mode: position
  stiffness: [800, 800, 800, 600, 250, 150, 50]
  damping: [80, 80, 80, 60, 25, 15, 5]
limits:
  velocity: [2.1, 2.1, 2.1, 2.5, 2.5, 2.5, 2.5]
  effort: [87, 87, 87, 87, 12, 12, 12]
```

The controller then consumes a normalized `RobotConfig` object rather than a robot-specific code path.

## Servo model in the simulator

For calibration testing you usually do **not** need to simulate the low-level motor electronics in detail.  
What you do need is:

- realistic joint position/velocity limits
- reasonable stiffness/damping
- smooth time-parameterized motion
- deterministic repeatability

So the v1 model should use the simulator’s articulation drives as a **servo-like abstraction**:

- position mode for most preset trajectories
- velocity mode for sweeps or reactive control
- optional effort mode for advanced tests

The articulation controller in Isaac Sim is explicitly designed to accept joint position, velocity, and effort commands.[isaac-articulation]

## End-effector and mount frames

Do not mount the phone directly to a visual mesh.  
Define these frames explicitly:

- `base_frame`
- `ee_frame`
- `tool0`
- `phone_mount_frame`
- `phone_camera_frame`
- `phone_imu_frame`

The phone mount should be a small rigid transform chain:

```text
base -> ... -> ee_frame -> tool0 -> phone_mount -> {camera_rgb, imu}
```

That gives you stable extrinsics and keeps the sensor mount decoupled from robot asset naming.

## Phone mount design

The phone rig should be configurable because different experiments may want:

- portrait vs landscape orientation
- camera centered on the wrist vs offset
- IMU center offset from camera center
- an additional observer camera on the wrist

Recommended config fields:

```yaml
mount:
  parent_frame: tool0
  translation_m: [0.02, 0.00, 0.10]
  rpy_deg: [0.0, -90.0, 90.0]
camera_rgb:
  frame: phone_camera
imu:
  frame: phone_imu
  translation_m: [0.00, 0.00, 0.01]
  rpy_deg: [0.0, 0.0, 0.0]
```

## Controller abstraction

The robot layer should provide **three command paths**:

### 1. Joint target execution

Best for deterministic preset playback.

Inputs:

- joint target array
- optional velocity targets
- duration or sample period

### 2. Task-space pose execution

Best for “look at tag” and arc-sweep patterns.

Inputs:

- desired end-effector pose
- interpolation profile
- controller gains

### 3. Streaming low-latency command path

Best for interactive or closed-loop experiments.

Inputs:

- queued command samples from ROS 2 or in-process command bus

## Why the arm choice matters for calibration testing

For this project, the arm is not being chosen for dexterous manipulation benchmarks.  
It is being chosen for:

- repeatable viewpoint motion
- stable wrist transform
- easy sensor mounting
- good documentation
- easy simulator support

That is why the “best” arm is often the one with the cleanest supported integration path.

## Recommended first two robot configs

### A. `franka_panda.yaml`

Use this as the default.

### B. `ur5e.yaml`

Use this as the first alternate config to prove the abstraction is real.

Even if the second robot is not feature-complete on day one, having the config there forces the code to stay generic.

## Optional future upgrade: motion-planning layer

Once the basics work, you can add a higher-level motion generation or planning layer.  
Isaac Sim documents motion-generation interfaces and, in newer lines, cuMotion integration paths for more advanced robot motion workflows.[isaac-motion][isaac-cumotion]  
That is a second-step optimization, not a prerequisite for v1.

## Sources

[isaac-franka]: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/core_api_tutorials/tutorial_core_adding_manipulator.html
[isaac-urdf]: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/importer_exporter/import_urdf.html
[isaac-ur10e]: https://docs.isaacsim.omniverse.nvidia.com/6.0.0/robot_setup_tutorials/tutorial_import_assemble_manipulator.html
[isaac-articulation]: https://docs.isaacsim.omniverse.nvidia.com/4.5.0/robot_simulation/articulation_controller.html
[isaac-motion]: https://docs.isaacsim.omniverse.nvidia.com/4.5.0/manipulators/concepts/index.html
[isaac-cumotion]: https://docs.isaacsim.omniverse.nvidia.com/6.0.0/cumotion/index.html
