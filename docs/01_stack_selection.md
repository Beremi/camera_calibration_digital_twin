# 01 — Stack selection and trade-offs

## Chosen baseline

**Simulation core:** NVIDIA Isaac Sim 5.1  
**Robotics middleware:** ROS 2 Humble or Jazzy  
**Orchestration API:** FastAPI + WebSockets  
**Room assets:** ReplicaCAD + optional Poly Haven additions  
**Robot default:** Franka Panda  
**Robot alternatives:** UR5/UR10e or any URDF/USD articulation  
**Tag layer:** AprilTag 2 / `36h11`  
**Tag detector implementation:** OpenCV `aruco` module with `DICT_APRILTAG_36h11`  
**Recording:** rosbag2 / MCAP

## Why Isaac Sim instead of building everything from scratch

The official Isaac Sim docs already show support for:

- Python scripting and standalone scripts,
- articulated robots and articulation controllers,
- camera sensors and distortion models,
- IMU sensors,
- robot assets,
- URDF import,
- ROS 2 bridge workflows.[isaac-python][isaac-articulation][isaac-cam][isaac-imu][isaac-assets][isaac-urdf][isaac-ros2]

That means the bulk of your engineering work can stay focused on:

- your experiment model,
- API contracts,
- motion presets,
- dataset generation,
- calibration validation.

This is better than spending the first month building foundational plumbing.

## Why Isaac Sim 5.1, not 6.0, as the implementation target

Isaac Sim 6.0 documentation currently labels 6.0 as an **Early Developer Release** and says the documentation is incomplete, with GA artifacts not yet available from that page.[isaac-6-edr]  
By contrast, Isaac Sim 5.1 has stable installation, requirements, camera, and ROS 2 pages with recent production documentation timestamps.[isaac-51-quick][isaac-51-req][isaac-cam]

So the blueprint uses:

- **5.1 as the build target now**
- **6.x as an upgrade path later**

This is a practical stability decision, not an ideological one.

## Why not SAPIEN as the primary stack?

SAPIEN is a real option and its official docs show a Python-first simulator with articulated robots and camera workflows.[sapien-docs][sapien-camera]  
If your only goal were a clean open-source research simulator core, SAPIEN would be a serious candidate.

I am not choosing it as the primary recommendation here because your requirements lean heavily on:

- multi-sensor publication,
- robot asset workflows,
- quick manipulator onboarding,
- ROS-facing experiment plumbing,
- digital-twin style sensor stacks.

Isaac Sim has stronger out-of-the-box documentation for that exact bundle.

## Why ROS 2 for the data plane

The Isaac Sim ROS 2 docs explicitly state that Isaac Sim connects to ROS through the ROS 2 bridge and recommend **ROS 2 Humble and Jazzy**.[isaac-ros2]  
That makes ROS 2 the right place for:

- camera topics,
- IMU topics,
- joint states,
- `/tf` / `/tf_static`,
- `/clock`,
- optional tag detection output topics.

This also makes it easy to plug in external calibration software or wrappers that already speak ROS.

## Why FastAPI for orchestration

FastAPI provides a typed Python API framework and native WebSocket support.[fastapi][fastapi-ws]  
That makes it a good control plane for:

- creating sessions,
- loading configs,
- starting/stopping runs,
- choosing presets,
- querying artifacts,
- streaming lightweight status to a dashboard.

Keep this distinction clear:

- **ROS 2 = robotics data plane**
- **FastAPI = orchestration/control plane**

## Why rosbag2 / MCAP for recording

ROS 2 documentation describes `ros2 bag` as the standard way to record and replay topics, services, and actions.[ros2-bag]  
The rosbag2 repository states that the storage plugin architecture includes `mcap` and `sqlite3`, and that the default storage is `mcap`.[rosbag2-github]

That makes MCAP a good baseline because it fits modern ROS 2 expectations and offline replay workflows.

## Why AprilTag 2 / 36h11

For your specific simulator, the most important tag properties are:

- distinctive orientation,
- simple pose estimation from a single tag,
- robustness at relatively small image footprints,
- stable open-source generation and detection.

OpenCV includes `DICT_APRILTAG_36h11` as a predefined dictionary and can generate canonical marker images from it.[opencv-aruco]  
The original AprilTag system was designed as a robust visual fiducial family for robotics, and the AprilTag 2 paper focuses on faster and more robust detection.[apriltag-olson][apriltag2]

Why `36h11` specifically:

- enough IDs for most lab/sim use,
- higher minimum Hamming distance than `36h10`,
- widely recognized legacy family.

## Why Franka Panda as the default robot

The Isaac Sim manipulator tutorial uses the **Franka Panda** as the reference manipulator example.[isaac-franka]  
That makes it the safest default for a scaffold because:

- the manipulator path is documented,
- the asset is known to the platform,
- the control examples exist.

This is a tooling-first choice.  
You are not claiming Franka is the only or universally best arm for calibration; you are choosing the one with the lowest simulator integration risk.

## Why the robot must be swapable by config

Calibration testing usually evolves from:

- wrist-mounted phone only,
- to wrist phone + scene camera,
- to second arm,
- to another manipulator with different reach or stiffness.

If the robot is not abstracted early, the simulator will calcify around one arm.  
So the robot abstraction should reduce a robot choice to:

- asset path
- articulation root
- joint order
- tool frame
- mount frame
- drive gains
- joint limits
- controller plugin name

That is why the repo includes robot configs instead of hard-coded robot assumptions.

## Final recommendation

If you want the shortest path to a useful system:

1. **Build on Isaac Sim 5.1**
2. **Use ROS 2 Humble/Jazzy**
3. **Use Franka Panda first**
4. **Model the phone as RGB + IMU on the wrist**
5. **Use `DICT_APRILTAG_36h11`**
6. **Record to MCAP**
7. **Publish orchestration through FastAPI**

That combination is the cleanest match to your requirements today.

## Sources

[isaac-python]: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/python_scripting/python_scripting_concepts.html
[isaac-articulation]: https://docs.isaacsim.omniverse.nvidia.com/4.5.0/robot_simulation/articulation_controller.html
[isaac-cam]: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/sensors/isaacsim_sensors_camera.html
[isaac-imu]: https://docs.isaacsim.omniverse.nvidia.com/4.5.0/sensors/isaacsim_sensors_physics_imu.html
[isaac-assets]: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/assets/usd_assets_overview.html
[isaac-urdf]: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/importer_exporter/import_urdf.html
[isaac-ros2]: https://docs.isaacsim.omniverse.nvidia.com/6.0.0/ros2_tutorials/ros2_landing_page.html
[isaac-6-edr]: https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/index.html
[isaac-51-quick]: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/quick-install.html
[isaac-51-req]: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html
[sapien-docs]: https://sapien.ucsd.edu/docs/2.2/index.html
[sapien-camera]: https://sapien.ucsd.edu/docs/2.2/tutorial/rendering/camera.html
[fastapi]: https://fastapi.tiangolo.com/
[fastapi-ws]: https://fastapi.tiangolo.com/advanced/websockets/
[ros2-bag]: https://docs.ros.org/en/rolling/Tutorials/Beginner-CLI-Tools/Recording-And-Playing-Back-Data/Recording-And-Playing-Back-Data.html
[rosbag2-github]: https://github.com/ros2/rosbag2
[opencv-aruco]: https://docs.opencv.org/4.x/de/d67/group__objdetect__aruco.html
[apriltag-olson]: https://april.eecs.umich.edu/media/pdfs/olson2011tags.pdf
[apriltag2]: https://april.eecs.umich.edu/pdfs/wang2016iros.pdf
[isaac-franka]: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/core_api_tutorials/tutorial_core_adding_manipulator.html
