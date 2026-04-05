# 07 — Human-readable API documentation

This file documents the **external contract** of the simulator.

## API philosophy

Use **ROS 2** for the machine-facing robotics streams.  
Use **REST + WebSocket** for the human-facing orchestration layer.

This keeps the interfaces easy to extend.

---

# A. ROS 2 topics

## Core timing and transforms

| Topic | Type | Direction | Notes |
|---|---|---|---|
| `/clock` | `rosgraph_msgs/Clock` | sim -> clients | Simulation time source |
| `/tf` | `tf2_msgs/TFMessage` | sim -> clients | Dynamic transforms |
| `/tf_static` | `tf2_msgs/TFMessage` | sim -> clients | Static transforms |

## Robot state

| Topic | Type | Direction | Notes |
|---|---|---|---|
| `/robot/joint_states` | `sensor_msgs/JointState` | sim -> clients | Ordered to match robot config |
| `/robot/tool_pose` | custom or TF-derived | sim -> clients | Optional convenience topic |

## Phone rig

| Topic | Type | Direction | Notes |
|---|---|---|---|
| `/phone/rgb/image_raw` | `sensor_msgs/Image` | sim -> clients | Primary synthetic phone video |
| `/phone/rgb/camera_info` | `sensor_msgs/CameraInfo` | sim -> clients | Intrinsics/distortion |
| `/phone/imu` | `sensor_msgs/Imu` | sim -> clients | Accel + gyro |

## Scene cameras

| Topic | Type | Direction | Notes |
|---|---|---|---|
| `/scene_cam/<name>/image_raw` | `sensor_msgs/Image` | sim -> clients | Extra scene views |
| `/scene_cam/<name>/camera_info` | `sensor_msgs/CameraInfo` | sim -> clients | Intrinsics for each scene cam |

## Optional ground-truth topics

| Topic | Type | Direction | Notes |
|---|---|---|---|
| `/sim/ground_truth/tag_poses` | custom | sim -> clients | Tag poses in world frame |
| `/sim/ground_truth/phone_pose` | `geometry_msgs/PoseStamped` | sim -> clients | Exact phone pose |
| `/sim/ground_truth/tool_pose` | `geometry_msgs/PoseStamped` | sim -> clients | Exact tool pose |

## Optional control topics

| Topic | Type | Direction | Notes |
|---|---|---|---|
| `/robot/command/joint_targets` | custom or trajectory msg | clients -> sim | For low-latency external control |
| `/robot/command/ee_pose` | custom | clients -> sim | Task-space command path |

---

# B. REST API

Base path: `/v1`

## `POST /v1/sessions`

Create a new simulator session.

### Request

```json
{
  "session_name": "franka_room_a",
  "headless": true
}
```

### Response

```json
{
  "session_id": "sess_001",
  "state": "IDLE"
}
```

## `POST /v1/sessions/{session_id}/scene`

Load a room/scene config.

### Request

```json
{
  "scene_config_path": "config/scene/replicacad_room_01.yaml"
}
```

### Response

```json
{
  "session_id": "sess_001",
  "state": "SCENE_READY",
  "scene_name": "replicacad_room_01"
}
```

## `POST /v1/sessions/{session_id}/robot`

Load robot config.

### Request

```json
{
  "robot_config_path": "config/robot/franka_panda.yaml"
}
```

## `POST /v1/sessions/{session_id}/phone_rig`

Attach the phone rig.

### Request

```json
{
  "device_config_path": "config/device/pixel_9a_phone.yaml"
}
```

## `POST /v1/sessions/{session_id}/tags`

Spawn tags.

### Request

```json
{
  "tag_config_path": "config/tags/apriltag36h11_single.yaml"
}
```

## `POST /v1/sessions/{session_id}/runs`

Create and optionally start a run.

### Request

```json
{
  "run_name": "sweep_5s_single_tag",
  "record": true,
  "record_storage": "mcap"
}
```

## `POST /v1/runs/{run_id}/presets/execute`

Execute a motion preset.

### Request

```json
{
  "preset_id": "vi_excitation",
  "duration_s": 5.0,
  "repeat": 1
}
```

### Response

```json
{
  "run_id": "run_001",
  "accepted": true,
  "queued": true
}
```

## `POST /v1/runs/{run_id}/recording/stop`

Stop recording and finalize artifacts.

## `GET /v1/runs/{run_id}`

Return run status and artifact paths.

### Response

```json
{
  "run_id": "run_001",
  "state": "COMPLETED",
  "artifacts": {
    "bag_dir": "runs/run_001/bag",
    "metadata_json": "runs/run_001/metadata/run.json",
    "preview_mp4": "runs/run_001/preview/phone.mp4"
  }
}
```

---

# C. WebSocket API

## `/ws/v1/sessions/{session_id}/events`

Pushes low-bandwidth status events such as:

- session state changes
- run started/stopped
- recorder armed/disarmed
- publisher health warnings
- detector service health

---

# D. Current browser-sim service

The currently implemented operator-facing app lives in `src/calib_sim/interactive/service.py`.

Useful endpoints:

| Path | Purpose |
|---|---|
| `/` | browser dashboard |
| `/health` | current config and service status |
| `/v1/config` | active interactive config |
| `/v1/catalog` | scene and robot-arm preset catalog |
| `/ws/live` | live sim snapshots and control messages |

### Example message

```json
{
  "event": "recording_started",
  "session_id": "sess_001",
  "run_id": "run_001",
  "sim_time_s": 12.500
}
```

## `/ws/v1/sessions/{session_id}/preview`

Optional lightweight preview stream metadata.  
Do not try to make this your primary image transport if ROS 2 is already present.

---

# D. Python SDK

The SDK should be tiny and explicit.  
It is a convenience wrapper over REST/WebSocket, not a second protocol.

Example:

```python
from calib_sim.sdk.client import CalibSimClient

client = CalibSimClient("http://localhost:8000")
session = client.create_session("demo", headless=True)
client.load_scene(session.session_id, "config/scene/replicacad_room_01.yaml")
client.load_robot(session.session_id, "config/robot/franka_panda.yaml")
client.attach_phone_rig(session.session_id, "config/device/pixel_9a_phone.yaml")
client.spawn_tags(session.session_id, "config/tags/apriltag36h11_single.yaml")
run = client.create_run(session.session_id, "vi_run", record=True)
client.execute_preset(run.run_id, "vi_excitation", duration_s=5.0)
```

---

# E. Tag detector service API

## `POST /v1/detect/frame`

Input: one image frame  
Output: list of detections

### Response shape

```json
{
  "frame_index": 0,
  "detections": [
    {
      "family": "36h11",
      "id": 42,
      "corners_xy": [[100.0, 120.0], [180.0, 120.0], [180.0, 200.0], [100.0, 200.0]],
      "center_xy": [140.0, 160.0],
      "points5_xy": [[100.0, 120.0], [180.0, 120.0], [180.0, 200.0], [100.0, 200.0], [140.0, 160.0]]
    }
  ]
}
```

## `POST /v1/detect/video`

Input: video file or path  
Output: detections grouped by frame/timestamp

## `/ws/v1/detect/live`

Optional live frame-by-frame detector feed.

---

# F. Versioning and extension rules

1. Keep REST paths versioned (`/v1/...`).
2. Never silently rename ROS topics; add aliases first.
3. Add new response fields as optional before making them required.
4. Keep detector output corner order documented and stable.
5. Preserve simulator truth channels even if the UI changes.

## Why these API choices are grounded

ROS 2 is the natural robotics data path because Isaac Sim’s bridge is official and recommended with Humble/Jazzy.[isaac-ros2]  
FastAPI supports typed Python APIs and WebSockets, which makes it a good orchestration/control layer.[fastapi][fastapi-ws]

## Sources

[isaac-ros2]: https://docs.isaacsim.omniverse.nvidia.com/6.0.0/ros2_tutorials/ros2_landing_page.html
[fastapi]: https://fastapi.tiangolo.com/
[fastapi-ws]: https://fastapi.tiangolo.com/advanced/websockets/
