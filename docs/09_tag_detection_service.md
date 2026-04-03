# 09 — AprilTag detection service

## Purpose

This module takes **live frames** or **recorded video** and returns AprilTag detections in a stable machine-readable format.

For your use case, every detection should include **five 2D points**:

1. corner 0
2. corner 1
3. corner 2
4. corner 3
5. center

plus the tag `id` and `family`.

## Why OpenCV for the first implementation

OpenCV’s `aruco` API includes the predefined dictionary `DICT_APRILTAG_36h11`, AprilTag-based corner refinement, and canonical marker image generation.[opencv-aruco]  
That makes it the simplest pure-Python-friendly path for a v1 service.

If you later want to compare against the AprilRobotics library, keep the detector behind an interface and add a second backend.

## Supported modes

### 1. Offline video mode

Input:

- path to MP4/MOV/AVI
- optional frame stride
- optional camera intrinsics
- optional tag size

Output:

- detections grouped by frame index / timestamp

### 2. Single-frame mode

Input:

- image bytes or numpy array
- optional intrinsics
- optional tag size

Output:

- detections for one frame

### 3. Live stream mode

Input:

- WebSocket frame stream or ROS bridge adapter

Output:

- per-frame detections and optional pose estimates

## Output contract

```json
{
  "frame_index": 12,
  "timestamp_s": 3.400,
  "detections": [
    {
      "family": "36h11",
      "id": 42,
      "corners_xy_clockwise": [
        [412.2, 181.0],
        [507.8, 184.4],
        [503.9, 279.6],
        [408.1, 276.2]
      ],
      "center_xy": [458.0, 230.3],
      "points5_xy": [
        [412.2, 181.0],
        [507.8, 184.4],
        [503.9, 279.6],
        [408.1, 276.2],
        [458.0, 230.3]
      ],
      "pose_camera": null,
      "quality": {
        "detector_backend": "opencv_aruco_apriltag36h11"
      }
    }
  ]
}
```

## Corner ordering

This must be documented and stable.  
The scaffold implementation uses OpenCV’s marker corner output and preserves the returned clockwise order.  
Do not let one client assume clockwise and another assume `lb-rb-rt-lt` without stating it.

## Optional pose estimation

If the caller provides:

- camera intrinsics
- distortion coefficients
- physical tag size

the service can estimate tag pose in camera coordinates using a square PnP method.  
This is useful for quick simulator-side sanity checks, but the core required output remains the **five points + id**.

## Performance notes

AprilTag detection in common Python stacks is usually CPU-side.  
That is acceptable in v1 because:

- the simulator itself is already GPU-heavy,
- the detector service is decoupled,
- offline analysis throughput matters more than low single-frame latency at the start.

If live throughput later becomes the bottleneck, add:

- frame decimation,
- resized preview pass,
- detector worker pool,
- alternate detector backend.

## Relationship to the simulator

The detector service should not know about Isaac Sim internals.  
It should only know about:

- pixels
- timestamps
- optional intrinsics
- optional tag size

That separation makes it reusable on real data later.

## Validation plan

Test the detector in three steps:

1. **Synthetic marker image test** — detect a generated canonical marker.
2. **Rendered still image test** — detect the tag in one simulator frame.
3. **Motion sequence test** — detect across a full preset run and compare with simulator truth.

## Why the chosen tag family is compatible

OpenCV lists `DICT_APRILTAG_36h11` among the predefined dictionaries and provides `generateImageMarker()` for canonical marker creation.[opencv-aruco]  
The AprilTag family itself is established in robotics and AprilTag 2 improved robustness and efficiency over the earlier detector.[apriltag-olson][apriltag2]

## Sources

[opencv-aruco]: https://docs.opencv.org/4.x/de/d67/group__objdetect__aruco.html
[apriltag-olson]: https://april.eecs.umich.edu/media/pdfs/olson2011tags.pdf
[apriltag2]: https://april.eecs.umich.edu/pdfs/wang2016iros.pdf
