# Camera Calibration Digital Twin

This repository currently supports one slim Isaac Sim workflow:

- simulator profile: `tabletop_replica`
- web app: the standard tabletop auto-demo app
- detector backend: `new_pupil`, backed by `pupil_apriltags`
- export script: `scripts/export_tabletop_autodemo_dataset.py`
- dataset bundle: `tabletop_autodemo_dataset/`

Older multi-profile, external detector, and broad experimental surfaces are not part of the supported path on this branch.

## Setup And Run

Run all commands from the repository root:

```bash
cd /home/beremi/repos/camera_calibration_digital_twin
```

The live app and fresh simulation export need:

- Linux workstation with a supported GPU
- Isaac Sim installed in the repository environment at `./.venv-isaac`
- Python packages from `pyproject.toml`, including `fastapi`, `uvicorn`, `opencv-python`, `numpy`, `pupil-apriltags`, `PyYAML`, and `jax`
- NVIDIA Omniverse EULA accepted through `OMNI_KIT_ACCEPT_EULA=YES`

Check that the Isaac environment exists and can see the key packages:

```bash
test -x ./.venv-isaac/bin/python
./.venv-isaac/bin/python - <<'PY'
import importlib.util
for name in ("isaacsim", "cv2", "numpy", "pupil_apriltags", "fastapi", "uvicorn"):
    print(f"{name}: {'ok' if importlib.util.find_spec(name) else 'missing'}")
PY
```

If the repo package or dependencies are not installed in `./.venv-isaac`, install them there:

```bash
./.venv-isaac/bin/python -m pip install -e .
```

Set the environment used by both the web app and exporter:

```bash
export PYTHONPATH=src
export OMNI_KIT_ACCEPT_EULA=YES
```

Start the tabletop web app:

```bash
./.venv-isaac/bin/python scripts/run_isaac_standard_demo.py --port 8013
```

Open:

```text
http://127.0.0.1:8013
```

The health endpoint is useful for checking that the service is alive:

```bash
curl http://127.0.0.1:8013/health
```

The app is locked to the `tabletop_replica` profile. It launches the tabletop auto-demo, shows one mounted camera preview plus three observer previews, supports robot controls and homing, and exposes IMU, vision, and actuation noise controls. The detector path is fixed to `new_pupil`.

## Export Phone-Matched Data

The export script creates three variants by default unless `--capture-only` is used.

- `clean/` is the ideal master capture.
- `phone_clean/` is derived from `clean/`; it applies the Pixel 9a landscape processed-video geometry and the V1 rolling-shutter image-space warp. The default Pixel 9a profile keeps the processed MP4 image pinhole because the real recorded video is already a processed phone output, not raw sensor imagery.
- `noisy/` is derived from `phone_clean/`; it keeps the same phone-video timestamps and GT, then applies deterministic Pixel 9a low-light RGB degradation and nominal phone IMU noise.
- Every variant has a `phone_capture/` mirror with `video.mp4`, `session.json`, `frames.csv`, `camera_results.csv`, and `imu.csv` shaped like the real phone capture app output.
- The sim export is intentionally landscape: `1280x720` video frames and `phone_capture/frames.csv` rows with `rotation_degrees=0`. The inspected real Pixel 9a reference remains recorded as portrait `720x1280`, and its original dimensions are retained only as sample metadata in `config/camera/pixel_9a_main.toml`.
- The default export mode is lightweight corner-only data.
- The full export is selected with `--full-analysis`.

Default corner-only export:

```bash
export PYTHONPATH=src
export OMNI_KIT_ACCEPT_EULA=YES
./.venv-isaac/bin/python scripts/export_tabletop_autodemo_dataset.py
```

This writes `tabletop_autodemo_dataset/clean/`, `tabletop_autodemo_dataset/phone_clean/`, `tabletop_autodemo_dataset/noisy/`, `dataset_export_manifest.json`, and `report.md`. In this mode, tag pose solving, mounted videos, `gt_path.csv`, and meter-based localization reports are intentionally skipped, but each variant still gets a phone-style `phone_capture/video.mp4`.

Full export:

```bash
export PYTHONPATH=src
export OMNI_KIT_ACCEPT_EULA=YES
./.venv-isaac/bin/python scripts/export_tabletop_autodemo_dataset.py --full-analysis
```

This creates the same variants, plus:

- `mounted_video.mp4`
- `mounted_video_timestamps.csv`
- `gt_path.csv`
- `localization/new_pupil/camera_pose_measurements_anchor_only.jsonl`
- `localization/new_pupil/camera_pose_measurements_multitag.jsonl`
- `localization/new_pupil/camera_pose_noise_calibration.json`
- `pixel_precision_report.md`
- root-level pixel precision plots and CSVs under `analysis/`

Useful exporter flags:

```bash
# Write somewhere other than tabletop_autodemo_dataset/
./.venv-isaac/bin/python scripts/export_tabletop_autodemo_dataset.py --output-root /tmp/tabletop_export

# Use a deterministic seed other than the default 7
./.venv-isaac/bin/python scripts/export_tabletop_autodemo_dataset.py --seed 123

# Reuse an existing clean Isaac run instead of capturing a fresh simulation
./.venv-isaac/bin/python scripts/export_tabletop_autodemo_dataset.py --source-run-dir /path/to/clean/run

# Capture and finalize only one clean variant at the output root
./.venv-isaac/bin/python scripts/export_tabletop_autodemo_dataset.py --capture-only --output-root /tmp/clean_capture

# Preserve the older two-variant clean/noisy export
./.venv-isaac/bin/python scripts/export_tabletop_autodemo_dataset.py --skip-phone-export

# Use a different phone camera TOML profile
./.venv-isaac/bin/python scripts/export_tabletop_autodemo_dataset.py --phone-profile config/camera/pixel_9a_main.toml
```

`--capture-only` cannot be combined with `--source-run-dir`.

## Identify Corners In Exported Frames

Use `raw/camera_frames.jsonl` to map a frame index to its image:

- `frame_index` is the mounted camera frame number.
- `rgb_path` is the relative path to the exported PNG, usually `raw/rgb/frame_XXXXXX.png`.
- `timestamp_s`, `sim_time_s`, and `sensor_time_s` are the frame timestamps.
- `visible_gt_tag_ids` lists GT tags that are fully visible after projection.

Use GT image-plane corners from:

```text
<variant>/gt/tag_image_projections.jsonl
```

Each row is one tag projected into one frame. Filter rows by `frame_index`, `tag_id`, and `visibility_flags.fully_visible == true`. The important fields are:

- `corners_xy`: four `[x, y]` pixel coordinates for the rendered visible AprilTag marker square
- `center_xy`: projected tag center in pixels
- `tag_id` and `tag_size_m`
- `visibility_flags`: whether the tag is in front of the camera, in frame, fully visible, or partially visible
- `projection_pose_frame_index`: the camera pose frame used for projection

Use detected Pupil corners from:

```text
<variant>/localization/new_pupil/detections.jsonl
```

In corner-only exports, the same detector output is also exposed at:

```text
<variant>/raw/detections.jsonl
```

Detector rows include:

- `corners_xy`: four detected `[x, y]` pixel coordinates
- `corner_order`: currently `clockwise_top_left_first`
- `tag_id`, `family`, `score`, and `detector_backend`
- `measurement_source`: native, retry, ROI recovery, or temporal tracking source
- `pose_camera_rvec` and `pose_camera_tvec_m` when pose solving is enabled

Pixel coordinates use image coordinates: `x` grows right, `y` grows down, and the origin is the top-left image pixel.

Minimal overlay example:

```python
import json
from pathlib import Path

import cv2
import numpy as np

variant = Path("tabletop_autodemo_dataset/clean")
frame_index = 0
tag_id = 0

frames = [
    json.loads(line)
    for line in (variant / "raw/camera_frames.jsonl").read_text().splitlines()
    if line.strip()
]
frame = next(row for row in frames if int(row["frame_index"]) == frame_index)
image = cv2.imread(str(variant / frame["rgb_path"]))

gt_rows = [
    json.loads(line)
    for line in (variant / "gt/tag_image_projections.jsonl").read_text().splitlines()
    if line.strip()
]
gt = next(
    row
    for row in gt_rows
    if int(row["frame_index"]) == frame_index
    and int(row["tag_id"]) == tag_id
    and row["visibility_flags"]["fully_visible"]
)

detection_path = variant / "localization/new_pupil/detections.jsonl"
if not detection_path.exists():
    detection_path = variant / "raw/detections.jsonl"
detections = [
    json.loads(line)
    for line in detection_path.read_text().splitlines()
    if line.strip()
]
det = next(
    row
    for row in detections
    if int(row["frame_index"]) == frame_index and int(row["tag_id"]) == tag_id
)

gt_poly = np.asarray(gt["corners_xy"], dtype=np.int32).reshape((-1, 1, 2))
det_poly = np.asarray(det["corners_xy"], dtype=np.int32).reshape((-1, 1, 2))

cv2.polylines(image, [gt_poly], isClosed=True, color=(0, 255, 0), thickness=3)
cv2.polylines(image, [det_poly], isClosed=True, color=(0, 0, 255), thickness=2)
cv2.imwrite("corner_overlay.png", image)
```

Green is GT projection and red is the detected Pupil quad.

## Exported Folder Structure

The root export folder is usually `tabletop_autodemo_dataset/`:

```text
tabletop_autodemo_dataset/
  clean/
  phone_clean/
  noisy/
  dataset_export_manifest.json
  report.md
  pixel_precision_report.md          # full-analysis export only
  analysis/                          # full-analysis export only
```

Root files:

- `dataset_export_manifest.json`: top-level export metadata, source capture ID, export mode, paths, and report locations.
- `report.md`: summary report. In corner-only mode this summarizes image-corner detection quality; in full-analysis mode it also includes localization summary tables.
- `pixel_precision_report.md`: full-analysis-only report for per-tag corner RMSE.
- `analysis/`: full-analysis-only CSVs, plots, and JSON summaries for pixel precision.

Each variant folder has this shape:

```text
clean/ or phone_clean/ or noisy/
  dataset_manifest.json
  manifest.json
  config_snapshot/
  raw/
  gt/
  localization/new_pupil/
  analysis/
  estimates/
  phone_capture/
  mounted_video.mp4                  # full-analysis export only
  mounted_video_timestamps.csv       # full-analysis export only
  gt_path.csv                        # full-analysis export only
```

Important per-variant files:

- `dataset_manifest.json`: export-facing manifest. It records the variant name, export mode, sensor rates, timestamp files, GT basis, noise policy, detector set, phone effect profile, phone-capture mirror paths, and convenience export paths when present.
- `manifest.json`: source Isaac run manifest with run ID and capture metadata.
- `config_snapshot/`: JSON copies of the scene, camera, IMU, robot, actuation, control, and estimation configs used for the run.
- `raw/rgb/frame_XXXXXX.png`: mounted camera RGB frames.
- `raw/camera_frames.jsonl`: one row per mounted frame with timestamps, image path, intrinsics, extrinsics, and visible GT tag IDs.
- `raw/imu.csv`: one row per IMU packet with timestamp, angular velocity, specific force, orientation, sensor time, and noise preset.
- `raw/commands.csv`, `raw/realized_joints.csv`, `raw/controller_diagnostics.csv`: robot command, joint, and controller traces.
- `gt/camera_gt.csv`: camera ground truth pose trace from the capture.
- `gt/camera_projection_gt.csv`: camera pose trace matched to image projection frames.
- `gt/camera_projection_pose_trace.jsonl`: raw/projection pose pairing details for each camera frame.
- `gt/imu_gt.csv`, `gt/joint_gt.csv`: ideal IMU and joint ground truth.
- `gt/tag_gt.json`: world-space tag layout, sizes, and anchor metadata.
- `gt/tag_render_geometry.json`: rendered AprilTag marker geometry used for corner projection.
- `gt/tag_image_projections.jsonl`: projected GT image corners and visibility flags for every tag/frame pair.
- `localization/new_pupil/detections.jsonl`: Pupil detector corner rows.
- `localization/new_pupil/detections_overlay.mp4`: quick visual check video with detected tag quads, corner numbers, IDs, scores, and per-frame counts.
- `localization/new_pupil/backend_summary.json` and `.csv`: detection metrics, GT basis, output paths, and full-analysis measurement summaries when enabled.
- `localization/new_pupil/camera_pose_measurements_anchor_only.jsonl`: full-analysis pose measurements using only the anchor tag.
- `localization/new_pupil/camera_pose_measurements_multitag.jsonl`: full-analysis pose measurements using all available tags.
- `localization/new_pupil/camera_pose_noise_calibration.json`: full-analysis residual/noise calibration summary.
- `phone_capture/video.mp4`: phone-style landscape MP4 mirror encoded as H.264/BT.709 with Pixel 9a-style sidecars.
- `phone_capture/frames.csv`: real app analysis-frame sidecar, using `640x480` landscape rows and `rotation_degrees=0`.
- `phone_capture/camera_results.csv`: Camera2-style exposure, ISO, rolling-shutter, focus, stabilization, and crop metadata.
- `phone_capture/imu.csv`: Android-style uncalibrated accelerometer and gyroscope rows with bias columns.
- `phone_capture/session.json`: Pixel 9a-style capture manifest with file sizes, counts, metadata, and quality fields.
- `mounted_video.mp4`: full-analysis mounted camera video encoded from `raw/rgb`.
- `mounted_video_timestamps.csv`: frame-to-video timestamp map.
- `gt_path.csv`: convenient camera GT path export, using projection-matched GT when available.

`clean/` is ideal pinhole. `phone_clean/` and `noisy/` share the same phone-video GT basis; `noisy/` changes RGB frames and `raw/imu.csv`.

## Supported Surface

What currently works:

- run the tabletop auto-demo in Isaac Sim
- inspect the mounted camera plus three observer cameras in the browser
- control the robot, home it, and toggle supported noise modes
- export ideal, phone-clean, and phone-noisy tabletop datasets
- compare `anchor_only` and `multitag` localization outputs in full-analysis mode
- compare detected image corners to rendered GT tag corners

The shorter workflow note is kept at `docs/tabletop_workflow.md`. The current
Pixel 9a tuning memo is in `docs/phone_camera_model_tuning.md`.
