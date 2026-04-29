# Tabletop Workflow

## Supported Surface

This slim branch supports one tabletop workflow only:

- profile: `tabletop_replica`
- motion mode: auto-demo / auto-path
- detector: `new_pupil`
- browser UI: standard tabletop app
- export: `tabletop_autodemo_dataset/`

## Start The App

```bash
export PYTHONPATH=src
export OMNI_KIT_ACCEPT_EULA=YES
./.venv-isaac/bin/python scripts/run_isaac_standard_demo.py --port 8013
```

Open `http://127.0.0.1:8013`.

What you should see:

- one mounted preview
- three observer previews
- robot controls
- noise controls for IMU, vision, and actuation

There are no profile, detector, or streaming selectors in the supported UI anymore.

## Export The Dataset

```bash
export PYTHONPATH=src
export OMNI_KIT_ACCEPT_EULA=YES
./.venv-isaac/bin/python scripts/export_tabletop_autodemo_dataset.py
```

That generates:

- `tabletop_autodemo_dataset/clean/`
- `tabletop_autodemo_dataset/phone_clean/`
- `tabletop_autodemo_dataset/noisy/`
- `tabletop_autodemo_dataset/report.md`

`clean/` is the ideal pinhole sim baseline. `phone_clean/` applies the Pixel 9a
processed-video geometry and the V1 rolling-shutter image-space warp. The
default Pixel 9a profile does not force Brown-Conrady RGB distortion because the
real MP4 is a processed phone video, not a raw sensor frame. `noisy/` is derived
from `phone_clean/` and adds the deterministic Pixel 9a low-light degradation. Every variant also gets a
`phone_capture/` mirror with `video.mp4`, `session.json`, `frames.csv`,
`camera_results.csv`, and `imu.csv` shaped like the real phone capture folder.
The exported sim video is landscape `1280x720` with `rotation_degrees=0`; the
portrait `720x1280` real reference dimensions are kept only as sample metadata
in the Pixel 9a TOML.

Use this when you need the full localization reports and convenience videos:

```bash
./.venv-isaac/bin/python scripts/export_tabletop_autodemo_dataset.py --full-analysis
```

Use this when an older two-variant `clean/` plus `noisy/` layout is needed:

```bash
./.venv-isaac/bin/python scripts/export_tabletop_autodemo_dataset.py --skip-phone-export
```

## Noise Model

The noisy export is deterministic and intended to be the landscape real-video
swap-in variant for the Pixel 9a low-light sample.

It uses:

- the same Pixel 9a processed-video geometry as `phone_clean/`
- luma/chroma noise in YCrCb space
- mild blur and softness
- ISO-driven gain variation
- chroma subsampling and compression behavior
- H.264/BT.709 phone-capture video output

It does not yet use:

- a full Pixel ISP or demosaic model
- exact depth-aware per-row rolling-shutter re-rendering
- real dropped analysis-frame timing
- a replay of the exact Android IMU hardware stack

## Sensor Rates

The supported tabletop dataset uses:

- mounted camera: `1280x720 @ 30 Hz`, using the Pixel 9a landscape processed-video crop
- IMU: `58.236 Hz`, matching the observed Pixel 9a uncalibrated accelerometer/gyroscope rate
- observer previews: lightweight monitoring views in the web app

## Main Outputs

Inside each dataset variant:

- `phone_capture/video.mp4`
- `phone_capture/session.json`
- `phone_capture/frames.csv`
- `phone_capture/camera_results.csv`
- `phone_capture/imu.csv`
- `raw/imu.csv`
- `raw/camera_frames.jsonl`
- `gt/tag_image_projections.jsonl`
- `localization/new_pupil/detections.jsonl`
- `localization/new_pupil/detections_overlay.mp4`
- `localization/new_pupil/backend_summary.json`
- `mounted_video.mp4` in `--full-analysis` exports
- `mounted_video_timestamps.csv` in `--full-analysis` exports
- `gt_path.csv` in `--full-analysis` exports

The checked-in summary is in:

- [tabletop_autodemo_dataset/report.md](/home/beremi/repos/camera_calibration_digital_twin/tabletop_autodemo_dataset/report.md)
