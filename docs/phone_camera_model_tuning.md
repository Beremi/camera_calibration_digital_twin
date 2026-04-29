# Pixel 9a Camera Model Tuning Memo

This memo records how the tabletop sim camera is tuned against the real Pixel 9a
capture checked into
[`Beremi/bayesian_camera_IMU_calibration`](https://github.com/Beremi/bayesian_camera_IMU_calibration).

External capture inspected:

- Repository HEAD: `8ae5a1232616273187badb6afaf13cde67b6d616`
- Capture path: `sample_capture/`
- Main files: `video.mp4`, `session.json`, `frames.csv`, `camera_results.csv`, `imu.csv`

## TLDR

The sim export now has three camera variants for the tabletop dataset:

- `clean/`: ideal pinhole sim baseline for existing workflows.
- `phone_clean/`: Pixel 9a processed-video geometry with landscape video
  intrinsics, V1 rolling-shutter image-space warp, and
  real-capture-shaped sidecars.
- `noisy/`: derived from `phone_clean/` with deterministic low-light RGB and IMU
  degradation tuned against the real Pixel 9a sample.

Use `noisy/phone_capture/` when you want algorithms to swap between sim output
and the real `sample_capture/` folder with the same high-level file contract.
The current sim contract is landscape, so algorithms that are hard-coded to the
portrait reference MP4 dimensions must normalize orientation at ingest.

The phone-matched variants are aligned to the real sample capture at the
active-array crop, intrinsics, metadata, codec, and nominal timing level:

- sim export frame: `1280x720` landscape
- inspected real MP4 frame: `720x1280` portrait
- nominal camera rate: `30 Hz`
- focal length: `4.53 mm`
- effective sim-export intrinsics: `fx=fy=862.11424 px`,
  `cx=640.194848 px`, `cy=362.42144 px`
- active-array source: `4000x3000`, center-cropped to a landscape `4000x2250`
  16:9 region, then scaled to `1280x720`
- stabilization: off in the real capture and not modeled as a stabilizing warp in
  the sim
- phone-capture mirror: `video.mp4`, `session.json`, `frames.csv`,
  `camera_results.csv`, and `imu.csv`

It still does not fully reproduce the phone camera pipeline. The remaining gap
is mainly the exact Pixel ISP and sensor behavior: demosaic/color pipeline,
auto-exposure convergence, depth-aware rolling shutter, exact timestamp jitter,
and exact Android IMU hardware behavior.

## Real Capture Facts

From `sample_capture/session.json`:

- Device: Google Pixel 9a, Android 16
- Camera preset: `FHD`
- Requested frame rate: `30-30`
- Timestamp source: `REALTIME`
- Video stabilization: `OFF`
- Optical stabilization: `OFF`
- Autofocus: `OFF`
- Focus distance: `0.2651071 diopters`
- Active array: `4000x3000`
- Sensor orientation: `90 deg`
- Focal length: `4.53 mm`
- Active-array intrinsics:
  - `fx=2694.107 px`
  - `fy=2694.107 px`
  - `cx=2000.6089 px`
  - `cy=1507.567 px`
- Lens distortion:
  - `[0.15084998, -0.44805366, 0.38709834, 0.0, 0.0]`

From `video.mp4`:

- Encoded stream: H.264 High, `yuv420p`, BT.709
- Encoded size: `720x1280`
- Average frame rate reported by FFmpeg: `29.854032835 Hz`
- Bit rate: about `11.92 Mbps`
- Duration: `7.871633 s`
- Decoded frames: `235`

From `camera_results.csv` and `frames.csv`:

- Frame duration: `33,488,214 ns`, about `29.861252 Hz`
- Rolling shutter skew: `10,681,500 ns`
- Exposure: `20,002,889 ns`
- ISO range in sample: `332` to `533`
- Analysis frames are logged as `640x480` with `rotation_degrees=90`; those are
  analysis-side frames, not the encoded MP4 size.

From `imu.csv`:

- Accelerometer stream: `58.236010 Hz`
- Gyroscope stream: `58.235830 Hz`
- The captured sensor rows are uncalibrated Android sensor rows with bias
  columns.

## Sim Settings And Export Path

`config/isaac/camera/phone_tabletop_demo.yaml` now uses the landscape
processed-video geometry chosen for the sim export:

```text
width_px: 1280
height_px: 720
rate_hz: 30.0
focal_length_mm: 4.53
horizontal_aperture_mm: 6.725791
vertical_aperture_mm: 3.783257
fx_px: 862.114240
fy_px: 862.114240
cx_px: 640.194848
cy_px: 362.421440
distortion_coefficients: [0, 0, 0, 0, 0]
```

That camera remains the ideal `clean/` basis. The exporter then uses
`config/camera/pixel_9a_main.toml` to post-process `phone_clean/` and `noisy/`.
Those phone variants update their camera snapshots and frame metadata to the
Pixel 9a processed-video intrinsics, recompute GT tag projections in the same
pixel coordinate system, and regenerate detector outputs against the
post-processed frames. The Pixel 9a distortion coefficients are retained as
source metadata, but the default profile has
`rendering.apply_lens_distortion_in_render = false`; this matches the real MP4
as a processed phone video rather than raw sensor imagery.

`config/camera/pixel_9a_main.toml` records the landscape processed-video mode
used for sim export while keeping the inspected portrait sample-capture metadata
in `sample_capture_*` fields:

```text
width_px = 1280
height_px = 720
encoded_rotation_degrees = 0
sample_capture_avg_frame_rate_hz = 29.854032835201505
sample_capture_bit_rate_bps = 11920098
```

`src/calib_sim/interactive/camera_model.py` now applies the encoded rotation
when deriving effective intrinsics from the active-array metadata.

`src/calib_sim/reporting/tabletop_autodemo_dataset.py` now adds the
phone-matched export path:

```text
clean        ideal pinhole render
phone_clean  clean plus Pixel 9a processed-video geometry and V1 rolling shutter warp
noisy        phone_clean plus deterministic low-light degradation and phone IMU noise
```

Each variant gets a `phone_capture/` folder shaped like the real sample capture.
When FFmpeg is available, `phone_capture/video.mp4` is encoded as H.264 High,
`yuv420p`, BT.709, no B-frames, at about `11.92 Mbps`. If FFmpeg is not
available, the exporter falls back to OpenCV video writing and marks the codec
metadata as degraded in the variant manifest.

`config/isaac/imu/phone_nominal.yaml` now uses the observed phone IMU rate:

```text
rate_hz: 58.236
read_gravity: true
imu_semantics: specific_force
```

## Intrinsics Derivation

The Pixel metadata is in active-array sensor coordinates. The real sample MP4 is
portrait, but the sim export intentionally uses the same 16:9 crop in landscape
coordinates so the tabletop view fits the simulator and preview UI naturally.

1. Start with active array `4000x3000`.
2. Use a 16:9 landscape crop for the video stream:

   ```text
   crop_x = 0
   crop_y = (3000 - 2250) / 2 = 375
   crop_width = 4000
   crop_height = 2250
   ```

3. Scale the crop to `1280x720`:

   ```text
   scale = 1280 / 4000 = 720 / 2250 = 0.32
   fx_landscape = 2694.107 * 0.32 = 862.11424
   fy_landscape = 2694.107 * 0.32 = 862.11424
   cx_landscape = 2000.6089 * 0.32 = 640.194848
   cy_landscape = (1507.567 - 375) * 0.32 = 362.42144
   ```

4. Use the landscape coordinates directly for sim export:

   ```text
   width = 1280
   height = 720
   fx = fy = 862.11424
   cx = 640.194848
   cy = 362.42144
   encoded_rotation_degrees = 0
   ```

5. Derive the Isaac camera apertures from the physical focal length:

   ```text
   horizontal_aperture_mm = 4.53 * 1280 / 862.11424 = 6.725791
   vertical_aperture_mm = 4.53 * 720 / 862.11424 = 3.783257
   ```

## What The Current Sim Covers

- The same 16:9 active-array crop as the real MP4, exported as landscape
  `1280x720` with `encoded_rotation_degrees=0`.
- Pixel 9a active-array metadata and center-crop geometry.
- Ideal pinhole projection in `clean/`.
- Pixel 9a processed-video intrinsics and matching GT/detector coordinates in
  `phone_clean/` and `noisy/`.
- Pixel distortion metadata is recorded, but RGB distortion is not forced unless
  the phone TOML explicitly enables `apply_lens_distortion_in_render`.
- Fixed focus matching the real capture metadata.
- Stabilization disabled.
- Nominal camera rate of `30 Hz`.
- Camera2-style exposure, ISO, frame-duration, rolling-shutter, focus,
  stabilization, and crop sidecars in `phone_capture/camera_results.csv`.
- Real-app-shaped `phone_capture/frames.csv` rows with landscape `640x480` and
  `rotation_degrees=0`.
- Pixel 9a-style `phone_capture/session.json`.
- H.264/BT.709 phone-capture video path when FFmpeg is available.
- Deterministic low-light export with YCrCb luma/chroma noise, mild blur,
  gain variation, chroma subsampling, and compression behavior.
- IMU sample rate closer to the observed Android uncalibrated streams.
- Android-style uncalibrated accelerometer/gyroscope mirror rows with bias
  columns in `phone_capture/imu.csv`.

## Remaining Discrepancies

These are the remaining V1 gaps that are not avoidable without a deeper sensor
renderer or a larger real-capture calibration set:

- The V1 rolling-shutter effect is an image-space, rotation-dominant warp from
  adjacent camera poses. It exports the correct `10.6815 ms` row-skew metadata,
  but it is not an exact per-row Isaac re-render and does not model
  depth/parallax rolling shutter perfectly.
- The real phone has a full ISP: demosaicing, denoising, sharpening, tone
  mapping, local contrast, color correction, YUV conversion, and encoder rate
  control. The sim approximates the visible low-light degradation after RGB
  rendering; it is not a full Pixel ISP implementation.
- The real capture has Camera2 exposure and ISO metadata. The exporter writes
  matching fixed exposure and deterministic ISO-like rows, but it does not
  reproduce auto-exposure convergence, saturation, or real sensor gain changes.
- The real frame sidecars include analysis-pipeline timing behavior. The sim
  keeps deterministic timestamps and does not replay dropped or delayed analysis
  frames by default.
- The real IMU rows come from Android uncalibrated sensors. The exporter writes
  Android-shaped rows and bias columns mapped from synthetic sim IMU packets,
  not a hardware replay.
- Scene content, lighting, lens flare, motion blur, rolling exposure, and
  hand-held phone micro-motion still depend on the simulated setup, not on a
  direct reconstruction of the real recording environment.

## Does This Fully Cover The Phone Footage?

No, not fully. It now covers the file contract, phone processed-video geometry
in the chosen landscape orientation, nominal timing, low-light visual
degradation, phone-style metadata, and phone-shaped IMU/video sidecars well
enough for V1 algorithm swapping.

The next things to iron out for tighter phone-footage matching are:

1. Replace the image-space rolling-shutter approximation with true per-row
   rendering or depth-aware reprojection.
2. Calibrate the RGB degradation from more real low-light frames, including
   color response, denoise/sharpening, H.264 rate control, and motion blur.
3. Model exposure/gain dynamics and saturation instead of fixed exposure with
   deterministic ISO sampling.
4. Add capture-timing irregularities when algorithms need to be tested against
   dropped or delayed analysis frames.
5. Validate the synthetic IMU statistics against longer real Pixel 9a motion
   captures.
