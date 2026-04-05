# Interactive Pose-Estimation Pipeline

Historical note:

- This document captures the development path before the current checkpoint was frozen.
- The current authoritative checkpoint state is [checkpoint_01.md](checkpoint_01.md).
- The current operator-facing browser app description is [app.md](app.md).

## Goal

The interactive browser sim now supports a full loop that looks much closer to a
real capture workflow:

1. Move the robot-mounted phone camera automatically with a repeatable demo path.
2. Record the raw phone video plus IMU and exact simulator ground
   truth.
3. Undistort the recorded video offline only when the chosen preset actually
   renders a distorted image.
4. Detect AprilTags on the corrected analysis replay.
5. Estimate pose from image measurements alone with a repo-inspired optimizer.
6. Compare the estimated pose to the simulator truth.

## Runtime Side

The default runtime entrypoint is
`src/calib_sim/interactive/service.py`, backed by
`src/calib_sim/interactive/sim.py`.

Important runtime choices:

- The default camera model is `config/camera/pixel_9a_main.toml`.
- That preset now models the processed phone video path instead of forcing a
  synthetic raw-like pre-correction distortion into the rendered image.
- If you want the stronger raw-like distortion on purpose, use
  `config/camera/pixel_9a_raw_distorted.toml`.
- The browser demo config is `config/interactive/browser_game_demo.yaml`.
- Auto-demo motion is enabled by default so a capture can be produced without
  manual servo input.
- The saved phone video has no point overlay or HUD burned into it.

## Recorded Artifacts

Each run under `output/interactive_runs/run_*/` contains:

- `phone_raw.mp4`: raw phone video
- `imu.csv`: accelerometer and gyroscope stream
- `camera_gt.csv`: world-space camera pose and servo values
- `samples.jsonl`: per-frame detections, exported measurement vectors, and true
  tag/camera geometry
- `analysis/phone_undistorted.mp4`: undistorted replay
- `analysis/phone_undistorted_detected.mp4`: undistorted replay with detected
  patterns and points
- `analysis/pose_estimates.jsonl`: per-frame optimizer output
- `analysis/report.md` and `analysis/summary.json`: aggregated results
- `analysis/representative_observation.json`: one fully expanded
  frame-and-pattern solve with seed details and Newton history

## Camera And Measurement Model

The main camera model is TOML-driven in
`src/calib_sim/interactive/camera_model.py`.

For the Pixel 9a preset:

- active array: `4000 x 3000`
- focal length: `4.53 mm`
- active-array intrinsics: `fx = fy = 2694.107 px`
- distortion coefficients: `[0.15084998, -0.44805366, 0.38709834, 0.0, 0.0]`
- output preview used by the browser demo: `960 x 540`

The exported 5-point measurement keeps the same logical ordering used for the
external `CameraPoseEstimation` workflow:

- `center`
- pattern corner 1
- pattern corner 2
- pattern corner 3
- pattern corner 4

Internally, the analysis stage also has to respect the detector corner order
produced by the synthetic flipped tag texture.

## Offline Estimation Method

The offline estimator lives in `src/calib_sim/interactive/analysis.py`.

Method summary:

1. Load the recorded raw video and the camera model stored in `metadata.json`.
2. If the selected preset rendered a distorted image, undistort each frame
   with the Pixel-style Brown-Conrady model; otherwise analyze the recorded
   frame directly.
3. Detect AprilTags on the corrected analysis frames.
4. Convert detected points into image-plane metric coordinates.
5. Build an initial pose seed from a trivial camera-obscura scale estimate.
6. Evaluate a second seed from OpenCV PnP on the undistorted corner
   correspondences.
7. Refine with a Newton line-search solver whose gradients and Hessians come
   from JAX autodiff instead of SymPy-generated operators.
8. Compare the solved pose to the recorded simulator ground truth.

## Example Result

This section is intentionally kept as a historical milestone, not the current best result.

An end-to-end demo run produced on **April 4, 2026** with the corrected
processed-video Pixel preset is available under:

- `output/interactive_runs/run_20260404_070618`

The recorded analysis summary for that run reports:

- `60` analyzed frames
- `116` solved tag observations
- mean position error: `3.6647364598839176 m`
- median position error: `3.4065517920297665 m`
- mean rotation error: `90.22390066699879 deg`
- mean image-plane reprojection RMSE: `1.976463338341209e-05 m`

The run now also includes visual diagnostics in the analysis folder:

- `representative_fit.png`
- `position_error_timeline.png`
- `reprojection_timeline.png`
- `optimizer_loss_trace.png`

The generated `analysis/report.md` also now contains a dedicated
"Representative Single-Pattern Solve" section that shows:

- the observed 5-point measurement
- the camera-obscura initial guess
- the seed competition between obscura and OpenCV PnP
- the full Newton convergence from the camera-obscura seed
- the convergence from the actually selected seed when it differs

## Interpretation

The important split in the current results is:

- The image-space fit is already good.
- The absolute pose comparison is still poor.

That means the recording, undistortion, redetection, and JAX optimization path
is functioning, but there is still a remaining convention/ambiguity mismatch in
the synthetic tag-to-camera frame used for truth comparison.

So the current implementation should be treated as:

- a working end-to-end recording and optimization pipeline
- a useful benchmark harness for future estimator work
- not yet the final trusted absolute-pose evaluator

## Recommended Next Step

The next improvement should be to unify the synthetic marker frame, detector
corner convention, and optimizer state into one explicitly documented right-
handed frame so that the strong reprojection fit also turns into a strong
physical pose match.
