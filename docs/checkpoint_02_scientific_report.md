# Checkpoint 02 Scientific Technical Report

## Title

Visual-Only Unknown-Map and Visual-Inertial Batch Estimation for the Browser-Backed Camera Calibration Digital Twin

## Abstract

This report documents the first repo checkpoint that supports full batch estimation over the browser-backed interactive simulator recordings rather than only per-tag fitting and known-map diagnostics. Two estimators are evaluated on the preserved benchmark run `run_20260406_083611`: a visual-only unknown-tag-map batch maximum-a-posteriori (MAP) estimator (V1) and a fused visual-inertial batch MAP estimator (V2). The visual estimator solves for one camera pose per frame and one rigid `SE(3)` pose per observed tag using 2D pixel-space AprilTag corner residuals. The fused estimator augments the state with per-frame velocity, gyroscope bias, and accelerometer bias and adds discrete IMU preintegration-style constraints. On the preserved run, V1 achieves `0.0006534916 m` mean position error, `0.0349735601 deg` mean rotation error, and `0.1060125621 px` mean reprojection RMSE. V2 slightly improves trajectory accuracy to `0.0006306947 m` and `0.0300133652 deg`, while IMU-only dead reckoning drifts to `474.1154805 m`. Under nominal and stress perturbations, fused estimation remains better than visual-only in trajectory accuracy, although the preserved best-case IMU benchmark still exhibits physically suspicious accelerometer bias estimates. Ground truth is used only for evaluation.

## 1. Problem Statement

The repo previously supported strong single-tag and known-map diagnostics on the preserved browser-sim benchmark, but it did not yet solve the harder problem required for research-grade calibration studies:

1. estimate camera trajectory over time without using known tag world poses in inference
2. estimate a rigid world pose for each observed tag jointly with trajectory
3. move from visual-only estimation to fused visual-inertial estimation
4. preserve a strict no-ground-truth-leak boundary between inference and evaluation

The core research question for Checkpoint 02 is therefore:

"Can the browser-backed simulator data products support a technically credible unknown-map bundle adjustment and a first fused visual-inertial batch estimator, with reproducible metrics and explicit uncertainty outputs, while preserving the current checkpoint behavior?"

## 2. Data Sources and No-GT Rule

### 2.1 Inference Inputs

The estimator consumes only the following run products:

- `samples.jsonl`
- `imu.csv`
- `metadata.json`
- camera and device config files referenced by `metadata.json`

### 2.2 Evaluation-Only Inputs

The following files are evaluation-only and are excluded from inference:

- `camera_gt.csv`
- `samples.jsonl["ground_truth"]`

### 2.3 No-GT Rule

The inference path is intentionally separated from evaluation:

- `load_batch_dataset(...)` does not read ground truth
- `load_evaluation_data(...)` is the only path that reads `camera_gt.csv`
- tag world truth is used only after solving, for metric computation and plots

This separation is a hard requirement for the scientific credibility of the reported results.

## 3. Benchmark Run and Experimental Setting

### 3.1 Preserved Run

| Field | Value |
| --- | --- |
| Run ID | `run_20260406_083611` |
| Config name | `tabletop_grab_challenge` |
| Device config | `config/device/pixel_9a_phone.yaml` |
| Camera model | `config/camera/pixel_9a_main.toml` |
| Preserved output directory | `output/interactive_runs/run_20260406_083611` |
| Frames analyzed | `240` |
| Tag detections used in inference | `714` |
| Logged IMU packets in preserved run | `240` |
| Camera frame rate | `12.0 Hz` |
| Camera render resolution | `960 x 540 px` |

### 3.2 Important Interpretation Note

The preserved run is still a compatibility benchmark:

- the dataset contains `240` IMU packets, which is effectively one logged packet per analyzed frame
- the device model still declares a nominal IMU capability of `58.236 Hz`
- therefore the headline fused result is best interpreted as a preserved synthetic compatibility benchmark, not yet a fully realistic async phone-grade VI benchmark

This matters when interpreting the strong fused trajectory accuracy.

### 3.3 Scene Geometry

| Object | Center [m] | Size [m] |
| --- | --- | --- |
| `table_top` | `(0.18, 0.60, 0.36)` | `(1.08, 0.72, 0.72)` |
| `target_block` | `(0.30, 0.56, 0.75)` | `(0.11, 0.11, 0.11)` |
| `tray` | `(-0.20, 0.64, 0.74)` | `(0.20, 0.14, 0.03)` |
| `pickup_box` | `(0.00, 0.64, 0.79)` | `(0.11, 0.11, 0.11)` |
| Wall width / height | `3.2 / 2.1` | n/a |
| Look-at point | `(0.04, 0.46, 0.78)` | n/a |

### 3.4 Robot Motion Parameters

| Parameter | Value |
| --- | --- |
| Robot preset | `long_reach_tabletop` |
| Base position [m] | `(0.0, 1.08, 0.72)` |
| Shoulder height [m] | `0.08` |
| Link lengths [m] | `(0.34, 0.30, 0.22)` |
| Initial servos [deg] | `(-36.0, 60.0, -56.0)` |
| Servo 1 limits [deg] | `[-55.0, 55.0]` |
| Servo 2 limits [deg] | `[-12.0, 86.0]` |
| Servo 3 limits [deg] | `[-82.0, 76.0]` |
| Servo speeds [deg/s] | `(66.0, 88.0, 108.0)` |
| Auto-demo mode | `keyframes` |
| Auto-demo duration [s] | `20.0` |

### 3.5 Auto-Demo Keyframes

| Time [s] | Servo 1 [deg] | Servo 2 [deg] | Servo 3 [deg] |
| --- | --- | --- | --- |
| 0.0 | -36.0 | 60.0 | -56.0 |
| 2.4 | -20.0 | 76.0 | -78.0 |
| 4.9 | -4.0 | 54.0 | -36.0 |
| 7.3 | 12.0 | 70.0 | -72.0 |
| 9.8 | 6.0 | 34.0 | -18.0 |
| 12.6 | -16.0 | 80.0 | -80.0 |
| 15.4 | -34.0 | 58.0 | -46.0 |
| 17.7 | -10.0 | 66.0 | -68.0 |
| 20.0 | -36.0 | 60.0 | -56.0 |

## 4. Tag Layout

Eight tags are defined in the scene, but only four are observed in the preserved recording and therefore only four enter the unknown-map solve.

| Tag ID | Family | Size [m] | Position [m] | Mount | Observed detections |
| --- | --- | --- | --- | --- | --- |
| 0 | `36h11` | `0.10` | `(-0.92, 0.00, 0.84)` | `wall` | `95` |
| 7 | `36h11` | `0.10` | `(-0.45, 0.00, 1.44)` | `wall` | `0` |
| 19 | `36h11` | `0.10` | `(-0.10, 0.00, 0.92)` | `wall` | `180` |
| 42 | `36h11` | `0.10` | `(0.38, 0.00, 1.22)` | `wall` | `0` |
| 88 | `36h11` | `0.10` | `(0.82, 0.00, 0.88)` | `wall` | `0` |
| 123 | `36h11` | `0.10` | `(0.92, 0.00, 1.54)` | `wall` | `0` |
| 155 | `36h11` | `0.06` | `(0.30, 0.56, 0.806)` | `top` | `240` |
| 201 | `36h11` | `0.06` | `(0.00, 0.64, 0.846)` | `top` | `199` |

## 5. Sensor and Device Parameters

### 5.1 Camera Parameters Used by the Estimator

| Parameter | Value |
| --- | --- |
| Camera model name | `pixel_9a_main` |
| Projection model | `pixel_processed_video_default` |
| Render distortion enabled | `false` |
| Output resolution [px] | `960 x 540` |
| Focal length [mm] | `4.53` |
| Sensor orientation [deg] | `90` |
| `fx` [px] | `646.58568` |
| `fy` [px] | `646.58568` |
| `cx` [px] | `480.146136` |
| `cy` [px] | `271.81608` |
| Distortion model | `opencv_brown_conrady` |
| Distortion coefficients | `(0.15084998, -0.44805366, 0.38709834, 0.0, 0.0)` |
| Legacy bridge order | `center, top_right, bottom_right, bottom_left, top_left` |
| Legacy bridge half extent [m] | `0.05` |

### 5.2 Device and IMU Parameters

| Parameter | Value |
| --- | --- |
| Device | `pixel_9a_phone` |
| Camera nominal rate [Hz] | `30.0` |
| IMU nominal rate [Hz] | `58.236` |
| IMU reads gravity | `true` |
| Camera frame | `phone_camera` |
| IMU frame | `phone_imu` |
| IMU translation wrt camera [m] | `(0.0, 0.0, 0.01)` |
| IMU roll-pitch-yaw wrt camera [deg] | `(0.0, 0.0, 0.0)` |
| Device accel noise std [m/s^2] | `(0.0047856453, 0.0047856453, 0.0047856453)` |
| Device gyro noise std [rad/s] | `(0.0012217305, 0.0012217305, 0.0012217305)` |

## 6. Estimation Methods

### 6.1 Visual-Only Unknown-Map Batch MAP (V1)

The V1 state consists of:

- one camera pose `T_WC_k` for every analyzed frame `k`
- one rigid tag pose `T_WT_j` for every observed tag `j`

No known tag world pose enters inference. Each visual measurement uses only the four ordered tag corners in pixel space.

The visual residual for a detection of tag `j` in frame `k` is:

`r_vis(k, j) = u_hat(k, j) - u_obs(k, j)`

where:

- `u_obs(k, j)` is the stacked 2D pixel measurement of the 4 tag corners
- `u_hat(k, j)` is the camera-model projection of the known local tag square corners transformed by `T_WT_j` and `T_WC_k`

Important modeling decisions:

- tag corners are observations, not free 3D landmarks
- no tag center residual is used
- residuals are computed in pixel space
- robust weighting is Huber-style
- the graph gauge is fixed by a strong prior on the anchor-frame camera pose

### 6.2 Visual-Inertial Batch MAP (V2)

The V2 state augments the trajectory with:

- camera/IMU pose per frame
- velocity per frame
- gyro bias per frame
- accel bias per frame
- the same rigid tag map as V1

The fused state per keyframe is therefore 15D:

`x_k = [pose(6), velocity(3), gyro_bias(3), accel_bias(3)]`

The visual residual reuses the same 2D pixel-corner model through the fixed camera-IMU transform `T_IC`. The inertial factor propagates state between consecutive keyframes with discrete preintegration-style semantics and penalizes:

- position mismatch
- rotation mismatch
- velocity mismatch
- gyro bias continuity
- accel bias continuity

### 6.3 Gauge Fixing and World Alignment

The solver world is not left gauge-free:

- the anchor frame is selected from image evidence
- that anchor pose is fixed to identity with a strong prior
- evaluation then aligns the solved world to `camera_gt.csv` using the first common frame only

This avoids hidden GT leakage while keeping trajectory and tag errors interpretable.

### 6.4 Bootstrap Initialization

Initialization follows a staged bootstrap:

1. select the anchor frame by visible-tag count, then border margin, then image spread, then earliest tie-break
2. set anchor pose to identity
3. initialize visible tags from anchor observations
4. expand neighboring camera poses from initialized tags
5. initialize unseen tags when first observed from an initialized camera pose
6. run V1 first
7. derive V2 velocity initialization by finite differencing the converged V1 trajectory
8. initialize gyro and accel biases to zero
9. rerun as fused batch MAP

For the preserved run, the selected anchor is frame `119` with four visible tags.

## 7. Numerical Parameters

### 7.1 Problem Size

| Quantity | V1 | V2 |
| --- | --- | --- |
| Frames | `240` | `240` |
| Observed tags | `4` | `4` |
| Camera pose variables | `240 x 6 = 1440` | included in IMU state |
| Tag pose variables | `4 x 6 = 24` | `4 x 6 = 24` |
| IMU-state variables | n/a | `240 x 15 = 3600` |
| Total state dimension | `1464` | `3624` |
| Visual factors | `714` | `714` |
| Visual scalar residuals | `714 x 4 x 2 = 5712` | `5712` |
| IMU factors | n/a | `159` |
| IMU scalar residuals | n/a | `159 x 15 = 2385` |

### 7.2 Solver Hyperparameters

| Parameter | Visual-only | Fused |
| --- | --- | --- |
| Optimizer | block Levenberg-Marquardt | block Levenberg-Marquardt |
| Backend | JAX with 64-bit enabled | JAX with 64-bit enabled |
| Initial damping | `1e-3` | `1e-3` |
| Maximum iterations | `18` | `14` |
| Pose prior translation std [m] | `1e-4` | `1e-4` |
| Pose prior rotation std [rad] | `1e-4` | `1e-4` |
| First velocity prior std [m/s] | n/a | `0.1` |
| First bias prior std | n/a | `0.05` |
| Gravity vector [m/s^2] | n/a | `(0.0, 0.0, -9.81)` |

### 7.3 Robust Visual Weighting

For the headline preserved-run solve, the estimator uses the named preset `vision_nominal` as resolved by the in-code loader registry in `src/calib_sim/estimation/noise_models.py`.

| Parameter | Value actually used |
| --- | --- |
| Corner noise sigma [px] | `0.55` |
| Detection drop probability | `0.03` |
| Quality scale | `1.0` |
| Huber delta [px] | `max(3 x 0.55, 1.0) = 1.65` |

Important note:

- the repo also contains `config/noise/vision_nominal.yaml`, but the preserved-run batch analysis resolves the named preset through the in-code registry
- the values above are the values actually used by the estimator entrypoints during analysis

### 7.4 Noise Presets Used for Ablations

The ablations are measurement-level perturbations of the preserved checkpoint data rather than re-simulated runs.

#### Vision perturbations

| Preset | Corner noise std [px] | Detection drop probability | Notes |
| --- | --- | --- | --- |
| ideal | `0.0` | `0.0` | raw preserved observations, but optimized with nominal visual weighting |
| nominal | `0.55` | `0.03` | loaded from the in-code `vision_nominal` preset |
| stress | `1.25` | `0.08` | hard-coded stress override in the reporting experiment |

#### IMU perturbations

| Preset | Accel noise std [m/s^2] | Gyro noise std [rad/s] | Accel RW std | Gyro RW std | Sample drop |
| --- | --- | --- | --- | --- | --- |
| ideal | `(0, 0, 0)` | `(0, 0, 0)` | `(0, 0, 0)` | `(0, 0, 0)` | `0.0` |
| nominal_phone | `(0.0047856453, 0.0047856453, 0.0047856453)` | `(0.0012217305, 0.0012217305, 0.0012217305)` | `(0.00012, 0.00012, 0.00012)` | `(2.5e-05, 2.5e-05, 2.5e-05)` | `0.0` |
| stress_phone | `(0.012, 0.012, 0.012)` | `(0.0032, 0.0032, 0.0032)` | `(0.00045, 0.00045, 0.00045)` | `(0.00012, 0.00012, 0.00012)` | `0.01` |

## 8. Results

### 8.1 Main Benchmark Results

| Case | Mean pos err [m] | Median pos err [m] | P95 pos err [m] | Max pos err [m] | Mean rot err [deg] | Mean reproj RMSE [px] | Iterations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Visual-only unknown-map | `0.0006534916` | `0.0005554802` | `0.0013354352` | `0.0053249662` | `0.0349735601` | `0.1060125621` | `10` |
| Visual-inertial fused | `0.0006306947` | `0.0005425187` | `0.0013433029` | `0.0029759182` | `0.0300133652` | `0.1066745304` | `5` |
| IMU-only dead reckoning | `474.1154805` | n/a | n/a | n/a | `5.9871722837` | `n/a` | `239` |
| Known-map diagnostic | `0.0007334817` | n/a | n/a | n/a | `0.0754082654` | `0.1211535329` | `9` |

### 8.2 Convergence and Runtime

| Solve | Initial cost | Final cost | Cost reduction | Mean runtime per iteration [ms] |
| --- | --- | --- | --- | --- |
| Visual-only unknown-map | `95290.862090` | `130.901744` | `727.96x` | `4617.20` |
| Visual-inertial fused | `1453198214648.756836` | `133.811721` | `1.0860e10 x` | `11753.43` |
| Known-map diagnostic | `99490.034890` | `150.442009` | `661.32x` | reported in run artifact |

Interpretation:

- V1 and V2 both reduce cost dramatically and converge in a small number of LM iterations
- the fused solve starts from a much larger initial cost because inertial consistency terms are added on top of the visual problem
- despite the very large initial fused cost, the final fused cost ends up close to the visual-only final cost

### 8.3 Visual-Only Tag Map Results

| Tag ID | Detections | Translation err [m] | Rotation err [deg] | Reprojection RMSE [px] | 95% radius [m] |
| --- | --- | --- | --- | --- | --- |
| 0 | `95` | `0.002189` | `0.0484` | `0.1440` | `0.005568` |
| 19 | `180` | `0.001653` | `0.1166` | `0.4568` | `0.003457` |
| 155 | `240` | `0.000680` | `0.0634` | `0.0882` | `0.003254` |
| 201 | `199` | `0.001228` | `0.0600` | `0.1076` | `0.003223` |

The top-mounted tabletop tags `155` and `201` dominate the observation count and are estimated most tightly.

### 8.4 Visual Residual Statistics

| Metric | Value |
| --- | --- |
| Mean reprojection RMSE [px] | `0.1060125621` |
| Robust axis sigma [px] | `0.0545` |
| Corner residual p50 [px] | `0.0668` |
| Corner residual p95 [px] | `0.2104` |
| Corner residual p99 [px] | `0.3160` |
| Robust inlier ratio | `0.9975490196` |
| Visual rejection count | `7` |

These residuals are comfortably sub-pixel and consistent with a clean synthetic benchmark.

### 8.5 Fused State Diagnostics

| Quantity | Value |
| --- | --- |
| Mean gyro-bias norm [rad/s] | `0.002991` |
| Max gyro-bias norm [rad/s] | `0.022068` |
| Mean accel-bias norm [m/s^2] | `17.018832` |
| Max accel-bias norm [m/s^2] | `17.362814` |
| Mean velocity norm [m/s] | `0.105012` |
| Max velocity norm [m/s] | `1.438885` |
| IMU factors | `159` |

The large accel-bias magnitude is the main warning sign in the current V2 model. The trajectory result is strong, but the bias estimate is not yet physically credible.

### 8.6 Uncertainty Summary for V1

| Quantity | Value |
| --- | --- |
| Camera pose count | `240` |
| Tag pose count | `4` |
| Mean camera 95% radius [m] | `0.007308` |
| Max camera 95% radius [m] | `0.018493` |
| Mean tag 95% radius [m] | `0.003876` |
| Max tag 95% radius [m] | `0.005568` |
| Position-sigma / error correlation | `0.076638` |

### 8.7 Empirical Coverage of V1 Uncertainty

| Nominal level [%] | Observed coverage [%] |
| --- | --- |
| 50 | `96.3889` |
| 68 | `99.1667` |
| 90 | `99.5833` |
| 95 | `99.5833` |

Interpretation:

- the Laplace uncertainty reported for V1 is conservative on the preserved run
- the conservative behavior is acceptable for this synthetic benchmark, but it shows that calibration of the uncertainty model still needs refinement

### 8.8 Noise Ablations

| Noise preset | Method | Mean pos err [m] | Mean rot err [deg] | Mean reproj RMSE [px] |
| --- | --- | --- | --- | --- |
| ideal | visual-only | `0.0006534916` | `0.0349735601` | `0.1060125621` |
| ideal | fused | `0.0006306947` | `0.0300133652` | `0.1066745304` |
| ideal | IMU-only | `474.1154805` | `5.9871722837` | `n/a` |
| nominal | visual-only | `0.0047840155` | `0.3126083542` | `0.6591097142` |
| nominal | fused | `0.0039651298` | `0.2656115079` | `0.6786496958` |
| nominal | IMU-only | `474.0648351` | `6.0610089243` | `n/a` |
| stress | visual-only | `0.0150807095` | `1.1495186631` | `1.4483209976` |
| stress | fused | `0.0128809165` | `0.9800072964` | `1.5229857819` |
| stress | IMU-only | `461.3673898` | `6.4136348627` | `n/a` |

Interpretation:

- fused is materially better than visual-only in trajectory accuracy under both nominal and stress perturbations
- IMU-only remains dramatically worse, as expected
- reprojection RMSE does not necessarily decrease under fusion because the fused estimator trades visual fit against inertial consistency

## 9. Figures

### 9.1 Visual-Only Convergence

![V1 convergence](../output/interactive_runs/run_20260406_083611/analysis/batch_v1_convergence.png)

### 9.2 Visual-Only 3D Trajectory and Tag Map

![V1 3D trajectory](../output/interactive_runs/run_20260406_083611/analysis/batch_v1_trajectory_3d.png)

### 9.3 Fused 3D Trajectory

![V2 3D trajectory](../output/interactive_runs/run_20260406_083611/analysis/batch_v2_trajectory_3d.png)

### 9.4 Position-Error Comparison

![Position-error comparison](../output/interactive_runs/run_20260406_083611/analysis/trajectory_position_error_compare.png)

### 9.5 Residual Histogram

![Residual histogram](../output/interactive_runs/run_20260406_083611/analysis/batch_v1_residual_hist.png)

### 9.6 Ablation Comparison

![Ablation position bars](../output/interactive_runs/run_20260406_083611/analysis/ablation_position_bars.png)

![Ablation rotation bars](../output/interactive_runs/run_20260406_083611/analysis/ablation_rotation_bars.png)

## 10. Discussion

The technical outcome of Checkpoint 02 is positive.

1. The repo now solves the intended unknown-map problem rather than relying on known world tags.
2. The visual-only unknown-map solution is accurate, stable, and sub-pixel on the preserved benchmark.
3. The fused estimator improves trajectory accuracy under nominal and stress perturbations.
4. IMU-only drift is now made explicit because the inference path no longer uses exact GT start velocity.

However, the checkpoint is not yet the end-state visual-inertial system.

1. The preserved run still uses a best-case IMU compatibility path, not a fully realistic async dataset.
2. The fused accel-bias estimate is too large to be interpreted as a credible phone bias estimate.
3. Visual-only uncertainty is available and useful, but fused uncertainty export is not yet at the same level.

## 11. Main Conclusions

Checkpoint 02 demonstrates that the browser-backed simulator can support a technically credible batch-estimation stack with:

- clean no-GT inference
- unknown-map bundle adjustment
- first fused visual-inertial batch estimation
- quantitative ablations
- uncertainty outputs
- reproducible reporting artifacts

The strongest immediate next step is not a new estimator family. It is improving physical realism and inertial consistency so that bias estimates become believable while preserving the current trajectory quality.

## 12. Reproducibility

Environment bootstrap:

```bash
./tools/bootstrap_sim_env.sh
```

Run the preserved benchmark analysis:

```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
from calib_sim.estimation.reporting import run_batch_estimation_analysis
run_batch_estimation_analysis('output/interactive_runs/run_20260406_083611')
PY
```

Focused regression test:

```bash
.venv/bin/python -m pytest -q tests/test_batch_estimation_regression.py
```

## 13. Source Artifacts

- `output/interactive_runs/run_20260406_083611/analysis/summary.json`
- `output/interactive_runs/run_20260406_083611/analysis/visual_map_report.md`
- `output/interactive_runs/run_20260406_083611/analysis/visual_inertial_report.md`
- `output/interactive_runs/run_20260406_083611/analysis/ablation_summary.json`
- `output/interactive_runs/run_20260406_083611/analysis/uncertainty_summary.json`
- `src/calib_sim/estimation/init_bootstrap.py`
- `src/calib_sim/estimation/graph_build.py`
- `src/calib_sim/estimation/solve_batch_map.py`
- `src/calib_sim/estimation/backends/jax_backend.py`
