# Batch Estimation

This repo now includes a batch estimation package under `src/calib_sim/estimation/` for unknown-map visual bundle adjustment and visual-inertial batch MAP on the interactive simulator recordings.

## Scope

Implemented stages:

- V1: visual-only unknown-tag-map batch MAP
- V2: visual-inertial batch MAP with per-frame velocity and one constant gyro / accel bias pair for the sequence
- Laplace-style uncertainty summaries for both the visual-only and fused solves
- ideal / nominal / stress ablations on recorded interactive runs

Not in scope for this pass:

- extrinsic self-calibration
- robot kinematic calibration factors in the main solve
- MCMC / full Bayesian sampling
- time-offset estimation

## Package Layout

Main modules:

- `src/calib_sim/estimation/dataset.py`
- `src/calib_sim/estimation/init_bootstrap.py`
- `src/calib_sim/estimation/graph_build.py`
- `src/calib_sim/estimation/solve_batch_map.py`
- `src/calib_sim/estimation/posterior.py`
- `src/calib_sim/estimation/eval_metrics.py`
- `src/calib_sim/estimation/reporting.py`

Factors and backends:

- `src/calib_sim/estimation/factors/tag_corner_factor.py`
- `src/calib_sim/estimation/factors/imu_preintegration.py`
- `src/calib_sim/estimation/factors/priors.py`
- `src/calib_sim/estimation/backends/jax_backend.py`
- `src/calib_sim/estimation/backends/jax_validation.py`
- `src/calib_sim/estimation/backends/gtsam_backend.py`

## Data Boundary

Inference inputs come from:

- `samples.jsonl`
- `imu.csv`
- `metadata.json`
- device and camera config files referenced by the run metadata

Evaluation-only inputs come from:

- `camera_gt.csv`
- `samples.jsonl["ground_truth"]`

The clean boundary is:

- `load_batch_dataset(...)` never reads ground truth
- `load_evaluation_data(...)` is the only GT-loading path
- `camera_gt.csv` must not affect inference

## Modeling Choices

Visual factors:

- one rigid `SE(3)` pose per tag
- 2D pixel residuals on the 4 tag corners
- no center residual in the main solve
- Huber-style robust weighting

V1 state:

- one camera pose per frame
- one tag pose per observed tag

V2 state:

- one pose per frame
- velocity per frame
- one constant gyro bias over the full sequence
- one constant accel bias over the full sequence
- fixed camera / IMU mount from the device config

Gauge handling:

- the solver anchors the graph on the bootstrap anchor frame, not the chronological first frame
- this matters on the preserved run because the best anchor frame is `119`

## Initialization

The intended sequence is:

1. choose an anchor frame with strong tag coverage
2. set the anchor pose to identity
3. initialize visible tags from the anchor observation
4. expand camera poses and unseen tags outward from initialized neighborhoods
5. solve V1 visual-only first
6. lift the converged V1 solution into a V2 inertial initial guess
7. add IMU, velocity, and bias states
8. solve the fused graph

Checkpoint 03 detail:

- the fused graph now uses a constant-bias model instead of per-frame bias states
- the anchor IMU-state prior is centered on the bootstrap pose and velocity, not a hard-coded zero-velocity state
- the simulator writes explicit IMU convention metadata so the report can state the inertial measurement model exactly

## Noise And Realism

Noise presets live under `config/noise/`:

- `imu_ideal.yaml`
- `imu_async_ideal.yaml`
- `imu_nominal_phone.yaml`
- `imu_stress_phone.yaml`
- `vision_nominal.yaml`
- `timing_nominal.yaml`
- `timing_vi_headline.yaml`

The simulator now supports:

- `checkpoint_01_compat` IMU recording
- async IMU emission through `src/calib_sim/interactive/imu_runtime.py`
- optional `imu_truth.csv` output in async realism mode
- explicit IMU convention fields in `metadata.json`:
  - `imu_measurement_convention`
  - `imu_timestamp_semantics`
  - `imu_gravity_handling`
  - `imu_sampling_mode`

Current realism note:

- async IMU emission is decoupled from frame cadence
- new async runs generate IMU truth at the IMU frame rather than the camera frame
- the motion truth is still interpolated from frame-step endpoints rather than a separate full physics integrator

## Current Checkpoints

Compatibility baseline:

- `output/interactive_runs/run_20260406_083611`

Headline async VI benchmark configuration:

- `config/interactive/tabletop_grab_challenge_vi_headline.yaml`

Checkpoint 03 reporting now treats these separately:

- the preserved run remains the compatibility and regression baseline
- the async fixed-rate run is the scientific headline benchmark

The recommended source of truth for exact metrics is now the generated run-level
scientific artifact bundle:

- `analysis/checkpoint_03_scientific_report.md`
- `analysis/trajectory_accuracy_table.csv`
- `analysis/parameter_plausibility_table.csv`
- `analysis/uncertainty_calibration_table.csv`
- `analysis/noise_sensitivity_table.csv`
- `analysis/factor_breakdown.json`

## Key Artifacts

Generated analysis outputs include:

- `analysis/visual_map_report.md`
- `analysis/visual_inertial_report.md`
- `analysis/checkpoint_03_scientific_report.md`
- `analysis/ablation_summary.json`
- `analysis/uncertainty_summary.json`
- `analysis/trajectory_accuracy_table.csv`
- `analysis/parameter_plausibility_table.csv`
- `analysis/uncertainty_calibration_table.csv`
- `analysis/noise_sensitivity_table.csv`
- `analysis/factor_breakdown.json`
- `analysis/batch_v1_pose_estimates.jsonl`
- `analysis/batch_v2_pose_estimates.jsonl`
- `analysis/imu_only_pose_estimates.jsonl`
- `analysis/batch_v1_convergence.png`
- `analysis/batch_v2_convergence.png`
- `analysis/batch_v1_trajectory_3d.png`
- `analysis/batch_v2_trajectory_3d.png`
- `analysis/trajectory_position_error_compare.png`
- `analysis/trajectory_rotation_error_compare.png`
- `analysis/error_cdf.png`
- `analysis/per_tag_error_bar.png`
- `analysis/per_tag_rotation_error_bar.png`
- `analysis/batch_v1_tag_uncertainty_bar.png`
- `analysis/ablation_pareto.png`
- `analysis/ablation_position_bars.png`
- `analysis/ablation_rotation_bars.png`
- `analysis/coverage_reliability.png`

The two primary markdown reports now include:

- summary tables instead of scalar-only bullets
- a single canonical estimation-problem section with state, measurements, models, priors, and MAP objective
- convergence sections with iteration counts, initial/final cost, and runtime per iteration
- 3D trajectory visualizations
- residual and reprojection-noise diagnostics
- per-tag map tables
- uncertainty coverage tables for both V1 and V2
- IMU bias and velocity timelines
- preset-by-preset ablation comparisons

## Caveats

- the preserved checkpoint IMU is still a synthetic best-case stream, so the main V2 headline solve uses the `imu_ideal` preset
- the preserved checkpoint IMU stream predates the IMU-frame lever-arm fix and should be read as a compatibility benchmark rather than the scientific headline VI dataset
- IMU-only dead reckoning drifts massively when initialized without ground-truth velocity and without visual corrections; that is expected and is now explicit in the saved reports
- the current Laplace uncertainty estimate is now exported for both the visual-only and fused solves
