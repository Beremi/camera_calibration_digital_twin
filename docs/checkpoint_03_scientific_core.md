# Checkpoint 03 Scientific Core

This note captures the canonical scientific model introduced in Checkpoint 03.
It is the repo-level reference for the model and reporting changes behind the
async visual-inertial benchmark and the generated scientific artifacts written
under each run's `analysis/` directory.

## Scope

Checkpoint 03 is focused on two changes:

1. make the simulator and estimator state the IMU convention explicitly
2. simplify the fused V2 estimator to one constant gyro bias and one constant
   accelerometer bias for the full sequence

The preserved run `output/interactive_runs/run_20260406_083611` remains the
compatibility baseline. The async headline benchmark is configured by:

- `config/interactive/tabletop_grab_challenge_vi_headline.yaml`
- `config/noise/imu_async_ideal.yaml`
- `config/noise/timing_vi_headline.yaml`

## Canonical Estimation Problem

The visual state is:

- one camera pose `T_WC_k` per analyzed frame
- one rigid world pose `T_WT_j` per observed tag

The fused state is:

- the visual state above
- one world velocity `v_k` per frame
- one constant gyro bias `b_g` shared across the sequence
- one constant accelerometer bias `b_a` shared across the sequence

Known constants are:

- camera intrinsics from the processed-video camera model snapshot
- fixed camera-to-IMU extrinsics from the device config
- gravity `g_W = (0, 0, -9.81) m/s^2`
- frame and IMU timestamps
- tag side lengths

The visual measurements are the raw pixel-space AprilTag corner observations
stored in `samples.jsonl` under `detections[*].corners_xy_clockwise`. The batch
estimator works directly in pixel space and does not use the legacy 5-point
metric image-plane export for inference.

The IMU measurements are explicitly declared in `metadata.json` using:

- `imu_measurement_convention`
- `imu_timestamp_semantics`
- `imu_gravity_handling`
- `imu_sampling_mode`

For the current async simulator path, the accelerometer convention is specific
force in the IMU body frame:

`tilde_f_k = R_IW_k (a_W_k - g_W) + b_a + n_a`

and the gyro convention is:

`tilde_omega_k = omega_k + b_g + n_g`

## Noise Taxonomy

Checkpoint 03 separates four categories cleanly:

1. synthetic data corruption
2. estimator likelihood model
3. robustification
4. priors / process model

Synthetic data corruption includes visual corner perturbation, dropped
detections, IMU white noise, and IMU bias random walk in the perturbed ablation
datasets. The likelihood model is the covariance used to whiten residuals
inside the solver. Huber robustification is applied only on top of the whitened
visual residual and is not itself a Gaussian likelihood. Initialization remains
separate from priors.

For the clean async headline solve, the optimizer's assumed visual likelihood
scale is the configured nominal pixel sigma from `vision_nominal`, while the
reported robust residual sigma in the generated report is an empirical
post-solve dispersion summary. Those two numbers should not be conflated.

## Priors and Gauge

The batch graphs use:

- an anchor-pose prior to remove the global 6-DoF gauge
- an initial velocity prior in V2
- one zero-mean gyro-bias prior in V2
- one zero-mean accelerometer-bias prior in V2

Metric scale remains observable through the known tag side lengths.

## Uncertainty

Uncertainty is reported as a local Laplace approximation built from the inverse
regularized normal matrix at the MAP point. Checkpoint 03 extends this export
beyond V1 pose and tag marginals to include:

- trajectory position and orientation marginals
- per-frame velocity marginals
- global gyro-bias marginal
- global accelerometer-bias marginal

The exported scalar 95% radius is a diagonalized marginal summary, not a full
Mahalanobis confidence ellipsoid.

The generated run-level scientific artifacts now include:

- `analysis/checkpoint_03_scientific_report.md`
- `analysis/trajectory_accuracy_table.csv`
- `analysis/parameter_plausibility_table.csv`
- `analysis/uncertainty_calibration_table.csv`
- `analysis/noise_sensitivity_table.csv`
- `analysis/factor_breakdown.json`

## Headline Outcome

The current headline async benchmark run is:

- `output/interactive_runs/run_20260407_143339`

Its generated scientific artifacts show:

- visual-only headline accuracy of about `1.04 mm` mean position error and
  `0.043 deg` mean rotation error on the clean async run
- fused headline accuracy of about `0.94 mm` mean position error and
  `0.038 deg` mean rotation error on the clean async run
- a physically plausible fused mean accelerometer-bias norm below the
  `2.0 m/s^2` plausibility threshold
- a small mean whitened IMU squared residual per factor on the repaired clean
  async run
- a 5-seed async sweep in which fused beats visual-only on mean position error
  under both the `nominal` and `stress` corruption regimes

For the current benchmark, the 5-seed aggregate means are approximately:

- `nominal`: visual-only `9.46 mm`, fused `7.63 mm`
- `stress`: visual-only `0.430 m`, fused `0.091 m`

## Compatibility vs Headline Runs

The preserved checkpoint run still uses the older frame-locked compatibility
IMU stream and is retained for regression stability. The new async headline
benchmark records IMU packets at an independent fixed rate and writes the IMU
convention explicitly into `metadata.json`.

That distinction matters scientifically: compatibility runs are still useful
for regression and artifact continuity, but the async headline run is the
primary basis for judging the physical plausibility of the fused model and its
reported uncertainty.

## Interpretation Notes

Two additional precision notes matter when discussing Checkpoint 03 publicly:

- the current fused IMU factor is a simplified discrete-time interval factor,
  not a full inertial preintegration model with lever-arm and higher-order
  covariance propagation
- the likelihood sweeps exported under each run's `analysis/` directory are
  local likelihood-calibration sweeps, not global hyperparameter searches
