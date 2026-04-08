# Checkpoint 03 Scientific Report

This checked-in note is the repo-level landing page for the current
Checkpoint 03 scientific result. It complements the generated run-level report
bundle under each recording's `analysis/` directory and captures the reporting
interpretation details that matter most for review.

## Scope

Checkpoint 03 establishes:

- a V1 visual-only unknown-tag-map batch MAP estimator
- a V2 visual-inertial batch MAP estimator with one constant gyro bias and one
  constant accelerometer bias per sequence
- Laplace-style posterior uncertainty export for both V1 and V2
- async ideal / nominal / stress ablations on the headline visual-inertial run

The preserved run `output/interactive_runs/run_20260406_083611` remains the
compatibility baseline. The async headline scientific run is
`output/interactive_runs/run_20260407_143339`.

## Headline Outcome

On the current repaired async headline run:

- visual-only mean position error is about `1.04 mm`
- fused mean position error is about `0.94 mm`
- visual-only mean rotation error is about `0.043 deg`
- fused mean rotation error is about `0.038 deg`
- fused mean whitened IMU squared residual per factor is small on the repaired
  clean run
- fused mean accelerometer-bias norm remains far below the `2.0 m/s^2`
  plausibility threshold

On the 5-seed async corruption sweep:

- `nominal`: fused beats visual-only on mean position error
- `stress`: fused beats visual-only on mean position error by a larger margin

## Scientific Interpretation

Checkpoint 03 is now methodologically coherent enough to serve as a real
scientific milestone rather than just an implementation note.

What is strongest:

- the report structure is now cleanly split into state, observations, forward
  models, likelihoods, priors, MAP objective, and posterior uncertainty
- the constant-bias fused model is physically much more plausible than the
  earlier per-frame bias design
- the async headline run is the primary scientific benchmark, rather than the
  older frame-locked compatibility path

What still needs careful wording:

- the current fused IMU factor is a simplified discrete-time interval factor,
  not a full inertial preintegration model with lever-arm and higher-order
  covariance propagation
- the reported scalar 95% position radius is a diagonalized marginal summary,
  not a full Mahalanobis confidence ellipsoid
- the likelihood sweeps are local likelihood-calibration probes around the
  repaired headline solve, not global hyperparameter optimization

## Reporting Precision Notes

Two distinct visual-noise numbers appear in Checkpoint 03 reporting and should
be kept separate:

- assumed visual likelihood sigma: the nominal pixel sigma actually used to
  whiten visual residuals in the optimizer
- empirical robust residual sigma: the realized post-solve residual dispersion
  measured from the fitted trajectory

They are not the same quantity and should not be reported as if they were.

Similarly, the uncertainty export should be interpreted in layers:

- posterior covariance is a local Laplace approximation around the MAP point
- scalar 95% radii are coarse diagonal summaries
- NEES and whitened squared error are stricter covariance-calibration tests

## Where To Look Next

The best local source of truth for the current run is the generated artifact
bundle under:

- `output/interactive_runs/run_20260407_143339/analysis/checkpoint_03_scientific_report.md`
- `output/interactive_runs/run_20260407_143339/analysis/trajectory_accuracy_table.csv`
- `output/interactive_runs/run_20260407_143339/analysis/parameter_plausibility_table.csv`
- `output/interactive_runs/run_20260407_143339/analysis/uncertainty_calibration_table.csv`
- `output/interactive_runs/run_20260407_143339/analysis/noise_sensitivity_table.csv`
- `output/interactive_runs/run_20260407_143339/analysis/likelihood_sweep_table.csv`
- `output/interactive_runs/run_20260407_143339/analysis/factor_breakdown.json`

For the repo-level modeling overview, pair this note with:

- `docs/checkpoint_03_scientific_core.md`
- `docs/estimation.md`
