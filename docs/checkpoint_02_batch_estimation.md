# Checkpoint 02 Batch Estimation

Checkpoint 02 preserves the first repo state with a working unknown-map batch estimator and a working visual-inertial batch estimator on top of the browser-backed interactive simulator data products.

Reference run:

- `output/interactive_runs/run_20260406_083611`

Primary new reports:

- `output/interactive_runs/run_20260406_083611/analysis/visual_map_report.md`
- `output/interactive_runs/run_20260406_083611/analysis/visual_inertial_report.md`
- `output/interactive_runs/run_20260406_083611/analysis/ablation_summary.json`
- `output/interactive_runs/run_20260406_083611/analysis/uncertainty_summary.json`

Key new figures:

- `output/interactive_runs/run_20260406_083611/analysis/batch_v1_convergence.png`
- `output/interactive_runs/run_20260406_083611/analysis/batch_v2_convergence.png`
- `output/interactive_runs/run_20260406_083611/analysis/batch_v1_trajectory_3d.png`
- `output/interactive_runs/run_20260406_083611/analysis/batch_v2_trajectory_3d.png`
- `output/interactive_runs/run_20260406_083611/analysis/trajectory_position_error_compare.png`
- `output/interactive_runs/run_20260406_083611/analysis/ablation_position_bars.png`
- `output/interactive_runs/run_20260406_083611/analysis/ablation_rotation_bars.png`

## What Changed

Compared with Checkpoint 01:

- the repo no longer stops at per-tag fitting and joint known-map world fitting
- V1 now solves camera trajectory plus tag map jointly without GT tag poses in inference
- V2 now adds IMU, velocity, and bias states in the batch solve
- the simulator supports async IMU realism modes while keeping `checkpoint_01_compat` as the default preserved behavior

## Headline Comparison

| Case | Mean position error [m] | Mean rotation error [deg] | Mean reprojection RMSE [px] | Iterations |
| --- | --- | --- | --- | --- |
| Visual-only unknown-map | `0.0006534916` | `0.0349735601` | `0.1060125621` | `10` |
| Visual-inertial fused | `0.0006306947` | `0.0300133652` | `0.1066745304` | `5` |
| IMU-only dead reckoning | `474.1154805` | `5.9871722837` | `n/a` | `239` |
| Known-map diagnostic | `0.0007334817` | `0.0754082654` | `0.1211535329` | `9` |

## V1 Results

Visual-only unknown-map batch MAP on the preserved checkpoint:

| Metric | Value |
| --- | --- |
| Mean position error | `0.0006534916 m` |
| Median position error | `0.0005554802 m` |
| P95 position error | `0.0013354352 m` |
| Max position error | `0.0053249662 m` |
| Mean rotation error | `0.0349735601 deg` |
| Mean reprojection RMSE | `0.1060125621 px` |
| Robust inlier ratio | `0.9975490196` |
| Visual rejection count | `7` |
| Cost reduction | `727.96x` |

Interpretation:

- the unknown-map solve stays close to the old known-map joint baseline on the clean preserved run
- reprojection error remains in the sub-pixel regime
- the generated uncertainty summary is conservative on this preserved run

Uncertainty highlights from `analysis/uncertainty_summary.json`:

| Quantity | Value |
| --- | --- |
| Camera pose count | `240` |
| Tag pose count | `4` |
| Mean camera 95% radius | `0.007308 m` |
| Max camera 95% radius | `0.018493 m` |
| Mean tag 95% radius | `0.003876 m` |
| Sigma-error correlation | `0.0766` |

The new V1 report now includes:

- convergence plot
- 3D trajectory view with aligned tag centers
- residual histogram and per-frame reprojection timeline
- per-tag translation, rotation, and uncertainty tables
- worst-frame table with realized error and predicted 95 percent radius

## V2 Results

Fused visual-inertial batch MAP on the exact preserved IMU stream:

| Metric | Fused | IMU-only |
| --- | --- | --- |
| Mean position error [m] | `0.0006306947` | `474.1154805` |
| Mean rotation error [deg] | `0.0300133652` | `5.9871722837` |
| Mean reprojection RMSE [px] | `0.1066745304` | `n/a` |
| Path length [m] | `2.0447` | `1417.2757` |
| Mean speed [m/s] | `0.1027` | `71.1603` |
| Max speed [m/s] | `0.2127` | `212.6503` |
| Cost reduction | `10860021828.44x` | `n/a` |

Interpretation:

- IMU-only drift is now explicit once GT velocity is removed from the main inference path
- fused remains slightly better than visual-only on the exact preserved stream
- the fused convergence and 3D comparison plots make that difference easier to inspect frame by frame

Bias-state diagnostics from the fused report:

| Quantity | Value |
| --- | --- |
| Mean gyro-bias norm | `0.002991 rad/s` |
| Max gyro-bias norm | `0.022068 rad/s` |
| Mean accel-bias norm | `17.018832 m/s^2` |
| Max accel-bias norm | `17.362814 m/s^2` |
| Mean velocity norm | `0.105012 m/s` |
| IMU factor count | `159` |

Important note:

- the fused trajectory is accurate on this exact preserved stream, but the large inferred accel bias shows the current IMU model is still absorbing consistency mismatch; treat the bias plots as diagnostics, not yet as phone-grade calibrated bias estimates

## Noise Ablations

Saved in:

- `output/interactive_runs/run_20260406_083611/analysis/ablation_summary.json`

Headline comparisons:

| Noise preset | Method | Mean position error [m] | Mean rotation error [deg] | Mean reprojection RMSE [px] |
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

These runs use the preserved checkpoint inputs plus measurement-level perturbations from the configured noise presets.

Useful readouts:

- `analysis/ablation_position_bars.png` shows fused staying below visual-only under nominal and stress noise
- `analysis/ablation_rotation_bars.png` shows the same trend in rotation
- `analysis/ablation_pareto.png` makes the visual-only / fused / IMU-only separation immediately visible

## Known-Map Diagnostic

The known-map diagnostic is retained for diagnosis only:

- mean position error: `0.0007334817 m`
- mean rotation error: `0.0754082654 deg`
- mean reprojection RMSE: `0.1211535329 px`

The important detail is that the diagnostic tag map is expressed in the batch solver's anchor-frame gauge, not raw simulator world coordinates.

It is still useful because:

- it gives a fast regression reference when the unknown-map solve changes
- its convergence trace can reveal whether a regression comes from map gauge handling or from the camera-side residual model
- it provides a sanity-check baseline for the V1 convergence plot

## Reporting Improvements

Compared with the first draft of Checkpoint 02 reporting, the preserved run now ships:

- markdown tables instead of scalar-only bullets
- explicit convergence summaries with cost reduction and iteration timing
- 3D trajectory visualizations for visual-only, fused, and IMU-only paths
- visual-only residual histograms and timelines
- per-tag error and uncertainty plots
- fused bias and velocity timelines
- preset-by-preset ablation bar charts
- a worst-frame table for targeted debugging

## Remaining Limits

- the async IMU realism path still uses interpolated motion truth rather than a separate physics-rate dynamics integrator
- fused uncertainty is not yet exported with the same depth as the visual-only Laplace summary
- the fused accel-bias estimate on the ideal preserved run is still physically suspicious and needs follow-up modeling work
- the old `analysis/report.md` remains the legacy long-form narrative; the new batch reports are additive rather than a full replacement
