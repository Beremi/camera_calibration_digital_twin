# Isaac Estimator Second Pass

This branch continues from the frozen first-pass publication milestone tagged
`isaac-first-pass-freeze`.

The first-pass suite under `output/isaac_runs/latest_first_pass_suite` remains
the immutable publication baseline. This branch is for estimator-quality
diagnosis and repair only.

## Scope

The active question on `feature/isaac-estimator-second-pass` is:

- why anchored closed-loop control remains stable while residual quality,
  auxiliary-tag map quality, and covariance calibration are poor

The branch intentionally does not add:

- new scenes
- new robot presets
- raw-log replay re-solving
- broad stress campaigns

## New Tools

- `scripts/analyze_isaac_estimator_quality.py`
  - writes:
    - `analysis/estimator_quality.json`
    - `analysis/estimator_quality.csv`
    - `analysis/tag_update_breakdown.csv`
    - `analysis/anchor_vs_aux_residuals.png`
    - `analysis/smoother_correction_timeline.png`
- `scripts/run_isaac_anchor_vio.py`
  - now supports second-pass isolation switches:
    - `--[no-]use-aux-tags-in-filter`
    - `--[no-]use-aux-tags-in-smoother`
    - `--[no-]use-aux-map-for-control`
    - `--vision-covariance-scale`
    - `--imu-process-covariance-scale`
    - `--post-relocalization-covariance-scale`

## Canonical Diagnostic Command

Run the analyzer on the frozen canonical run:

```bash
source .venv/bin/activate
python scripts/analyze_isaac_estimator_quality.py \
  output/isaac_runs/first_pass_fused_closed-loop_servo_nominal_seed_007
```

## Isolation Matrix

The intended second-pass isolation matrix on seed `007` is:

1. `visual / closed-loop / anchor-only`
2. `fused / closed-loop / anchor-only`
3. `visual / closed-loop / anchor+aux`
4. `fused / closed-loop / anchor+aux`

Use the new switches so anchor-only means:

- `use_aux_tags_in_filter = false`
- `use_aux_tags_in_smoother = false`
- `use_aux_map_for_control = false`

## Current Diagnostic Focus

The new runtime/reporting path now separates:

- anchor vs auxiliary reprojection quality
- native PnP vs fallback detections
- residuals before vs after anchor relocalization
- accepted vs rejected auxiliary-tag updates
- smoother feedback correction norms

That bundle is the required evidence base before changing the online filter or
replacing the smoother core.
