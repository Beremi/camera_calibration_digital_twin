# Isaac Second-Pass Handoff

This note replaces the older pre-rerun collaborator handoff that still
described the branch as blocked by catastrophic fused intermittent-anchor
failure.

## Current State

- branch: `feature/isaac-estimator-second-pass`
- publication packet head: `6ef8807`
- preserved pre-draft checkpoint: `b3f23c9`
- frozen first-pass baseline: `output/isaac_runs/latest_first_pass_suite`
- refreshed second-pass suite: `output/isaac_runs/latest_second_pass_suite`

## Locked Fused Configuration

- backend: `lightweight`
- aux policy:
  - `use_aux_tags_in_filter = false`
  - `use_aux_tags_in_smoother = false`
  - `use_aux_map_for_control = false`
- covariance scales:
  - `imu_process_covariance_scale = 8.0`
  - `vision_covariance_scale = 2.0`
  - `post_relocalization_covariance_scale = 2.0`
- suppression-aware fused overrides:
  - `suppression_propagation_mode = gyro_only`
  - `suppression_imu_specific_force_gate_mps2 = 10.0`
  - `dropout_post_reacquisition_covariance_scale = 8.0`

## What The Refreshed Suite Shows

- nominal / fused:
  - mean position error: `0.01303 m`
  - mean waypoint error: `0.02293 m`
  - coverage / NEES: `99.72% / 4.25`
- nominal / visual:
  - mean position error: `0.01388 m`
  - mean waypoint error: `0.01979 m`
  - coverage / NEES: `99.79% / 5.01`
- intermittent anchor / fused:
  - mean position error: `0.03072 m`
  - mean waypoint error: `0.02250 m`
  - coverage / NEES: `88.82% / 14.02`
- intermittent anchor / visual:
  - mean position error: `0.01983 m`
  - mean waypoint error: `0.01874 m`
  - coverage / NEES: `90.49% / 10.56`

## Honest Claim

The second-pass result is a stabilization result, not a broad fused win.
Fused is competitive on mean position error in the nominal condition and
remains well calibrated. Under intermittent anchor suppression it no longer
fails catastrophically and completes all runs, but visual still leads on the
current waypoint and intermittent-anchor error metrics.

## Artifacts To Open First

- paper PDF:
  - `report_tex/second_pass_publication_draft.pdf`
- publication summary:
  - `output/isaac_runs/latest_second_pass_suite/analysis/publication_build_summary.json`
- suite summary:
  - `output/isaac_runs/latest_second_pass_suite/analysis/suite_summary.json`
- dashboard:
  - `output/isaac_runs/latest_second_pass_suite/presentation/index.html`
- lock:
  - `docs/second_pass_draft_lock.json`

## What Is Already Done

- fused intermittent-anchor blocker cleared
- focused nominal retune completed
- full 18-run suite rerun from the promoted lock
- second-pass PDF rebuilt from fresh suite artifacts
- local presentation bundle and dashboard regenerated
- lock marked `draft_ready = true`

## Best Next Work

- manuscript polish around the narrower stabilization claim
- optional task-facing metric expansion:
  - tolerance-ball success rates
  - commanded-vs-realized path deviation
- broader science only after this draft package is reviewed
