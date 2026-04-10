# Isaac Second-Pass Handoff

This note replaces the older pre-rerun collaborator handoff that still
described the branch as blocked by catastrophic fused intermittent-anchor
failure.

## Current State

- branch: `feature/isaac-estimator-second-pass`
- canonical zig-zag science packet head: `a18a7aa`
- prior Draft v0 packaging milestone: `6ec3cec`
- publication packet head: `6ef8807`
- science rerun packet head: `a1b0d4f`
- preserved pre-draft checkpoint: `b3f23c9`
- frozen first-pass baseline: `output/isaac_runs/latest_first_pass_suite`
- refreshed second-pass suite: `output/isaac_runs/latest_second_pass_suite`

The commit labels are intentional rather than contradictory: `a18a7aa` changed
the canonical benchmark surface to the five-point zig-zag task, `6ec3cec` was
the earlier review-bundle closure packet for the straight-line Draft v0
package, `6ef8807` closed the initial publication/media packet, and
`a1b0d4f` is still the science rerun packet that first cleared the
intermittent-anchor blocker.

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

## What The Canonical Zig-Zag Suite Shows

- nominal / fused:
  - mean position error: `0.01268 m`
  - mean waypoint error: `0.03234 m`
  - coverage / NEES: `99.79% / 4.21`
- nominal / visual:
  - mean position error: `0.01379 m`
  - mean waypoint error: `0.02822 m`
  - coverage / NEES: `99.79% / 4.99`
- intermittent anchor / fused:
  - mean position error: `0.03112 m`
  - mean waypoint error: `0.03208 m`
  - coverage / NEES: `87.01% / 14.24`
- intermittent anchor / visual:
  - mean position error: `0.01779 m`
  - mean waypoint error: `0.02826 m`
  - coverage / NEES: `90.49% / 7.90`
- servo stress / fused:
  - mean position error: `0.01261 m`
  - mean waypoint error: `0.04421 m`
  - coverage / NEES: `99.79% / 4.20`
- servo stress / visual:
  - mean position error: `0.01252 m`
  - mean waypoint error: `0.03953 m`
  - coverage / NEES: `99.79% / 4.72`

## Honest Claim

The second-pass result is a stabilization result, not a broad fused win.
Fused is competitive on mean position error in the nominal zig-zag condition
and remains well calibrated. Under intermittent anchor suppression it no
longer fails catastrophically, but visual still leads on the current waypoint
and intermittent-anchor error metrics. The broader 3D path makes that trade-off
easier to inspect spatially, but it does not change the underlying claim.

## Control-Facing Read

- nominal:
  - 5 cm waypoint success is `1.0` for visual and `0.6` for fused
  - mean commanded-vs-realized EE path deviation is large and similar for both:
    `0.981 m` visual versus `0.974 m` fused
- intermittent anchor:
  - 5 cm waypoint success is `1.0` for visual and `0.6` for fused
  - mean commanded-vs-realized EE path deviation remains similar:
    `0.979 m` visual versus `0.965 m` fused
- servo stress:
  - 5 cm waypoint success is `0.4` for visual and `0.533` for fused
  - dropped-command duration rises equally for both estimators to `1.947 s`

The useful read is that the repaired fused lock is now stable enough to expose
control trade-offs cleanly on the larger zig-zag path. The present benchmark
still favors visual on waypoint success in nominal and intermittent-anchor
conditions, while the servo-stress condition is closer to a draw.

## Artifacts To Open First

- paper PDF:
  - `report_tex/second_pass_publication_draft.pdf`
- publication summary:
  - `output/isaac_runs/latest_second_pass_suite/analysis/publication_build_summary.json`
- suite summary:
  - `output/isaac_runs/latest_second_pass_suite/analysis/suite_summary.json`
- control-success summary:
  - `output/isaac_runs/latest_second_pass_suite/analysis/control_success_summary.json`
- evidence tables:
  - `docs/isaac_second_pass_evidence_tables.json`
- dashboard:
  - `output/isaac_runs/latest_second_pass_suite/presentation/index.html`
- repo-relative review manifest:
  - `docs/second_pass_review_manifest.json`
- lock:
  - `docs/second_pass_draft_lock.json`

## What Is Already Done

- fused intermittent-anchor blocker cleared
- focused nominal retune completed
- full 18-run suite rerun from the promoted lock
- second-pass PDF rebuilt from fresh suite artifacts
- local presentation bundle and dashboard regenerated
- lock marked `draft_ready = true`
- zig-zag diagnostics and control-success reporting added to the review package

## Best Next Work

- manuscript polish around the narrower stabilization claim
- review the repair trajectory, suppression ablation, and control-success tables
  before opening any new science packet
- broader science only after this draft package is reviewed
