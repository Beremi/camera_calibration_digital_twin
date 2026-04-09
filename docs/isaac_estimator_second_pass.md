# Isaac Estimator Second Pass

## Current Work Packet

- current commit SHA: `4c565ae`
- preserved pre-draft checkpoint: `b3f23c9`
- current milestone: `second-pass first-draft artifact closure`

This branch continues from the frozen first-pass publication milestone tagged
`isaac-first-pass-freeze`.

The first-pass suite under `output/isaac_runs/latest_first_pass_suite` remains
the immutable publication baseline. This branch is for estimator-quality
diagnosis and repair only.

## Current Closure Status

The second-pass first-draft closure attempt is currently blocked by the fused
intermittent-anchor condition, not by publication plumbing.

What is true right now:

- the full 18-run draft suite was executed once under the provisional
  anchor-only fused lock
- that suite showed catastrophic fused intermittent-anchor instability while
  visual remained well behaved
- two fused-path fixes have since landed during this work packet:
  - the inertial helper now uses standard constant-acceleration position
    kinematics instead of the old doubled position update
  - the runtime now samples synthetic IMU motion from the live camera pose
    instead of the end-effector pose
- those fixes improved fused nominal seed-`007` anchor-only accuracy to
  `0.01556 m` mean position error with sane calibration, but the
  intermittent-anchor fused rerun is still catastrophic at about `248.74 m`
  mean position error and `1.42e7` pose NEES

The practical conclusion is simple: the branch is not blocked on tables,
figures, or media anymore. It is blocked on one remaining fused runtime or
mechanization defect that still makes the dropout condition scientifically
indefensible for the first draft.

## Preserved Checkpoint

The pre-draft checkpoint for this branch is commit `b3f23c9`.

It is preserved in:

- `docs/second_pass_draft_lock.json`
- `output/isaac_runs/second_pass_checkpoint_isolation_seed_007/`

That checkpoint captures the six-run seed-007 isolation bundle that identified
the current branch-level root cause as a fused mechanization or weighting issue.

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

Build the six-run isolation bundle with:

```bash
source .venv/bin/activate
python scripts/build_isaac_second_pass_isolation_bundle.py
```

This writes:

- `output/isaac_runs/latest_second_pass_isolation/summary.csv`
- `output/isaac_runs/latest_second_pass_isolation/summary.json`
- `docs/isaac_second_pass_isolation_summary.md`

## Current Status

The seed-`007` isolation bundle has now been executed on real 8-second Isaac
runs. The current automatic classification is:

- `fused_mechanization_or_weighting_bug`

The evidence for that call is simple:

- anchor-only visual is already numerically sane
- anchor-only fused is still more than 10% worse than anchor-only visual on
  mean position error
- aux-enabled runs stay well behaved rather than exploding
- reprojection metrics are now sane rather than the absurd first-pass values

Representative bundle numbers from
`output/isaac_runs/latest_second_pass_isolation/summary.json`:

- `visual / anchor-only`: position error `0.0166 m`, waypoint error `0.0202 m`,
  coverage `99.59%`, NEES `8.67`
- `fused / anchor-only`: position error `0.0190 m`, waypoint error `0.0228 m`,
  coverage `97.94%`, NEES `9.48`
- `visual / aux-for-control`: aux map error `0.0256 m`
- `fused / aux-for-control`: aux map error `0.0269 m`

That means the aux-tag path is no longer the first suspect on this branch.
The next estimator change should focus on fused process/measurement weighting
before broadening the campaign.

## Current Tuning Note

A small fused-only sweep on the same canonical seed-`007` setup already shows
that increasing `imu_process_covariance_scale` is a useful next-step lever.

Observed follow-up runs:

- `second_pass_fused_anchor_only_seed_007_imu_scale_2`
  - position error `0.01624 m`
  - coverage `98.54%`
  - NEES `5.70`
- `second_pass_fused_anchor_only_seed_007_imu_scale_4`
  - position error `0.01621 m`
  - coverage `98.75%`
  - NEES `5.58`
- `second_pass_fused_aux_control_seed_007_imu_scale_4`
  - position error `0.01632 m`
  - aux map error `0.0240 m`
  - coverage `98.54%`
  - NEES `5.62`

This sweep materially improves fused state accuracy and calibration, but it
does not yet close the full waypoint-error gap to visual closed-loop. That is
the current boundary before a broader second-pass science suite.

## Current Diagnostic Focus

The new runtime/reporting path now separates:

- anchor vs auxiliary reprojection quality
- native PnP vs fallback detections
- residuals before vs after anchor relocalization
- accepted vs rejected auxiliary-tag updates
- smoother feedback correction norms

That bundle is the required evidence base before changing the online filter or
replacing the smoother core.

## First-Draft Closure Path

The next closure target on this branch is the minimum second-pass publication
draft, not a broad benchmark campaign. The intended draft bundle is:

- one locked fused nominal configuration selected by
  `scripts/run_isaac_second_pass_tuning.py`
- one 18-run suite executed by `scripts/run_isaac_second_pass_draft_suite.py`
- one separate second-pass artifact bundle under
  `output/isaac_runs/latest_second_pass_suite/analysis/`
- one separate draft paper compiled from
  `report_tex/second_pass_publication_draft.tex`
- one local-only presentation bundle under
  `output/isaac_runs/latest_second_pass_suite/presentation/`

## Current Draft-Lock Note

The current provisional draft lock now points to the best real fused nominal
candidate seen so far:

- backend: `lightweight`
- reference run:
  `second_pass_tuning_anchor_only_lightweight_seed_007_imu_8p0_vision_4p0_post_1p0_gyro_default_accel_default`
- runtime switches:
  - `use_aux_tags_in_filter = false`
  - `use_aux_tags_in_smoother = false`
  - `use_aux_map_for_control = false`

That choice is provisional rather than celebratory. It is the strongest real
seed-`007` fused nominal configuration collected so far, but the nominal
three-seed comparison still shows fused trailing visual on both mean position
error and mean waypoint error even while fused remains well calibrated
(`coverage ~= 99.6%`, `pose NEES ~= 4.2`). The remaining closure question is
therefore not whether the branch is stable enough to benchmark, but whether the
final nominal backend/weighting choice can narrow that gap enough for the first
draft claim to be strong on the clean condition.

That note now needs one explicit caveat: the current lock and suite membership
should be treated as diagnostic rather than draft-final. The lock still points
to the best nominal fused candidate collected so far, but the dropout condition
remains unresolved even after the latest fused propagation fixes. The next
closure move is therefore a rerun of the draft suite only after the fused
intermittent-anchor path is numerically sane.
