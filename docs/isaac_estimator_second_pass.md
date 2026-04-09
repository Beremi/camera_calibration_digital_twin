# Isaac Estimator Second Pass

## Current Work Packet

- current packet head: `a1b0d4f`
- previous packet head: `3a4a582`
- working baseline commit SHA: `334e5a2`
- preserved pre-draft checkpoint: `b3f23c9`
- current milestone: `suite rerun complete; publication refresh pending`
- control dropout bundle: `output/isaac_runs/latest_second_pass_dropout_debug`

This branch continues from the frozen first-pass publication milestone tagged
`isaac-first-pass-freeze`.

The first-pass suite under `output/isaac_runs/latest_first_pass_suite` remains
the immutable publication baseline. This branch is for estimator-quality
diagnosis and repair only.

## Current Closure Status

The estimator-side blocker for the first usable second-pass draft is now
cleared. The fused intermittent-anchor condition is no longer catastrophic
under the promoted suppression strategy, and the full 18-run suite has been
rerun from the refreshed draft lock. Publication closure is now the remaining
work: regenerate the second-pass PDF, figures, and media bundle from the new
suite artifacts and tighten the written claims around what the rerun actually
shows.

What is true right now:

- the historical control bundle under
  `output/isaac_runs/latest_second_pass_dropout_debug` remains preserved as the
  blocked baseline
- the validated suppression winner has now been promoted into the real draft
  lock:
  - `suppression_propagation_mode = gyro_only`
  - `suppression_imu_specific_force_gate_mps2 = 10.0`
  - `dropout_post_reacquisition_covariance_scale = 8.0`
  - anchor-only runtime policy with aux disabled in filter, smoother, and
    control
- a focused nominal retune selected the current fused nominal reference:
  - run id:
    `second_pass_tuning_anchor_only_lightweight_seed_007_imu_8p0_vision_2p0_post_2p0_gyro_default_accel_default_supp_gyro_only_gate_10p0_covinfl_8p0`
  - mean position error: `0.01293 m`
  - mean waypoint error: `0.02374 m`
  - empirical 95% coverage: `99.79%`
  - pose NEES: `4.23`
- the full 18-run suite has now been rerun from that promoted lock:
  - nominal / fused:
    - mean position error: `0.01303 m`
    - mean waypoint error: `0.02293 m`
    - empirical 95% coverage: `99.72%`
    - pose NEES: `4.25`
  - nominal / visual:
    - mean position error: `0.01388 m`
    - mean waypoint error: `0.01979 m`
    - empirical 95% coverage: `99.79%`
    - pose NEES: `5.01`
  - intermittent anchor / fused:
    - mean position error: `0.03072 m`
    - mean waypoint error: `0.02250 m`
    - empirical 95% coverage: `88.82%`
    - pose NEES: `14.02`
  - intermittent anchor / visual:
    - mean position error: `0.01983 m`
    - mean waypoint error: `0.01874 m`
    - empirical 95% coverage: `90.49%`
    - pose NEES: `10.56`
  - servo stress / fused:
    - mean position error: `0.01328 m`
    - mean waypoint error: `0.02646 m`
    - empirical 95% coverage: `99.79%`
    - pose NEES: `4.26`
  - servo stress / visual:
    - mean position error: `0.01390 m`
    - mean waypoint error: `0.02366 m`
    - empirical 95% coverage: `99.79%`
    - pose NEES: `5.01`
- there is no catastrophic fused intermittent-anchor row in the refreshed
  suite
- nominal fused stayed healthy and is within the requested 10% position-error
  window relative to visual at the suite level
- future runs now persist the actual overridden estimator/control configuration
  into the saved run manifest rather than the raw YAML defaults

The practical conclusion is now straightforward: the branch has a credible
post-fix 18-run science suite and a real fused draft lock. The next packet is
publication closure, not more estimator plumbing unless the refreshed paper
review surfaces a new inconsistency.

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
to the best nominal fused candidate collected so far, and the latest nominal
follow-up run stayed healthy after the suppression-window propagation fixes.
But the dropout condition remains unresolved even after those fixes. The next
closure move is therefore a rerun of the draft suite only after the fused
intermittent-anchor path is numerically sane under the production fused method,
not merely under the diagnostic `no_imu_during_suppression` probe.
