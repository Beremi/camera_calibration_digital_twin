# Isaac Estimator Second Pass

## Current Work Packet

- canonical science packet head: `a18a7aa`
- prior Draft v0 packaging milestone: `6ec3cec`
- previous science rerun packet head: `a1b0d4f`
- working baseline commit SHA: `334e5a2`
- preserved pre-draft checkpoint: `b3f23c9`
- current milestone: `zig-zag stabilization package ready for review`
- control dropout bundle: `output/isaac_runs/latest_second_pass_dropout_debug`

This branch continues from the frozen first-pass publication milestone tagged
`isaac-first-pass-freeze`.

The first-pass suite under `output/isaac_runs/latest_first_pass_suite` remains
the immutable publication baseline. This branch is for estimator-quality
diagnosis and repair only.

## Current Closure Status

The estimator-side blocker for the first usable second-pass draft is now
cleared. The canonical branch package is the five-point zig-zag task promoted
at `a18a7aa`: the fused intermittent-anchor condition is no longer
catastrophic under the promoted suppression strategy, the full 18-run zig-zag
suite has been rerun from the refreshed draft lock, and the second-pass PDF
plus local presentation bundle have now been regenerated from those refreshed
artifacts. The next work on this branch is manuscript polish and claim
discipline, not another estimator-repair packet.

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
  - mean position error: `0.01268 m`
  - mean waypoint error: `0.03234 m`
  - empirical 95% coverage: `99.79%`
  - pose NEES: `4.21`
- the full 18-run suite has now been rerun from that promoted lock:
  - nominal / fused:
    - mean position error: `0.01268 m`
    - mean waypoint error: `0.03234 m`
    - empirical 95% coverage: `99.79%`
    - pose NEES: `4.21`
  - nominal / visual:
    - mean position error: `0.01379 m`
    - mean waypoint error: `0.02822 m`
    - empirical 95% coverage: `99.79%`
    - pose NEES: `4.99`
  - intermittent anchor / fused:
    - mean position error: `0.03112 m`
    - mean waypoint error: `0.03208 m`
    - empirical 95% coverage: `87.01%`
    - pose NEES: `14.24`
  - intermittent anchor / visual:
    - mean position error: `0.01779 m`
    - mean waypoint error: `0.02826 m`
    - empirical 95% coverage: `90.49%`
    - pose NEES: `7.90`
  - servo stress / fused:
    - mean position error: `0.01261 m`
    - mean waypoint error: `0.04421 m`
    - empirical 95% coverage: `99.79%`
    - pose NEES: `4.20`
  - servo stress / visual:
    - mean position error: `0.01252 m`
    - mean waypoint error: `0.03953 m`
    - empirical 95% coverage: `99.79%`
    - pose NEES: `4.72`
- there is no catastrophic fused intermittent-anchor row in the refreshed
  suite
- nominal fused stayed healthy and is within the requested 10% position-error
  window relative to visual at the suite level
- the second-pass draft package now builds locally from the rerun suite:
  - PDF:
    `report_tex/second_pass_publication_draft.pdf`
  - publication summary:
    `output/isaac_runs/latest_second_pass_suite/analysis/publication_build_summary.json`
  - presentation bundle:
    `output/isaac_runs/latest_second_pass_suite/presentation/`
- future runs now persist the actual overridden estimator/control configuration
  into the saved run manifest rather than the raw YAML defaults

The practical conclusion is now straightforward: the branch has a credible
post-fix 18-run zig-zag science suite, a real fused draft lock, a compiled
draft PDF, and a local media/dashboard bundle that all point to the same
narrower stabilization story. Visual remains the stronger waypoint-tracking
baseline, but the second-pass branch is now publication-usable rather than
blocked.

## Preserved Checkpoint

The pre-draft checkpoint for this branch is commit `b3f23c9`.

It is preserved in:

- `docs/second_pass_draft_lock.json`
- `output/isaac_runs/second_pass_checkpoint_isolation_seed_007/`

That checkpoint captures the six-run seed-007 isolation bundle that identified
the current branch-level root cause as a fused mechanization or weighting issue.

## Current Review Package

The current review bundle is intentionally small and repo-centered:

- quickstart:
  - `docs/isaac_second_pass_handoff.md`
- current findings:
  - `docs/isaac_second_pass_key_findings.md`
- figure/video guidance:
  - `docs/isaac_second_pass_figure_notes.md`
- frozen evidence tables:
  - `docs/isaac_second_pass_evidence_tables.json`
- review manifest:
  - `docs/second_pass_review_manifest.json`
- draft lock:
  - `docs/second_pass_draft_lock.json`
- manuscript source:
  - `report_tex/second_pass_publication_draft.tex`
- built manuscript:
  - `report_tex/second_pass_publication_draft.pdf`
- dashboard:
  - `output/isaac_runs/latest_second_pass_suite/presentation/index.html`

## Review Bundle Regeneration

Use the wrapper below to rebuild the current review package from the promoted
second-pass draft lock:

```bash
source .venv/bin/activate
python scripts/build_isaac_second_pass_review_bundle.py
```

This regenerates:

- the second-pass publication build
- the local media bundle
- the static dashboard
- the repo-relative review manifest
- the evidence tables and control-success summaries consumed by the manuscript
  and dashboard

## Artifact Policy

The canonical second-pass review outputs live under
`output/isaac_runs/latest_second_pass_suite`, but those large generated
artifacts remain local. The repo-level sources of truth are therefore:

- the draft lock
- the review manifest
- the execution log
- the manuscript source

## Historical Packet Archive

Historical packet-by-packet notes, including the isolation bundle, dropout
debug packets, suppression-propagation repair, and publication-closure
timeline, now live in:

- `docs/isaac_second_pass_packet_history.md`
