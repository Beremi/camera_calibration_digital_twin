# Isaac First-Pass Expert Handoff

This handoff is for the current branch:

- `feature/isaac-runtime-first-pass`

It describes where the scientifically complete first Isaac pass lives in the repo, what is verified, which local artifacts matter, and what still needs expert follow-through.

## Commit Scope Vs Local Workspace

The checkpoint commit for this handoff includes:

- source, config, tests, and documentation for the first-pass Isaac runtime
- the expert handoff itself
- the existing small checked-in plumbing artifact bundle already tracked in git

The checkpoint commit does not include the larger local live-run workspace artifacts under `output/isaac_runs/`.
Those runs remain available in the shared workspace on this machine and are referenced below, but they are intentionally left out of the checkpoint commit to avoid turning the branch into a large binary artifact dump.

## Start Here

Read these in order:

1. `README.md`
2. `docs/README.md`
3. `docs/estimation.md`
4. `report_tex/publication_report_template.tex`
5. `src/calib_sim/isaac/runtime/main_loop.py`

If you want the shortest path to the current publication inputs, open:

- `output/isaac_runs/latest_first_pass_suite/analysis/suite_summary.json`
- `output/isaac_runs/latest_first_pass_suite/analysis/report_data/paper_artifacts.tex`
- `report_tex/publication_report_template.pdf`

If you want the shortest path to the small checked-in plumbing sample that exists in git history, open:

- `output/isaac_runs/run_20260408_062541/manifest.json`
- `output/isaac_runs/run_20260408_062541/analysis/metrics.json`

## Current Publication Inputs

The suite-level publication link is:

- `output/isaac_runs/latest_first_pass_suite`

The per-run fallback link is:

- `output/isaac_runs/latest_complete`

The small checked-in plumbing sample that is safe to inspect from the checkpoint commit is:

- `output/isaac_runs/run_20260408_062541`

The canonical fused nominal live run that anchors the paper and diagnostics is:

- `output/isaac_runs/first_pass_fused_closed-loop_servo_nominal_seed_007`

The suite executor and regeneration details are tracked in:

- `docs/isaac_first_pass_results.md`

## Repo Map

### Runtime and Bootstrapping

- `src/calib_sim/isaac/app.py`
  - loads YAML configs and creates the runtime
- `src/calib_sim/isaac/runtime/main_loop.py`
  - standalone Isaac orchestrator
  - owns startup, warmup, stepping, sensing, filtering, smoothing, control, and logging
- `src/calib_sim/sim/runtime.py`
  - Isaac bootstrap / compatibility wrapper layer

### Scene, Robot, and Sensors

- `config/isaac/scene/anchor_room.yaml`
  - first-pass anchor-room scene
  - note: tag overlap bug was fixed here
- `config/isaac/robot/franka_phone_head.yaml`
  - first-pass headline robot preset
- `config/isaac/camera/phone_main.yaml`
  - current mounted-view camera config that reliably sees the anchor
- `config/isaac/imu/phone_nominal.yaml`
  - first-pass IMU semantics and nominal noise preset
- `src/calib_sim/isaac/stage_builder.py`
  - programmatic scene build and tag geometry/material generation
- `src/calib_sim/isaac/robot_builder.py`
  - robot/articulation setup and mount prim handling
- `src/calib_sim/isaac/sensors.py`
  - `IsaacCameraBinding`
  - `IsaacImuBinding`

### Front End and Estimation

- `src/calib_sim/tag_service/detector.py`
  - OpenCV AprilTag detector plus recovery/fallback logic
  - important: native detections now use the corrected corner/object-point convention
- `src/calib_sim/isaac/frontend/apriltag_frontend.py`
  - wraps detections into Isaac measurement packets
  - important: fallback matched quads are logged but do not contribute PnP poses
- `src/calib_sim/isaac/estimation/online_filter.py`
  - anchored online filter
  - important: anchor updates now relocalize hard enough to stop drift
- `src/calib_sim/isaac/estimation/fixed_lag_smoother.py`
  - first-pass lag smoother
- `src/calib_sim/isaac/estimation/state_defs.py`
  - filter/smoother/uncertainty state dataclasses

### Control and Actuation

- `config/isaac/control/path_tracking.yaml`
  - current first-pass waypoint path in the actual mounted-camera workspace
- `config/isaac/actuation/servo_nominal.yaml`
  - nominal software servo corruption preset
- `src/calib_sim/isaac/control/path_tracker.py`
  - waypoint-following logic
- `src/calib_sim/isaac/control/safety_gates.py`
  - visibility/uncertainty gating
- `src/calib_sim/isaac/actuation/servo_model.py`
  - command corruption / realized actuation model

### Logging, Replay, and Reporting

- `src/calib_sim/isaac/logging/schemas.py`
  - raw/GT/estimate packet definitions
- `src/calib_sim/isaac/logging/writer.py`
  - append-only artifact writer
- `src/calib_sim/isaac/runtime/replay.py`
  - raw/estimate/GT loaders
  - note: no-GT boundary is enforced at load time
  - note: replay is analysis/report regeneration only on this branch, not a raw-log estimator re-solve
- `src/calib_sim/reporting/__init__.py`
  - completeness gating and latest-link promotion
- `src/calib_sim/reporting/isaac_report_metrics.py`
  - metric extraction
- `src/calib_sim/reporting/isaac_report_tables.py`
  - TeX/CSV table generation
- `src/calib_sim/reporting/isaac_report_figures.py`
  - report figure generation
- `scripts/run_isaac_anchor_vio.py`
  - live experiment entrypoint
- `scripts/replay_isaac_anchor_vio.py`
  - replay / report regeneration entrypoint
  - note: analysis-only on this branch, not a full estimator re-solve path
- `scripts/run_isaac_ablation_suite.py`
  - executes the first-pass 2x2 matrix, 5-seed closed-loop reproducibility sweep, and actuation comparison
- `src/calib_sim/reporting/isaac_first_pass_suite.py`
  - aggregates per-run metrics into suite CSV/JSON/TeX publication artifacts
- `scripts/generate_isaac_report_artifacts.py`
  - publication artifact generation with completeness enforcement

### Tests

Most important tests for the first pass:

- `tests/test_isaac_runtime_smoke.py`
  - real Isaac boot/step/shutdown and artifact write
- `tests/test_isaac_frontend_pose.py`
  - corrected tag pose convention
- `tests/test_isaac_stage_spec.py`
  - catches overlapping fiducials
- `tests/test_imu_semantics.py`
  - specific-force convention
- `tests/test_filter_consistency.py`
  - basic anchored filter sanity
- `tests/test_isaac_servo_and_reporting.py`
  - report generation and actuator corruption checks
- `tests/test_isaac_report_regression.py`
  - publication template compile regression

## Local Artifact Directories Worth Inspecting

These are useful in the shared workspace and explain how the current result was reached:

- `output/isaac_runs/first_pass_fused_closed-loop_servo_nominal_seed_007`
  - canonical headline run used by `latest_first_pass_suite`
- `output/isaac_runs/first_pass_fused_closed-loop_none_seed_007`
  - actuation comparison partner for the canonical headline cell
- `output/isaac_runs/first_pass_visual_closed-loop_servo_nominal_seed_007`
  - visual closed-loop comparison cell
- `output/isaac_runs/first_pass_fused_open-loop_servo_nominal_seed_007`
  - fused open-loop comparison cell
- `output/isaac_runs/first_pass_visual_open-loop_servo_nominal_seed_007`
  - visual open-loop comparison cell
- `output/isaac_runs/first_real_pass_nominal_v2`
  - older pre-suite milestone run retained for historical comparison
- `output/isaac_runs/first_real_pass_recheck`
  - earlier live run after the tag-pose fix, before control/path cleanup
- `output/isaac_runs/texture_probe_g6`
  - the key visual diagnosis run where native anchor detection became reliable
- `output/isaac_runs/first_pass_probe_camfix5`
  - useful for seeing the earlier weird tag layout / camera debugging phase

There are many other probe/debug directories under `output/isaac_runs/`. They are intentionally left in the shared workspace for expert inspection, but they are not all authoritative. Treat the `first_pass_*` suite runs and `latest_first_pass_suite` as the authoritative first-pass publication inputs.

## How To Reproduce The Current First Pass

### Live Isaac Run

```bash
source .venv-isaac/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
python scripts/run_isaac_anchor_vio.py \
  --headless \
  --duration-s 8.0 \
  --run-id first_pass_fused_closed-loop_servo_nominal_seed_007 \
  --estimator-mode fused \
  --controller-mode closed-loop \
  --bootstrap-control-policy hold_until_first_detection
```

### Run The Full First-Pass Suite

```bash
source .venv-isaac/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
python scripts/run_isaac_ablation_suite.py --headless
```

### Regenerate Report Artifacts

```bash
source .venv/bin/activate
python scripts/generate_isaac_report_artifacts.py output/isaac_runs/first_pass_fused_closed-loop_servo_nominal_seed_007
```

### Replay / Rebuild Analysis

```bash
source .venv/bin/activate
python scripts/replay_isaac_anchor_vio.py output/isaac_runs/first_pass_fused_closed-loop_servo_nominal_seed_007
```

### Compile The Paper

```bash
source .venv/bin/activate
cd report_tex
latexmk -pdf -interaction=nonstopmode publication_report_template.tex
```

## What Is Verified

Verified in the current workspace:

- a real live Isaac run produces nonzero camera, IMU, detection, command, estimate, and uncertainty artifacts
- the full 2x2 estimator/controller matrix exists at seed `7`
- the 5-seed closed-loop reproducibility sweep exists for both `visual` and `fused`
- the minimum `none` vs `servo_nominal` actuation comparison exists for the fused closed-loop headline cell
- the anchor tag is visible in every frame of the canonical headline run
- native anchor PnP is aligned with GT after the corner/size fix
- `latest_complete` now promotes only from runs that pass completeness checks
- `latest_first_pass_suite` is populated and drives the paper before `latest_complete`
- report artifact generation works without `--allow-incomplete` on the canonical run and the suite bundle
- the publication PDF compiles against the suite artifact bundle
- publication-mode closed-loop control uses `hold_until_first_detection`, not GT bootstrap
- the canonical fused closed-loop run reaches completion fraction `1.0`, IK failure fraction `0.0`, and anchor visible fraction `1.0`

## Known Limitations

These are the main honest gaps remaining for expert follow-up:

- `scripts/replay_isaac_anchor_vio.py` does not yet re-solve the estimator from raw logs; it currently rebuilds analysis/report outputs
- the first-pass branch keeps replay analysis-only to avoid scope creep into a second-system rewrite
- the suite-driven publication path is intentionally narrow and does not yet cover broader ideal/nominal/stress campaigns
- the broader `output/isaac_runs/` tree contains many exploratory runs that have not been cleaned up or curated

## Best Next Expert Steps

1. Verify the suite summary under `latest_first_pass_suite` against the intended canonical run IDs.
2. Inspect the remaining estimator-quality metrics, especially trajectory error and uncertainty calibration, now that controller semantics are stable.
3. Decide whether the next branch should prioritize raw-log estimator re-solve or broader stress/visibility campaigns.
