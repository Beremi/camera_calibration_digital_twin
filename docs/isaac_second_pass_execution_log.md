# Isaac Second-Pass Execution Log

## 2026-04-09 Baseline Guardrails

- commit: `4c565ae`
- milestone: `second-pass first-draft artifact closure`

### Passed commands

1. `.venv`

```bash
python scripts/verify_isaac_first_pass_suite.py
```

Outcome:
- passed
- artifact source: `latest_first_pass_suite`
- canonical run: `first_pass_fused_closed-loop_servo_nominal_seed_007`

2. `.venv`

```bash
python scripts/build_isaac_first_pass_publication.py
```

Outcome:
- passed
- `report_tex/publication_report_template.pdf` exists
- placeholders remaining: `false`

3. `.venv`

```bash
pytest -q tests/test_isaac_reprojection_metric_consistency.py \
          tests/test_isaac_second_pass_isolation_bundle.py \
          tests/test_isaac_windowed_anchor_ba.py
```

Outcome:
- passed
- result: `3 passed`

## 2026-04-09 Tuning And Draft-Suite Execution

- commit: `4c565ae`

### Real tuning sweep

1. `.venv-isaac`

```bash
python scripts/run_isaac_second_pass_tuning.py --headless --execute-missing
```

Outcome:
- completed for the lightweight global sweep
- anchor-only grid executed over:
  - `imu_process_covariance_scale in {1,2,4,8}`
  - `vision_covariance_scale in {1,2,4}`
  - `post_relocalization_covariance_scale in {1,2,4}`
- best real anchor-only fused candidate:
  - run id: `second_pass_tuning_anchor_only_lightweight_seed_007_imu_8p0_vision_4p0_post_1p0_gyro_default_accel_default`
  - mean position error: `0.01625 m`
  - mean waypoint error: `0.02176 m`
  - coverage: `99.58%`
  - pose NEES: `4.19`
- best real aux-for-control fused candidate:
  - run id: `second_pass_tuning_aux_for_control_lightweight_seed_007_imu_8p0_vision_4p0_post_1p0_gyro_default_accel_default`
  - mean position error: `0.01630 m`
  - mean waypoint error: `0.02270 m`
  - coverage: `99.58%`
  - pose NEES: `4.20`
- saved summary artifacts:
  - `output/isaac_runs/latest_second_pass_tuning/summary.json`
  - `output/isaac_runs/latest_second_pass_tuning/summary.csv`
  - `output/isaac_runs/latest_second_pass_tuning/top_fused_candidates.csv`

2. `.venv-isaac`

```bash
python scripts/run_isaac_anchor_vio.py \
  --run-id second_pass_tuning_aux_estimation_only_lightweight_seed_007_imu_8p0_vision_4p0_post_1p0 \
  ...
```

Outcome:
- completed
- purpose: test whether aux estimation helps while aux-map control is disabled
- result:
  - mean position error: `0.01631 m`
  - mean waypoint error: `0.02254 m`
  - coverage: `99.58%`
  - pose NEES: `4.20`
- conclusion:
  - aux-map control is part of the nominal gap, but turning it off alone does not make fused nominal competitive with visual

3. `.venv-isaac`

```bash
python scripts/run_isaac_second_pass_draft_suite.py \
  --headless --execute-missing \
  --conditions nominal_full_anchor \
  --seeds 7 11 17
```

Outcome:
- executed all six nominal draft runs under the provisional anchor-only lock policy
- representative nominal means from real runs:
  - visual:
    - mean position error: `0.01392 m`
    - mean waypoint error: `0.01964 m`
    - coverage: `99.79%`
    - pose NEES: `5.01`
  - fused:
    - mean position error: `0.01623 m`
    - mean waypoint error: `0.02244 m`
    - coverage: `99.58%`
    - pose NEES: `4.19`
- completion fraction was `1.0` for all six runs
- note:
  - the runner originally tried to build the full 18-run suite at the end and failed because dropout/stress runs were not present yet
  - the runner has since been patched so partial condition execution no longer crashes or overwrites the draft lock

### Current backend comparison

4. `.venv-isaac`

```bash
python scripts/run_isaac_anchor_vio.py \
  --run-id second_pass_tuning_anchor_only_windowed_ba_seed_007_imu_8p0_vision_4p0_post_1p0 \
  --smoother-backend windowed_ba \
  ...
```

Outcome:
- completed
- purpose: compare `windowed_ba` against the current best anchor-only lightweight fused nominal run on seed `007`
- result:
  - `lightweight`
    - mean position error: `0.01625 m`
    - mean waypoint error: `0.02176 m`
    - coverage: `99.58%`
    - pose NEES: `4.19`
  - `windowed_ba`
    - mean position error: `0.01620 m`
    - mean waypoint error: `0.02297 m`
    - coverage: `99.58%`
    - pose NEES: `4.19`
- conclusion:
  - `windowed_ba` slightly improves reprojection quality but hurts waypoint
    tracking on the nominal control task while taking about `1067 s`
    wall-clock for a single `8.0 s` run
  - it is not the draft-suite backend

5. `.venv-isaac`

```bash
python scripts/run_isaac_anchor_vio.py \
  --run-id second_pass_tuning_anchor_only_lightweight_seed_007_imu_16p0_vision_4p0_post_1p0_gyro_default_accel_default \
  ...
```

Outcome:
- completed
- purpose: last lightweight anchor-only covariance tangent after the backend comparison
- result:
  - mean position error: `0.01637 m`
  - mean waypoint error: `0.02237 m`
  - coverage: `99.58%`
  - pose NEES: `4.10`
- conclusion:
  - increasing global IMU process scaling from `8` to `16` degrades nominal
    tracking even though NEES decreases slightly
  - the best practical lightweight nominal setting remains
    `imu_process_covariance_scale = 8.0`,
    `vision_covariance_scale = 4.0`,
    `post_relocalization_covariance_scale = 1.0`

### Current full-suite run

6. `.venv-isaac`

```bash
python scripts/run_isaac_second_pass_draft_suite.py --headless --execute-missing
```

Outcome:
- completed
- wrote:
  - `output/isaac_runs/latest_second_pass_suite/analysis/suite_summary.json`
  - `draft_nominal_table.csv`
  - `draft_dropout_table.csv`
  - `draft_actuation_stress_table.csv`
  - `draft_uncertainty_table.csv`
  - `draft_map_quality_table.csv`
- key result:
  - nominal remained stable for both estimators
  - fused intermittent-anchor collapsed catastrophically under the provisional
    anchor-only lock:
    - mean position error: about `251.69 m`
    - coverage: about `14.72%`
    - pose NEES: about `1.46e7`
  - visual intermittent-anchor remained sane:
    - mean position error: about `0.01648 m`
    - coverage: about `92.71%`
    - pose NEES: about `6.30`
- conclusion:
  - the current anchor-only fused lock is not a viable first-draft selection
  - publication/media closure was intentionally paused pending a smaller fused
    runtime fix

### Focused dropout diagnostics after the failed suite

7. `.venv-isaac` + `.venv`

```bash
python scripts/run_isaac_anchor_vio.py \
  --run-id second_pass_explore_fused_intermittent_aux_estimation_only_seed_007 \
  ...
python scripts/analyze_isaac_estimator_quality.py \
  output/isaac_runs/second_pass_explore_fused_intermittent_aux_estimation_only_seed_007
```

Outcome:
- completed
- purpose: test whether enabling auxiliary tags for estimation only rescues the
  intermittent-anchor fused collapse
- result:
  - mean position error: `253.46 m`
  - mean waypoint error: `0.02286 m`
  - coverage: `14.58%`
  - pose NEES: `1.48e7`
  - mean smoother correction norm: `126.61 m`
- conclusion:
  - the dropout collapse is not explained by the current draft lock's aux-map
    control setting

8. `.venv-isaac` + `.venv`

```bash
python scripts/run_isaac_anchor_vio.py \
  --run-id second_pass_explore_fused_intermittent_aux_for_control_seed_007 \
  ...
python scripts/analyze_isaac_estimator_quality.py \
  output/isaac_runs/second_pass_explore_fused_intermittent_aux_for_control_seed_007
```

Outcome:
- completed
- purpose: test whether fully aux-enabled fused control behaves materially
  differently under intermittent anchor
- result:
  - mean position error: `247.55 m`
  - mean waypoint error: `0.02309 m`
  - coverage: `14.58%`
  - pose NEES: `1.41e7`
  - mean smoother correction norm: `105.35 m`
- conclusion:
  - the fused dropout failure persists across anchor-only, aux-estimation-only,
    and aux-for-control policies

### Fused-path runtime fixes and post-fix spot checks

9. `.venv`

```bash
pytest -q tests/test_estimation_factors.py \
          tests/test_imu_semantics.py \
          tests/test_isaac_mode_semantics.py
```

Outcome:
- passed
- result: `18 passed`
- code changes covered by this slice:
  - inertial position integration now uses standard constant-acceleration
    kinematics
  - `_process_imu(...)` now prefers the camera pose for synthetic IMU motion
    sampling when a live camera binding is available

10. `.venv-isaac` + `.venv`

```bash
python scripts/run_isaac_anchor_vio.py \
  --run-id second_pass_explore2_fused_nominal_anchor_only_seed_007 \
  ...
python scripts/analyze_isaac_estimator_quality.py \
  output/isaac_runs/second_pass_explore2_fused_nominal_anchor_only_seed_007
```

Outcome:
- completed
- purpose: confirm nominal fused behavior after the fused-path propagation fixes
- result:
  - mean position error: `0.01556 m`
  - mean waypoint error: `0.02211 m`
  - coverage: `99.58%`
  - pose NEES: `4.13`
- conclusion:
  - fused nominal improves modestly, but still does not close the full nominal
    gap to visual

11. `.venv-isaac` + `.venv`

```bash
python scripts/run_isaac_anchor_vio.py \
  --run-id second_pass_explore2_fused_intermittent_anchor_only_seed_007 \
  ...
python scripts/analyze_isaac_estimator_quality.py \
  output/isaac_runs/second_pass_explore2_fused_intermittent_anchor_only_seed_007
```

Outcome:
- completed
- purpose: check whether the fused-path fixes rescue intermittent-anchor
  stability for the current nominal lock
- result:
  - mean position error: `248.74 m`
  - mean waypoint error: `0.02272 m`
  - coverage: `14.79%`
  - pose NEES: `1.42e7`
- conclusion:
  - the branch is still blocked on a deeper fused intermittent-anchor defect
  - the second-pass draft was not advanced to publication/media closure after
    this point because the science bar is still unmet

### Dropout-debug pack implementation pass

12. system Python

```bash
python -m py_compile \
  scripts/run_isaac_anchor_vio.py \
  scripts/run_isaac_second_pass_dropout_debug.py \
  src/calib_sim/isaac/runtime/main_loop.py \
  src/calib_sim/isaac/estimation/online_filter.py \
  src/calib_sim/isaac/logging/writer.py \
  src/calib_sim/reporting/isaac_second_pass_dropout_debug.py \
  tests/test_isaac_second_pass_dropout_debug.py \
  tests/test_isaac_second_pass_switches.py \
  tests/_isaac_test_helpers.py
```

Outcome:
- passed
- purpose: quick syntax validation after wiring the dropout-debug runtime,
  reporting, and tests

13. `.venv`

```bash
pytest -q tests/test_isaac_second_pass_dropout_debug.py \
          tests/test_isaac_second_pass_switches.py \
          tests/test_estimation_factors.py \
          tests/test_imu_semantics.py \
          tests/test_isaac_mode_semantics.py \
          tests/test_isaac_second_pass_tuning.py \
          tests/test_isaac_second_pass_suite.py \
          tests/test_isaac_second_pass_media_bundle.py
```

Outcome:
- passed
- result: `26 passed`
- purpose: keep the fused-path regression slice green while adding the new
  dropout-debug runner, trace schema, and root-cause classifier

14. `.venv`

```bash
python scripts/verify_isaac_first_pass_suite.py
python scripts/build_isaac_first_pass_publication.py
```

Outcome:
- passed
- purpose: confirm the frozen first-pass publication path remains untouched
- result:
  - suite verify: `ok = true`
  - publication build: `artifact_source = latest_first_pass_suite`
  - placeholders remaining: `false`

15. `.venv`

```bash
python scripts/run_isaac_second_pass_dropout_debug.py --dry-run
```

Outcome:
- passed
- purpose: verify the six fixed debug runs and their special dropout flags are
  exposed by the new runner before launching real Isaac jobs
- result:
 - planned runs:
    - `second_pass_dropout_debug_visual_seed_007`
    - `second_pass_dropout_debug_fused_seed_007`
    - `second_pass_dropout_debug_fused_no_reacq_seed_007`
    - `second_pass_dropout_debug_fused_no_imu_during_suppression_seed_007`
    - `second_pass_dropout_debug_fused_covinfl_seed_007`
    - `second_pass_dropout_debug_fused_clipcorr_seed_007`

## 2026-04-09 Dropout Propagation Follow-Up

- starting head: `bff0408`
- preserved pre-draft checkpoint: `b3f23c9`
- active milestone: `fused-dropout stabilization before draft closure`

### Real dropout-pack reruns and propagation fixes

16. `.venv-isaac`

```bash
rm -rf output/isaac_runs/second_pass_dropout_debug_* \
       output/isaac_runs/latest_second_pass_dropout_debug
python scripts/run_isaac_second_pass_dropout_debug.py --headless --execute-missing
```

Outcome:
- completed after fixing a runtime `NameError` in
  `src/calib_sim/isaac/runtime/main_loop.py`
- first real bundle showed:
  - visual reference was sane
  - baseline fused remained blocked
  - `fused_no_imu_during_suppression` cleared the blocker bar
  - `fused_covinfl` and `fused_clipcorr` did not rescue the run
- conclusion:
  - the blocker is not the smoother and not a pure reacquisition-only failure
  - the dominant mechanism is suppression-window propagation

17. local code change + `.venv`

```bash
pytest -q tests/test_imu_semantics.py tests/test_isaac_mode_semantics.py
```

Outcome:
- passed
- purpose: lock in the two new propagation-side fixes:
  - synthetic IMU conditioning in `src/calib_sim/isaac/sensors.py`
  - suppression-window specific-force gating in
    `src/calib_sim/isaac/runtime/main_loop.py`
- effect:
  - production fused intermittent-anchor improved from catastrophic suite
    failure to an interpretable but still blocked regime

18. `.venv-isaac`

```bash
rm -rf output/isaac_runs/second_pass_dropout_debug_* \
       output/isaac_runs/latest_second_pass_dropout_debug
python scripts/run_isaac_second_pass_dropout_debug.py --headless --execute-missing
```

Outcome:
- completed
- current bundle:
  - `output/isaac_runs/latest_second_pass_dropout_debug/summary.json`
  - `output/isaac_runs/latest_second_pass_dropout_debug/summary.csv`
  - `docs/isaac_second_pass_dropout_debug.md`
- key results:
  - `second_pass_dropout_debug_visual_seed_007`
    - mean position error: `0.01555 m`
    - coverage: `99.79%`
    - pose NEES: `4.21`
  - `second_pass_dropout_debug_fused_seed_007`
    - mean position error: `0.15517 m`
    - mean waypoint error: `0.02421 m`
    - coverage: `76.04%`
    - pose NEES: `195.95`
    - root-cause hint: `propagation_process_problem`
  - `second_pass_dropout_debug_fused_no_imu_during_suppression_seed_007`
    - mean position error: `0.01367 m`
    - coverage: `99.38%`
    - pose NEES: `4.06`
    - blocker clear: `true`
- conclusion:
  - the production fused baseline is materially improved but still blocked
  - the branch must not reopen the 18-run suite, second-pass PDF, or polished
    media yet

19. `.venv-isaac`

```bash
python scripts/run_isaac_anchor_vio.py \
  --headless \
  --duration-s 8.0 \
  --seed 7 \
  --run-id second_pass_followup_fused_nominal_anchor_only_seed_007 \
  --estimator-mode fused \
  --controller-mode closed-loop \
  --bootstrap-control-policy hold_until_first_detection \
  --smoother-backend lightweight \
  --vision-covariance-scale 4.0 \
  --imu-process-covariance-scale 8.0 \
  --post-relocalization-covariance-scale 1.0 \
  --no-use-aux-tags-in-filter \
  --no-use-aux-tags-in-smoother \
  --no-use-aux-map-for-control \
  --no-promote-global-latest
```

Outcome:
- completed
- purpose: recheck nominal fused seed-`007` after the suppression-window
  propagation fixes before considering any further dropout iteration
- result:
  - mean position error: `0.01284 m`
  - mean waypoint error: `0.02392 m`
  - empirical 95% coverage: `99.79%`
  - pose NEES: `3.97`
- conclusion:
  - the latest propagation-side fixes did not regress nominal fused stability

20. `.venv`

```bash
pytest -q tests/test_imu_semantics.py \
          tests/test_isaac_mode_semantics.py \
          tests/test_isaac_second_pass_switches.py \
          tests/test_isaac_second_pass_dropout_debug.py \
          tests/test_isaac_second_pass_tuning.py \
          tests/test_isaac_second_pass_suite.py \
          tests/test_isaac_second_pass_media_bundle.py
python scripts/verify_isaac_first_pass_suite.py
python scripts/build_isaac_first_pass_publication.py
```

Outcome:
- passed
- result:
  - regression slice: `24 passed`
  - first-pass suite verify: `ok = true`
  - first-pass publication build:
    - `artifact_source = latest_first_pass_suite`
    - `placeholders_remaining = false`
- extra note:
  - `scripts/run_isaac_anchor_vio.py --validate-config-only` now writes a run
    manifest that matches the actual overridden estimator/control config rather
    than the raw YAML defaults

## 2026-04-09 Suppression-Window Propagation Stabilization

- starting head: `3a4a582`
- previous packet head: `bff0408`
- active milestone: `suppression-window propagation stabilization for fused intermittent-anchor`

### Packet implementation and probe execution

21. `.venv`

```bash
python -m py_compile \
  scripts/analyze_isaac_suppression_windows.py \
  scripts/run_isaac_anchor_vio.py \
  scripts/run_isaac_second_pass_dropout_debug.py \
  src/calib_sim/isaac/estimation/online_filter.py \
  src/calib_sim/isaac/runtime/main_loop.py \
  src/calib_sim/isaac/logging/writer.py \
  src/calib_sim/reporting/isaac_suppression_windows.py \
  src/calib_sim/reporting/isaac_second_pass_dropout_debug.py \
  tests/test_isaac_suppression_windows.py \
  tests/test_isaac_second_pass_dropout_debug.py \
  tests/test_isaac_second_pass_switches.py \
  tests/test_isaac_mode_semantics.py
pytest -q tests/test_isaac_suppression_windows.py \
          tests/test_isaac_second_pass_dropout_debug.py \
          tests/test_isaac_second_pass_switches.py \
          tests/test_isaac_mode_semantics.py
```

Outcome:
- passed
- result: `23 passed`
- packet additions:
  - `scripts/analyze_isaac_suppression_windows.py`
  - suppression-window summary artifacts under each run's `analysis/`
  - `suppression_propagation_mode` routed through config, runner, manifest,
    and runtime with `full_imu`, `gyro_only`, `constant_velocity`, and
    `freeze`

22. `.venv`

```bash
for run_dir in output/isaac_runs/second_pass_dropout_debug_*; do
  python scripts/analyze_isaac_suppression_windows.py "$run_dir"
done
```

Outcome:
- completed on the existing six-run control bundle
- conclusion:
  - the blocked production fused control run remained
    `mean_state_drift_dominant`
  - the helper outputs are now available under each run's `analysis/` as:
    - `suppression_window_summary.csv`
    - `suppression_window_summary.json`
    - `suppression_window_note.md`

23. `.venv-isaac`

```bash
python scripts/run_isaac_second_pass_dropout_debug.py \
  --headless --execute-missing \
  --output-root output/isaac_runs/probes/gate_10 \
  --docs-path output/isaac_runs/probes/gate_10/dropout_debug.md \
  --suppression-imu-specific-force-gate-mps2 10.0
```

Outcome:
- completed
- key result:
  - tighter suppression specific-force gating helped but did not clear the
    production fused baseline by itself:
    - mean position error: `0.07647 m`
    - empirical 95% coverage: `78.96%`
    - pose NEES: `52.70`
- conclusion:
  - gate tightening is a real lever, but it is not sufficient alone

24. `.venv-isaac`

```bash
python scripts/run_isaac_second_pass_dropout_debug.py \
  --headless --execute-missing \
  --output-root output/isaac_runs/probes/gyro_only_gate_10 \
  --docs-path output/isaac_runs/probes/gyro_only_gate_10/dropout_debug.md \
  --suppression-imu-specific-force-gate-mps2 10.0 \
  --suppression-propagation-mode gyro_only
```

Outcome:
- completed
- key result:
  - `gyro_only` by itself remained blocked
  - `gyro_only + dropout_post_reacquisition_covariance_scale = 8.0`
    cleared the bar on seed `007`:
    - mean position error: `0.03091 m`
    - empirical 95% coverage: `88.96%`
    - pose NEES: `10.10`
  - `gyro_only + correction clipping` also cleared, but covariance inflation
    was the cleaner winner candidate
- conclusion:
  - the packet winner should be validated as
    `gyro_only + gate_10 + covinfl`, not `gyro_only` alone

25. `.venv-isaac` + `.venv`

```bash
python scripts/run_isaac_anchor_vio.py \
  --headless \
  --output-root output/isaac_runs/probes/gyro_only_gate_10_validation \
  --run-id second_pass_followup_fused_nominal_anchor_only_seed_007_gyro_only_covinfl_gate_10 \
  --seed 7 \
  --duration-s 8.0 \
  --estimator-mode fused \
  --controller-mode closed-loop \
  --bootstrap-control-policy hold_until_first_detection \
  --smoother-backend lightweight \
  --vision-covariance-scale 4.0 \
  --imu-process-covariance-scale 8.0 \
  --post-relocalization-covariance-scale 1.0 \
  --suppression-imu-specific-force-gate-mps2 10.0 \
  --suppression-propagation-mode gyro_only \
  --dropout-post-reacquisition-covariance-scale 8.0 \
  --no-use-aux-tags-in-filter \
  --no-use-aux-tags-in-smoother \
  --no-use-aux-map-for-control \
  --no-promote-global-latest
python scripts/analyze_isaac_estimator_quality.py \
  output/isaac_runs/probes/gyro_only_gate_10_validation/second_pass_followup_fused_nominal_anchor_only_seed_007_gyro_only_covinfl_gate_10
python scripts/run_isaac_anchor_vio.py \
  --headless \
  --visibility-config config/isaac/visibility/anchor_dropout_nominal.yaml \
  --output-root output/isaac_runs/probes/gyro_only_gate_10_validation \
  --run-id second_pass_validation_fused_intermittent_anchor_only_seed_011_gyro_only_covinfl_gate_10 \
  --seed 11 \
  --duration-s 8.0 \
  --estimator-mode fused \
  --controller-mode closed-loop \
  --bootstrap-control-policy hold_until_first_detection \
  --smoother-backend lightweight \
  --vision-covariance-scale 4.0 \
  --imu-process-covariance-scale 8.0 \
  --post-relocalization-covariance-scale 1.0 \
  --suppression-imu-specific-force-gate-mps2 10.0 \
  --suppression-propagation-mode gyro_only \
  --dropout-post-reacquisition-covariance-scale 8.0 \
  --no-use-aux-tags-in-filter \
  --no-use-aux-tags-in-smoother \
  --no-use-aux-map-for-control \
  --no-promote-global-latest
python scripts/analyze_isaac_estimator_quality.py \
  output/isaac_runs/probes/gyro_only_gate_10_validation/second_pass_validation_fused_intermittent_anchor_only_seed_011_gyro_only_covinfl_gate_10
python scripts/run_isaac_anchor_vio.py \
  --headless \
  --visibility-config config/isaac/visibility/anchor_dropout_nominal.yaml \
  --output-root output/isaac_runs/probes/gyro_only_gate_10_validation \
  --run-id second_pass_validation_fused_intermittent_anchor_only_seed_017_gyro_only_covinfl_gate_10 \
  --seed 17 \
  --duration-s 8.0 \
  --estimator-mode fused \
  --controller-mode closed-loop \
  --bootstrap-control-policy hold_until_first_detection \
  --smoother-backend lightweight \
  --vision-covariance-scale 4.0 \
  --imu-process-covariance-scale 8.0 \
  --post-relocalization-covariance-scale 1.0 \
  --suppression-imu-specific-force-gate-mps2 10.0 \
  --suppression-propagation-mode gyro_only \
  --dropout-post-reacquisition-covariance-scale 8.0 \
  --no-use-aux-tags-in-filter \
  --no-use-aux-tags-in-smoother \
  --no-use-aux-map-for-control \
  --no-promote-global-latest
python scripts/analyze_isaac_estimator_quality.py \
  output/isaac_runs/probes/gyro_only_gate_10_validation/second_pass_validation_fused_intermittent_anchor_only_seed_017_gyro_only_covinfl_gate_10
python scripts/analyze_isaac_suppression_windows.py \
  output/isaac_runs/probes/gyro_only_gate_10_validation/second_pass_followup_fused_nominal_anchor_only_seed_007_gyro_only_covinfl_gate_10
python scripts/analyze_isaac_suppression_windows.py \
  output/isaac_runs/probes/gyro_only_gate_10_validation/second_pass_validation_fused_intermittent_anchor_only_seed_011_gyro_only_covinfl_gate_10
python scripts/analyze_isaac_suppression_windows.py \
  output/isaac_runs/probes/gyro_only_gate_10_validation/second_pass_validation_fused_intermittent_anchor_only_seed_017_gyro_only_covinfl_gate_10
```

Outcome:
- completed
- nominal fused follow-up stayed healthy:
  - mean position error: `0.01278 m`
  - mean waypoint error: `0.02408 m`
  - empirical 95% coverage: `99.79%`
  - pose NEES: `3.97`
- intermittent-anchor fused validation also cleared on seeds `011` and `017`:
  - seed `011`:
    - mean position error: `0.03560 m`
    - mean waypoint error: `0.02235 m`
    - empirical 95% coverage: `86.04%`
    - pose NEES: `13.02`
  - seed `017`:
    - mean position error: `0.02935 m`
    - mean waypoint error: `0.02203 m`
    - empirical 95% coverage: `93.75%`
    - pose NEES: `10.42`
- conclusion:
  - the current packet produced a validated production-like suppression winner:
    `gyro_only + gate_10 + covinfl`
  - the next packet can promote that winner into the draft lock and reopen
    focused retuning and the 18-run suite

26. `.venv`

```bash
pytest -q tests/test_estimation_factors.py \
          tests/test_imu_semantics.py \
          tests/test_isaac_mode_semantics.py \
          tests/test_isaac_second_pass_tuning.py \
          tests/test_isaac_second_pass_suite.py \
          tests/test_isaac_second_pass_media_bundle.py \
          tests/test_isaac_second_pass_dropout_debug.py \
          tests/test_isaac_second_pass_switches.py \
          tests/test_isaac_suppression_windows.py
python scripts/verify_isaac_first_pass_suite.py
python scripts/build_isaac_first_pass_publication.py
```

Outcome:
- passed
- result:
  - regression slice: `37 passed`
  - first-pass suite verify:
    - `artifact_source = latest_first_pass_suite`
    - `ok = true`
  - first-pass publication build:
    - `artifact_source = latest_first_pass_suite`
    - `pdf_exists = true`
    - `placeholders_remaining = false`
- conclusion:
  - the suppression-propagation packet winner did not regress the frozen
    first-pass publication path
