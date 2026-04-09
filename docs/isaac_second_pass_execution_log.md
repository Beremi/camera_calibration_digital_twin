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
