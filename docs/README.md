# Documentation

This directory now mixes three things:

- long-lived project design notes
- preserved browser-simulator checkpoints
- the first real Isaac Sim pass and its reporting path

## Start Here

- [../README.md](../README.md) — top-level setup, bootstrap, and run commands
- [isaac_first_pass_expert_handoff.md](isaac_first_pass_expert_handoff.md) — expert-facing map of the current first-pass Isaac branch, local artifacts, and next technical risks
- [checkpoint_01.md](checkpoint_01.md) — current checkpoint report and metrics
- [checkpoint_02_batch_estimation.md](checkpoint_02_batch_estimation.md) — batch-estimation milestone report and metrics
- [checkpoint_02_scientific_report.md](checkpoint_02_scientific_report.md) — scientific-style technical report for the batch-estimation checkpoint
- [checkpoint_03_scientific_report.md](checkpoint_03_scientific_report.md) — repo-level landing page for the repaired async headline scientific result
- [checkpoint_03_scientific_core.md](checkpoint_03_scientific_core.md) — canonical scientific model, IMU convention, and Checkpoint 03 artifact map
- [estimation.md](estimation.md) — estimation package layout, no-GT rule, and artifact map
- [../report_tex/publication_report_template.pdf](../report_tex/publication_report_template.pdf) — publication template compiled against the current Isaac artifact bundle
- [intermediate_report/README.md](intermediate_report/README.md) — self-contained LaTeX report bundle for Checkpoint 01
- [intermediate_report/main.pdf](intermediate_report/main.pdf) — built PDF version of the checkpoint report
- [app.md](app.md) — browser app behavior, controls, artifacts, and configuration surface

## Current vs Historical

- `checkpoint_01.md` is the current authoritative frozen report for the interactive sim state represented by `output/interactive_runs/run_20260406_083611`.
- as of April 8, 2026, the current workspace also contains a complete first-pass Isaac run under `output/isaac_runs/first_real_pass_nominal_v2`, with `output/isaac_runs/latest_complete` promoted to that run.
- the Isaac publication path now lives in `report_tex/`; `docs/intermediate_report/` remains the historical Checkpoint 01 browser report track.
- `11_interactive_pose_estimation_pipeline.md` and `12_pose_estimation_investigation.md` are historical development notes. They remain useful for understanding how the estimator evolved, but their old metrics are superseded by `checkpoint_01.md`.
- `00` through `10` are design/reference notes for the broader project and repo structure.

## Map

- [00_system_goals.md](00_system_goals.md)
- [01_stack_selection.md](01_stack_selection.md)
- [02_architecture.md](02_architecture.md)
- [03_build_plan.md](03_build_plan.md)
- [04_scene_assets_and_tags.md](04_scene_assets_and_tags.md)
- [05_robot_arm_and_swapability.md](05_robot_arm_and_swapability.md)
- [06_sensors_recording_and_time_sync.md](06_sensors_recording_and_time_sync.md)
- [07_apis.md](07_apis.md)
- [08_motion_presets.md](08_motion_presets.md)
- [09_tag_detection_service.md](09_tag_detection_service.md)
- [10_repo_tree.md](10_repo_tree.md)
- [11_interactive_pose_estimation_pipeline.md](11_interactive_pose_estimation_pipeline.md)
- [12_pose_estimation_investigation.md](12_pose_estimation_investigation.md)
- [checkpoint_02_batch_estimation.md](checkpoint_02_batch_estimation.md)
- [checkpoint_02_scientific_report.md](checkpoint_02_scientific_report.md)
- [checkpoint_03_scientific_report.md](checkpoint_03_scientific_report.md)
- [checkpoint_03_scientific_core.md](checkpoint_03_scientific_core.md)
- [estimation.md](estimation.md)
- [intermediate_report/README.md](intermediate_report/README.md)
- [intermediate_report/main.pdf](intermediate_report/main.pdf)
