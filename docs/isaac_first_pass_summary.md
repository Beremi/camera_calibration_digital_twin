# Isaac First-Pass Summary

This branch is frozen around the suite at [`output/isaac_runs/latest_first_pass_suite`](/home/beremi/repos/camera_calibration_digital_twin/output/isaac_runs/latest_first_pass_suite). The authoritative canonical run is [`first_pass_fused_closed-loop_servo_nominal_seed_007`](/home/beremi/repos/camera_calibration_digital_twin/output/isaac_runs/first_pass_fused_closed-loop_servo_nominal_seed_007), and the frozen run set is recorded in [`first_pass_suite_lock.json`](/home/beremi/repos/camera_calibration_digital_twin/docs/first_pass_suite_lock.json).

## Runtime And Data Volume

The canonical fused closed-loop run covers `8.0 s` and writes `240` camera frames, `1600` IMU packets, `669` tag detections, `400` commands, `400` controller-diagnostic rows, `480` filter states, `80` smoother states, and `480` uncertainty states. Publication builds now consume suite artifacts from [`latest_first_pass_suite`](/home/beremi/repos/camera_calibration_digital_twin/output/isaac_runs/latest_first_pass_suite), not ad hoc single-run fallbacks.

## 2x2 Matrix

The first-pass comparison matrix separates estimator and controller modes on the same scene, robot, path, rates, actuation preset, and seed. Open-loop runs stop at `0.50` completion with mean waypoint error `0.048 m` for both visual and fused estimation. Closed-loop runs complete the path with zero IK failures; visual closed-loop reports mean position error `0.0455 m` and mean waypoint error `0.0212 m`, while fused closed-loop reports mean position error `0.0471 m` and mean waypoint error `0.0241 m`.

## Reproducibility

The closed-loop five-seed sweep over seeds `11, 17, 23, 31, 47` is stable for both estimators. Visual closed-loop achieves mean position error `0.0453 +/- 0.0005 m`, mean waypoint error `0.0196 +/- 0.0006 m`, and `1.00` completion fraction with zero IK failures. Fused closed-loop achieves mean position error `0.0461 +/- 0.0005 m`, mean waypoint error `0.0234 +/- 0.0007 m`, and `1.00` completion fraction with zero IK failures.

## Actuation Comparison

The minimal first-pass actuation comparison uses fused closed-loop seed `007` with presets `none` and `servo_nominal`. In this frozen suite, `servo_nominal` is the better-behaved setting: mean waypoint error `0.0241 m`, completion `1.00`, and mean actuator tracking error `0.0069`. The `none` preset reaches only `0.50` completion with mean waypoint error `0.0505 m` and mean actuator tracking error `0.1287`.

## Controller Diagnostics

The canonical run records zero IK failures, dominant safety reason `path_complete`, anchor visible fraction `1.00`, mean anchor innovation norm `0.269`, and `22` anchor relocalizations. The controller stays on estimate-driven control after a single initial `bootstrap_hold` tick, then uses estimated state for the remaining `399` control updates.

## Uncertainty And Residual Quality

The canonical run reports mean 95% position radius `0.0246 m`, empirical 95% coverage `68.5%`, pose NEES `69.5`, and sigma/error correlation `0.660`. This means the first-pass covariance output is informative but still overconfident. The residual table shows mean reprojection RMSE `49,717.5 px`, p95 reprojection RMSE `190,534.8 px`, mean anchor innovation norm `0.269`, anchor PnP success fraction `1.00`, and fallback-only frame fraction `0.00`. In other words, translational tracking is usable, but orientation / map consistency remains weak under the current lightweight filter+smoother stack.

## Map Quality

Across the smoother snapshots in the canonical run, the auxiliary-tag map reports mean position error `0.757 m` and p95 position error `1.510 m`, with anchor visible fraction `1.00` and `22` anchor relocalizations. This is the clearest first-pass sign that auxiliary-tag refinement is not yet a publication-grade mapping result even though the anchored closed-loop trajectory is stable enough for the current benchmark.

## Honest Limitations

Replay on this branch is still analysis-only rather than a raw-log re-solver. The first-pass suite is intentionally narrow: one canonical scene, one robot preset, one short actuation comparison, and no broad stress or visibility campaign yet. Real-phone transfer and a stronger offline estimation backend remain future-branch work.
