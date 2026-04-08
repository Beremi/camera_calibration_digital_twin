# Isaac Second-Pass Isolation Summary

This note is generated from the six canonical `second_pass_*` isolation runs.

- Root-cause classification: `fused_mechanization_or_weighting_bug`
- Decision note: Fused anchor-only is more than 10% worse than visual anchor-only on mean position error.

| Run ID | Estimator | Aux Filter | Aux Smoother | Aux Control | Pos Err [m] | Waypoint Err [m] | Anchor RMSE [px] | Aux RMSE [px] | Coverage [%] | NEES | Aux Map Err [m] |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| second_pass_visual_closed_loop_anchor_only_seed_007 | visual | off | off | off | 0.0166 | 0.0202 | 0.04 | 1.99 | 99.59 | 8.67 | 0.0000 |
| second_pass_fused_closed_loop_anchor_only_seed_007 | fused | off | off | off | 0.0190 | 0.0228 | 0.04 | 1.83 | 97.94 | 9.48 | 0.0000 |
| second_pass_visual_closed_loop_aux_estimation_only_seed_007 | visual | on | on | off | 0.0167 | 0.0198 | 0.34 | 2.25 | 99.59 | 8.70 | 0.0272 |
| second_pass_fused_closed_loop_aux_estimation_only_seed_007 | fused | on | on | off | 0.0190 | 0.0232 | 0.29 | 2.05 | 96.91 | 9.49 | 0.0249 |
| second_pass_visual_closed_loop_aux_for_control_seed_007 | visual | on | on | on | 0.0166 | 0.0204 | 0.34 | 2.22 | 99.59 | 8.68 | 0.0256 |
| second_pass_fused_closed_loop_aux_for_control_seed_007 | fused | on | on | on | 0.0190 | 0.0236 | 0.31 | 2.09 | 96.71 | 9.51 | 0.0269 |

The frozen first-pass publication bundle under `output/isaac_runs/latest_first_pass_suite` was not modified by this second-pass isolation workflow.

