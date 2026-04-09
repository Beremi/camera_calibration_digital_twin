# Isaac Second-Pass Dropout Debug Pack

This bundle tracks the seed-007 fused intermittent-anchor blocker before the second-pass draft suite is rerun.

- summary csv: `/home/beremi/repos/camera_calibration_digital_twin/output/isaac_runs/latest_second_pass_dropout_debug/summary.csv`
- summary json: `/home/beremi/repos/camera_calibration_digital_twin/output/isaac_runs/latest_second_pass_dropout_debug/summary.json`

## Runs

- `second_pass_dropout_debug_visual_seed_007`, mode=visual_reference, pos=0.017617555208978928, way=0.019785564651336345, coverage=96.875, nees=4.581378051790227, hint=stable_reference
- `second_pass_dropout_debug_fused_seed_007`, mode=fused_baseline, pos=0.1551700701675676, way=0.02421285760291589, coverage=76.04166666666666, nees=195.95499022278034, hint=propagation_process_problem
- `second_pass_dropout_debug_fused_no_reacq_seed_007`, mode=fused_no_reacquisition, pos=692.7857840816636, way=0.05300121148098716, coverage=0.4166666666666667, nees=81649893.60492586, hint=propagation_process_problem
- `second_pass_dropout_debug_fused_no_imu_during_suppression_seed_007`, mode=fused_no_imu_during_suppression, pos=0.013669403114194975, way=0.021429745298426353, coverage=99.375, nees=4.057626038703267, hint=stable_reference
- `second_pass_dropout_debug_fused_covinfl_seed_007`, mode=fused_reacquisition_covariance_inflation, pos=0.1557320526919672, way=0.0240847787557051, coverage=75.625, nees=195.8501785584854, hint=undetermined
- `second_pass_dropout_debug_fused_clipcorr_seed_007`, mode=fused_reacquisition_correction_clipping, pos=0.18695587314463508, way=0.023668974070898505, coverage=74.375, nees=308.4671781134249, hint=undetermined

## Current Read

- blocker-clear runs: second_pass_dropout_debug_visual_seed_007, second_pass_dropout_debug_fused_no_imu_during_suppression_seed_007
- current production fused baseline is improved but still blocked:
  - mean position error: `0.15517 m`
  - empirical 95% coverage: `76.04%`
  - pose NEES: `195.95`
  - dominant hint: `propagation_process_problem`
- `second_pass_dropout_debug_fused_no_reacq_seed_007` is diagnostic only and is
  not an acceptable publication method
- the next estimator iteration should stay on suppression-window propagation
  plausibility or split gyro/accel process weighting; do not reopen the draft
  suite, second-pass PDF, or polished media yet
- do not rerun the 18-run second-pass suite or rebuild the second-pass draft PDF until at least one fused intermittent-anchor variant clears the interim blocker bar.

## Validated Candidate

The control bundle above remains the branch source of truth for the blocked
baseline. A separate probe family under
`output/isaac_runs/probes/gyro_only_gate_10_validation` has now validated one
production-like fused suppression candidate:

- `suppression_propagation_mode = gyro_only`
- `suppression_imu_specific_force_gate_mps2 = 10.0`
- `dropout_post_reacquisition_covariance_scale = 8.0`

Validated intermittent-anchor fused runs:

- seed `007`:
  - `second_pass_dropout_debug_fused_covinfl_seed_007`
  - mean position error: `0.03091 m`
  - empirical 95% coverage: `88.96%`
  - pose NEES: `10.10`
- seed `011`:
  - `second_pass_validation_fused_intermittent_anchor_only_seed_011_gyro_only_covinfl_gate_10`
  - mean position error: `0.03560 m`
  - empirical 95% coverage: `86.04%`
  - pose NEES: `13.02`
- seed `017`:
  - `second_pass_validation_fused_intermittent_anchor_only_seed_017_gyro_only_covinfl_gate_10`
  - mean position error: `0.02935 m`
  - empirical 95% coverage: `93.75%`
  - pose NEES: `10.42`

Nominal fused follow-up for the same candidate:

- `second_pass_followup_fused_nominal_anchor_only_seed_007_gyro_only_covinfl_gate_10`
  - mean position error: `0.01278 m`
  - empirical 95% coverage: `99.79%`
  - pose NEES: `3.97`

This means the next packet can promote the candidate into the draft lock and
rerun the second-pass suite, but the draft suite, PDF, and presentation bundle
should remain deferred until that rerun happens.
