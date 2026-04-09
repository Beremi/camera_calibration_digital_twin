# Isaac Second-Pass Packet History

This note archives the packet-by-packet branch history that used to live inside
`docs/isaac_estimator_second_pass.md`. The main second-pass note now stays
focused on the current review package and current branch state.

## Preserved Pre-Draft Checkpoint

- checkpoint commit: `b3f23c9`
- preserved isolation bundle:
  - `output/isaac_runs/second_pass_checkpoint_isolation_seed_007/`
- branch-level classification from that checkpoint:
  - `fused_mechanization_or_weighting_bug`

## Historical Packet Timeline

### 1. Isolation And Mechanization Diagnosis

- seed-007 isolation bundle established that anchor-only visual was already
  sane while anchor-only fused still trailed visual
- auxiliary-tag instability and the earlier reprojection-metric bug were ruled
  out as the dominant branch-level blocker

### 2. Dropout-Blocker Investigation

- the first full 18-run suite under the provisional fused lock showed
  catastrophic fused intermittent-anchor failure while visual remained sane
- dedicated seed-007 dropout-debug runs localized the remaining fused defect to
  suppression-window propagation rather than a smoother swap or pure
  reacquisition-only failure

### 3. Suppression-Propagation Repair

- inertial propagation and synthetic IMU semantics were corrected
- the production-style suppression winner emerged as:
  - `suppression_propagation_mode = gyro_only`
  - `suppression_imu_specific_force_gate_mps2 = 10.0`
  - `dropout_post_reacquisition_covariance_scale = 8.0`
- this candidate cleared the blocker on seeds `007`, `011`, and `017` while
  preserving a healthy nominal fused recheck

### 4. Lock Promotion And Suite Rerun

- the promoted suppression winner was written into the real draft lock
- a focused nominal retune selected the current fused nominal reference
- the full 18-run suite was rerun from that promoted lock
- the rerun suite no longer contained a catastrophic fused intermittent-anchor
  row

### 5. Publication Closure

- the second-pass PDF was rebuilt from `latest_second_pass_suite`
- the local media bundle and dashboard were regenerated from the same suite
- the branch moved from “science blocker cleared” to “review bundle ready”

## Where To Read The Current Story

- current package status:
  - `docs/isaac_estimator_second_pass.md`
- collaborator quickstart:
  - `docs/isaac_second_pass_handoff.md`
- current findings:
  - `docs/isaac_second_pass_key_findings.md`
- current figure/video guidance:
  - `docs/isaac_second_pass_figure_notes.md`
- detailed command history:
  - `docs/isaac_second_pass_execution_log.md`
