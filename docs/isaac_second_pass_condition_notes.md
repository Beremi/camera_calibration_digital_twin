# Isaac Second-Pass Condition Notes

This note is the discussion-side companion to
`output/isaac_runs/latest_second_pass_suite/analysis/suite_summary.json`.

The rows below summarize the refreshed 18-run suite executed from the promoted
fused draft lock at commit `a1b0d4f`. Representative run IDs are the current
suite curation stored in `docs/second_pass_draft_lock.json`.

## Nominal Full Anchor

### Visual

- representative run: `second_pass_draft_visual_nominal_full_anchor_seed_017`
- completion fraction: `1.0` across seeds `7, 11, 17`
- anchor visible raw / effective: `1.0 / 1.0`
- anchor update suppressed fraction: `0.0`
- mean waypoint error: `0.01979 m`
- mean position error: `0.01388 m`
- empirical coverage / NEES: `99.79% / 5.01`
- qualitative note:
  - anchor-only visual remains clean, repeatable, and slightly better on
    waypoint error than the promoted fused nominal lock
  - the visual nominal row remains a strong control reference rather than a
    failure case the fused path needed to rescue

### Fused

- representative run: `second_pass_draft_fused_nominal_full_anchor_seed_007`
- completion fraction: `1.0` across seeds `7, 11, 17`
- anchor visible raw / effective: `1.0 / 1.0`
- anchor update suppressed fraction: `0.0`
- mean waypoint error: `0.02293 m`
- mean position error: `0.01303 m`
- empirical coverage / NEES: `99.72% / 4.25`
- qualitative note:
  - the promoted `gyro_only + gate_10 + covinfl` fused lock stayed healthy on
    the clean nominal case after the focused retune
  - fused is now within the requested nominal position window and is slightly
    better than visual on mean position error, but it still trails visual on
    waypoint error

## Intermittent Anchor

### Visual

- representative run: `second_pass_draft_visual_intermittent_anchor_seed_017`
- completion fraction: `1.0` across seeds `7, 11, 17`
- anchor visible raw / effective: `1.0 / 0.80`
- anchor update suppressed fraction: `0.20`
- mean waypoint error: `0.01874 m`
- mean position error: `0.01983 m`
- empirical coverage / NEES: `90.49% / 10.56`
- qualitative note:
  - visual remains stable through the deterministic suppression windows and
    stays close to the configured visibility schedule
  - the intermittent-anchor visual row remains a strong practical baseline
    even though its calibration is weaker than the nominal case

### Fused

- representative run: `second_pass_draft_fused_intermittent_anchor_seed_011`
- completion fraction: `1.0` across seeds `7, 11, 17`
- anchor visible raw / effective: `1.0 / 0.80`
- anchor update suppressed fraction: `0.20`
- mean waypoint error: `0.02250 m`
- mean position error: `0.03072 m`
- empirical coverage / NEES: `88.82% / 14.02`
- qualitative note:
  - this row is no longer catastrophic and is now scientifically interpretable
    under the promoted suppression strategy
  - fused still trails visual on both current error metrics and uncertainty
    calibration in this condition, but the branch blocker was cleared because
    the intermittent-anchor fused path is now stable enough to defend in the
    paper

## Servo Stress

### Visual

- representative run: `second_pass_draft_visual_servo_stress_seed_007`
- completion fraction: `1.0` across seeds `7, 11, 17`
- anchor visible raw / effective: `1.0 / 1.0`
- anchor update suppressed fraction: `0.0`
- mean waypoint error: `0.02366 m`
- mean position error: `0.01390 m`
- empirical coverage / NEES: `99.79% / 5.01`
- qualitative note:
  - visual stays stable under the current stress preset and remains the better
    waypoint-tracking controller on this condition
  - the stress preset increases tracking difficulty without causing estimator
    collapse

### Fused

- representative run: `second_pass_draft_fused_servo_stress_seed_017`
- completion fraction: `1.0` across seeds `7, 11, 17`
- anchor visible raw / effective: `1.0 / 1.0`
- anchor update suppressed fraction: `0.0`
- mean waypoint error: `0.02646 m`
- mean position error: `0.01328 m`
- empirical coverage / NEES: `99.79% / 4.26`
- qualitative note:
  - fused remains well calibrated and slightly better than visual on mean
    position error under servo stress
  - the current fused lock still trails visual on waypoint error, so the draft
    should frame this condition as a stability/calibration check rather than a
    fused win
