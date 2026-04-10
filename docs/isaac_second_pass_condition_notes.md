# Isaac Second-Pass Condition Notes

This note is the discussion-side companion to
`output/isaac_runs/latest_second_pass_suite/analysis/suite_summary.json`.

The rows below summarize the canonical five-point zig-zag 18-run suite
promoted at commit `a18a7aa`. Representative run IDs are the current suite
curation stored in `docs/second_pass_draft_lock.json`.

## Nominal Full Anchor

### Visual

- representative run: `second_pass_draft_visual_nominal_full_anchor_seed_017`
- completion fraction: `1.0` across seeds `7, 11, 17`
- anchor visible raw / effective: `1.0 / 1.0`
- anchor update suppressed fraction: `0.0`
- mean waypoint error: `0.02822 m`
- mean position error: `0.01379 m`
- empirical coverage / NEES: `99.79% / 4.99`
- qualitative note:
  - anchor-only visual remains clean, repeatable, and slightly better on
    waypoint error than the promoted fused lock
  - the visual nominal row remains the stronger control reference even though
    fused is now slightly better on mean position error

### Fused

- representative run: `second_pass_draft_fused_nominal_full_anchor_seed_007`
- completion fraction: `1.0` across seeds `7, 11, 17`
- anchor visible raw / effective: `1.0 / 1.0`
- anchor update suppressed fraction: `0.0`
- mean waypoint error: `0.03234 m`
- mean position error: `0.01268 m`
- empirical coverage / NEES: `99.79% / 4.21`
- qualitative note:
  - the promoted `gyro_only + gate_10 + covinfl` fused lock stayed healthy on
    the broader zig-zag nominal case after the focused retune
  - fused is now slightly better than visual on mean position error, but it
    still trails visual on waypoint error

## Intermittent Anchor

### Visual

- representative run: `second_pass_draft_visual_intermittent_anchor_seed_017`
- completion fraction: `1.0` across seeds `7, 11, 17`
- anchor visible raw / effective: `1.0 / 0.80`
- anchor update suppressed fraction: `0.20`
- mean waypoint error: `0.02826 m`
- mean position error: `0.01779 m`
- empirical coverage / NEES: `90.49% / 7.90`
- qualitative note:
  - visual remains stable through the deterministic suppression windows and
    stays close to the configured visibility schedule
  - the intermittent-anchor visual row remains the stronger practical baseline
    on the current zig-zag path

### Fused

- representative run: `second_pass_draft_fused_intermittent_anchor_seed_011`
- completion fraction: `0.9333` across seeds `7, 11, 17`
- anchor visible raw / effective: `1.0 / 0.80`
- anchor update suppressed fraction: `0.20`
- mean waypoint error: `0.03208 m`
- mean position error: `0.03112 m`
- empirical coverage / NEES: `87.01% / 14.24`
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
- completion fraction: `0.6` across seeds `7, 11, 17`
- anchor visible raw / effective: `1.0 / 1.0`
- anchor update suppressed fraction: `0.0`
- mean waypoint error: `0.03953 m`
- mean position error: `0.01252 m`
- empirical coverage / NEES: `99.79% / 4.72`
- qualitative note:
  - visual stays stable under the current stress preset and remains the better
    waypoint-tracking controller on this condition
  - the stress preset increases path-tracking difficulty without causing
    estimator collapse

### Fused

- representative run: `second_pass_draft_fused_servo_stress_seed_017`
- completion fraction: `0.6` across seeds `7, 11, 17`
- anchor visible raw / effective: `1.0 / 1.0`
- anchor update suppressed fraction: `0.0`
- mean waypoint error: `0.04421 m`
- mean position error: `0.01261 m`
- empirical coverage / NEES: `99.79% / 4.20`
- qualitative note:
  - fused remains well calibrated and nearly matches visual on mean position
    error under servo stress
  - the current fused lock still trails visual on waypoint error, so the draft
    should frame this condition as a stability/calibration check rather than a
    fused win
