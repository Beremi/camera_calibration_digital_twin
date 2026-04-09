# Isaac Second-Pass Condition Notes

This note is the discussion-side companion to
`output/isaac_runs/latest_second_pass_suite/analysis/suite_summary.json`.

The currently recorded suite rows come from the first full 18-run draft-suite
attempt under the provisional anchor-only fused lock. They are still useful for
discussion, but they should be treated as diagnostic rather than draft-final
because the fused path has since received two runtime fixes and still needs a
full suite rerun once intermittent-anchor stability is repaired.

## Nominal Full Anchor

### Visual

- representative run: `second_pass_draft_visual_nominal_full_anchor_seed_017`
- completion fraction: `1.0` across seeds `7, 11, 17`
- mean waypoint error: `0.01964 m`
- mean position error: `0.01392 m`
- empirical coverage / NEES: `99.79% / 5.01`
- qualitative note:
  - anchor-only visual is clean, repeatable, and slightly better than the
    current fused nominal candidate on the nominal condition
  - residuals remain numerically sane and there is no obvious controller
    instability in the nominal runs

### Fused

- representative run: `second_pass_draft_fused_nominal_full_anchor_seed_007`
- completion fraction: `1.0` across seeds `7, 11, 17`
- mean waypoint error: `0.02244 m`
- mean position error: `0.01623 m`
- empirical coverage / NEES: `99.58% / 4.19`
- qualitative note:
  - fused remains stable and well calibrated on the nominal condition, but it
    still trails visual on both position and waypoint error with the current
    lightweight anchor-only lock
  - this keeps the draft story centered on “fusion helps under harder
    perturbations” rather than “fusion wins everywhere”

## Intermittent Anchor

### Visual

- representative run: `second_pass_draft_visual_intermittent_anchor_seed_017`
- anchor visible raw / effective: `1.0 / 0.80`
- anchor update suppressed fraction: `0.20`
- mean waypoint error: `0.01978 m`
- mean position error: `0.01648 m`
- empirical coverage / NEES: `92.71% / 6.30`
- qualitative note:
  - visual remains stable through the deterministic suppression windows
  - the effective anchor visibility matches the configured schedule and the
    estimator remains numerically sane

### Fused

- representative run: `second_pass_draft_fused_intermittent_anchor_seed_011`
- anchor visible raw / effective: `0.7403 / 0.5750`
- anchor update suppressed fraction: `0.20`
- mean waypoint error: `0.02281 m`
- mean position error: `251.69 m`
- empirical coverage / NEES: `14.72% / 1.46e7`
- qualitative note:
  - this is the current branch blocker
  - realized control still completes, but the fused estimator diverges so
    badly that the condition is not publication-ready
  - post-fix exploratory reruns on seed `007` still show the same collapse,
    so this section should be rerun only after the fused intermittent-anchor
    path is fixed

## Servo Stress

### Visual

- representative run: `second_pass_draft_visual_servo_stress_seed_007`
- completion fraction: `1.0`
- mean waypoint error: `0.02377 m`
- mean position error: `0.01381 m`
- empirical coverage / NEES: `99.79% / 4.98`
- qualitative note:
  - visual stays stable under the current stress preset, with modestly worse
    waypoint tracking than the nominal case

### Fused

- representative run: `second_pass_draft_fused_servo_stress_seed_017`
- completion fraction: `1.0`
- mean waypoint error: `0.02734 m`
- mean position error: `0.01593 m`
- empirical coverage / NEES: `99.58% / 4.16`
- qualitative note:
  - fused remains stable and well calibrated under stress, but it does not yet
    outperform visual on the current control-facing metrics
