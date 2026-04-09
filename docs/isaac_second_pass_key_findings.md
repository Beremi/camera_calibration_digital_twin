# Isaac Second-Pass Key Findings

This note reflects the refreshed 18-run second-pass suite executed from the
promoted fused draft lock at commit `a1b0d4f`.

## Nominal Condition

The nominal condition now supports a credible second-pass stabilization claim.
Visual averages `0.01388 m` mean position error and `0.01979 m` mean waypoint
error, while fused averages `0.01303 m` and `0.02293 m`. That means fused is
competitive on mean position error and remains well calibrated, but visual
still leads on waypoint error. The clean nominal story is therefore no longer
that fused is obviously weaker; it is that fused has become defensible and
reproducible without yet becoming the better control baseline on every metric.

## Intermittent-Anchor Condition

The branch blocker was cleared here. In the old blocked suite, fused
intermittent-anchor diverged catastrophically. In the refreshed suite, visual
averages `0.01983 m` mean position error, `90.49%` coverage, and `10.56` pose
NEES, while fused averages `0.03072 m`, `88.82%`, and `14.02`. Fused does not
beat visual on the current error metrics, but it is no longer scientifically
indefensible: it completes all runs and stays numerically interpretable through
the scheduled suppression windows.

## Servo-Stress Condition

The servo-stress condition now reads as a stability and calibration check
rather than a fused win. Visual averages `0.01390 m` mean position error and
`0.02366 m` mean waypoint error, while fused averages `0.01328 m` and
`0.02646 m`. Fused is slightly better on mean position error and remains well
calibrated, but visual again keeps the stronger waypoint-tracking baseline.

## Uncertainty Calibration

The second-pass branch now has a credible calibration story. Nominal fused
coverage is `99.72%` with pose NEES `4.25`, and servo-stress fused remains at
`99.79%` coverage with pose NEES `4.26`. Under intermittent anchor, fused no
longer collapses to absurd values; it holds `88.82%` coverage and `14.02` pose
NEES. That is weaker than visual under the same condition, but it is no longer
the catastrophic overconfidence failure that blocked the draft.

## Remaining Weaknesses

The remaining weakness is practical rather than existential. The branch now has
a repaired fused runtime policy, a tuned fused nominal lock, and a rerun suite
that is fit for publication closure. What it does not yet show is a broad fused
advantage. Visual still leads on waypoint error and remains stronger on the
current intermittent-anchor error metrics. The honest second-pass paper claim
is therefore a stabilization result: fusion is now defensible and reproducible
in the anchored benchmark, not universally better.
