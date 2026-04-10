# Isaac Second-Pass Key Findings

This note reflects the canonical five-point zig-zag second-pass suite promoted
at commit `a18a7aa`.

## Nominal Condition

The nominal zig-zag condition now supports a credible stabilization claim.
Visual averages `0.01379 m` mean position error and `0.02822 m` mean waypoint
error, while fused averages `0.01268 m` and `0.03234 m`. That means fused is
slightly better on mean position error and remains well calibrated, but visual
still leads on waypoint error. The clean nominal story is therefore no longer
that fused is obviously weaker; it is that fused has become defensible and
reproducible on a broader 3D path without yet becoming the better control
baseline on every metric.

## Intermittent-Anchor Condition

The branch blocker was cleared here. In the blocked suite, fused
intermittent-anchor diverged catastrophically. In the canonical zig-zag suite,
visual averages `0.01779 m` mean position error, `0.02826 m` mean waypoint
error, `90.49%` coverage, and `7.90` pose NEES, while fused averages
`0.03112 m`, `0.03208 m`, `87.01%`, and `14.24`. Fused does not beat visual on
the current error metrics, but it is no longer scientifically indefensible: it
remains numerically interpretable through the scheduled suppression windows and
no longer collapses catastrophically.

## Servo-Stress Condition

The servo-stress condition reads mainly as a stability and calibration check.
Visual averages `0.01252 m` mean position error and `0.03953 m` mean waypoint
error, while fused averages `0.01261 m` and `0.04421 m`. Both estimators keep
the path controllable through the stress preset, but visual again retains the
cleaner waypoint-tracking behavior.

## Uncertainty Calibration

The second-pass branch now has a credible calibration story. Nominal fused
coverage is `99.79%` with pose NEES `4.21`, and servo-stress fused remains at
`99.79%` coverage with pose NEES `4.20`. Under intermittent anchor, fused no
longer collapses to absurd values; it holds `87.01%` coverage and `14.24` pose
NEES. That is weaker than visual under the same condition, but it is no longer
the catastrophic overconfidence failure that blocked the draft.

## Control-Facing Read

The new control-success tables reinforce the same interpretation. On the
zig-zag task, fused now reaches the workspace goals across a more spatial path
without reopening the old dropout failure, but visual still keeps the stronger
waypoint-tracking baseline on the current benchmark. In nominal and
intermittent-anchor conditions, visual reaches all waypoints within `5 cm`
while fused reaches `60%` of them. Under servo stress the picture tightens:
fused reaches `53.3%` within `5 cm` versus `40.0%` for visual, while both
estimators see the same large dropped-command burden from the stress preset.
The present paper is therefore best read as a repaired benchmark and
stabilized fused baseline, not as a demonstration that fusion now dominates
control quality.

## Remaining Weaknesses

The remaining weakness is practical rather than existential. The branch now has
a repaired fused runtime policy, a tuned fused nominal lock, a rerun suite, and
a reviewable paper/media package. What it does not yet show is a broad fused
advantage. Visual still leads on waypoint error and remains stronger on the
current intermittent-anchor error metrics. The honest second-pass paper claim
is therefore a stabilization result: fusion is now defensible and reproducible
in the anchored benchmark, not universally better.
