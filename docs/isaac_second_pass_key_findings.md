# Isaac Second-Pass Key Findings

This note now reflects the current state after the first full 18-run suite
attempt and the follow-up fused-path reruns on 2026-04-09.

## Nominal Condition

The nominal three-seed batch is stable and fully complete for both estimators,
but it still does not support the strongest second-pass claim on the clean
condition. Under the provisional anchor-only lock, visual averages
`0.01392 m` mean position error and `0.01964 m` mean waypoint error, while
fused averages `0.01623 m` and `0.02244 m`. A later post-fix seed-007 fused
nominal rerun improved to `0.01556 m` mean position error and `0.02211 m`
waypoint error, so the nominal fused path is getting better, but it is still
not yet within the intended `5%` competitiveness bar against visual.

## Intermittent-Anchor Condition

The first full suite attempt produced the clearest remaining blocker on the
branch. Visual under intermittent anchor stayed numerically sane with
`0.01648 m` mean position error, `92.71%` empirical coverage, and `6.30` pose
NEES. Fused under the same condition collapsed catastrophically at about
`251.69 m` mean position error, `14.72%` empirical coverage, and
`1.46e7` pose NEES. Follow-up fused reruns with auxiliary tags enabled and with
two runtime fixes still remained catastrophic at roughly `248–253 m` mean
position error. The current conclusion is therefore that intermittent-anchor
fused stability is the unresolved blocker for the first draft.

## Servo-Stress Condition

The first full suite attempt did complete the servo-stress condition. Visual
averaged `0.01381 m` mean position error and `0.02377 m` mean waypoint error,
while fused averaged `0.01593 m` and `0.02734 m`. That means the current fused
configuration does not yet show a clear control advantage under the present
servo-stress preset, even though both estimators still complete the path.

## Uncertainty Calibration

The clean nominal condition is no longer grossly overconfident. Fused nominal
coverage stays around `99.58%` with pose NEES around `4.19`, compared with
visual at `99.79%` and `5.01`, and the post-fix seed-007 fused rerun remained
well calibrated at `4.13` NEES. The calibration story changes completely under
intermittent anchor, where fused still collapses to about `14.7%` empirical
coverage and `1.4e7` pose NEES. The branch therefore no longer has a general
uncertainty-calibration problem; it has a fused dropout-specific failure mode.

## Remaining Failure Modes

The remaining weakness is now narrower but more serious. The lightweight fused
nominal configuration is stable, completes the path, and keeps uncertainty
sane, but it still loses to visual on the easy condition and still fails
catastrophically under intermittent anchor even after:

- switching the inertial position update to standard constant-acceleration
  kinematics
- sampling synthetic IMU motion from the camera pose instead of the
  end-effector pose

That means the next branch action should not be more publication polishing. It
should be a focused fix to the fused intermittent-anchor runtime path before the
18-run suite is treated as draft-final.
