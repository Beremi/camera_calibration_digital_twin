# Isaac Second-Pass Open Questions

This note tracks the questions that should remain explicit even after the first
second-pass draft closes.

## Fusion Regime

The main scientific question is still whether fusion becomes clearly helpful
once anchor visibility is degraded, or whether the tuned fused configuration is
also meaningfully better on the clean nominal condition. The current answer is
"not yet": nominal fused is still slightly behind visual, and intermittent
anchor currently exposes a fused-specific failure mode instead of a fused
advantage. The next scientific step is therefore to fix that fused dropout path
before asking broader robustness questions.

## Smoother Backend

The branch now supports both the `lightweight` and `windowed_ba` smoother
backends. The current evidence says `windowed_ba` is not the draft blocker:
its nominal waypoint error was worse than `lightweight`, and the catastrophic
intermittent-anchor failure already appears in the anchor-only fused path with
no smoother feedback. The open question is whether the BA backend should remain
available for later mapping work, not whether it closes the present dropout
bug.

## Control Metric Adequacy

Mean waypoint error is currently the main control-quality headline metric. The
current suite shows why it is not sufficient by itself: fused intermittent
anchor still reports a normal waypoint error even while estimator position error
and NEES explode. The open question is therefore no longer just whether
waypoint error is adequate, but how to pair it with estimator-quality plots so
the paper does not accidentally hide a fused failure behind stable realized
control.

## Next Branch Direction

Once the fused intermittent-anchor defect is fixed, the next branch choice
should remain explicit:

- continue with robustness and visibility campaigns if the draft suite already
  shows a clear scientific story,
- or pivot toward replay re-solving only if the live runtime still hides the
  estimator failure mode too much to debug confidently.

The current default still favors robustness and visibility work over replay
re-solving, but there is now one sharper prerequisite question: is the
remaining fused dropout failure caused by another runtime frame/propagation bug,
or by a deeper estimator-model mismatch that will require a larger fused
mechanization rewrite?
