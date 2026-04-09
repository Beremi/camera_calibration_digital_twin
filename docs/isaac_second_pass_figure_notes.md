# Isaac Second-Pass Figure And Video Notes

This note records what each second-pass draft asset now shows after the
refreshed rerun suite. The key change from the earlier blocked branch state is
that intermittent-anchor fused is no longer a catastrophic failure case.

## Paper Figures

### `nominal_trajectory_compare.png`

- Shows: representative visual vs fused top-down trajectory on the nominal full-anchor condition.
- Notice: fused now tracks the nominal path competitively on mean position error without reintroducing the first-pass instability.
- Cite in paper: nominal-results paragraph in the Results section.

### `dropout_trajectory_compare.png`

- Shows: representative visual vs fused top-down trajectory with deterministic anchor-update suppression.
- Notice: fused no longer diverges catastrophically through the suppression windows.
- Current reality: visual still looks cleaner on the current error metrics, so this figure supports a stabilization claim, not a fused-win claim.
- Cite in paper: intermittent-anchor paragraph in the Results section.

### `stress_trajectory_compare.png`

- Shows: representative visual vs fused top-down trajectory under the `servo_stress` actuation preset.
- Notice: both estimators remain stable under the harder actuation path, with fused slightly better on mean position error and visual still better on waypoint error.
- Cite in paper: actuation-stress paragraph in the Results section.

### `coverage_nees_compare.png`

- Shows: suite-level coverage and NEES summary.
- Notice: second-pass fused calibration is now numerically sane across the suite, including the intermittent-anchor condition.
- Cite in paper: uncertainty-discussion paragraph in the Results or Discussion section.

### `anchor_vs_aux_residuals_nominal.png`

- Shows: residual-quality comparison for the representative nominal visual and fused runs.
- Notice: the second-pass branch removed the first-pass residual pathology without needing to claim a broad fused-control win.
- Cite in paper: nominal residual-quality paragraph near the nominal table.

### `anchor_vs_aux_residuals_dropout.png`

- Shows: residual-quality comparison for the representative intermittent-anchor runs.
- Notice: reduced anchor availability no longer produces the old fused branch blocker, even though visual remains stronger on the current control-facing metrics.
- Cite in paper: intermittent-anchor analysis paragraph.

### `smoother_feedback_compare.png`

- Shows: smoother-correction timelines for the representative nominal visual and fused runs.
- Notice: the chosen backend and lock no longer produce the large corrective jumps that previously made the fused path hard to trust.
- Cite in paper: tuning or estimator-backend discussion paragraph.

### `tuning_heatmap_fused.png`

- Shows: compact fused-tuning summary across the explored covariance settings.
- Notice: the promoted fused nominal lock was selected from generated evidence and a focused retune rather than a hand-picked parameter set.
- Cite in paper: second-pass tuning protocol section.

## Dashboard And Presentation Videos

### `hero_demo.mp4`

- Shows: the nominal fused run with an overview-style diagnostic overlay.
- Notice: the branch now has a stable, readable end-to-end fused demo rather than only recovery/debug artifacts.
- Cite in dashboard: opening hero section.

### `visual_vs_fused_nominal.mp4`

- Shows: split-screen nominal comparison of visual and fused closed-loop behavior.
- Notice: fused is competitive on mean position error, while visual still holds the cleaner waypoint-tracking baseline.
- Cite in dashboard: nominal finding box.

### `visual_vs_fused_dropout.mp4`

- Shows: split-screen intermittent-anchor comparison.
- Notice: fused remains stable and interpretable while anchor updates are suppressed on schedule.
- Current reality: this is now a paper-claim asset, but it supports a repaired-baseline story rather than a stronger fused advantage claim.
- Cite in dashboard: intermittent-anchor finding box.

### `actuation_stress_demo.mp4`

- Shows: fused nominal versus fused `servo_stress` behavior.
- Notice: the stress preset changes the positioning task without causing estimator collapse, and the second-pass result is best framed as robustness plus calibration rather than a fused waypoint win.
- Cite in dashboard: servo-stress finding box.

### `observer_phone_diagnostics.mp4`

- Shows: observer-view imagery paired with the diagnostic context used to interpret the nominal fused run.
- Notice: the link between motion, residual quality, and smoother feedback on the repaired fused path.
- Cite in dashboard: diagnostics section.

### `paper_teaser_second_pass.mp4`

- Shows: stitched overview of the nominal, intermittent-anchor, and servo-stress second-pass story.
- Notice: the high-level takeaway is stabilization: nominal fused is competitive on mean position error, intermittent-anchor fused is no longer catastrophic, and visual remains the stronger waypoint baseline.
- Cite in dashboard: top-level presentation or lab-review link.
