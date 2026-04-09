# Isaac Second-Pass Figure And Video Notes

This note records what each second-pass draft asset is supposed to show and what
the reader should notice first.

Current status note:

- the first full suite-generated figure set is diagnostic, not draft-final
- after the 2026-04-09 fused-path fixes, the nominal and intermittent-anchor
  figures should be rerendered once the fused dropout instability is resolved

## Paper Figures

### `nominal_trajectory_compare.png`

- Shows: representative visual vs fused top-down trajectory on the nominal full-anchor condition.
- Notice: whether fused stays close to the visual baseline without losing completion or introducing obvious trajectory wobble.
- Cite in paper: nominal-results paragraph in the Results section.

### `dropout_trajectory_compare.png`

- Shows: representative visual vs fused top-down trajectory with deterministic anchor-update suppression.
- Notice: whether fused degrades more gracefully through the dropout windows than visual-only.
- Current reality: the first generated version instead shows the branch blocker, namely catastrophic fused estimator drift while realized control still completes.
- Cite in paper: intermittent-anchor paragraph in the Results section.

### `stress_trajectory_compare.png`

- Shows: representative visual vs fused top-down trajectory under the `servo_stress` actuation preset.
- Notice: whether estimator differences remain visible once the actuation path becomes harder.
- Cite in paper: actuation-stress paragraph in the Results section.

### `coverage_nees_compare.png`

- Shows: suite-level coverage and NEES summary.
- Notice: whether fused calibration is still overconfident or moves closer to nominal after second-pass tuning.
- Cite in paper: uncertainty-discussion paragraph in the Results or Discussion section.

### `anchor_vs_aux_residuals_nominal.png`

- Shows: residual-quality comparison for the representative nominal visual and fused runs.
- Notice: whether the second-pass branch truly removed the first-pass residual pathology instead of only masking it with anchor relocalization.
- Cite in paper: nominal residual-quality paragraph near the nominal table.

### `anchor_vs_aux_residuals_dropout.png`

- Shows: residual-quality comparison for the representative intermittent-anchor runs.
- Notice: whether anchor suppression hurts visual-only and fused in the same way or reveals a clearer fusion advantage.
- Cite in paper: intermittent-anchor analysis paragraph.

### `smoother_feedback_compare.png`

- Shows: smoother-correction timelines for the representative nominal visual and fused runs.
- Notice: whether the chosen backend refines the state cleanly or still produces large corrective jumps.
- Cite in paper: tuning or estimator-backend discussion paragraph.

### `tuning_heatmap_fused.png`

- Shows: compact fused-tuning summary across the explored covariance/backend settings.
- Notice: that the nominal fused configuration was selected from generated evidence rather than a hand-picked parameter set.
- Cite in paper: second-pass tuning protocol section.

## Dashboard And Presentation Videos

### `hero_demo.mp4`

- Shows: the nominal fused run with an overview-style diagnostic overlay.
- Notice: the end-to-end system behavior and the fact that the branch now has a readable demo artifact, not just tables.
- Cite in dashboard: opening hero section.

### `visual_vs_fused_nominal.mp4`

- Shows: split-screen nominal comparison of visual and fused closed-loop behavior.
- Notice: whether fusion is at least competitive on the clean baseline.
- Cite in dashboard: nominal finding box.

### `visual_vs_fused_dropout.mp4`

- Shows: split-screen intermittent-anchor comparison.
- Notice: how the two estimators behave while anchor updates are suppressed on schedule.
- Current reality: until the fused dropout bug is fixed, this video is more useful as an internal debugging asset than as a paper-claim asset.
- Cite in dashboard: intermittent-anchor finding box.

### `actuation_stress_demo.mp4`

- Shows: fused nominal versus fused `servo_stress` behavior.
- Notice: how harder actuation changes the positioning task and how much uncertainty grows.
- Cite in dashboard: servo-stress finding box.

### `observer_phone_diagnostics.mp4`

- Shows: observer-view imagery paired with the diagnostic context used to interpret the nominal fused run.
- Notice: the link between motion, residual quality, and smoother feedback.
- Cite in dashboard: diagnostics section.

### `paper_teaser_second_pass.mp4`

- Shows: stitched overview of the nominal, dropout, and servo-stress second-pass story.
- Notice: the high-level scientific takeaway in under one minute.
- Cite in dashboard: top-level presentation or lab-review link.
