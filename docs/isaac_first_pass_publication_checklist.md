# Isaac First-Pass Publication Checklist

Use this checklist before treating the first-pass paper as the authoritative frozen milestone.

## Frozen Suite

- `docs/first_pass_suite_lock.json` is present and committed.
- `python scripts/verify_isaac_first_pass_suite.py` passes.
- The canonical run id in the lock, suite summary, and paper artifacts is `first_pass_fused_closed-loop_servo_nominal_seed_007`.
- `output/isaac_runs/latest_first_pass_suite` is the publication source of truth.

## Publication Build

- `python scripts/build_isaac_first_pass_publication.py` passes.
- `output/isaac_runs/latest_first_pass_suite/analysis/publication_build_summary.json` exists.
- The build summary reports `artifact_source = latest_first_pass_suite`.
- `placeholders_remaining` is `false`.
- `report_tex/publication_report_template.pdf` compiles successfully.

## Artifacts

- `output/isaac_runs/latest_first_pass_suite/analysis/report_data/paper_artifacts.tex` exists.
- The suite bundle includes all first-pass table macros:
  - `\IsaacFirstPassCompletenessRows`
  - `\IsaacFirstPassMatrixRows`
  - `\IsaacFirstPassReproducibilityRows`
  - `\IsaacFirstPassControllerRows`
  - `\IsaacFirstPassActuationRows`
  - `\IsaacFirstPassUncertaintyRows`
  - `\IsaacFirstPassResidualRows`
  - `\IsaacFirstPassMapRows`
- The suite bundle includes the first-pass figures needed by the paper.

## Documentation

- `README.md` points to `latest_first_pass_suite` as the frozen source of truth.
- `docs/README.md` points to the lock, summary, and publication checklist.
- `docs/isaac_first_pass_results.md` lists only the frozen suite runs as publication inputs.
- `docs/isaac_first_pass_summary.md` matches the current suite summary and canonical metrics.
