# Intermediate Report

This folder is a self-contained LaTeX bundle for the current Checkpoint 01 state.

Status note:

- this remains the historical browser-simulator / Checkpoint 01 report
- the Isaac-oriented paper draft now lives under `report_tex/`
- do not duplicate Isaac results or Isaac-only tables here; link readers to the `report_tex/` track instead

Main files:

- `main.tex` — report source
- `main.pdf` — rebuilt sendable PDF snapshot of the checkpoint
- `refs.bib` — bibliography
- `img/` — copied checkpoint figures plus generated environment views
- `build.sh` — helper to compile with `latexmk` or `pdflatex` + `bibtex`

The report documents:

- the simulation environment
- the fiducial-tag inventory and rationale
- the Pixel 9a-based camera and IMU model
- the offline pose-estimation pipeline
- the preserved Checkpoint 01 results from `output/interactive_runs/run_20260406_083611`
- the corrected IMU-only trajectory reconstruction from the exact logging path

Build locally with a TeX distribution installed:

```bash
cd docs/intermediate_report
./build.sh
```

Expected tools:

- `latexmk` preferred, or
- `pdflatex` and `bibtex`

The PDF in this folder was rebuilt locally from the current source after the final IMU logging fix was folded into Checkpoint 01.
