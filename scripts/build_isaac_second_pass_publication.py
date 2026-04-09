#!/usr/bin/env python3
"""Regenerate and compile the second-pass publication draft."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from calib_sim.reporting.isaac_second_pass_suite import (
    DEFAULT_SECOND_PASS_DRAFT_LOCK,
    generate_second_pass_suite_artifacts,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "isaac_runs"
DEFAULT_REPORT_TEX_DIR = REPO_ROOT / "report_tex"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--report-tex-dir", default=str(DEFAULT_REPORT_TEX_DIR))
    parser.add_argument("--lock-path", default=str(DEFAULT_SECOND_PASS_DRAFT_LOCK))
    return parser.parse_args()


def _run_latexmk(report_tex_dir: Path) -> Path:
    latexmk = shutil.which("latexmk")
    if latexmk is None:
        raise FileNotFoundError("latexmk is required to build the second-pass publication draft.")
    subprocess.run(
        [latexmk, "-g", "-pdf", "-interaction=nonstopmode", "second_pass_publication_draft.tex"],
        cwd=str(report_tex_dir),
        check=True,
        capture_output=True,
        text=True,
    )
    return report_tex_dir / "second_pass_publication_draft.pdf"


def build_second_pass_publication(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    report_tex_dir: Path = DEFAULT_REPORT_TEX_DIR,
    lock_path: Path = DEFAULT_SECOND_PASS_DRAFT_LOCK,
) -> dict[str, object]:
    suite_payload = generate_second_pass_suite_artifacts(
        output_root,
        lock_path=lock_path,
        regenerate_run_artifacts=False,
        artifact_source="latest_second_pass_suite",
    )
    paper_artifacts_path = Path(suite_payload["paper_artifacts_tex"]).resolve()
    placeholders_remaining = r"\ArtifactPending{}" in paper_artifacts_path.read_text(encoding="utf-8")
    pdf_path = _run_latexmk(report_tex_dir).resolve()

    suite_dir = Path(suite_payload["suite_dir"]).resolve()
    build_summary = {
        "artifact_source": "latest_second_pass_suite",
        "suite_dir": str(suite_dir),
        "suite_summary_json": str(Path(suite_payload["summary_json"]).resolve()),
        "paper_artifacts_tex": str(paper_artifacts_path),
        "pdf_path": str(pdf_path),
        "pdf_exists": pdf_path.exists(),
        "placeholders_remaining": bool(placeholders_remaining),
    }
    summary_path = suite_dir / "analysis" / "publication_build_summary.json"
    summary_path.write_text(json.dumps(build_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    build_summary["publication_build_summary"] = str(summary_path.resolve())
    return build_summary


def main() -> int:
    args = parse_args()
    payload = build_second_pass_publication(
        output_root=Path(args.output_root).resolve(),
        report_tex_dir=Path(args.report_tex_dir).resolve(),
        lock_path=Path(args.lock_path).resolve(),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["pdf_exists"] and not payload["placeholders_remaining"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
