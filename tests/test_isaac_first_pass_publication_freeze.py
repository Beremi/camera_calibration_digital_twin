"""Freeze/build regression for the frozen first-pass publication path."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from calib_sim.reporting.isaac_first_pass_suite import generate_first_pass_suite_artifacts

from tests._isaac_test_helpers import make_fake_first_pass_suite, make_first_pass_suite_lock


LATEXMK = shutil.which("latexmk")


pytestmark = pytest.mark.skipif(LATEXMK is None, reason="latexmk is required for the publication freeze build test.")


def test_first_pass_publication_build_uses_suite_artifacts_and_clears_placeholders(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_root = make_fake_first_pass_suite(tmp_path / "output" / "isaac_runs")
    lock_path = make_first_pass_suite_lock(tmp_path / "docs")
    report_tex_dir = tmp_path / "report_tex"
    shutil.copytree(repo_root / "report_tex", report_tex_dir)
    generate_first_pass_suite_artifacts(output_root, regenerate_run_artifacts=False)

    verify_result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "verify_isaac_first_pass_suite.py"),
            "--lock-path",
            str(lock_path),
            "--suite-dir",
            str(output_root / "latest_first_pass_suite"),
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )
    verify_payload = json.loads(verify_result.stdout)
    assert verify_payload["ok"] is True

    build_result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "build_isaac_first_pass_publication.py"),
            "--lock-path",
            str(lock_path),
            "--suite-dir",
            str(output_root / "latest_first_pass_suite"),
            "--output-root",
            str(output_root),
            "--report-tex-dir",
            str(report_tex_dir),
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )
    build_payload = json.loads(build_result.stdout)
    assert build_payload["artifact_source"] == "latest_first_pass_suite"
    assert build_payload["placeholders_remaining"] is False
    assert Path(build_payload["pdf_path"]).exists()

    build_summary = json.loads(
        (output_root / "latest_first_pass_suite" / "analysis" / "publication_build_summary.json").read_text(encoding="utf-8")
    )
    assert build_summary["artifact_source"] == "latest_first_pass_suite"

    paper_artifacts = (
        output_root / "latest_first_pass_suite" / "analysis" / "report_data" / "paper_artifacts.tex"
    ).read_text(encoding="utf-8")
    assert r"\ArtifactPending{}" not in paper_artifacts
