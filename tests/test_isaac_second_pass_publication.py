"""Second-pass publication draft regression."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests._isaac_test_helpers import make_second_pass_draft_lock, make_second_pass_draft_suite_runs


LATEXMK = shutil.which("latexmk")


pytestmark = pytest.mark.skipif(LATEXMK is None, reason="latexmk is required for the second-pass publication build test.")


def test_second_pass_publication_build_uses_suite_artifacts(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_root = tmp_path / "output" / "isaac_runs"
    docs_dir = tmp_path / "docs"
    report_tex_dir = tmp_path / "report_tex"
    make_second_pass_draft_suite_runs(output_root)
    lock_path = make_second_pass_draft_lock(docs_dir)
    shutil.copytree(repo_root / "report_tex", report_tex_dir)

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "build_isaac_second_pass_publication.py"),
            "--output-root",
            str(output_root),
            "--report-tex-dir",
            str(report_tex_dir),
            "--lock-path",
            str(lock_path),
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["artifact_source"] == "latest_second_pass_suite"
    assert payload["placeholders_remaining"] is False
    assert Path(payload["pdf_path"]).exists()
    assert (docs_dir / "isaac_second_pass_evidence_tables.json").exists()

    updated_lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert updated_lock["draft_selection"]["publication_build_summary_path"] == payload["publication_build_summary"]
    assert updated_lock["draft_selection"]["publication_pdf_path"] == payload["pdf_path"]
