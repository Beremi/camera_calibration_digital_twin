"""Second-pass review-bundle regression."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests._isaac_test_helpers import make_second_pass_draft_lock, make_second_pass_draft_suite_runs


LATEXMK = shutil.which("latexmk")


pytestmark = pytest.mark.skipif(LATEXMK is None, reason="latexmk is required for the second-pass review bundle test.")


def test_second_pass_review_bundle_builds_manifest_and_updates_lock(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_root = tmp_path / "output" / "isaac_runs"
    docs_dir = tmp_path / "docs"
    report_tex_dir = tmp_path / "report_tex"
    make_second_pass_draft_suite_runs(output_root)
    lock_path = make_second_pass_draft_lock(docs_dir)
    shutil.copytree(repo_root / "report_tex", report_tex_dir)
    manifest_path = docs_dir / "second_pass_review_manifest.json"

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "build_isaac_second_pass_review_bundle.py"),
            "--output-root",
            str(output_root),
            "--report-tex-dir",
            str(report_tex_dir),
            "--lock-path",
            str(lock_path),
            "--manifest-path",
            str(manifest_path),
            "--max-frames",
            "12",
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert Path(manifest_path).exists()
    assert payload["draft_pdf_path"]
    assert payload["presentation_manifest_path"]
    assert payload["dashboard_path"]
    assert payload["canonical_science_packet_commit"] == "a18a7aa"
    assert payload["evidence_tables_path"] == "docs/isaac_second_pass_evidence_tables.json"

    updated_lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert updated_lock["checkpoint"]["draft_selection_status"] == "review_bundle_ready"
    assert updated_lock["draft_selection"]["draft_ready"] is True
    assert updated_lock["draft_selection"]["review_manifest_path"] == str(manifest_path.resolve())
