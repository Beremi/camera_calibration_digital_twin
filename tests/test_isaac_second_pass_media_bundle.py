"""Second-pass presentation bundle regression."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from calib_sim.reporting.isaac_second_pass_suite import generate_second_pass_suite_artifacts

from tests._isaac_test_helpers import make_second_pass_draft_lock, make_second_pass_draft_suite_runs


def test_second_pass_media_bundle_and_dashboard_build(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_root = tmp_path / "output" / "isaac_runs"
    docs_dir = tmp_path / "docs"
    make_second_pass_draft_suite_runs(output_root)
    lock_path = make_second_pass_draft_lock(docs_dir)
    generate_second_pass_suite_artifacts(output_root, lock_path=lock_path, regenerate_run_artifacts=False)

    media_result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "render_isaac_media_bundle.py"),
            "--output-root",
            str(output_root),
            "--max-frames",
            "12",
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )
    media_payload = json.loads(media_result.stdout)
    assert Path(media_payload["videos"]["hero_demo"]).exists()
    assert Path(media_payload["videos"]["visual_vs_fused_nominal"]).exists()
    assert Path(media_payload["videos"]["visual_vs_fused_dropout"]).exists()

    dashboard_result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "build_isaac_presentation_dashboard.py"),
            "--output-root",
            str(output_root),
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )
    dashboard_payload = json.loads(dashboard_result.stdout)
    assert Path(dashboard_payload["dashboard_path"]).exists()
