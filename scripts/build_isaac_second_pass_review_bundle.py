#!/usr/bin/env python3
"""Build the full second-pass review bundle from the refreshed suite artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="output/isaac_runs")
    parser.add_argument("--report-tex-dir", default="report_tex")
    parser.add_argument("--lock-path", default="docs/second_pass_draft_lock.json")
    parser.add_argument("--manifest-path", default="docs/second_pass_review_manifest.json")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--max-frames", type=int, default=120)
    return parser.parse_args()


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_json_command(args: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        args,
        cwd=str(REPO_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _current_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def build_review_bundle(
    *,
    output_root: Path,
    report_tex_dir: Path,
    lock_path: Path,
    manifest_path: Path,
    fps: int,
    max_frames: int,
) -> dict[str, Any]:
    publication_payload = _run_json_command(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_isaac_second_pass_publication.py"),
            "--output-root",
            str(output_root),
            "--report-tex-dir",
            str(report_tex_dir),
            "--lock-path",
            str(lock_path),
        ]
    )
    media_payload = _run_json_command(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "render_isaac_media_bundle.py"),
            "--output-root",
            str(output_root),
            "--lock-path",
            str(lock_path),
            "--fps",
            str(int(fps)),
            "--max-frames",
            str(int(max_frames)),
        ]
    )
    dashboard_payload = _run_json_command(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_isaac_presentation_dashboard.py"),
            "--output-root",
            str(output_root),
        ]
    )

    lock_payload = _load_json(lock_path)
    checkpoint = lock_payload.setdefault("checkpoint", {})
    draft_selection = lock_payload.setdefault("draft_selection", {})
    checkpoint["current_work_packet_commit"] = _current_head()
    checkpoint["draft_selection_status"] = "review_bundle_ready"
    draft_selection["draft_ready"] = True
    draft_selection["suite_artifacts_stale"] = False
    draft_selection["media_artifacts_stale"] = False
    draft_selection["draft_suite_status"] = "review_bundle_ready"
    draft_selection["draft_suite_status_reason"] = (
        "The refreshed second-pass suite, PDF build, dashboard, and local presentation bundle "
        "have all been regenerated from the promoted fused draft lock and are ready for review."
    )
    draft_selection["publication_build_summary_path"] = _repo_relative(
        Path(publication_payload["publication_build_summary"])
    )
    draft_selection["publication_pdf_path"] = _repo_relative(Path(publication_payload["pdf_path"]))
    draft_selection["presentation_manifest_path"] = _repo_relative(
        Path(media_payload["suite_summary_json"]).resolve().parent.parent / "presentation" / "presentation_manifest.json"
    )
    tuning_summary_path = draft_selection.get("tuning_summary_path")
    if tuning_summary_path:
        draft_selection["tuning_summary_path"] = _repo_relative(Path(str(tuning_summary_path)))
    draft_selection["review_manifest_path"] = _repo_relative(manifest_path)
    draft_selection["draft_readiness_note"] = (
        "The second-pass review bundle is now ready from the refreshed suite artifacts: the promoted "
        "fused draft lock, the rebuilt paper PDF, the regenerated dashboard/media bundle, and the "
        "review manifest all point to the same stabilization result."
    )
    _write_json(lock_path, lock_payload)

    manifest = {
        "packet_head": checkpoint["current_work_packet_commit"],
        "artifact_policy": (
            "Second-pass review outputs under output/isaac_runs remain local/generated artifacts. "
            "Regenerate them after cloning with scripts/build_isaac_second_pass_review_bundle.py."
        ),
        "lock_path": _repo_relative(lock_path),
        "review_manifest_path": _repo_relative(manifest_path),
        "draft_pdf_path": _repo_relative(Path(publication_payload["pdf_path"])),
        "publication_build_summary_path": _repo_relative(Path(publication_payload["publication_build_summary"])),
        "presentation_manifest_path": draft_selection["presentation_manifest_path"],
        "dashboard_path": _repo_relative(Path(dashboard_payload["dashboard_path"])),
        "suite_summary_json": _repo_relative(Path(publication_payload["suite_summary_json"])),
        "representative_run_ids": dict(draft_selection.get("representative_run_ids", {})),
        "recommended_start_points": [
            "docs/isaac_second_pass_handoff.md",
            "docs/isaac_second_pass_key_findings.md",
            "report_tex/second_pass_publication_draft.tex",
            "report_tex/second_pass_publication_draft.pdf",
            "output/isaac_runs/latest_second_pass_suite/presentation/index.html",
        ],
        "commands": [
            "source .venv/bin/activate",
            "python scripts/build_isaac_second_pass_review_bundle.py",
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    args = parse_args()
    payload = build_review_bundle(
        output_root=(REPO_ROOT / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root).resolve(),
        report_tex_dir=(REPO_ROOT / args.report_tex_dir).resolve() if not Path(args.report_tex_dir).is_absolute() else Path(args.report_tex_dir).resolve(),
        lock_path=(REPO_ROOT / args.lock_path).resolve() if not Path(args.lock_path).is_absolute() else Path(args.lock_path).resolve(),
        manifest_path=(REPO_ROOT / args.manifest_path).resolve() if not Path(args.manifest_path).is_absolute() else Path(args.manifest_path).resolve(),
        fps=int(args.fps),
        max_frames=int(args.max_frames),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
