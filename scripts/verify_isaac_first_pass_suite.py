#!/usr/bin/env python3
"""Verify that the frozen first-pass suite still matches the authoritative lock."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK_PATH = REPO_ROOT / "docs" / "first_pass_suite_lock.json"
DEFAULT_SUITE_DIR = REPO_ROOT / "output" / "isaac_runs" / "latest_first_pass_suite"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def verify_first_pass_suite(lock_path: Path = DEFAULT_LOCK_PATH, suite_dir: Path = DEFAULT_SUITE_DIR) -> dict[str, Any]:
    lock_payload = _load_json(lock_path)
    suite_summary_path = suite_dir / "analysis" / "suite_summary.json"
    if not suite_summary_path.exists():
        raise FileNotFoundError(f"Missing suite summary: {suite_summary_path}")
    suite_summary = _load_json(suite_summary_path)
    paper_artifacts_path = suite_dir / "analysis" / "report_data" / "paper_artifacts.tex"

    errors: list[str] = []
    canonical_run_id = str(suite_summary.get("canonical_run_id", ""))
    if canonical_run_id != str(lock_payload["canonical_run_id"]):
        errors.append(
            f"canonical_run_id mismatch: expected {lock_payload['canonical_run_id']}, got {canonical_run_id or 'missing'}"
        )

    matrix_run_ids = sorted(str(row.get("run_id", "")) for row in suite_summary.get("matrix_runs", []))
    expected_matrix_run_ids = sorted(str(run_id) for run_id in lock_payload.get("expected_matrix_run_ids", []))
    if matrix_run_ids != expected_matrix_run_ids:
        errors.append(f"matrix run IDs mismatch: expected {expected_matrix_run_ids}, got {matrix_run_ids}")

    actuation_run_ids = sorted(str(row.get("run_id", "")) for row in suite_summary.get("actuation", []))
    expected_actuation_run_ids = sorted(str(run_id) for run_id in lock_payload.get("expected_actuation_run_ids", []))
    if actuation_run_ids != expected_actuation_run_ids:
        errors.append(f"actuation run IDs mismatch: expected {expected_actuation_run_ids}, got {actuation_run_ids}")

    reproducibility_rows = {
        f"{row.get('estimator_mode', '')}_{str(row.get('controller_mode', '')).replace('-', '_')}": row
        for row in suite_summary.get("reproducibility", [])
    }
    for key, expected in lock_payload.get("expected_reproducibility", {}).items():
        row = reproducibility_rows.get(key)
        if row is None:
            errors.append(f"missing reproducibility row for {key}")
            continue
        actual_seeds = [int(seed) for seed in row.get("seeds", [])]
        expected_seeds = [int(seed) for seed in expected.get("seeds", [])]
        if actual_seeds != expected_seeds:
            errors.append(f"{key} seeds mismatch: expected {expected_seeds}, got {actual_seeds}")
        actual_run_ids = sorted(str(run_id) for run_id in row.get("run_ids", []))
        expected_run_ids = sorted(str(run_id) for run_id in expected.get("run_ids", []))
        if actual_run_ids != expected_run_ids:
            errors.append(f"{key} run IDs mismatch: expected {expected_run_ids}, got {actual_run_ids}")

    publication_summary = suite_summary.get("publication", {})
    artifact_source = str(publication_summary.get("artifact_source", ""))
    if artifact_source and artifact_source != "latest_first_pass_suite":
        errors.append(f"publication artifact source mismatch: expected latest_first_pass_suite, got {artifact_source}")
    if not paper_artifacts_path.exists():
        errors.append(f"missing paper artifacts file: {paper_artifacts_path}")

    return {
        "ok": not errors,
        "canonical_run_id": canonical_run_id,
        "lock_path": str(lock_path),
        "suite_dir": str(suite_dir),
        "suite_summary_path": str(suite_summary_path),
        "paper_artifacts_tex": str(paper_artifacts_path),
        "artifact_source": artifact_source or "latest_first_pass_suite",
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock-path", default=str(DEFAULT_LOCK_PATH))
    parser.add_argument("--suite-dir", default=str(DEFAULT_SUITE_DIR))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = verify_first_pass_suite(Path(args.lock_path).resolve(), Path(args.suite_dir).resolve())
    except Exception as exc:  # pragma: no cover - exercised via CLI error path
        payload = {
            "ok": False,
            "lock_path": str(Path(args.lock_path).resolve()),
            "suite_dir": str(Path(args.suite_dir).resolve()),
            "errors": [str(exc)],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
