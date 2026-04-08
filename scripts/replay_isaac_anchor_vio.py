#!/usr/bin/env python3
"""Replay one Isaac run and regenerate report artifacts."""

from __future__ import annotations

import argparse
import json
import sys

from calib_sim.reporting import generate_isaac_report_artifacts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Regenerate artifacts for a partial run without enforcing completeness thresholds.",
    )
    args = parser.parse_args()
    try:
        payload = generate_isaac_report_artifacts(args.run_dir, allow_incomplete=bool(args.allow_incomplete))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
