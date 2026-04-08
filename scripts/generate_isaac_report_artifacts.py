#!/usr/bin/env python3
"""Generate report metrics, tables, and figures for one Isaac run."""

from __future__ import annotations

import argparse
import json
import sys

from calib_sim.reporting import generate_isaac_report_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Isaac run directory under output/isaac_runs/<run_id>")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Generate artifacts even when the run does not meet completeness thresholds.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        payload = generate_isaac_report_artifacts(args.run_dir, allow_incomplete=bool(args.allow_incomplete))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
