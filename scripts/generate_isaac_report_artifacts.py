#!/usr/bin/env python3
"""Generate report metrics, tables, and figures for one Isaac run."""

from __future__ import annotations

import argparse
import json

from calib_sim.reporting import generate_isaac_report_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Isaac run directory under output/isaac_runs/<run_id>")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = generate_isaac_report_artifacts(args.run_dir)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
