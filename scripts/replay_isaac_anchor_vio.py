#!/usr/bin/env python3
"""Replay one Isaac run and regenerate report artifacts."""

from __future__ import annotations

import argparse
import json

from calib_sim.reporting import generate_isaac_report_artifacts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args()
    payload = generate_isaac_report_artifacts(args.run_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
