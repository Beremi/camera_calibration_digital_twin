#!/usr/bin/env python3
"""Generate suppression-window summaries for one dropout-debug run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from calib_sim.reporting.isaac_suppression_windows import generate_suppression_window_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = generate_suppression_window_artifacts(Path(args.run_dir))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
