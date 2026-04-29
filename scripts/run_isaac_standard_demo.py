#!/usr/bin/env python3
"""Launch the tabletop-only standard Isaac demo service."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8013)
    parser.add_argument("--log-level", default="info")
    return parser.parse_args()


def main() -> int:
    import uvicorn

    args = parse_args()
    uvicorn.run(
        "calib_sim.isaac.standard_demo_service:app",
        host=str(args.host),
        port=int(args.port),
        log_level=str(args.log_level),
        reload=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
