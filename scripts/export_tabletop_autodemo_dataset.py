#!/usr/bin/env python3
"""Export the tabletop auto-demo dataset bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from calib_sim.reporting.tabletop_autodemo_dataset import DEFAULT_DATASET_ROOT, export_tabletop_autodemo_dataset


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--capture-only",
        action="store_true",
        help="Capture and finalize only the clean master run at --output-root, then exit.",
    )
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=None,
        help="Optional existing clean source run for debugging; if omitted the script captures a fresh auto_demo run.",
    )
    parser.add_argument(
        "--full-analysis",
        action="store_true",
        help="Opt back into the heavier localization/pose-analysis export. By default the script exports image corners only.",
    )
    parser.add_argument(
        "--skip-phone-export",
        action="store_true",
        help="Preserve the older two-variant clean/noisy export without phone_clean or phone_capture mirrors.",
    )
    parser.add_argument(
        "--phone-profile",
        type=Path,
        default=Path("config/camera/pixel_9a_main.toml"),
        help="Phone camera TOML profile used for phone_clean/noisy post-processing and phone_capture sidecars.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if bool(args.capture_only) and args.source_run_dir is not None:
        raise SystemExit("--capture-only cannot be combined with --source-run-dir.")
    payload = export_tabletop_autodemo_dataset(
        output_root=args.output_root,
        profile_key="tabletop_replica",
        seed=int(args.seed),
        source_run_dir=args.source_run_dir,
        capture_only=bool(args.capture_only),
        corners_only=not bool(args.full_analysis),
        skip_phone_export=bool(args.skip_phone_export),
        phone_profile_path=args.phone_profile,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
