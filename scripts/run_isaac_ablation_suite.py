#!/usr/bin/env python3
"""Plan or execute the Isaac ablation suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise-presets", nargs="+", default=["ideal", "nominal", "stress"])
    parser.add_argument("--actuation-modes", nargs="+", default=["none", "lag_only", "lag_deadband", "lag_deadband_backlash_delay", "jitter_drop"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 17, 23, 31, 47])
    parser.add_argument("--output-plan", default=None)
    args = parser.parse_args()

    plan = []
    for noise_preset in args.noise_presets:
        for actuation_mode in args.actuation_modes:
            for seed in args.seeds:
                run_id = f"{noise_preset}_{actuation_mode}_seed_{seed:03d}"
                plan.append(
                    {
                        "run_id": run_id,
                        "noise_preset": noise_preset,
                        "actuation_mode": actuation_mode,
                        "seed": int(seed),
                    }
                )
    payload = {"planned_runs": plan, "count": len(plan)}
    if args.output_plan:
        Path(args.output_plan).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
