# Isaac First-Pass Results

This file curates the canonical run IDs and regeneration commands for the first scientifically complete Isaac pass on `feature/isaac-runtime-first-pass`.

## Publication Inputs

The publication report prefers the suite-level artifact bundle:

- `output/isaac_runs/latest_first_pass_suite`

The frozen suite definition is locked in:

- `docs/first_pass_suite_lock.json`

The per-run fallback remains:

- `output/isaac_runs/latest_complete`

## Canonical Matrix Runs

These four runs form the paper's estimator/controller 2x2 matrix at the canonical seed `7`:

- `output/isaac_runs/first_pass_visual_open-loop_servo_nominal_seed_007`
- `output/isaac_runs/first_pass_visual_closed-loop_servo_nominal_seed_007`
- `output/isaac_runs/first_pass_fused_open-loop_servo_nominal_seed_007`
- `output/isaac_runs/first_pass_fused_closed-loop_servo_nominal_seed_007`

The canonical headline run is:

- `output/isaac_runs/first_pass_fused_closed-loop_servo_nominal_seed_007`

## Reproducibility Runs

Closed-loop visual reproducibility seeds:

- `output/isaac_runs/first_pass_visual_closed-loop_servo_nominal_seed_011`
- `output/isaac_runs/first_pass_visual_closed-loop_servo_nominal_seed_017`
- `output/isaac_runs/first_pass_visual_closed-loop_servo_nominal_seed_023`
- `output/isaac_runs/first_pass_visual_closed-loop_servo_nominal_seed_031`
- `output/isaac_runs/first_pass_visual_closed-loop_servo_nominal_seed_047`

Closed-loop fused reproducibility seeds:

- `output/isaac_runs/first_pass_fused_closed-loop_servo_nominal_seed_011`
- `output/isaac_runs/first_pass_fused_closed-loop_servo_nominal_seed_017`
- `output/isaac_runs/first_pass_fused_closed-loop_servo_nominal_seed_023`
- `output/isaac_runs/first_pass_fused_closed-loop_servo_nominal_seed_031`
- `output/isaac_runs/first_pass_fused_closed-loop_servo_nominal_seed_047`

## Actuation Comparison

The minimum first-pass actuation comparison uses the fused closed-loop controller at seed `7`:

- `output/isaac_runs/first_pass_fused_closed-loop_none_seed_007`
- `output/isaac_runs/first_pass_fused_closed-loop_servo_nominal_seed_007`

## Historical And Debug-Only Runs

Runs whose purpose is diagnosis rather than publication should not be used as headline paper inputs:

- `output/isaac_runs/first_pass_probe_control_fix`
- `output/isaac_runs/first_pass_probe_control_fix2`
- `output/isaac_runs/first_pass_probe_control_fix2_8s`
- `output/isaac_runs/first_real_pass_nominal_v2`
- older exploratory runs already listed in `docs/isaac_first_pass_expert_handoff.md`

## Regeneration Commands

Run the full first-pass suite:

```bash
source .venv-isaac/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
python scripts/run_isaac_ablation_suite.py --headless
```

Verify that the frozen suite still matches the lock:

```bash
source .venv/bin/activate
python scripts/verify_isaac_first_pass_suite.py
```

Regenerate the per-run report bundle for the canonical headline run:

```bash
source .venv/bin/activate
python scripts/generate_isaac_report_artifacts.py output/isaac_runs/first_pass_fused_closed-loop_servo_nominal_seed_007
```

Replay the canonical run for analysis/report regeneration only:

```bash
source .venv/bin/activate
python scripts/replay_isaac_anchor_vio.py output/isaac_runs/first_pass_fused_closed-loop_servo_nominal_seed_007
```

Build the frozen first-pass publication bundle:

```bash
source .venv/bin/activate
python scripts/build_isaac_first_pass_publication.py
```
