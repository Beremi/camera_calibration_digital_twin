# Pose Estimation Investigation

Historical note:

- This document records the investigation phase that led to the current checkpointed estimator behavior.
- The current checkpoint report is [checkpoint_01.md](checkpoint_01.md).
- The values in this note are still useful for debugging history, but they are not the current headline metrics.

This note explains why the single-tag offline pose estimate was much worse than
expected, what was fixed, and what still remains.

## Short Answer

The bad world-space error came from multiple issues stacked together:

1. The simulator was writing an improper reflected camera frame instead of a
   true rotation.
2. The optimizer was trying to use an OpenCV PnP seed that cannot be converted
   directly into the current y-up Rodrigues state without introducing a
   reflection.
3. The remaining image measurements are still noisy relative to the tag size in
   the image.
4. Even with perfect synthetic points, the current obscura-seeded Newton solve
   is not robust enough to reliably reach the true pose.

So the answer to "is the minimization problem properly set up?" is:

- `No` for the original version.
- `Better now`, but still not fully reliable.

## Pattern Size Audit

The tag size information is internally consistent in the current sim:

- config tag size: `0.10 m`
- ground-truth exported tag size: `0.10 m`
- camera-pose-export half extent: `0.05 m`
- optimizer object points: `[-0.05, +0.05]` square corners around the tag origin

Relevant files:

- `config/interactive/browser_game_demo.yaml`
- `config/camera/pixel_9a_main.toml`
- `src/calib_sim/interactive/sim.py`
- `src/calib_sim/interactive/analysis.py`

So the large pose error was not caused by a wrong tag size.

## Forward Model Audit

I also checked the exact model used inside the minimizer itself:

- state: tag-to-camera translation plus Rodrigues rotation
- object geometry: 10 cm square tag with the same 5-point ordering used in the
  analysis
- projection: pinhole image-plane coordinates in meters

The key question was:

- If I feed the recorded ground-truth pose into the minimizer's forward model,
  does it reproduce the simulator's exact synthetic measurement?

For the latest corrected run (`output/interactive_runs/run_20260404_103727`),
the answer is now `yes`.

Across all `114` solved tag observations:

- mean RMSE between minimizer prediction and stored synthetic ground-truth
  measurement: `7.63e-11 m` on the image plane
- max RMSE between minimizer prediction and stored synthetic ground-truth
  measurement: `1.19e-10 m`
- mean equivalent pixel error: about `1.09e-5 px`

So the exact forward model is now self-consistent with the simulator truth to
numerical precision.

For comparison, when the same true pose is compared against the detected
measurements instead of the stored synthetic truth:

- mean RMSE: `2.25e-5 m`
- mean equivalent pixel error: `3.2161 px`
- max equivalent pixel error: `4.3460 px`

That is an important separation:

- the forward projection model itself is now correct
- the remaining mismatch comes from the measurement side and from the optimizer
  finding a different low-reprojection pose, not from the exact camera operator
  failing to reproduce the simulator's own geometry

## Root Cause 1: Reflected Camera Frame

The old `_look_at_rotation()` implementation built a left-handed frame:

- stored camera rotation determinant: `-1`
- stored `camera_rotation_tc` determinant: `-1`

That means the "ground truth rotation" was not a valid rotation matrix and
could not be represented by the optimizer's Rodrigues state.

This has now been fixed by:

- making `_look_at_rotation()` right-handed
- removing the compensating horizontal flip from the rendered tag texture

After that fix:

- camera rotation determinant became `+1`
- tag-relative ground truth pose could be reprojected consistently again

## Root Cause 2: Invalid PnP Seed Conversion

OpenCV `solvePnP` works in the standard computer-vision camera frame:

- `x` right
- `y` down
- `z` forward

The current optimizer state uses:

- `x` right
- `y` up
- `z` forward

The map between those two camera frames is a reflection, not a proper
rotation. That means a raw OpenCV pose cannot be dropped into a Rodrigues
rotation state directly.

This showed up clearly in the fresh corrected run:

- representative `opencv_pnp` seed position error: `2.6341 m`
- representative `opencv_pnp` seed reprojection error: `22.3043 px`
- representative obscura seed position error: `0.9632 m`
- representative obscura seed reprojection error: `2.1796 px`

So the PnP seed was not helping. It was pulling the optimizer into a lower-loss
but physically worse basin. The current analysis now disables that seed.

## Root Cause 3: The Observation Is Noisy

For the latest corrected run:

- run: `output/interactive_runs/run_20260404_103727`
- mean detector corner error vs synthetic ground truth: `5.0483 px`
- median detector corner error vs synthetic ground truth: `5.0733 px`
- mean observed tag edge length: `32.7380 px`
- median observed tag edge length: `31.7212 px`
- mean reprojection error of the true pose against detected measurements:
  `3.2155 px`

That is a hard regime:

- the tag is only about `32 px` wide in the image
- corner noise is about `5 px`

So the true pose is already several pixels away from the detected corners.
Because the optimizer only sees image measurements, it can prefer a different
3D pose that explains those noisy corners better.

## Root Cause 4: The Current Minimizer Still Misses the True Pose

This is the most important remaining finding.

I tested the current optimizer with:

- the true synthetic image-plane points
- the true tag size
- the corrected right-handed ground truth
- only the obscura seed

Expected result:

- the optimizer should return the ground-truth pose or something extremely
  close to it

Actual result on the representative observation:

- ground-truth position: `[0.188049, 0.401244, 1.166516] m`
- optimized position from perfect points: `[-0.197009, 0.077047, 1.260550] m`
- position error even with perfect points: `0.5121 m`

That means the remaining problem is not only detector noise.

It also means the current minimization setup is still not robust enough:

- single obscura seed
- planar tag
- Rodrigues state near a `pi`-rotation initialization
- no stronger global strategy or alternative parameterization

## What Improved

Three stages are now available as concrete evidence:

1. Original broken pipeline:
   - run: `output/interactive_runs/run_20260404_070618`
   - mean position error: `3.6647 m`
   - mean reprojection error: `2.8211 px`

2. After fixing the reflected camera frame:
   - run: `output/interactive_runs/run_20260404_103335`
   - mean position error: `1.2755 m`
   - mean reprojection error: `0.8402 px`

3. After also removing the invalid PnP seed and keeping the corrected geometry:
   - run: `output/interactive_runs/run_20260404_103727`
   - mean position error: `0.8597 m`
   - mean reprojection error: `1.1372 px`

So the investigation already found and fixed real bugs. The pose estimate is
still not good enough, but it is much less wrong than before.

## Current Best Interpretation

The current remaining error is mainly caused by:

1. A single small planar tag gives a weak image-only pose signal.
2. The detected corners are noisy relative to the tag footprint.
3. The current Newton formulation is still too local and not seeded in a way
   that reliably reaches the true branch.

## Recommended Next Steps

The highest-value next fixes are:

1. Rewrite the estimator in one consistent standard camera convention
   (`x` right, `y` down, `z` forward) so OpenCV PnP/IPPE seeds are valid.
2. Use a proper multi-start strategy:
   - homography/IPPE branch A
   - homography/IPPE branch B
   - obscura seed
3. Fit only the four real corner observations, or keep the 5th point only if
   it is derived projectively in a fully consistent way.
4. Use both visible tags together in one frame instead of solving each tag
   independently.
5. Add a multi-frame bundle adjustment using the recorded motion stream as a
   temporal prior.

Until those are done, the current pipeline should be treated as:

- useful for debugging and comparing formulations
- good enough to expose pose-model problems
- not yet a trusted absolute-pose estimator
