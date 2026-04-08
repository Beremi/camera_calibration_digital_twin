# Next Workplan: Bayesian Visual–Inertial Tag-Map Estimation and Robot Calibration

This document is the recommended next workplan for the current checkpoint described in `docs/checkpoint_01.md` and `output/interactive_runs/run_20260406_083611`.

It is written to be directly actionable for implementation in this repository.

---

## 0. Executive Decision

### Recommended next estimator
Do **not** jump directly from the current per-frame camera-only solver to a full "everything at once" Bayesian sampler.

The best next step is:

1. **Build a batch MAP factor graph / smoother**
   - unknown camera/IMU trajectory over time
   - unknown **static tag poses** in the world
   - IMU biases over time
   - optional camera–IMU extrinsic as a later stage
   - optional robot kinematic correction as a later stage

2. **Use the batch MAP result as the center of a Laplace posterior**
   - extract marginal covariances / confidence ellipsoids
   - validate posterior calibration by Monte Carlo coverage

3. **Only after that** consider full sampling over a reduced parameter block
   - e.g. robot joint offsets, camera–IMU extrinsic, time offset
   - while conditioning on a good linearization point

### Why I am changing your proposed formulation slightly

Your proposed idea is close, but I recommend three changes:

#### Change A — Estimate one rigid pose per tag/pattern, not free 3D points for all corners
You wrote:

- the calibration patterns location will be unknown
- we build the location of everything, static positions of patterns and the movement of camera

That is correct in spirit, but the best state is **not**
"one free 3D position per observed corner."

Instead, use:

- one unknown rigid transform `T_WT_j` per tag / pattern `j`
- fixed known local geometry for that tag, e.g. four corners in its local frame

This keeps the "pattern location unknown" property you want, but:
- enforces rigidity automatically
- reduces state dimension a lot
- improves identifiability
- makes the graph cleaner
- avoids shape drift in noisy data

If later a *board* contains several tags with known relative layout, estimate one board pose and keep all tag offsets fixed inside that board.

#### Change B — Use 4 corner reprojection residuals in **pixel space**, not the current 5-point metric-image-plane residual
Your current solver uses:
- center
- 4 corners
- image-plane metric coordinates

For the new batch estimator, I recommend:
- use the **4 tag corners only**
- optimize in **pixel coordinates**
- keep the camera projection model explicit in the factor

Why:
- the tag center is not an independent measurement once corners are known
- using center as an independent residual double-counts information unless you model correlated covariance
- detector noise is easiest to specify and tune in pixels
- robust losses and gating thresholds are also more natural in pixels

#### Change C — Use batch MAP + Laplace first, not full MCMC first
A full Bayesian sampler over:
- all camera poses
- all tag poses
- IMU biases
- extrinsics
- optional robot parameters

is expensive and hard to initialize.

For this problem, the practical sequence is:
- solve sparse nonlinear MAP
- compute Hessian / marginal covariances
- validate uncertainty calibration
- only then, if needed, sample a reduced block

This is still a perfectly valid Bayesian inverse workflow for a large robotics problem.

---

## 1. What the next estimator should estimate

### 1.1 Stage V1 (recommended immediate target)
Estimate:

- camera / IMU trajectory over time
- static tag poses
- IMU biases
- optionally velocity states

Fix:
- camera intrinsics
- distortion parameters
- camera–IMU extrinsic
- robot parameters

This stage answers:

> Can the repo recover a static tag map and a smooth fused trajectory from unknown map geometry + visual tag observations + IMU?

This is the best immediate jump from the current checkpoint.

### 1.2 Stage V2
Add estimation of:

- camera–IMU extrinsic `T_IC`
- global camera–IMU time offset `delta_t`

Still fix:
- robot geometry
- camera intrinsics

This stage answers:

> Can the system self-calibrate the phone rig in the synthetic benchmark?

### 1.3 Stage V3
Add:

- robot parameter corrections `theta_robot`
  - start with joint zero offsets only
  - then maybe selected link-length or mounting parameters
- hand/phone transform if modeled through the robot chain

This stage answers:

> Can the system calibrate the robot / phone rig jointly using all modalities?

### 1.4 Stage V4 (paper-grade extension)
Add a realism-complete sim and later real-data transfer:
- rolling shutter
- asynchronous sensor clocks
- visual outliers / missed detections
- IMU scale / axis misalignment
- motion blur
- timestamp jitter
- real Android metadata path

---

## 2. State variables

Use a world frame `W`.

### 2.1 Trajectory states
At each camera frame or keyframe `k`, estimate:

- `R_WI_k` : IMU orientation
- `p_WI_k` : IMU position
- `v_W_k` : IMU/world velocity
- `b_g_k` : gyro bias
- `b_a_k` : accel bias

Bundle these into:

```text
x_k = {R_WI_k, p_WI_k, v_W_k, b_g_k, b_a_k}
```

### 2.2 Static tag map
For each tag `j`, estimate one rigid pose:

```text
l_j = T_WT_j in SE(3)
```

In the tag frame `T_j`, keep the 4 corners fixed from tag size `s_j`:

```text
c_1 = [-s_j/2, -s_j/2, 0]
c_2 = [ s_j/2, -s_j/2, 0]
c_3 = [ s_j/2,  s_j/2, 0]
c_4 = [-s_j/2,  s_j/2, 0]
```

If you keep the tag center for plotting, compute it from the pose; do not optimize on it as an independent observation.

### 2.3 Rig parameters
Later stages may include:

- `T_IC` : IMU-to-camera extrinsic
- `delta_t` : camera–IMU time offset
- `kappa` : selected intrinsics/distortion nuisance parameters
- `theta_robot` : robot correction parameters
  - first: joint zero offsets
  - later: selected link lengths, base pose, mount transform

---

## 3. Gauge freedoms and identifiability

If both the camera trajectory and the tag map are unknown, the problem has gauge freedom.

### 3.1 In V1 (no robot factor)
Fix the gauge by imposing strong priors on the first state:

- `R_WI_0 = I`
- `p_WI_0 = 0`
- optionally `yaw_0 = 0`
- weak prior on `v_0 = 0` if initialization supports it

This defines the world frame.

### 3.2 In V3 (robot factor active)
Use the robot base frame as world:
- `W = B` (robot base)
- then tag poses are learned directly in robot/world coordinates

### 3.3 Important practical note
Do **not** estimate all of the following simultaneously at first:
- unknown tag map
- full trajectory
- IMU biases
- camera–IMU extrinsic
- time offset
- rolling shutter
- full robot DH/POE corrections
- camera intrinsics

That is too ambitious for the current motion and simulator.

The correct order is:
1. unknown map + trajectory + biases
2. add extrinsic / time offset
3. add selected robot corrections
4. add richer nuisance parameters

---

## 4. Measurement model

## 4.1 Visual tag-corner factor

For frame `k`, tag `j`, corner `m`, let the observed pixel be:

```text
z_kjm in R^2
```

Predicted pixel:

```text
u_hat_kjm = pi(kappa, T_CI * T_IW_k * T_WT_j * c_m)
```

where:
- `pi` is the full camera projection function
- `c_m` is the known tag corner in the tag-local frame
- `T_IW_k = inverse(T_WI_k)`

Residual:

```text
r_vis_kjm = z_kjm - u_hat_kjm
```

Use a robust penalty:
- Huber first
- later Cauchy or Student-t if you inject larger outliers

Recommended covariance:
- start with `Sigma_vis = sigma_px^2 I`
- later make `sigma_px` depend on tag size in the image, incidence angle, and blur score

## 4.2 IMU factor

For accelerometer and gyroscope samples between states `k` and `k+1`, use IMU preintegration.

Continuous-time measurement model:

```text
omega_tilde(t) = omega(t) + b_g(t) + n_g(t)
a_tilde(t)     = R_IW(t) * (a_W(t) - g) + b_a(t) + n_a(t)
```

Bias evolution:

```text
dot(b_g) = n_wg
dot(b_a) = n_wa
```

In discrete form:
- use preintegrated delta rotation
- preintegrated delta velocity
- preintegrated delta position
- bias Jacobians

If you stay custom/JAX, implement standard Forster-style IMU preintegration logic.
If you want fastest robust implementation, use GTSAM's IMU factor stack.

## 4.3 Optional robot factor (V3)
If robot joint states `q_k` are known, define:

```text
T_WI_robot(q_k, theta_robot) = T_WB * FK(q_k; theta_robot) * T_EI
```

Residual:

```text
r_robot_k = Log( inverse(T_WI_k) * T_WI_robot(q_k, theta_robot) )
```

Use:
- tight covariance in pure simulation
- looser covariance once encoder noise / compliance / mount flex are simulated

---

## 5. MAP objective

The batch MAP problem is

```text
arg min_X  J(X)
```

with

```text
J(X) =
sum visual_factors rho_vis( r_vis^T Sigma_vis^-1 r_vis )
+ sum imu_factors           r_imu^T Sigma_imu^-1 r_imu
+ sum bias_walk_factors     r_bias^T Sigma_bias^-1 r_bias
+ sum prior_factors         r_prior^T Sigma_prior^-1 r_prior
+ sum robot_factors         r_robot^T Sigma_robot^-1 r_robot   (V3+)
```

where `X` contains all unknown states.

### Why this is the right first Bayesian estimator
This gives you:
- a principled likelihood
- a principled prior
- a sparse graph
- a tractable batch solver
- posterior covariance via the linearized Hessian

After convergence, approximate the posterior as:

```text
p(X | y) approx N(X_MAP, H^-1)
```

where `H` is the Gauss-Newton / LM Hessian at the optimum.

That is the right posterior object to validate first.

---

## 6. Recommended solver stack

## 6.1 Recommended implementation choice: GTSAM first
Even though the current repo already uses JAX/Newton for per-frame solves, I recommend:

### Use GTSAM for the first full fused estimator
Reasons:
- factor graphs are its native abstraction
- it already supports Lie-group state variables
- it already supports IMU preintegration
- it already supports batch optimization and marginals
- Python bindings are available
- this reduces research risk dramatically

### Keep the current JAX solver path
Use the existing JAX projection code and per-frame solver as:
- a validation oracle for visual factors
- unit tests for reprojection consistency
- synthetic sanity checks

## 6.2 Acceptable alternative: custom JAX factor graph
If you strongly want a single-language stack:
- keep everything in Python/JAX
- implement sparse block Jacobians
- use LM rather than raw Newton
- add Schur support later if needed

This is feasible, but it is more engineering work and more failure-prone.

### Recommendation
Use:
- **GTSAM for the fused estimator**
- **current JAX code for residual validation and synthetic tests**

---

## 7. Initialization plan

Good initialization matters more than the exact solver.

## 7.1 V1 initialization (unknown map + trajectory)
Use the current tags and detections to bootstrap:

### Step A — choose anchor frame
Select the frame with:
- the largest number of visible tags
- good tag spread in the image
- low obliqueness if possible

Set:
- `T_WC_anchor = I`
- then derive `T_WI_anchor` from fixed `T_IC`

### Step B — initialize tag poses
For each visible tag in the anchor frame:
- solve tag pose from known tag size + observed corners
- store `T_WT_j`

### Step C — initialize neighboring camera poses
For the next frames:
- if at least one initialized tag is visible, solve camera pose from those tags
- use PnP / direct tag reprojection fit
- propagate forward and backward

### Step D — initialize unseen tags
Whenever a frame pose is known and a new tag appears:
- initialize that tag pose from the current frame

### Step E — initialize velocities
Use:
- finite differences of consecutive positions for V1 in sim
- or robot/ground-truth only for debugging, not for final inference

### Step F — initialize biases
Start with:
- `b_g = 0`
- `b_a = 0`

### Step G — run visual-only bundle first
Before adding IMU:
- optimize camera poses + tag poses
- verify residuals and map consistency

### Step H — add IMU and rerun
Then:
- insert velocities and bias states
- add preintegrated IMU factors
- optimize again

This two-stage initialization is safer than dropping everything into one solve immediately.

---

## 8. Realistic noise to add to the simulator

Your current checkpoint is too clean:
- IMU is effectively ideal
- frame timing is aligned
- lens distortion is not rendered
- tag measurements are near-perfect except detector effects

The next estimator will look unrealistically good unless the sim adds more realism.

## 8.1 IMU realism (must do first)

### Current limitation
Your preserved checkpoint records IMU once per simulation frame (`12 Hz`) and noise-free.
That is fine for debugging, but not for a calibration paper.

### Required changes
Implement a separate IMU stream:
- independent IMU clock
- rate `>= 58.236 Hz` to match the current device preset
- ideally `100 Hz` if the simulator cost is acceptable

### Noise model
Use the Kalibr-style model:

```text
omega_tilde = omega + b_g + n_g
a_tilde     = a + b_a + n_a
b_g[k+1]    = b_g[k] + n_bg * sqrt(dt)
b_a[k+1]    = b_a[k] + n_ba * sqrt(dt)
```

Include:
- white noise
- bias random walk

### Parameter source
Do **not** guess the final noise numbers from the current YAML alone.

Instead:
1. use your current device preset as a temporary nominal setting
2. record a long stationary phone dataset with your Android app
3. fit Allan deviation parameters
4. convert those to Kalibr-style densities and random walks
5. use those values in sim

### Practical recommendation
Maintain three noise presets:
- `ideal`
- `nominal_phone`
- `stress_phone`

Suggested stress knobs:
- 1x
- 3x
- 10x nominal noise

This is especially important because low-cost MEMS models derived from static data are usually optimistic.

## 8.2 Visual realism (next after IMU)

### Minimal realism first
Before photorealism, add measurement-layer realism:

#### Corner noise
Sample pixel noise per detected corner:
- clean: `sigma = 0.2 px`
- nominal: `sigma = 0.5 px`
- hard: `sigma = 1.0 px`

#### Visibility failures
Randomly inject:
- missed detections
- dropped tags
- partial tag visibility
- blur-dependent corner degradation

#### Tag-dependent covariance
Increase visual noise when:
- tag is far
- tag occupies few pixels
- tag is highly oblique
- motion magnitude is large

This is cheap and already useful.

### Later realism
After the measurement-layer noise is working, consider:
- motion blur
- rolling shutter
- exposure changes
- lens distortion in render path
- JPEG/MP4 compression artifacts

Do not block the estimator on full photorealism.

## 8.3 Time realism
Add:
- IMU/camera timestamp offset `delta_t`
- optional small timestamp jitter
- sensor start delay

Start with:
- fixed offset only
Later:
- add random jitter in logging, but do not try to infer per-sample jitter

## 8.4 Robot realism (V3 prep)
If you will later use robot factors, simulate:
- encoder noise
- joint zero offsets
- small mount transform error
- optional compliance/flex

Start with:
- joint zero offsets only

That is enough for a first robot calibration study.

---

## 9. Repository implementation plan

This is the concrete implementation sequence I recommend.

## 9.1 New package structure

```text
src/calib_sim/estimation/
    __init__.py
    dataset.py
    types.py
    init_bootstrap.py
    noise_models.py
    graph_build.py
    solve_batch_map.py
    posterior.py
    eval_metrics.py

src/calib_sim/estimation/factors/
    tag_corner_factor.py
    imu_preintegration.py
    robot_fk_factor.py
    priors.py

src/calib_sim/estimation/backends/
    gtsam_backend.py
    jax_validation.py
```

Optional config additions:

```text
config/noise/
    imu_ideal.yaml
    imu_nominal_phone.yaml
    imu_stress_phone.yaml
    vision_nominal.yaml
    timing_nominal.yaml
```

## 9.2 Data ingestion

### Input sources already present
Use:
- `samples.jsonl` for frame-level detections and camera model snapshot
- `imu.csv` for inertial data
- `metadata.json` for scene / camera / robot config
- `camera_gt.csv` **only for evaluation**, never for inference

### New internal dataset object
Create a clean dataset abstraction:

```python
class BatchCalibrationDataset:
    camera_frames: list[CameraFrame]
    imu_packets: list[ImuPacket]
    detections: list[TagDetection]
    camera_model: CameraModel
    robot_joints: list[RobotJointSample] | None
```

### Important rule
Do not let ground truth leak into the estimator path.
Ground truth should only appear in:
- metrics
- plots
- debugging assertions
- synthetic exact-measurement tests

## 9.3 Factor graph builder
Build a graph with:
- one state per camera frame at first
- one preintegrated IMU factor between consecutive camera states
- one 2D corner factor per visible tag corner
- priors on first state and initial biases

Later add:
- `T_IC`
- `delta_t`
- robot factors

## 9.4 Posterior extraction
After solve:
- compute marginal covariance for
  - selected trajectory nodes
  - tag poses
  - biases
  - extrinsics
  - robot parameters

Store:
- full summary JSON
- marginal diagonal statistics
- 3-sigma intervals
- per-parameter uncertainty plots

---

## 10. Detailed work packages

## WP1 — Clean visual-only unknown-map bundle adjustment
### Goal
Prove that the system can reconstruct:
- static tag map
- full camera trajectory

from tag detections alone, without known global tag poses.

### Tasks
- [ ] Create anchor-frame bootstrap
- [ ] Convert current detections to 4-corner pixel observations
- [ ] Build visual-only graph
- [ ] Solve for all frame poses and all tag poses
- [ ] Compare to current joint-world estimator using GT
- [ ] Plot:
  - trajectory error
  - tag-pose error
  - reprojection RMSE
  - per-tag uncertainty

### Exit criteria
- median trajectory position error remains low under clean synthetic data
- tag map is recovered consistently
- reprojection error matches visual noise level
- no GT is needed for inference

## WP2 — Realistic IMU stream and fused VI batch
### Goal
Fuse unknown-map visual graph with IMU.

### Tasks
- [ ] Decouple IMU from frame cadence
- [ ] Simulate white noise + bias random walk
- [ ] Implement IMU preintegration
- [ ] Add velocity and bias states
- [ ] Add IMU factors to the graph
- [ ] Compare:
  - visual-only
  - IMU-only
  - fused VI

### Exit criteria
- fused estimator is more stable than visual-only under detection dropout
- fused estimator is less drifting than IMU-only under noisy IMU
- estimated biases track injected biases sensibly

## WP3 — Time offset and rig extrinsic
### Goal
Upgrade from "fused trajectory" to "rig calibration."

### Tasks
- [ ] Add fixed scalar `delta_t`
- [ ] Interpolate pose during reprojection / preintegration as needed
- [ ] Add `T_IC` as an estimated variable
- [ ] Use strong prior around known nominal sim transform
- [ ] Run identifiability stress tests

### Exit criteria
- recovered `T_IC` matches injected transform
- recovered `delta_t` matches injected offset
- posterior uncertainty grows correctly when motion is less informative

## WP4 — Robot calibration factor
### Goal
Turn the estimator into robot calibration, not only VI-tag SLAM.

### Tasks
- [ ] Define robot parameter block `theta_robot`
- [ ] Start with joint zero offsets only
- [ ] Add FK-based factor tying trajectory to robot kinematics
- [ ] Use robot base as the world frame
- [ ] Add priors from nominal arm preset
- [ ] Test against injected joint-offset errors

### Exit criteria
- recovered joint offsets match injected offsets
- camera trajectory and tag map remain consistent
- uncertainty contracts when more tag coverage is available

## WP5 — Monte Carlo Bayesian validation
### Goal
Validate that the uncertainty is meaningful.

### Tasks
- [ ] Run many synthetic trials per noise condition
- [ ] Collect coverage of nominal 95% intervals
- [ ] Compute error-vs-uncertainty calibration curves
- [ ] Track solver failures and mode switches
- [ ] Identify overconfident and underconfident regimes

### Exit criteria
- 95% intervals have approximately correct coverage
- stress presets visibly widen posterior intervals
- failure cases are localized to known degeneracies

---

## 11. What *not* to do next

### Do not:
- [ ] estimate every tag corner as a free world landmark
- [ ] keep using the tag center as an independent residual
- [ ] start with MCMC over the full state
- [ ] estimate full robot geometry in the first fused graph
- [ ] keep the IMU at 12 Hz if you want meaningful inertial fusion
- [ ] delay the project waiting for photorealistic rendering
- [ ] use `camera_gt.csv` or exact initial velocity in the main inference path

These would either make the problem harder than needed or produce misleadingly optimistic results.

---

## 12. Motion-plan changes needed for identifiability

Your current motion is smooth and useful, but it is not ideal for the richer calibration stages.

### For V1
Current motion is good enough.

### For V2/V3
Add a more informative motion program with these properties:

- pauses at several poses
- changes in depth to the tabletop tags
- left-right sweeps
- up-down sweeps
- faster angular changes for gyro excitation
- mild acceleration reversals
- segments where near and far tags are both visible

### Suggested 20 s replacement
- 0–2 s: stationary hold
- 2–5 s: slow lateral sweep with wall tags
- 5–8 s: approach tabletop tags
- 8–10 s: hold over tabletop
- 10–13 s: retreat and elevate
- 13–16 s: faster sweep with direction change
- 16–18 s: second hold
- 18–20 s: return

### Why
This will give:
- better visual depth diversity
- better IMU excitation
- easier time-offset estimation
- cleaner observability tests

---

## 13. How to evaluate success

At each stage, report all of these:

### Trajectory metrics
- position RMSE
- rotation RMSE
- final drift
- error over time

### Tag-map metrics
- tag pose translation error
- tag pose rotation error
- per-tag uncertainty

### Measurement metrics
- reprojection RMSE
- robust inlier ratio
- visual outlier rejection count

### IMU metrics
- bias estimation error
- bias uncertainty
- sensitivity to dropout and rate changes

### Bayesian diagnostics
- 95% interval coverage
- marginal standard deviations
- normalized residual histograms
- failure cases / non-convergence count

### Ablations
Always compare:
- visual only
- IMU only
- fused
- known map vs unknown map
- ideal noise vs nominal vs stress

---

## 14. Minimum publishable arc from this repo

The cleanest research arc from the current checkpoint is:

### Paper arc A (strongest immediate path)
**Unknown-tag-map visual–inertial batch calibration in a robot-mounted phone simulator**
- unknown static tag poses
- unknown fused trajectory
- realistic phone-like IMU noise
- posterior uncertainty
- Monte Carlo validation

### Paper arc B (next)
**Joint robot / phone calibration with unknown scene map**
- adds joint-offset calibration
- adds phone rig extrinsic
- uncertainty-aware comparisons

### Paper arc C (later)
**Bridge from fiducials to markerless features**
- replace tags with tracked natural landmarks / reference points
- preserve the same batch back-end
- compare to fiducial-ground-truth baselines

Given the current repository maturity, Arc A is the best immediate target.

---

## 15. First 2 weeks of implementation

## Week 1
- [ ] Create `src/calib_sim/estimation/` package
- [ ] Implement dataset loader from `samples.jsonl`, `imu.csv`, `metadata.json`
- [ ] Convert detections to 4-corner pixel observations
- [ ] Implement anchor bootstrap for unknown tag map
- [ ] Implement visual-only batch graph
- [ ] Produce:
  - trajectory plot
  - tag-map plot
  - reprojection plots

## Week 2
- [ ] Decouple IMU rate from frame cadence in sim
- [ ] Add white noise + bias random walk
- [ ] Implement IMU preintegration path
- [ ] Add fused visual–IMU graph
- [ ] Run ideal / nominal / stress Monte Carlo
- [ ] Produce:
  - fused-vs-visual-only comparison
  - bias estimation plots
  - coverage plots

If Week 2 succeeds, the project is already in a much stronger research state.

---

## 16. Recommended immediate coding order inside this repo

1. `dataset.py`
2. `init_bootstrap.py`
3. `tag_corner_factor.py`
4. `solve_batch_map.py` (visual only)
5. `noise_models.py`
6. `imu_preintegration.py`
7. `solve_batch_map.py` (fused)
8. `posterior.py`
9. `eval_metrics.py`
10. `robot_fk_factor.py`

That order minimizes the chance of getting blocked.

---

## 17. Final recommendation

If I were steering this repo, I would do exactly this:

### Phase 1
Replace the current known-world tag solve with a **visual-only unknown-tag-map bundle adjustment**.

### Phase 2
Add a **realistic asynchronous IMU** and build a **batch visual–inertial smoother**.

### Phase 3
Add **camera–IMU extrinsic and time offset**.

### Phase 4
Add **robot kinematic correction factors**.

### Phase 5
Validate uncertainty by **Laplace posterior + Monte Carlo coverage**.

That is the shortest path from your current checkpoint to a technically credible Bayesian calibration result.

---

## 18. Short reference guide for the design choices

Use these as the conceptual templates while implementing:

- **Koide & Menegatti (2019)** — direct reprojection-error hand–eye calibration with latent pattern pose
- **Kalibr** — practical camera–IMU calibration and IMU noise model
- **Yang et al. (2022)** — observability and degeneracies for visual–inertial self-calibration
- **Ferguson et al. (2023)** — unified robot and IMU self-calibration
- **Colakovic-Bencerić et al. (2022)** — on-manifold hand–eye optimization on `SE(3)`
- **JCR / ARC-Calib / Kalib** — recent markerless joint calibration directions

For this repository today, however, do not start with markerless methods.
Use the current AprilTag benchmark to build the correct Bayesian back-end first.