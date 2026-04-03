# 08 — Motion presets for calibration testing

The preset library should be deterministic and parameterized.  
Every preset should be executable with different durations while preserving the same semantic motion pattern.

## Motion design goals

For camera/IMU calibration, good motion presets should:

- excite **translation** and **rotation**
- keep the target visible most of the time
- avoid jerk spikes that create unrealistic IMU artifacts
- stay inside joint and collision limits
- be easy to replay exactly

## Preset families

## A. 1-second presets

These are short impulses or micro-sequences.

### `snap_yaw_1s`

- small wrist-centered yaw sweep
- minimal translation
- use for quick detector smoke tests

### `nod_pitch_1s`

- pitch-forward / pitch-back
- small z translation
- useful for verifying IMU gyro signs and camera motion blur behavior

### `micro_arc_1s`

- short curved motion around a fixed target look-at point

## B. 2-second presets

### `figure8_small_2s`

- small lateral figure eight
- modest yaw and roll coupling

### `push_pull_roll_2s`

- translate toward tag, back away, add a small roll oscillation

## C. 5-second presets

### `calib_sweep_basic_5s`

The first serious preset.

- left/right translation
- up/down translation
- moderate yaw/pitch
- keeps target near center of the image

### `spiral_focus_5s`

- small spiral in task space around a look-at point
- slowly changing orientation
- good for detector stability testing

## D. 10-second presets

### `vi_excitation_10s`

Primary visual-inertial excitation preset.

- Lissajous-like translation in x/y/z
- coupled yaw/pitch/roll oscillations
- keeps the tag inside the central field of view most of the time
- the best default for calibration runs

### `frustum_box_10s`

- visit image-frustum corners deliberately
- good for stressing intrinsics and distortion estimation

## Trajectory representation

Presets should generate a sequence of target samples:

```yaml
preset_id: vi_excitation
sample_rate_hz: 120
samples:
  - t: 0.000
    xyz_m: [0.45, 0.00, 0.55]
    rpy_deg: [0.0, 0.0, 0.0]
  - t: 0.0083
    xyz_m: [0.451, 0.003, 0.548]
    rpy_deg: [0.4, -0.2, 0.1]
```

The robot controller then turns those into joint commands.

## Recommended parameterization

Represent a motion preset as:

- `duration_s`
- `sample_rate_hz`
- `look_at_target`
- `translation_amplitude_xyz`
- `rotation_amplitude_rpy_deg`
- `frequency_hz`
- `phase_offsets`
- `smoothing_profile`

This lets you define one preset family and instantiate 1/2/5/10 s variants.

## Example formulas

### Lissajous-style preset

```text
x(t) = x0 + ax * sin(2π f1 t + φ1)
y(t) = y0 + ay * sin(2π f2 t + φ2)
z(t) = z0 + az * sin(2π f3 t + φ3)

roll(t)  = ar * sin(2π fr t + φr)
pitch(t) = ap * sin(2π fp t + φp)
yaw(t)   = ayw * sin(2π fy t + φy)
```

These are not “the” correct formulas; they are a good reusable scaffold.

## Safety/validity filters

Every generated preset should pass:

- joint limit check
- self-collision check
- world collision check
- visibility fraction threshold
- minimum tag pixel footprint threshold

If a preset violates any check, reject it before the run begins.

## Suggested v1 preset set

| Preset | Nominal duration | Main use |
|---|---:|---|
| `snap_yaw` | 1 s | detector smoke test |
| `figure8_small` | 2 s | short motion integration test |
| `calib_sweep_basic` | 5 s | first useful calibration run |
| `vi_excitation` | 10 s | main calibration benchmark |

## Recording policy

Every preset execution should automatically log:

- preset id
- duration
- seed / phase values
- control sample rate
- visibility statistics
- final robot state
- whether a run was aborted or clipped

## Why keep preset ids stable

Calibration benchmarking gets messy fast if motion patterns change but keep the same name.  
Treat preset ids as part of the public contract.

If you materially change a preset, create a new id, for example:

- `vi_excitation_v1`
- `vi_excitation_v2`

## Practical first milestone

Implement only these three first:

- `snap_yaw_1s`
- `calib_sweep_basic_5s`
- `vi_excitation_10s`

That is enough to cover smoke, medium, and full runs.
