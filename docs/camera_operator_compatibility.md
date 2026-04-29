# Camera Operator Compatibility

This note compares the camera operator in
[`Beremi/CameraPoseEstimation`](https://github.com/Beremi/CameraPoseEstimation)
with the tabletop camera and AprilTag export used in this simulation repository.

External repository inspected:

- URL: `https://github.com/Beremi/CameraPoseEstimation`
- HEAD: `1be4553c79fdd404c5a48e3bccfbad169fc37542`
- Main operator files: `+operators/kamera.m`, `+operators/F_val.m`, `+operators/F_grad.m`, `+operators/F_hess.m`
- Generator reference: `sympy_generator/kamera_operators_grad_hess.ipynb`

## Bottom Line

The external `kamera` operator is not drop-in compliant with the current tabletop
sim export.

It can still be used for an inverse problem on calibration patterns, but only
after an explicit adapter or after regenerating the operator for this sim. The
main differences are units, focal model, pattern geometry, point order, pose
parameterization, and image coordinate conventions.

For best fidelity on current sim data, regenerate or reimplement the inverse
operator from this repository's pinhole projection and AprilTag marker-face
geometry instead of using the legacy operator unchanged.

## External CameraPoseEstimation Operator

The external operator maps a 6D pose-like vector to 5 image-plane points:

```text
input:  [X, Y, Z, alpha, beta, gama]
units:  X/Y/Z in meters, alpha/beta/gama in radians
output: [x0, y0, x1, y1, x2, y2, x3, y3, x4, y4] in millimeters
```

The generated notebook states that the output is image-plane millimeters. The
operator is hard-coded for a specific 5-point square calibration pattern:

```text
center       = [ 0.00,  0.00, 0]
top_right    = [ 0.05,  0.05, 0]
bottom_right = [ 0.05, -0.05, 0]
bottom_left  = [-0.05, -0.05, 0]
top_left     = [-0.05,  0.05, 0]
```

Important fixed constants:

- focal length: `ff = 0.008 m` (`8 mm`)
- robot arm plus end-effector length: `L = 0.3095 m`
- generated denominator uses `0.3015 m`, corresponding to `L - ff`
- pattern half extent: `0.05 m`
- rotation order in the generator: `R = Rx(alpha) * Ry(beta) * Rz(gama)`
- optical direction reference: `v = [0, 0, -1]`
- image-plane basis reference: `n = [0, 1, 0]`

The objective helpers `F_val`, `F_grad`, and `F_hess` minimize residuals against
the same 10-value image-plane-millimeter observation vector.

## Current Tabletop Sim Camera

The supported tabletop mounted camera is configured in
`config/isaac/camera/phone_tabletop_demo.yaml`:

```text
resolution: 1280 x 720
rate:       30 Hz
fx, fy:     862.11424 px, 862.11424 px
cx, cy:     640.194848 px, 362.42144 px
model:      OpenCV pinhole
distortion: [0, 0, 0, 0, 0]  # ideal clean render
```

The camera YAML also carries physical render settings:

```text
focal_length_mm:        4.53
horizontal_aperture_mm: 6.725791
vertical_aperture_mm:   3.783257
```

Those values are consistent with `fx ~= fy ~= 862.114 px` after rounding:

```text
fx = focal_length_mm / horizontal_aperture_mm * width_px  ~= 862.114 px
fy = focal_length_mm / vertical_aperture_mm   * height_px ~= 862.114 px
```

The source phone TOML at `config/camera/pixel_9a_main.toml` contains active-array
distortion metadata and the Pixel 9a sample-capture crop/rotation metadata. The
export now has three relevant geometry modes:

```text
clean/        ideal processed-video pinhole settings
phone_clean/  Pixel 9a processed-video intrinsics plus updated GT
noisy/        same processed-video coordinate system as phone_clean/
```

The external `kamera` operator remains closest to the `clean/` pinhole variant.
For `phone_clean/` and `noisy/`, use the exported variant metadata to decide
whether distortion is active. With the default Pixel 9a profile, RGB remains
pinhole because the MP4 is treated as processed phone video.

## Current Sim Projection And Export

The export code projects GT tag geometry in
`src/calib_sim/reporting/tabletop_autodemo_dataset.py`.

The projection model is:

```text
u = fx * (X / Z) + cx
v = fy * (-(Y / Z)) + cy
```

where the repo camera frame is:

```text
+X right
+Y up
+Z forward
```

Image pixels use the usual image convention:

```text
x right
y down
origin at top-left pixel
```

Isaac/USD camera poses use `+X right, +Y up, -Z forward`, so the export helper
flips only the forward axis before applying the repo pinhole projection.

The exported AprilTag GT corners are written to:

```text
<variant>/gt/tag_image_projections.jsonl
```

The detector corners are written to:

```text
<variant>/localization/new_pupil/detections.jsonl
```

In corner-only exports, the detector output is also available at:

```text
<variant>/raw/detections.jsonl
```

Current exported corner order is:

```text
top_left, top_right, bottom_right, bottom_left
```

The exported corners are the rendered visible AprilTag marker face, not the
outer printed board. The marker is smaller than the printed tag:

```text
marker_side_to_printed_tag_side = 0.8205128205128205
```

So the effective marker half extents are:

```text
printed tag 0.10 m -> marker half extent 0.041025641025641026 m
printed tag 0.06 m -> marker half extent 0.024615384615384612 m
```

This matters because the external operator assumes a square half extent of
`0.05 m`.

## Compliance Matrix

| Topic | External `CameraPoseEstimation` operator | Current tabletop sim | Drop-in compliant? |
| --- | --- | --- | --- |
| Observation units | image-plane millimeters | pixels in exported JSONL | No |
| Intrinsics | fixed `8 mm` focal length | `fx=fy=862.11424 px`, physical focal `4.53 mm` | No |
| Distortion | no explicit Brown-Conrady distortion path in operator | `clean/` is zero distortion; default `phone_clean/` and `noisy/` are processed-video pinhole with Pixel distortion metadata recorded but not applied | Compatible after using the exported variant intrinsics |
| Observation count | 5 points, 10 scalar values | 4 tag corners plus `center_xy` available per tag | Adaptable |
| Point order | `center, top_right, bottom_right, bottom_left, top_left` | corners are `top_left, top_right, bottom_right, bottom_left`; center separate | No, but easy to reorder |
| Pattern geometry | fixed 0.10 m square, half extent `0.05 m` | AprilTag marker face is `0.082051282 m` for 0.10 m printed tags | No |
| Coordinate frame | custom operator frame with `v=[0,0,-1]`, `n=[0,1,0]`, `R=Rx Ry Rz` | repo camera frame `+X right, +Y up, +Z forward`; USD camera frame flips forward | No |
| Pose variables | `[X,Y,Z,alpha,beta,gama]` around legacy robot/pattern model | exported camera pose is world pose `px,py,pz,qw,qx,qy,qz` | No |
| Objective | generated residuals against `kamera(...)` observations | export provides detections, GT projections, and optional pose measurements | Adaptable |
| Inverse use | solves the legacy 5-point operator inverse | sim needs a pinhole AprilTag/pose inverse or adapted observation vector | Not directly |

## Conversion Recipe For Experiments

Use this only if you intentionally want to test the legacy operator against sim
data. It is an adapter, not a guarantee of physical equivalence.

1. Choose one tag or calibration pattern.

   For exported data, start with a single row from
   `gt/tag_image_projections.jsonl` or `localization/new_pupil/detections.jsonl`.
   Filter by `frame_index`, `tag_id`, and, for GT, fully visible rows where
   `visibility_flags.fully_visible == true`.

2. Build the 5-point observation.

   The sim gives corners in this order:

   ```text
   top_left, top_right, bottom_right, bottom_left
   ```

   The legacy operator expects:

   ```text
   center, top_right, bottom_right, bottom_left, top_left
   ```

   Use the exported `center_xy` for GT rows, or compute the detector center as
   the mean or diagonal intersection of detected corners.

3. Convert pixels to image-plane coordinates.

   For the `clean/` sim pinhole camera:

   ```text
   x_norm =  (u - cx_px) / fx_px
   y_norm = -(v - cy_px) / fy_px
   ```

   To express observations in physical image-plane millimeters for the current
   sim camera:

   ```text
   x_mm = focal_length_mm * x_norm
   y_mm = focal_length_mm * y_norm
   ```

   with:

   ```text
   focal_length_mm = 4.53
   fx_px = fy_px = 862.11424
   cx_px = 640.194848
   cy_px = 362.42144
   ```

   If you use the legacy operator unchanged, you may instead scale normalized
   coordinates by the legacy focal length:

   ```text
   x_legacy_mm = 8.0 * x_norm
   y_legacy_mm = 8.0 * y_norm
   ```

   That only matches the legacy output scale; it does not fix geometry or pose
   convention differences.

4. Decide which pattern square the inverse should model.

   For the legacy operator unchanged, the pattern half extent is `0.05 m`.

   For sim AprilTag marker-face corners, regenerate/adapt the operator to use:

   ```text
   half_extent_m = 0.041025641025641026  # for printed 0.10 m tags
   half_extent_m = 0.024615384615384612  # for printed 0.06 m tags
   ```

   If you instead want the printed board square, use the printed tag size, but
   compare against projected/exported printed-board corners rather than current
   marker-face corners.

5. Align pose variables.

   The legacy operator state is:

   ```text
   [X, Y, Z, alpha, beta, gama]
   ```

   The sim full export provides projection-matched camera GT in:

   ```text
   <variant>/gt/camera_projection_gt.csv
   ```

   with:

   ```text
   px, py, pz, qw, qx, qy, qz
   ```

   This quaternion world pose is not the same parameterization as the legacy
   operator. A valid adapter must choose a common reference frame, convert the
   quaternion into the same `Rx(alpha) * Ry(beta) * Rz(gama)` convention, and
   account for the Isaac/USD forward-axis flip.

## Recommended Path

For direct use on current exported AprilTag detections:

1. Do not feed `corners_xy` directly into `kamera` or `F_val`.
2. Reorder and convert the observations first.
3. Prefer regenerating the operator with this sim's constants:
   `fx/fy/cx/cy` or equivalent `focal_length_mm`, the correct distortion mode
   for the chosen variant, the correct AprilTag marker half extent, and this
   repository's camera-frame convention.

For best sim fidelity:

- Base the inverse problem on this repository's pinhole projection path:
  `src/calib_sim/reporting/tabletop_autodemo_dataset.py`.
- Use the same AprilTag marker geometry helpers from
  `src/calib_sim/tag_service/detector.py`.
- Use `gt/tag_image_projections.jsonl` for GT image-plane references and
  `localization/new_pupil/detections.jsonl` for measured detector corners.
- Treat the legacy `CameraPoseEstimation` code as a useful optimizer/operator
  template, not as a drop-in camera model for this sim.

## Quick Compatibility Verdict

The external operator can be used for an inverse calibration-pattern problem in
this project only after an adapter or regeneration step. The minimum adapter
must handle:

- pixel-to-millimeter conversion
- point reordering
- center-point construction for detector rows
- AprilTag marker-face size versus legacy square size
- pose-frame and Euler-angle convention conversion
- focal-length and projection-model mismatch

Without those steps, the inverse will be solving a different camera/operator
problem than the one exported by the tabletop simulation.
