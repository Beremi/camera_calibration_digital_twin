# 04 — Scene assets and AprilTag placement

## Room asset strategy

For your simulator, room assets should satisfy three criteria:

1. **furnished indoor scenes**
2. **usable licensing**
3. **reasonable import path into the simulator**

ReplicaCAD is a strong default because the official dataset page describes furnished apartment variations intended for interactive simulation and released under **CC BY 4.0**.[replicacad]  
Isaac Sim’s asset converter supports conversion from formats such as `.gltf`, `.obj`, and `.fbx` to USD, which is the right pipeline for bringing third-party assets into the simulator.[isaac-formats]

### Recommended scene strategy

Start with:

- **1–2 ReplicaCAD rooms** for reproducibility
- optionally add a small set of **Poly Haven** props or HDRIs for better visual diversity without licensing pain.[polyhaven]

Do **not** begin by randomizing hundreds of environments.  
For calibration development, repeatability matters more than scale.

## Tag strategy for your use case

Your stated constraint is important:

- the pattern should be **small** (around 10 × 10 cm at most),
- it should be **recognizable from farther away**,
- fine detail is not useful.

That implies the default target should **not** be a dense checkerboard or dense ChArUco board.

### Best default target mode

Use a **single AprilTag 36h11** with physical edge length around **0.08–0.10 m**.

Why:

- one tag gives an orientation-resolved square target,
- the pattern is distinctive,
- pose can be estimated from a single tag,
- the detector logic is simple,
- the tag is still large enough to be useful at moderate distance.

### Board mode

Also support a **sparse AprilGrid** layout for richer calibration runs.  
Kalibr’s calibration target guidance recommends Aprilgrid because partial visibility is acceptable and the target pose is fully resolved without flips.[kalibr-targets]

So the simulator should support two target types:

- `single_tag`
- `aprilgrid`

## Tag families and generation

OpenCV’s predefined dictionary list includes `DICT_APRILTAG_36h11`, and the module can generate a canonical marker image using `generateImageMarker()`.[opencv-aruco]

The generated images in `assets/` were created from that path:

- [`../assets/apriltag36h11_id0.png`](../assets/apriltag36h11_id0.png)
- [`../assets/apriltag36h11_id42.png`](../assets/apriltag36h11_id42.png)

### Practical recommendation for ids

Reserve ids semantically:

- `0–49` for single-tag scenes
- `50–199` for board scenes
- `200+` for future experiments

That way, downstream code can infer intent from the id range if useful.

## Placement rules

### For a single wall-mounted tag

Good defaults:

- mount vertically on a wall or stand
- center at roughly camera height or slightly above
- keep at least one meter of unobstructed view volume
- ensure the tag plane normal points toward the expected robot workspace

### For a sparse board

Good defaults:

- 2 × 2 or 3 × 3 sparse layout
- larger spacing than traditional dense boards
- all tags coplanar
- known board frame at the board center

### Avoid these mistakes

- glossy surfaces behind or over the tag
- tiny tags on giant walls where they occupy too few pixels
- tags flush against cluttered textured backgrounds
- tags placed where the wrist camera sees them only at extreme oblique angles

## Suggested geometric defaults

### Single-tag mode

- `family`: `36h11`
- `id`: `0`
- `edge_length_m`: `0.10`
- `board_pose_world`: fixed wall transform

### Sparse board mode

- `family`: `36h11`
- `rows`: `2`
- `cols`: `2`
- `tag_edge_m`: `0.06`
- `gap_m`: `0.03`

These are design choices, not official standards.  
They are chosen to match your “small but still visible” constraint.

## How to represent tags in the simulator

The simplest representation is:

- a **thin rectangular plane** with a tag texture
- a known transform for the tag frame
- optional backside collision disabled
- matte material to reduce unrealistic specular artifacts

If you want to test lighting robustness later, keep the material configurable so you can create:

- matte paper
- laminated paper
- foam board
- acrylic sign

## Coordinate conventions

Define the tag-local coordinate frame explicitly and keep it consistent:

- origin at tag center
- `+x` to the tag’s right
- `+y` up
- `+z` out of the tag plane toward the camera when front-facing

Even if downstream detectors use a different internal convention, your simulator truth model must be unambiguous.

## Example config

```yaml
family: DICT_APRILTAG_36h11
mode: single_tag
tag_id: 0
edge_length_m: 0.10
material:
  roughness: 0.9
  metallic: 0.0
placement:
  parent_prim: /World/Room/WestWall
  translation_m: [0.0, 1.45, 0.02]
  rpy_deg: [0.0, 0.0, 0.0]
```

## Visibility testing

Before trusting a motion preset, validate:

- minimum pixel side length over the trajectory
- fraction of frames where the tag is visible
- max obliquity angle
- occlusion ratio

A simple simulator-side “visibility validator” saves a lot of wasted recorded data.

## Why this matches the literature and tooling

The original AprilTag work emphasizes robust fiducial detection for robotics use, including low-resolution and cluttered scenes.[apriltag-olson]  
AprilTag 2 focuses on reduced false positives and improved efficiency, making it practical on computation-limited systems.[apriltag2]  
For board-style calibration targets, Kalibr recommends Aprilgrid because it tolerates partial visibility and avoids symmetry flips.[kalibr-targets]

## Sources

[replicacad]: https://aihabitat.org/datasets/replica_cad/
[isaac-formats]: https://docs.isaacsim.omniverse.nvidia.com/4.5.0/assets/formats.html
[polyhaven]: https://polyhaven.com/
[opencv-aruco]: https://docs.opencv.org/4.x/de/d67/group__objdetect__aruco.html
[kalibr-targets]: https://github.com/ethz-asl/kalibr/wiki/calibration-targets
[apriltag-olson]: https://april.eecs.umich.edu/media/pdfs/olson2011tags.pdf
[apriltag2]: https://april.eecs.umich.edu/pdfs/wang2016iros.pdf
