"""Static stage description helpers and programmatic scene building."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from calib_sim.estimation._geometry import rotation_matrix_from_rpy_deg
from calib_sim.isaac.tag_builder import TagPoseSpec

REPO_ROOT = Path(__file__).resolve().parents[3]


def _repo_path(path_like: str | Path) -> Path:
    candidate = Path(path_like)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def _load_yaml_payload(path_like: str | Path) -> dict[str, Any]:
    resolved_path = _repo_path(path_like)
    payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping payload in {resolved_path}")
    return payload


def _matrix_to_quaternion_wxyz(rotation: np.ndarray) -> tuple[float, float, float, float]:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (matrix[2, 1] - matrix[1, 2]) / s
        y = (matrix[0, 2] - matrix[2, 0]) / s
        z = (matrix[1, 0] - matrix[0, 1]) / s
        return (float(w), float(x), float(y), float(z))
    diagonal = np.diag(matrix)
    if diagonal[0] > diagonal[1] and diagonal[0] > diagonal[2]:
        s = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
        return (
            float((matrix[2, 1] - matrix[1, 2]) / s),
            float(0.25 * s),
            float((matrix[0, 1] + matrix[1, 0]) / s),
            float((matrix[0, 2] + matrix[2, 0]) / s),
        )
    if diagonal[1] > diagonal[2]:
        s = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
        return (
            float((matrix[0, 2] - matrix[2, 0]) / s),
            float((matrix[0, 1] + matrix[1, 0]) / s),
            float(0.25 * s),
            float((matrix[1, 2] + matrix[2, 1]) / s),
        )
    s = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
    return (
        float((matrix[1, 0] - matrix[0, 1]) / s),
        float((matrix[0, 2] + matrix[2, 0]) / s),
        float((matrix[1, 2] + matrix[2, 1]) / s),
        float(0.25 * s),
    )


def _normalize_vector(vector: np.ndarray, *, default: np.ndarray) -> np.ndarray:
    candidate = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(candidate))
    if norm < 1e-12:
        return np.asarray(default, dtype=np.float64).reshape(candidate.shape)
    return candidate / norm


def _look_at_rotation_wxyz(
    eye_world_m: np.ndarray,
    target_world_m: np.ndarray,
    *,
    up_hint_world_m: np.ndarray | None = None,
) -> tuple[float, float, float, float]:
    eye = np.asarray(eye_world_m, dtype=np.float64).reshape(3)
    target = np.asarray(target_world_m, dtype=np.float64).reshape(3)
    forward = _normalize_vector(target - eye, default=np.array([0.0, 0.0, -1.0], dtype=np.float64))
    up_hint = np.asarray([0.0, 0.0, 1.0] if up_hint_world_m is None else up_hint_world_m, dtype=np.float64).reshape(3)
    if abs(float(np.dot(forward, _normalize_vector(up_hint, default=np.array([0.0, 0.0, 1.0], dtype=np.float64))))) > 0.97:
        up_hint = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    right = _normalize_vector(np.cross(forward, up_hint), default=np.array([1.0, 0.0, 0.0], dtype=np.float64))
    up = _normalize_vector(np.cross(right, forward), default=np.array([0.0, 0.0, 1.0], dtype=np.float64))
    return _matrix_to_quaternion_wxyz(np.column_stack((right, up, -forward)))


def _segment_rotation_wxyz(
    start_world_m: np.ndarray,
    end_world_m: np.ndarray,
    *,
    up_hint_world_m: np.ndarray | None = None,
) -> tuple[float, float, float, float]:
    start = np.asarray(start_world_m, dtype=np.float64).reshape(3)
    end = np.asarray(end_world_m, dtype=np.float64).reshape(3)
    forward = _normalize_vector(end - start, default=np.array([1.0, 0.0, 0.0], dtype=np.float64))
    up_hint = np.asarray([0.0, 0.0, 1.0] if up_hint_world_m is None else up_hint_world_m, dtype=np.float64).reshape(3)
    if abs(float(np.dot(up_hint, forward))) > 0.97:
        up_hint = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    up = _normalize_vector(
        up_hint - np.dot(up_hint, forward) * forward,
        default=np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )
    left = _normalize_vector(np.cross(up, forward), default=np.array([0.0, 1.0, 0.0], dtype=np.float64))
    up = _normalize_vector(np.cross(forward, left), default=np.array([0.0, 0.0, 1.0], dtype=np.float64))
    return _matrix_to_quaternion_wxyz(np.column_stack((forward, left, up)))


def _color_vector(raw: Any, *, default: tuple[float, float, float]) -> np.ndarray:
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        return np.asarray(default, dtype=np.float64)
    values = np.asarray([float(raw[0]), float(raw[1]), float(raw[2])], dtype=np.float64)
    if float(np.max(np.abs(values))) > 1.0:
        values = values / 255.0
    return np.clip(values, 0.0, 1.0)


def _vector3(raw: Any, *, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        return tuple(float(value) for value in default)
    return (float(raw[0]), float(raw[1]), float(raw[2]))


def _color_triplets(
    raw: Any,
    *,
    default: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float], ...]:
    colors: list[tuple[float, float, float]] = []
    entries = raw if isinstance(raw, (list, tuple)) else ()
    for index, fallback in enumerate(default):
        entry = entries[index] if index < len(entries) else fallback
        color = _color_vector(entry, default=fallback)
        colors.append(tuple(float(value) for value in color.tolist()))
    return tuple(colors)


def _tags_overlap(lhs: "StageTagSpec", rhs: "StageTagSpec") -> bool:
    lhs_position = np.asarray(lhs.position_world_m, dtype=np.float64).reshape(3)
    rhs_position = np.asarray(rhs.position_world_m, dtype=np.float64).reshape(3)
    center_distance = float(np.linalg.norm(lhs_position - rhs_position))
    min_separation = 0.5 * (float(lhs.size_m) + float(rhs.size_m))
    return center_distance < max(min_separation - 1.0e-6, 0.0)


def _generate_tag_texture(tag_id: int, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker_size_px = 512
    quiet_zone_px = 56
    marker = cv2.aruco.generateImageMarker(dictionary, int(tag_id), marker_size_px)
    marker = cv2.copyMakeBorder(
        marker,
        quiet_zone_px,
        quiet_zone_px,
        quiet_zone_px,
        quiet_zone_px,
        cv2.BORDER_CONSTANT,
        value=255,
    )
    cv2.imwrite(str(destination), marker)
    return destination


def _tag_binary_grid(tag_id: int, destination: Path, *, grid_size: int = 10) -> np.ndarray:
    texture_path = _generate_tag_texture(tag_id, destination)
    image = cv2.imread(str(texture_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load generated tag texture: {texture_path}")
    resized = cv2.resize(image, (int(grid_size), int(grid_size)), interpolation=cv2.INTER_NEAREST)
    return np.asarray(resized < 127, dtype=bool)


def _create_textured_tag_material(*, stage: Any, material_prim_path: str, texture_path: Path) -> Any:
    from pxr import Sdf, UsdShade

    material = UsdShade.Material.Define(stage, material_prim_path)
    shader = UsdShade.Shader.Define(stage, f"{material_prim_path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    texture = UsdShade.Shader.Define(stage, f"{material_prim_path}/Texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(texture_path)))
    texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
    texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    primvar = UsdShade.Shader.Define(stage, f"{material_prim_path}/Primvar")
    primvar.CreateIdAttr("UsdPrimvarReader_float2")
    primvar.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    primvar.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(primvar.ConnectableAPI(), "result")
    texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _create_tag_plane(
    *,
    stage: Any,
    prim_path: str,
    tag_id: int,
    tag_size_m: float,
    position_world_m: tuple[float, float, float],
    rotation_wt: np.ndarray,
    texture_path: Path,
) -> None:
    from isaacsim.core.api.objects import FixedCuboid
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    position = np.asarray(position_world_m, dtype=np.float64).reshape(3)
    orientation_wxyz = np.asarray(_matrix_to_quaternion_wxyz(rotation_wt), dtype=np.float64)
    board_thickness_m = 0.003
    tag_root = UsdGeom.Xform.Define(stage, prim_path)
    tag_xform = UsdGeom.Xformable(tag_root)
    tag_xform.ClearXformOpOrder()
    tag_xform.AddTranslateOp().Set(Gf.Vec3d(*position.tolist()))
    tag_xform.AddOrientOp().Set(
        Gf.Quatf(
            float(orientation_wxyz[0]),
            Gf.Vec3f(
                float(orientation_wxyz[1]),
                float(orientation_wxyz[2]),
                float(orientation_wxyz[3]),
            ),
        )
    )
    FixedCuboid(
        prim_path=f"{prim_path}/board",
        name=f"{Path(prim_path).name}_board",
        position=np.zeros(3, dtype=np.float64),
        orientation=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        scale=np.asarray([float(tag_size_m), float(tag_size_m), board_thickness_m], dtype=np.float64),
        color=np.asarray([1.0, 1.0, 1.0], dtype=np.float64),
        size=1.0,
    )
    material = _create_textured_tag_material(
        stage=stage,
        material_prim_path=f"/World/Looks/tag_{int(tag_id)}",
        texture_path=texture_path,
    )
    half_extent_m = 0.5 * float(tag_size_m)
    local_z_m = 0.5 * board_thickness_m + 1.0e-3
    local_corners = np.array(
        [
            [-half_extent_m, -half_extent_m, local_z_m],
            [half_extent_m, -half_extent_m, local_z_m],
            [half_extent_m, half_extent_m, local_z_m],
            [-half_extent_m, half_extent_m, local_z_m],
        ],
        dtype=np.float64,
    )
    mesh = UsdGeom.Mesh.Define(stage, f"{prim_path}/face")
    mesh.CreatePointsAttr([Gf.Vec3f(*corner.tolist()) for corner in local_corners])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateNormalsAttr([Gf.Vec3f(0.0, 0.0, 1.0)] * 4)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateExtentAttr(
        [
            Gf.Vec3f(-half_extent_m, -half_extent_m, local_z_m),
            Gf.Vec3f(half_extent_m, half_extent_m, local_z_m),
        ]
    )
    primvars = UsdGeom.PrimvarsAPI(mesh)
    uv = primvars.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    uv.Set(
        [
            Gf.Vec2f(0.0, 0.0),
            Gf.Vec2f(1.0, 0.0),
            Gf.Vec2f(1.0, 1.0),
            Gf.Vec2f(0.0, 1.0),
        ]
    )
    UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(material)


@dataclass(slots=True)
class StageTagSpec:
    tag_id: int
    prim_path: str
    size_m: float
    position_world_m: tuple[float, float, float]
    orientation_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    texture_path: str | None = None
    is_anchor: bool = False

    def pose_spec(self) -> TagPoseSpec:
        rotation = rotation_matrix_from_rpy_deg(self.orientation_rpy_deg)
        return TagPoseSpec(
            tag_id=int(self.tag_id),
            size_m=float(self.size_m),
            position_world_m=tuple(float(value) for value in self.position_world_m),
            rotation_wt=tuple(tuple(float(entry) for entry in row) for row in rotation.tolist()),
            is_anchor=bool(self.is_anchor),
        )


@dataclass(slots=True)
class StageBoxSpec:
    name: str
    prim_path: str
    center_world_m: tuple[float, float, float]
    size_m: tuple[float, float, float]
    color_rgb: tuple[float, float, float] = (0.75, 0.75, 0.75)


@dataclass(slots=True)
class ObserverCameraSpec:
    name: str
    prim_path: str
    position_world_m: tuple[float, float, float]
    look_at_world_m: tuple[float, float, float]
    width_px: int = 640
    height_px: int = 360
    fov_deg: float = 60.0
    rate_hz: float | None = None


@dataclass(slots=True)
class StageArmProxySpec:
    name: str
    prim_path: str
    base_position_world_m: tuple[float, float, float]
    shoulder_height_m: float
    link_lengths_m: tuple[float, float, float]
    initial_servos_deg: tuple[float, float, float]
    pedestal_size_m: tuple[float, float, float] = (0.14, 0.14, 0.06)
    link_thickness_m: float = 0.045
    joint_size_m: float = 0.06
    tool_size_m: tuple[float, float, float] = (0.10, 0.06, 0.04)
    base_color_rgb: tuple[float, float, float] = (0.38, 0.41, 0.46)
    link_colors_rgb: tuple[tuple[float, float, float], ...] = (
        (0.82, 0.85, 0.88),
        (0.66, 0.70, 0.75),
        (0.53, 0.58, 0.64),
    )
    joint_color_rgb: tuple[float, float, float] = (0.20, 0.23, 0.28)
    tool_color_rgb: tuple[float, float, float] = (0.34, 0.37, 0.42)

    def joint_chain_world(self) -> tuple[tuple[float, float, float], ...]:
        q1_rad, q2_rad, q3_rad = np.radians(np.asarray(self.initial_servos_deg, dtype=np.float64).reshape(3))
        l1_m, l2_m, l3_m = np.asarray(self.link_lengths_m, dtype=np.float64).reshape(3)
        base = np.asarray(self.base_position_world_m, dtype=np.float64).reshape(3)
        shoulder = base + np.array([0.0, 0.0, float(self.shoulder_height_m)], dtype=np.float64)

        first_reach = float(l1_m) * float(np.cos(q2_rad))
        first_rise = float(l1_m) * float(np.sin(q2_rad))
        elbow = shoulder + np.array(
            [first_reach * np.cos(q1_rad), first_reach * np.sin(q1_rad), first_rise],
            dtype=np.float64,
        )

        second_reach = float(l2_m) * float(np.cos(q2_rad + q3_rad))
        second_rise = float(l2_m) * float(np.sin(q2_rad + q3_rad))
        wrist = elbow + np.array(
            [second_reach * np.cos(q1_rad), second_reach * np.sin(q1_rad), second_rise],
            dtype=np.float64,
        )

        tool_reach = float(l3_m) * float(np.cos(q2_rad + q3_rad))
        tool_rise = float(l3_m) * float(np.sin(q2_rad + q3_rad))
        tool = wrist + np.array(
            [tool_reach * np.cos(q1_rad), tool_reach * np.sin(q1_rad), tool_rise],
            dtype=np.float64,
        )
        return tuple(tuple(float(value) for value in point.tolist()) for point in (base, shoulder, elbow, wrist, tool))


@dataclass(slots=True)
class IsaacStageSpec:
    """Repository-level description of the intended Isaac stage."""

    stage_path: str
    robot_prim_path: str
    camera_prim_path: str
    imu_prim_path: str
    physics_rate_hz: float
    observer_camera_prim_paths: tuple[str, ...] = ()
    tags: tuple[StageTagSpec, ...] = field(default_factory=tuple)
    scene_boxes: tuple[StageBoxSpec, ...] = field(default_factory=tuple)
    observer_cameras: tuple[ObserverCameraSpec, ...] = field(default_factory=tuple)
    arm_proxy: StageArmProxySpec | None = None
    table_center_world_m: tuple[float, float, float] = (0.0, 0.0, -0.025)
    table_scale_m: tuple[float, float, float] = (0.7, 0.7, 0.05)
    room_center_world_m: tuple[float, float, float] = (0.0, 0.0, 0.5)
    room_scale_m: tuple[float, float, float] = (2.0, 2.0, 1.0)

    def validate(self) -> None:
        if self.physics_rate_hz <= 0.0:
            raise ValueError("physics_rate_hz must be positive.")
        anchor_count = sum(1 for tag in self.tags if tag.is_anchor)
        if anchor_count != 1:
            raise ValueError("Stage spec must contain exactly one anchor tag.")
        if len(self.tags) < 3:
            raise ValueError("First-pass Isaac stage requires one anchor and at least two auxiliary tags.")
        for index, lhs in enumerate(self.tags):
            for rhs in self.tags[index + 1 :]:
                if _tags_overlap(lhs, rhs):
                    raise ValueError(
                        f"Stage spec contains overlapping tags: tag {lhs.tag_id} at {lhs.position_world_m} "
                        f"overlaps tag {rhs.tag_id} at {rhs.position_world_m}."
                    )

    @property
    def tag_pose_specs(self) -> tuple[TagPoseSpec, ...]:
        return tuple(tag.pose_spec() for tag in self.tags)


@dataclass(slots=True)
class StageBuildArtifacts:
    stage_spec: IsaacStageSpec
    tag_pose_specs: dict[int, TagPoseSpec]
    warnings: list[str] = field(default_factory=list)

    def anchor_tag_pose(self) -> TagPoseSpec:
        for tag in self.tag_pose_specs.values():
            if tag.is_anchor:
                return tag
        raise ValueError("Missing anchor tag pose.")


def _replica_arm_defaults(scene_config: dict[str, Any]) -> dict[str, Any]:
    arm_config_path = scene_config.get("replica_source_arm_config")
    if not arm_config_path and scene_config.get("replica_source_scene_config"):
        scene_payload = _load_yaml_payload(str(scene_config["replica_source_scene_config"]))
        robot_payload = scene_payload.get("robot", {}) if isinstance(scene_payload.get("robot"), dict) else {}
        arm_config_path = robot_payload.get("preset_path")
    if not arm_config_path:
        return {}
    return _load_yaml_payload(str(arm_config_path))


def _stage_arm_proxy_from_config(scene_config: dict[str, Any]) -> StageArmProxySpec | None:
    entry = scene_config.get("arm_proxy", {})
    if entry is None:
        entry = {}
    if not isinstance(entry, dict):
        raise ValueError("arm_proxy must be a mapping when provided.")
    if entry and not bool(entry.get("enabled", True)):
        return None
    defaults = _replica_arm_defaults(scene_config)
    if not entry and not defaults:
        return None
    link_colors_default = (
        (0.82, 0.85, 0.88),
        (0.66, 0.70, 0.75),
        (0.53, 0.58, 0.64),
    )
    base_color_default = (0.38, 0.41, 0.46)
    joint_color_default = (0.20, 0.23, 0.28)
    tool_color_default = (0.34, 0.37, 0.42)
    base_color = _color_vector(entry.get("base_color_rgb", entry.get("base_color_bgr")), default=base_color_default)
    joint_color = _color_vector(entry.get("joint_color_rgb", entry.get("joint_color_bgr")), default=joint_color_default)
    tool_color = _color_vector(entry.get("tool_color_rgb", entry.get("tool_color_bgr")), default=tool_color_default)
    return StageArmProxySpec(
        name=str(entry.get("name", defaults.get("label", defaults.get("name", "replica_arm")))),
        prim_path=str(entry.get("prim_path", "/World/ReplicaArm")),
        base_position_world_m=_vector3(
            entry.get("base_position_world_m", defaults.get("base_position_m")),
            default=(0.0, 1.08, 0.72),
        ),
        shoulder_height_m=float(entry.get("shoulder_height_m", defaults.get("shoulder_height_m", 0.08))),
        link_lengths_m=_vector3(
            entry.get("link_lengths_m", defaults.get("link_lengths_m")),
            default=(0.34, 0.30, 0.22),
        ),
        initial_servos_deg=_vector3(
            entry.get("initial_servos_deg", defaults.get("initial_servos_deg")),
            default=(-36.0, 60.0, -56.0),
        ),
        pedestal_size_m=_vector3(entry.get("pedestal_size_m"), default=(0.14, 0.14, 0.06)),
        link_thickness_m=float(entry.get("link_thickness_m", 0.045)),
        joint_size_m=float(entry.get("joint_size_m", 0.06)),
        tool_size_m=_vector3(entry.get("tool_size_m"), default=(0.10, 0.06, 0.04)),
        base_color_rgb=tuple(float(value) for value in base_color.tolist()),
        link_colors_rgb=_color_triplets(entry.get("link_colors_rgb", entry.get("link_colors_bgr")), default=link_colors_default),
        joint_color_rgb=tuple(float(value) for value in joint_color.tolist()),
        tool_color_rgb=tuple(float(value) for value in tool_color.tolist()),
    )


def stage_spec_from_config(scene_config: dict[str, Any]) -> IsaacStageSpec:
    tags: list[StageTagSpec] = []
    for entry in scene_config.get("tags", []):
        if not isinstance(entry, dict):
            continue
        tags.append(
            StageTagSpec(
                tag_id=int(entry["tag_id"]),
                prim_path=str(entry["prim_path"]),
                size_m=float(entry["size_m"]),
                position_world_m=tuple(float(value) for value in entry.get("position_world_m", (0.0, 0.0, 0.0))),
                orientation_rpy_deg=tuple(float(value) for value in entry.get("orientation_rpy_deg", (0.0, 0.0, 0.0))),
                texture_path=None if not entry.get("texture_path") else str(entry["texture_path"]),
                is_anchor=bool(entry.get("is_anchor", False)),
            )
        )
    boxes: list[StageBoxSpec] = []
    for entry in scene_config.get("scene_boxes", []):
        if not isinstance(entry, dict):
            continue
        color_rgb = _color_vector(entry.get("color_rgb", entry.get("color_bgr")), default=(0.75, 0.75, 0.75))
        boxes.append(
            StageBoxSpec(
                name=str(entry.get("name", Path(str(entry.get("prim_path", "/World/Box"))).name)),
                prim_path=str(entry.get("prim_path", f"/World/Environment/{entry.get('name', 'box')}")),
                center_world_m=tuple(float(value) for value in entry.get("center_world_m", (0.0, 0.0, 0.0))),
                size_m=tuple(float(value) for value in entry.get("size_m", (0.1, 0.1, 0.1))),
                color_rgb=tuple(float(value) for value in color_rgb.tolist()),
            )
        )
    observer_cameras: list[ObserverCameraSpec] = []
    for entry in scene_config.get("observer_cameras", []):
        if not isinstance(entry, dict):
            continue
        observer_cameras.append(
            ObserverCameraSpec(
                name=str(entry.get("name", Path(str(entry.get("prim_path", "/World/Observer/camera"))).name)),
                prim_path=str(entry.get("prim_path", f"/World/Observer/{entry.get('name', 'camera')}")),
                position_world_m=tuple(float(value) for value in entry.get("position_world_m", (0.0, 0.0, 0.0))),
                look_at_world_m=tuple(float(value) for value in entry.get("look_at_world_m", (0.0, 0.0, 0.0))),
                width_px=int(entry.get("width_px", entry.get("width", 640))),
                height_px=int(entry.get("height_px", entry.get("height", 360))),
                fov_deg=float(entry.get("fov_deg", 60.0)),
                rate_hz=None if entry.get("rate_hz") in (None, "") else float(entry.get("rate_hz")),
            )
        )
    spec = IsaacStageSpec(
        stage_path=str(scene_config.get("stage_path", "")),
        robot_prim_path=str(scene_config["robot_prim_path"]),
        camera_prim_path=str(scene_config["camera_prim_path"]),
        imu_prim_path=str(scene_config["imu_prim_path"]),
        physics_rate_hz=float(scene_config["physics_rate_hz"]),
        observer_camera_prim_paths=tuple(
            str(value) for value in (
                scene_config.get("observer_camera_prim_paths")
                or [camera.prim_path for camera in observer_cameras]
            )
        ),
        tags=tuple(tags),
        scene_boxes=tuple(boxes),
        observer_cameras=tuple(observer_cameras),
        arm_proxy=_stage_arm_proxy_from_config(scene_config),
        table_center_world_m=tuple(float(value) for value in scene_config.get("table_center_world_m", (0.0, 0.0, -0.025))),
        table_scale_m=tuple(float(value) for value in scene_config.get("table_scale_m", (0.7, 0.7, 0.05))),
        room_center_world_m=tuple(float(value) for value in scene_config.get("room_center_world_m", (0.0, 0.0, 0.5))),
        room_scale_m=tuple(float(value) for value in scene_config.get("room_scale_m", (2.0, 2.0, 1.0))),
    )
    spec.validate()
    return spec


def _create_scene_box(
    *,
    box: StageBoxSpec,
) -> None:
    from isaacsim.core.api.objects import FixedCuboid

    FixedCuboid(
        prim_path=str(box.prim_path),
        name=str(box.name),
        position=np.asarray(box.center_world_m, dtype=np.float64),
        scale=np.asarray(box.size_m, dtype=np.float64),
        color=np.asarray(box.color_rgb, dtype=np.float64),
        size=1.0,
    )


def _create_observer_camera(
    *,
    stage: Any,
    camera_spec: ObserverCameraSpec,
) -> None:
    from pxr import Gf, UsdGeom

    camera = UsdGeom.Camera.Define(stage, camera_spec.prim_path)
    camera_xform = UsdGeom.Xformable(camera.GetPrim())
    camera_xform.ClearXformOpOrder()
    orientation_wxyz = _look_at_rotation_wxyz(
        np.asarray(camera_spec.position_world_m, dtype=np.float64),
        np.asarray(camera_spec.look_at_world_m, dtype=np.float64),
    )
    camera_xform.AddTranslateOp().Set(Gf.Vec3d(*camera_spec.position_world_m))
    camera_xform.AddOrientOp().Set(
        Gf.Quatf(
            float(orientation_wxyz[0]),
            Gf.Vec3f(
                float(orientation_wxyz[1]),
                float(orientation_wxyz[2]),
                float(orientation_wxyz[3]),
            ),
        )
    )
    horizontal_aperture_mm = 20.955
    focal_length_mm = 0.5 * horizontal_aperture_mm / np.tan(np.radians(float(camera_spec.fov_deg)) / 2.0)
    camera.CreateHorizontalApertureAttr(horizontal_aperture_mm)
    camera.CreateVerticalApertureAttr(horizontal_aperture_mm * float(camera_spec.height_px) / max(float(camera_spec.width_px), 1.0))
    camera.CreateFocalLengthAttr(float(focal_length_mm))
    camera.CreateClippingRangeAttr((0.01, 25.0))


def _create_arm_proxy(
    *,
    stage: Any,
    arm_proxy: StageArmProxySpec,
) -> None:
    from isaacsim.core.api.objects import FixedCuboid
    from pxr import UsdGeom

    UsdGeom.Xform.Define(stage, arm_proxy.prim_path)
    base, shoulder, elbow, wrist, tool = (
        np.asarray(point, dtype=np.float64).reshape(3) for point in arm_proxy.joint_chain_world()
    )
    pedestal_size = np.asarray(arm_proxy.pedestal_size_m, dtype=np.float64)
    joint_size = float(arm_proxy.joint_size_m)
    mast_center = base + np.array([0.0, 0.0, 0.5 * float(arm_proxy.shoulder_height_m)], dtype=np.float64)
    FixedCuboid(
        prim_path=f"{arm_proxy.prim_path}/pedestal",
        name=f"{arm_proxy.name}_pedestal",
        position=base - np.array([0.0, 0.0, 0.5 * pedestal_size[2]], dtype=np.float64),
        scale=pedestal_size,
        color=np.asarray(arm_proxy.base_color_rgb, dtype=np.float64),
        size=1.0,
    )
    FixedCuboid(
        prim_path=f"{arm_proxy.prim_path}/mast",
        name=f"{arm_proxy.name}_mast",
        position=mast_center,
        scale=np.asarray([0.8 * joint_size, 0.8 * joint_size, float(arm_proxy.shoulder_height_m)], dtype=np.float64),
        color=np.asarray(arm_proxy.base_color_rgb, dtype=np.float64),
        size=1.0,
    )
    joint_positions = {
        "base_joint": base,
        "shoulder_joint": shoulder,
        "elbow_joint": elbow,
        "wrist_joint": wrist,
    }
    for joint_name, joint_position in joint_positions.items():
        FixedCuboid(
            prim_path=f"{arm_proxy.prim_path}/{joint_name}",
            name=f"{arm_proxy.name}_{joint_name}",
            position=joint_position,
            scale=np.asarray([joint_size, joint_size, joint_size], dtype=np.float64),
            color=np.asarray(arm_proxy.joint_color_rgb, dtype=np.float64),
            size=1.0,
        )
    link_pairs = ((shoulder, elbow), (elbow, wrist), (wrist, tool))
    for index, (start, end) in enumerate(link_pairs):
        segment = end - start
        length = float(np.linalg.norm(segment))
        if length <= 1.0e-9:
            continue
        FixedCuboid(
            prim_path=f"{arm_proxy.prim_path}/link_{index + 1}",
            name=f"{arm_proxy.name}_link_{index + 1}",
            position=0.5 * (start + end),
            orientation=np.asarray(_segment_rotation_wxyz(start, end), dtype=np.float64),
            scale=np.asarray([length, float(arm_proxy.link_thickness_m), float(arm_proxy.link_thickness_m)], dtype=np.float64),
            color=np.asarray(arm_proxy.link_colors_rgb[min(index, len(arm_proxy.link_colors_rgb) - 1)], dtype=np.float64),
            size=1.0,
        )
    FixedCuboid(
        prim_path=f"{arm_proxy.prim_path}/tool",
        name=f"{arm_proxy.name}_tool",
        position=tool,
        orientation=np.asarray(_segment_rotation_wxyz(wrist, tool), dtype=np.float64),
        scale=np.asarray(arm_proxy.tool_size_m, dtype=np.float64),
        color=np.asarray(arm_proxy.tool_color_rgb, dtype=np.float64),
        size=1.0,
    )


def build_anchor_room_geometry(
    *,
    scene_config: dict[str, Any],
    generated_asset_dir: str | Path,
) -> StageBuildArtifacts:
    """Build the minimum programmatic benchmark scene."""

    stage_spec = stage_spec_from_config(scene_config)
    stage_spec.validate()

    from isaacsim.core.api.objects import FixedCuboid
    from isaacsim.core.utils.prims import is_prim_path_valid
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import UsdGeom, UsdLux

    generated_asset_dir = Path(generated_asset_dir).resolve()
    stage = get_current_stage()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/Tags")
    if not is_prim_path_valid("/World/DomeLight"):
        dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome.CreateIntensityAttr(240.0)
        dome.CreateColorAttr((1.0, 1.0, 1.0))
    if not is_prim_path_valid("/World/KeyLight"):
        key = UsdLux.SphereLight.Define(stage, "/World/KeyLight")
        key.CreateIntensityAttr(2600.0)
        key.CreateRadiusAttr(0.12)
        key.AddTranslateOp().Set((0.0, 0.0, 1.2))
        key.CreateColorAttr((1.0, 1.0, 1.0))

    if stage_spec.scene_boxes:
        for box in stage_spec.scene_boxes:
            if not is_prim_path_valid(str(box.prim_path)):
                _create_scene_box(box=box)
    else:
        table_path = "/World/Table"
        if not is_prim_path_valid(table_path):
            FixedCuboid(
                prim_path=table_path,
                name="table",
                position=np.asarray(stage_spec.table_center_world_m, dtype=np.float64),
                scale=np.asarray(stage_spec.table_scale_m, dtype=np.float64),
                color=np.asarray([0.35, 0.28, 0.20], dtype=np.float64),
                size=1.0,
            )

    for observer_camera in stage_spec.observer_cameras:
        if not is_prim_path_valid(observer_camera.prim_path):
            _create_observer_camera(stage=stage, camera_spec=observer_camera)
    if stage_spec.arm_proxy is not None and not is_prim_path_valid(stage_spec.arm_proxy.prim_path):
        _create_arm_proxy(stage=stage, arm_proxy=stage_spec.arm_proxy)

    pose_specs: dict[int, TagPoseSpec] = {}
    warnings: list[str] = []
    for tag in stage_spec.tags:
        rotation = rotation_matrix_from_rpy_deg(tag.orientation_rpy_deg)
        texture_path = Path(tag.texture_path).resolve() if tag.texture_path else generated_asset_dir / f"apriltag36h11_id{tag.tag_id}.png"
        if not texture_path.exists():
            _generate_tag_texture(int(tag.tag_id), texture_path)
        _create_tag_plane(
            stage=stage,
            prim_path=tag.prim_path,
            tag_id=int(tag.tag_id),
            tag_size_m=float(tag.size_m),
            position_world_m=tag.position_world_m,
            rotation_wt=rotation,
            texture_path=texture_path,
        )
        pose_specs[int(tag.tag_id)] = tag.pose_spec()
        if not texture_path.exists():
            warnings.append(f"Texture asset for tag {tag.tag_id} was generated but could not be verified on disk.")

    return StageBuildArtifacts(stage_spec=stage_spec, tag_pose_specs=pose_specs, warnings=warnings)


__all__ = [
    "IsaacStageSpec",
    "ObserverCameraSpec",
    "StageArmProxySpec",
    "StageBoxSpec",
    "StageBuildArtifacts",
    "StageTagSpec",
    "build_anchor_room_geometry",
    "stage_spec_from_config",
]
