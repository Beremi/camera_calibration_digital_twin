"""Static stage description helpers and programmatic scene building."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from calib_sim.estimation._geometry import rotation_matrix_from_rpy_deg
from calib_sim.isaac.tag_builder import TagPoseSpec


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


def _tag_world_corners_xy(tag: "StageTagSpec") -> np.ndarray:
    rotation = rotation_matrix_from_rpy_deg(tag.orientation_rpy_deg)
    half = 0.5 * float(tag.size_m)
    local_corners = np.array(
        [
            [-half, -half, 0.0],
            [half, -half, 0.0],
            [half, half, 0.0],
            [-half, half, 0.0],
        ],
        dtype=np.float64,
    )
    position = np.asarray(tag.position_world_m, dtype=np.float64).reshape(3)
    world_corners = (rotation @ local_corners.T).T + position
    return np.asarray(world_corners[:, :2], dtype=np.float64)


def _convex_polygons_overlap_xy(lhs: np.ndarray, rhs: np.ndarray) -> bool:
    def _axes(points: np.ndarray) -> list[np.ndarray]:
        axes: list[np.ndarray] = []
        for start, end in zip(points, np.roll(points, -1, axis=0)):
            edge = np.asarray(end - start, dtype=np.float64)
            normal = np.asarray([-edge[1], edge[0]], dtype=np.float64)
            norm = float(np.linalg.norm(normal))
            if norm > 1e-12:
                axes.append(normal / norm)
        return axes

    for axis in _axes(lhs) + _axes(rhs):
        lhs_projection = lhs @ axis
        rhs_projection = rhs @ axis
        if float(np.max(lhs_projection)) <= float(np.min(rhs_projection)) + 1e-9:
            return False
        if float(np.max(rhs_projection)) <= float(np.min(lhs_projection)) + 1e-9:
            return False
    return True


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
class IsaacStageSpec:
    """Repository-level description of the intended Isaac stage."""

    stage_path: str
    robot_prim_path: str
    camera_prim_path: str
    imu_prim_path: str
    physics_rate_hz: float
    observer_camera_prim_paths: tuple[str, ...] = ()
    tags: tuple[StageTagSpec, ...] = field(default_factory=tuple)
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
            lhs_xy = _tag_world_corners_xy(lhs)
            for rhs in self.tags[index + 1 :]:
                rhs_xy = _tag_world_corners_xy(rhs)
                if _convex_polygons_overlap_xy(lhs_xy, rhs_xy):
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
    spec = IsaacStageSpec(
        stage_path=str(scene_config.get("stage_path", "")),
        robot_prim_path=str(scene_config["robot_prim_path"]),
        camera_prim_path=str(scene_config["camera_prim_path"]),
        imu_prim_path=str(scene_config["imu_prim_path"]),
        physics_rate_hz=float(scene_config["physics_rate_hz"]),
        observer_camera_prim_paths=tuple(str(value) for value in scene_config.get("observer_camera_prim_paths", [])),
        tags=tuple(tags),
        table_center_world_m=tuple(float(value) for value in scene_config.get("table_center_world_m", (0.0, 0.0, -0.025))),
        table_scale_m=tuple(float(value) for value in scene_config.get("table_scale_m", (0.7, 0.7, 0.05))),
        room_center_world_m=tuple(float(value) for value in scene_config.get("room_center_world_m", (0.0, 0.0, 0.5))),
        room_scale_m=tuple(float(value) for value in scene_config.get("room_scale_m", (2.0, 2.0, 1.0))),
    )
    spec.validate()
    return spec


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
    "StageBuildArtifacts",
    "StageTagSpec",
    "build_anchor_room_geometry",
    "stage_spec_from_config",
]
