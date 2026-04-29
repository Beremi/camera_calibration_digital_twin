"""Factor-graph style bookkeeping for the batch estimators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from calib_sim.estimation.types import BatchCalibrationDataset, ImuPacket, TagDetectionObservation


@dataclass(slots=True)
class VisualFactorEntry:
    frame_index: int
    tag_id: int
    observed_corners_px: np.ndarray
    tag_size_m: float
    quality: dict[str, Any]


@dataclass(slots=True)
class PosePriorEntry:
    variable_kind: str
    index: int
    mean_vector: np.ndarray
    sqrt_information: np.ndarray


@dataclass(slots=True)
class ImuFactorEntry:
    start_frame_index: int
    end_frame_index: int
    packet_indices: tuple[int, ...]
    dt_s: tuple[float, ...]


@dataclass(slots=True)
class BatchGraphDefinition:
    frame_indices: tuple[int, ...]
    tag_ids: tuple[int, ...]
    visual_factors: tuple[VisualFactorEntry, ...]
    pose_priors: tuple[PosePriorEntry, ...]
    imu_factors: tuple[ImuFactorEntry, ...]


def _camera_frame_indices(dataset: BatchCalibrationDataset) -> tuple[int, ...]:
    return tuple(frame.frame_index for frame in dataset.camera_frames)


def _observed_tag_ids(dataset: BatchCalibrationDataset) -> tuple[int, ...]:
    return tuple(sorted({int(detection.tag_id) for detection in dataset.tag_detections}))


def build_visual_graph(
    dataset: BatchCalibrationDataset,
    *,
    anchor_frame_index: int | None = None,
    first_pose_translation_std_m: float = 1e-4,
    first_pose_rotation_std_rad: float = 1e-4,
) -> BatchGraphDefinition:
    frame_indices = _camera_frame_indices(dataset)
    tag_ids = _observed_tag_ids(dataset)
    visual_factors = tuple(
        VisualFactorEntry(
            frame_index=int(detection.frame_index),
            tag_id=int(detection.tag_id),
            observed_corners_px=np.asarray(detection.corners_xy_clockwise_px, dtype=np.float64).reshape(4, 2),
            tag_size_m=float(detection.physical_edge_length_m),
            quality=dict(detection.quality),
        )
        for detection in dataset.tag_detections
    )
    translation_info = np.eye(3, dtype=np.float64) / max(first_pose_translation_std_m, 1e-9)
    rotation_info = np.eye(3, dtype=np.float64) / max(first_pose_rotation_std_rad, 1e-9)
    sqrt_information = np.block(
        [
            [translation_info, np.zeros((3, 3), dtype=np.float64)],
            [np.zeros((3, 3), dtype=np.float64), rotation_info],
        ]
    )
    pose_priors = (
        PosePriorEntry(
            variable_kind="camera_pose",
            index=int(frame_indices[0] if anchor_frame_index is None else anchor_frame_index),
            mean_vector=np.zeros(6, dtype=np.float64),
            sqrt_information=sqrt_information,
        ),
    )
    return BatchGraphDefinition(
        frame_indices=frame_indices,
        tag_ids=tag_ids,
        visual_factors=visual_factors,
        pose_priors=pose_priors,
        imu_factors=(),
    )


def build_visual_inertial_graph(
    dataset: BatchCalibrationDataset,
    *,
    anchor_frame_index: int | None = None,
    first_pose_translation_std_m: float = 1e-4,
    first_pose_rotation_std_rad: float = 1e-4,
    first_velocity_std_mps: float = 0.1,
    first_bias_std: float = 0.01,
    first_pose_mean_vector: np.ndarray | None = None,
    first_velocity_mean_mps: np.ndarray | None = None,
) -> BatchGraphDefinition:
    base_graph = build_visual_graph(
        dataset,
        anchor_frame_index=anchor_frame_index,
        first_pose_translation_std_m=first_pose_translation_std_m,
        first_pose_rotation_std_rad=first_pose_rotation_std_rad,
    )
    imu_packets = tuple(dataset.imu_packets)
    packet_timestamps = np.array([packet.timestamp_s for packet in imu_packets], dtype=np.float64)
    imu_factors: list[ImuFactorEntry] = []
    for start_frame, end_frame in zip(dataset.camera_frames[:-1], dataset.camera_frames[1:]):
        start_time_s = float(start_frame.timestamp_s)
        end_time_s = float(end_frame.timestamp_s)
        in_segment = np.where((packet_timestamps > start_time_s + 1e-12) & (packet_timestamps <= end_time_s + 1e-12))[0]
        if in_segment.size == 0:
            continue
        last_time_s = start_time_s
        dt_s = []
        for packet_index in in_segment.tolist():
            packet = imu_packets[packet_index]
            current_time_s = float(packet.timestamp_s)
            dt_s.append(max(current_time_s - last_time_s, 1e-9))
            last_time_s = current_time_s
        imu_factors.append(
            ImuFactorEntry(
                start_frame_index=int(start_frame.frame_index),
                end_frame_index=int(end_frame.frame_index),
                packet_indices=tuple(int(index) for index in in_segment.tolist()),
                dt_s=tuple(float(value) for value in dt_s),
            )
        )

    first_state_sqrt_information = np.diag(
        [
            1.0 / max(first_pose_translation_std_m, 1e-9),
            1.0 / max(first_pose_translation_std_m, 1e-9),
            1.0 / max(first_pose_translation_std_m, 1e-9),
            1.0 / max(first_pose_rotation_std_rad, 1e-9),
            1.0 / max(first_pose_rotation_std_rad, 1e-9),
            1.0 / max(first_pose_rotation_std_rad, 1e-9),
            1.0 / max(first_velocity_std_mps, 1e-9),
            1.0 / max(first_velocity_std_mps, 1e-9),
            1.0 / max(first_velocity_std_mps, 1e-9),
        ]
    )
    global_bias_sqrt_information = np.diag(
        [
            1.0 / max(first_bias_std, 1e-9),
            1.0 / max(first_bias_std, 1e-9),
            1.0 / max(first_bias_std, 1e-9),
            1.0 / max(first_bias_std, 1e-9),
            1.0 / max(first_bias_std, 1e-9),
            1.0 / max(first_bias_std, 1e-9),
        ]
    )
    pose_priors = base_graph.pose_priors + (
        PosePriorEntry(
            variable_kind="imu_state",
            index=int(base_graph.frame_indices[0] if anchor_frame_index is None else anchor_frame_index),
            mean_vector=np.concatenate(
                (
                    np.asarray(first_pose_mean_vector if first_pose_mean_vector is not None else np.zeros(6, dtype=np.float64), dtype=np.float64).reshape(6),
                    np.asarray(first_velocity_mean_mps if first_velocity_mean_mps is not None else np.zeros(3, dtype=np.float64), dtype=np.float64).reshape(3),
                ),
                axis=0,
            ),
            sqrt_information=first_state_sqrt_information,
        ),
        PosePriorEntry(
            variable_kind="imu_global_bias",
            index=0,
            mean_vector=np.zeros(6, dtype=np.float64),
            sqrt_information=global_bias_sqrt_information,
        ),
    )
    return BatchGraphDefinition(
        frame_indices=base_graph.frame_indices,
        tag_ids=base_graph.tag_ids,
        visual_factors=base_graph.visual_factors,
        pose_priors=pose_priors,
        imu_factors=tuple(imu_factors),
    )
