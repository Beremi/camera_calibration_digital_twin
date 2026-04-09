"""Figure generation for Isaac reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _float_or_none(value: Any) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def _vector_or_none(value: Any) -> np.ndarray | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    return np.asarray([float(value[0]), float(value[1]), float(value[2])], dtype=np.float64)


def _position_from_gt_row(row: dict[str, Any]) -> np.ndarray | None:
    required_keys = ("px", "py", "pz")
    if all(key in row for key in required_keys):
        values = [_float_or_none(row[key]) for key in required_keys]
        if all(value is not None for value in values):
            return np.asarray(values, dtype=np.float64)
    return None


def _save_text_figure(path: Path, *, title: str, lines: list[str]) -> None:
    canvas = np.full((420, 1080, 3), 252, dtype=np.uint8)
    cv2.putText(canvas, title, (28, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (22, 26, 32), 2, cv2.LINE_AA)
    y = 90
    for line in lines:
        cv2.putText(canvas, line[:110], (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (62, 66, 72), 1, cv2.LINE_AA)
        y += 34
    cv2.imwrite(str(path), canvas)


def _plot_canvas(title: str, subtitle: str | None = None) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    canvas = np.full((420, 1080, 3), 252, dtype=np.uint8)
    left, right, top, bottom = 90, 40, 60, 70
    cv2.putText(canvas, title, (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (22, 26, 32), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(canvas, subtitle, (24, 404), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (82, 90, 98), 1, cv2.LINE_AA)
    origin_x = left
    origin_y = 420 - bottom
    cv2.line(canvas, (origin_x, top), (origin_x, origin_y), (160, 166, 172), 1, cv2.LINE_AA)
    cv2.line(canvas, (origin_x, origin_y), (1080 - right, origin_y), (160, 166, 172), 1, cv2.LINE_AA)
    return canvas, (left, right, top, bottom)


def _save_line_plot(path: Path, *, title: str, x_values: list[float], y_values: list[float], y_label: str) -> None:
    if not x_values or not y_values or len(x_values) != len(y_values):
        _save_text_figure(path, title=title, lines=["No data available for this figure."])
        return
    xs = np.asarray(x_values, dtype=np.float64)
    ys = np.asarray(y_values, dtype=np.float64)
    finite = np.isfinite(xs) & np.isfinite(ys)
    if not np.any(finite):
        _save_text_figure(path, title=title, lines=["No finite samples available."])
        return
    xs = xs[finite]
    ys = ys[finite]
    canvas, (left, right, top, bottom) = _plot_canvas(title, y_label)
    origin_x = left
    origin_y = 420 - bottom
    plot_width = 1080 - left - right
    plot_height = 420 - top - bottom
    min_x, max_x = float(np.min(xs)), float(np.max(xs))
    min_y, max_y = float(np.min(ys)), float(np.max(ys))
    if abs(max_x - min_x) < 1e-12:
        max_x = min_x + 1.0
    if abs(max_y - min_y) < 1e-12:
        max_y = min_y + 1.0
    points = []
    for x, y in zip(xs, ys):
        x_px = origin_x + int(round((x - min_x) / (max_x - min_x) * plot_width))
        y_px = origin_y - int(round((y - min_y) / (max_y - min_y) * plot_height))
        points.append((x_px, y_px))
    cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, (53, 102, 188), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def _save_xy_plot(
    path: Path,
    *,
    title: str,
    tracks: list[tuple[list[float], list[float], tuple[int, int, int]]],
    point_sets: list[tuple[list[tuple[float, float]], tuple[int, int, int]]] | None = None,
    footer: str,
) -> None:
    valid_tracks = [(xs, ys, color) for xs, ys, color in tracks if xs and ys and len(xs) == len(ys)]
    valid_points = point_sets or []
    all_x = [value for xs, _, _ in valid_tracks for value in xs] + [x for points, _ in valid_points for x, _ in points]
    all_y = [value for _, ys, _ in valid_tracks for value in ys] + [y for points, _ in valid_points for _, y in points]
    if not all_x or not all_y:
        _save_text_figure(path, title=title, lines=["No planar trajectory data available."])
        return

    canvas, (left, right, top, bottom) = _plot_canvas(title, footer)
    origin_x = left
    origin_y = 420 - bottom
    plot_width = 1080 - left - right
    plot_height = 420 - top - bottom
    min_x, max_x = float(min(all_x)), float(max(all_x))
    min_y, max_y = float(min(all_y)), float(max(all_y))
    if abs(max_x - min_x) < 1e-12:
        max_x = min_x + 1.0
    if abs(max_y - min_y) < 1e-12:
        max_y = min_y + 1.0

    def project(x_value: float, y_value: float) -> tuple[int, int]:
        x_px = origin_x + int(round((x_value - min_x) / (max_x - min_x) * plot_width))
        y_px = origin_y - int(round((y_value - min_y) / (max_y - min_y) * plot_height))
        return x_px, y_px

    for xs, ys, color in valid_tracks:
        points = [project(x_value, y_value) for x_value, y_value in zip(xs, ys)]
        cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
    for points, color in valid_points:
        for x_value, y_value in points:
            cv2.circle(canvas, project(x_value, y_value), 5, color, -1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def _save_histogram(path: Path, *, title: str, values: list[float], x_label: str) -> None:
    if not values:
        _save_text_figure(path, title=title, lines=["No histogram samples available."])
        return
    samples = np.asarray(values, dtype=np.float64)
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        _save_text_figure(path, title=title, lines=["No finite histogram samples available."])
        return
    counts, edges = np.histogram(finite, bins=min(12, max(4, finite.size)))
    canvas, (left, right, top, bottom) = _plot_canvas(title, x_label)
    origin_x = left
    origin_y = 420 - bottom
    plot_width = 1080 - left - right
    plot_height = 420 - top - bottom
    max_count = max(int(np.max(counts)), 1)
    bin_width = plot_width / len(counts)
    for index, count in enumerate(counts):
        x0 = int(origin_x + index * bin_width)
        x1 = int(origin_x + (index + 1) * bin_width - 4)
        y1 = origin_y
        y0 = origin_y - int(round(count / max_count * plot_height))
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (53, 102, 188), -1)
    cv2.imwrite(str(path), canvas)


def _draw_polyline(
    canvas: np.ndarray,
    points: list[tuple[int, int]],
    *,
    color: tuple[int, int, int],
    thickness: int = 2,
    dashed: bool = False,
) -> None:
    if len(points) < 2:
        return
    if not dashed:
        cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)
        return
    for index in range(len(points) - 1):
        if index % 2 == 1:
            continue
        cv2.line(canvas, points[index], points[index + 1], color, thickness, cv2.LINE_AA)


def _save_stacked_series_figure(
    path: Path,
    *,
    title: str,
    subplots: list[dict[str, Any]],
    footer: str,
) -> None:
    if not subplots:
        _save_text_figure(path, title=title, lines=["No subplot definitions were provided."])
        return

    subplot_height = 210
    canvas_height = 78 + subplot_height * len(subplots) + 54
    canvas_width = 1280
    canvas = np.full((canvas_height, canvas_width, 3), 252, dtype=np.uint8)
    cv2.putText(canvas, title, (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.92, (22, 26, 32), 2, cv2.LINE_AA)
    cv2.putText(canvas, footer, (24, canvas_height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (82, 90, 98), 1, cv2.LINE_AA)

    for subplot_index, subplot in enumerate(subplots):
        top = 58 + subplot_index * subplot_height
        left = 88
        right = 26
        bottom = 44
        plot_height = subplot_height - bottom - 18
        plot_width = canvas_width - left - right
        origin_x = left
        origin_y = top + plot_height
        cv2.rectangle(canvas, (origin_x, top), (origin_x + plot_width, origin_y), (226, 230, 236), 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            str(subplot.get("title", f"subplot {subplot_index + 1}")),
            (origin_x, top - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.64,
            (22, 26, 32),
            1,
            cv2.LINE_AA,
        )
        cv2.line(canvas, (origin_x, top), (origin_x, origin_y), (160, 166, 172), 1, cv2.LINE_AA)
        cv2.line(canvas, (origin_x, origin_y), (origin_x + plot_width, origin_y), (160, 166, 172), 1, cv2.LINE_AA)

        series = list(subplot.get("series", []))
        finite_x: list[float] = []
        finite_y: list[float] = []
        for item in series:
            xs = np.asarray(item.get("x_values", []), dtype=np.float64)
            ys = np.asarray(item.get("y_values", []), dtype=np.float64)
            if xs.size == 0 or ys.size == 0 or xs.size != ys.size:
                continue
            finite = np.isfinite(xs) & np.isfinite(ys)
            if not np.any(finite):
                continue
            finite_x.extend(xs[finite].tolist())
            finite_y.extend(ys[finite].tolist())
        if not finite_x or not finite_y:
            cv2.putText(
                canvas,
                "No finite samples available.",
                (origin_x + 12, top + 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (96, 100, 106),
                1,
                cv2.LINE_AA,
            )
            continue

        min_x, max_x = float(min(finite_x)), float(max(finite_x))
        min_y, max_y = float(min(finite_y)), float(max(finite_y))
        if abs(max_x - min_x) < 1e-12:
            max_x = min_x + 1.0
        if abs(max_y - min_y) < 1e-12:
            max_y = min_y + 1.0

        def project(x_value: float, y_value: float) -> tuple[int, int]:
            x_px = origin_x + int(round((x_value - min_x) / (max_x - min_x) * plot_width))
            y_px = origin_y - int(round((y_value - min_y) / (max_y - min_y) * plot_height))
            return x_px, y_px

        legend_x = origin_x + 8
        legend_y = top + 18
        for item in series:
            xs = np.asarray(item.get("x_values", []), dtype=np.float64)
            ys = np.asarray(item.get("y_values", []), dtype=np.float64)
            if xs.size == 0 or ys.size == 0 or xs.size != ys.size:
                continue
            finite = np.isfinite(xs) & np.isfinite(ys)
            if not np.any(finite):
                continue
            points = [project(float(x_value), float(y_value)) for x_value, y_value in zip(xs[finite], ys[finite])]
            color = tuple(int(value) for value in item.get("color_bgr", (53, 102, 188)))
            dashed = bool(item.get("dashed", False))
            _draw_polyline(canvas, points, color=color, thickness=2, dashed=dashed)
            cv2.line(canvas, (legend_x, legend_y), (legend_x + 18, legend_y), color, 2, cv2.LINE_AA)
            cv2.putText(
                canvas,
                str(item.get("label", "series")),
                (legend_x + 24, legend_y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (48, 52, 58),
                1,
                cv2.LINE_AA,
            )
            legend_x += 160
            if legend_x > canvas_width - 200:
                legend_x = origin_x + 8
                legend_y += 18

        cv2.putText(
            canvas,
            str(subplot.get("y_label", "")),
            (18, top + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (96, 100, 106),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(path), canvas)


def _rotation_matrix_from_rvec(value: Any) -> np.ndarray | None:
    rvec = _vector_or_none(value)
    if rvec is None:
        return None
    rotation, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    return np.asarray(rotation, dtype=np.float64).reshape(3, 3)


def _rotation_matrix_or_none(value: Any) -> np.ndarray | None:
    if value in ("", None):
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3, 3):
        return None
    return array


def _load_tag_truth_lookup(path: Path) -> dict[int, dict[str, np.ndarray]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    lookup: dict[int, dict[str, np.ndarray]] = {}
    for row in payload.get("tags", []):
        if not isinstance(row, dict):
            continue
        tag_id = row.get("tag_id")
        position = _vector_or_none(row.get("position_world_m"))
        rotation = _rotation_matrix_or_none(row.get("rotation_wt"))
        if tag_id is None or position is None:
            continue
        lookup[int(tag_id)] = {
            "position_world_m": position,
            "rotation_wt": rotation if rotation is not None else np.eye(3, dtype=np.float64),
        }
    return lookup


def _camera_frame_records(run_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in _read_jsonl_rows(run_dir / "raw" / "camera_frames.jsonl"):
        timestamp = _float_or_none(row.get("timestamp_s"))
        rgb_path = row.get("rgb_path")
        if timestamp is None or not isinstance(rgb_path, str) or not rgb_path:
            continue
        records.append(
            {
                "timestamp_s": float(timestamp),
                "frame_index": int(row.get("frame_index", len(records))),
                "image_path": run_dir / rgb_path,
            }
        )
    return records


def _nearest_record_index(times: np.ndarray, target_time_s: float) -> int | None:
    if times.size == 0 or not np.isfinite(target_time_s):
        return None
    return int(np.argmin(np.abs(times - float(target_time_s))))


def _load_image(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return cv2.imread(str(path))


def _save_image_grid_montage(
    path: Path,
    *,
    title: str,
    labeled_images: list[tuple[str, np.ndarray]],
) -> None:
    valid = [(label, image) for label, image in labeled_images if image is not None]
    if not valid:
        _save_text_figure(path, title=title, lines=["No RGB frames were available for this montage."])
        return
    columns = min(3, len(valid))
    rows = (len(valid) + columns - 1) // columns
    tile_width = 360
    tile_height = 220
    canvas = np.full((80 + rows * (tile_height + 52), 24 + columns * (tile_width + 18), 3), 248, dtype=np.uint8)
    cv2.putText(canvas, title, (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (22, 26, 32), 2, cv2.LINE_AA)
    for index, (label, image) in enumerate(valid):
        row = index // columns
        column = index % columns
        x0 = 20 + column * (tile_width + 18)
        y0 = 56 + row * (tile_height + 52)
        resized = cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_LINEAR)
        canvas[y0 : y0 + tile_height, x0 : x0 + tile_width] = resized
        cv2.rectangle(canvas, (x0, y0), (x0 + tile_width, y0 + tile_height), (210, 214, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, label, (x0, y0 + tile_height + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (48, 52, 58), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def _save_sim_stills(
    run_dir: Path,
    *,
    output_first_frame_path: Path,
    output_waypoint_montage_path: Path,
    controller_rows: list[dict[str, str]],
    waypoint_count: int,
) -> None:
    frame_records = _camera_frame_records(run_dir)
    if not frame_records:
        _save_text_figure(output_first_frame_path, title="Representative Simulation First Frame", lines=["No RGB frames were logged for this run."])
        _save_text_figure(output_waypoint_montage_path, title="Simulation Waypoint Arrivals", lines=["No RGB frames were logged for this run."])
        return

    first_frame = _load_image(Path(frame_records[0]["image_path"]))
    if first_frame is None:
        _save_text_figure(output_first_frame_path, title="Representative Simulation First Frame", lines=["The first RGB frame could not be decoded."])
    else:
        cv2.imwrite(str(output_first_frame_path), first_frame)

    controller_times: list[float] = []
    arrival_labels: list[str] = []
    previous_index: int | None = None
    for row in controller_rows:
        timestamp = _float_or_none(row.get("timestamp_s"))
        raw_index = row.get("waypoint_index")
        if timestamp is None or raw_index in ("", None):
            continue
        waypoint_index = int(raw_index)
        if previous_index is None:
            previous_index = waypoint_index
            continue
        if waypoint_index <= previous_index:
            continue
        for completed_index in range(previous_index + 1, min(waypoint_index, waypoint_count) + 1):
            controller_times.append(float(timestamp))
            arrival_labels.append(f"reach point {completed_index}")
        previous_index = waypoint_index

    frame_times = np.asarray([float(row["timestamp_s"]) for row in frame_records], dtype=np.float64)
    labeled_images: list[tuple[str, np.ndarray]] = []
    if first_frame is not None:
        labeled_images.append(("first frame", first_frame))
    for label, timestamp in zip(arrival_labels, controller_times):
        index = _nearest_record_index(frame_times, timestamp)
        if index is None:
            continue
        image = _load_image(Path(frame_records[index]["image_path"]))
        if image is not None:
            labeled_images.append((label, image))
    _save_image_grid_montage(output_waypoint_montage_path, title="Simulation Frames Along The Anchored Path", labeled_images=labeled_images)


def _save_position_estimation_figure(
    path: Path,
    *,
    filter_rows: list[dict[str, Any]],
    camera_gt_rows: list[dict[str, str]],
) -> None:
    gt_samples: list[tuple[float, np.ndarray]] = []
    for row in camera_gt_rows:
        timestamp = _float_or_none(row.get("timestamp_s"))
        position = _position_from_gt_row(row)
        if timestamp is not None and position is not None:
            gt_samples.append((float(timestamp), position))
    filter_samples: list[tuple[float, np.ndarray]] = []
    for row in filter_rows:
        timestamp = _float_or_none(row.get("timestamp_s"))
        position = _vector_or_none(row.get("position_world_m"))
        if timestamp is not None and position is not None:
            filter_samples.append((float(timestamp), position))
    if not gt_samples or not filter_samples:
        _save_text_figure(path, title="Position Estimation Timeline", lines=["Filter states and camera ground truth are both required for this figure."])
        return

    gt_times = np.asarray([item[0] for item in gt_samples], dtype=np.float64)
    times = [item[0] for item in filter_samples]
    estimate_components = {
        "x": [float(item[1][0]) for item in filter_samples],
        "y": [float(item[1][1]) for item in filter_samples],
        "z": [float(item[1][2]) for item in filter_samples],
    }
    gt_components = {"x": [], "y": [], "z": []}
    for timestamp in times:
        index = _nearest_record_index(gt_times, timestamp)
        if index is None:
            gt_components["x"].append(float("nan"))
            gt_components["y"].append(float("nan"))
            gt_components["z"].append(float("nan"))
            continue
        position = gt_samples[index][1]
        gt_components["x"].append(float(position[0]))
        gt_components["y"].append(float(position[1]))
        gt_components["z"].append(float(position[2]))
    colors = {
        "estimate": (53, 102, 188),
        "ground_truth": (188, 92, 60),
    }
    _save_stacked_series_figure(
        path,
        title="Camera Position Estimation In Time",
        subplots=[
            {
                "title": "world x",
                "y_label": "m",
                "series": [
                    {"label": "estimate", "x_values": times, "y_values": estimate_components["x"], "color_bgr": colors["estimate"]},
                    {
                        "label": "ground truth",
                        "x_values": times,
                        "y_values": gt_components["x"],
                        "color_bgr": colors["ground_truth"],
                        "dashed": True,
                    },
                ],
            },
            {
                "title": "world y",
                "y_label": "m",
                "series": [
                    {"label": "estimate", "x_values": times, "y_values": estimate_components["y"], "color_bgr": colors["estimate"]},
                    {
                        "label": "ground truth",
                        "x_values": times,
                        "y_values": gt_components["y"],
                        "color_bgr": colors["ground_truth"],
                        "dashed": True,
                    },
                ],
            },
            {
                "title": "world z",
                "y_label": "m",
                "series": [
                    {"label": "estimate", "x_values": times, "y_values": estimate_components["z"], "color_bgr": colors["estimate"]},
                    {
                        "label": "ground truth",
                        "x_values": times,
                        "y_values": gt_components["z"],
                        "color_bgr": colors["ground_truth"],
                        "dashed": True,
                    },
                ],
            },
        ],
        footer="Estimated and ground-truth camera position components in the anchored world frame.",
    )


def _save_imu_measurements_figure(path: Path, *, imu_rows: list[dict[str, str]]) -> None:
    times: list[float] = []
    gyro_components = {"wx": [], "wy": [], "wz": []}
    accel_components = {"ax": [], "ay": [], "az": []}
    for row in imu_rows:
        timestamp = _float_or_none(row.get("timestamp_s"))
        if timestamp is None:
            continue
        gyro = [_float_or_none(row.get(key)) for key in ("wx", "wy", "wz")]
        accel = [_float_or_none(row.get(key)) for key in ("ax", "ay", "az")]
        if any(value is None for value in gyro + accel):
            continue
        times.append(float(timestamp))
        for key, value in zip(("wx", "wy", "wz"), gyro):
            gyro_components[key].append(float(value))
        for key, value in zip(("ax", "ay", "az"), accel):
            accel_components[key].append(float(value))
    if not times:
        _save_text_figure(path, title="Measured IMU Signals", lines=["No IMU packet stream was available for this figure."])
        return
    _save_stacked_series_figure(
        path,
        title="Measured IMU Signals",
        subplots=[
            {
                "title": "angular velocity",
                "y_label": "rad/s",
                "series": [
                    {"label": "wx", "x_values": times, "y_values": gyro_components["wx"], "color_bgr": (53, 102, 188)},
                    {"label": "wy", "x_values": times, "y_values": gyro_components["wy"], "color_bgr": (92, 176, 101)},
                    {"label": "wz", "x_values": times, "y_values": gyro_components["wz"], "color_bgr": (188, 92, 60)},
                ],
            },
            {
                "title": "specific force",
                "y_label": "m/s^2",
                "series": [
                    {"label": "ax", "x_values": times, "y_values": accel_components["ax"], "color_bgr": (53, 102, 188)},
                    {"label": "ay", "x_values": times, "y_values": accel_components["ay"], "color_bgr": (92, 176, 101)},
                    {"label": "az", "x_values": times, "y_values": accel_components["az"], "color_bgr": (188, 92, 60)},
                ],
            },
        ],
        footer="Raw measured IMU packets logged by the Isaac runtime.",
    )


def _save_pattern_world_positions_figure(
    path: Path,
    *,
    filter_rows: list[dict[str, Any]],
    detection_rows: list[dict[str, Any]],
    tag_truth_lookup: dict[int, dict[str, np.ndarray]],
) -> None:
    filter_samples: list[tuple[float, np.ndarray, np.ndarray]] = []
    for row in filter_rows:
        timestamp = _float_or_none(row.get("timestamp_s"))
        position = _vector_or_none(row.get("position_world_m"))
        rotation = _rotation_matrix_or_none(row.get("rotation_wi"))
        if timestamp is None or position is None or rotation is None:
            continue
        filter_samples.append((float(timestamp), position, rotation))
    if not filter_samples or not detection_rows or not tag_truth_lookup:
        _save_text_figure(path, title="Calibration Pattern World Positions", lines=["Filter poses, detections, and tag ground truth are required for this figure."])
        return

    filter_times = np.asarray([item[0] for item in filter_samples], dtype=np.float64)
    by_tag: dict[int, dict[str, list[float]]] = {}
    for row in detection_rows:
        timestamp = _float_or_none(row.get("timestamp_s"))
        tag_id = row.get("tag_id")
        translation_ct = _vector_or_none(row.get("pose_camera_tvec_m"))
        if timestamp is None or tag_id in ("", None) or translation_ct is None:
            continue
        tag_id_int = int(tag_id)
        if tag_id_int not in tag_truth_lookup:
            continue
        index = _nearest_record_index(filter_times, float(timestamp))
        if index is None:
            continue
        camera_position_world, rotation_wc = filter_samples[index][1], filter_samples[index][2]
        tag_position_world = camera_position_world + rotation_wc @ translation_ct
        bucket = by_tag.setdefault(
            tag_id_int,
            {
                "times": [],
                "x": [],
                "y": [],
                "z": [],
            },
        )
        bucket["times"].append(float(timestamp))
        bucket["x"].append(float(tag_position_world[0]))
        bucket["y"].append(float(tag_position_world[1]))
        bucket["z"].append(float(tag_position_world[2]))
    if not by_tag:
        _save_text_figure(path, title="Calibration Pattern World Positions", lines=["No pose-ready detections were available for the calibration patterns."])
        return

    subplots: list[dict[str, Any]] = []
    component_colors = {"x": (53, 102, 188), "y": (92, 176, 101), "z": (188, 92, 60)}
    reference_colors = {"x": (132, 164, 220), "y": (150, 204, 156), "z": (220, 156, 132)}
    for tag_id in sorted(by_tag):
        gt_position = tag_truth_lookup[tag_id]["position_world_m"]
        bucket = by_tag[tag_id]
        subplots.append(
            {
                "title": f"tag {tag_id} estimated world position",
                "y_label": "m",
                "series": [
                    {"label": "x estimate", "x_values": bucket["times"], "y_values": bucket["x"], "color_bgr": component_colors["x"]},
                    {"label": "y estimate", "x_values": bucket["times"], "y_values": bucket["y"], "color_bgr": component_colors["y"]},
                    {"label": "z estimate", "x_values": bucket["times"], "y_values": bucket["z"], "color_bgr": component_colors["z"]},
                    {
                        "label": "x gt",
                        "x_values": bucket["times"],
                        "y_values": [float(gt_position[0])] * len(bucket["times"]),
                        "color_bgr": reference_colors["x"],
                        "dashed": True,
                    },
                    {
                        "label": "y gt",
                        "x_values": bucket["times"],
                        "y_values": [float(gt_position[1])] * len(bucket["times"]),
                        "color_bgr": reference_colors["y"],
                        "dashed": True,
                    },
                    {
                        "label": "z gt",
                        "x_values": bucket["times"],
                        "y_values": [float(gt_position[2])] * len(bucket["times"]),
                        "color_bgr": reference_colors["z"],
                        "dashed": True,
                    },
                ],
            }
        )
    _save_stacked_series_figure(
        path,
        title="Calibration Pattern Position Estimates In Time",
        subplots=subplots,
        footer="Each subplot shows the world-position estimate recovered for one tag from detections and the camera state estimate.",
    )


def _save_pattern_relative_camera_figure(path: Path, *, detection_rows: list[dict[str, Any]]) -> None:
    by_tag: dict[int, dict[str, list[float]]] = {}
    for row in detection_rows:
        timestamp = _float_or_none(row.get("timestamp_s"))
        tag_id = row.get("tag_id")
        rotation_ct = _rotation_matrix_from_rvec(row.get("pose_camera_rvec"))
        translation_ct = _vector_or_none(row.get("pose_camera_tvec_m"))
        if timestamp is None or tag_id in ("", None) or rotation_ct is None or translation_ct is None:
            continue
        camera_position_tag = -rotation_ct.T @ translation_ct
        bucket = by_tag.setdefault(
            int(tag_id),
            {
                "times": [],
                "x": [],
                "y": [],
                "z": [],
            },
        )
        bucket["times"].append(float(timestamp))
        bucket["x"].append(float(camera_position_tag[0]))
        bucket["y"].append(float(camera_position_tag[1]))
        bucket["z"].append(float(camera_position_tag[2]))
    if not by_tag:
        _save_text_figure(path, title="Relative Camera Position To Each Pattern", lines=["Pose-ready detections are required for this figure."])
        return
    subplots = [
        {
            "title": f"tag {tag_id} camera position in tag frame",
            "y_label": "m",
            "series": [
                {"label": "x", "x_values": bucket["times"], "y_values": bucket["x"], "color_bgr": (53, 102, 188)},
                {"label": "y", "x_values": bucket["times"], "y_values": bucket["y"], "color_bgr": (92, 176, 101)},
                {"label": "z", "x_values": bucket["times"], "y_values": bucket["z"], "color_bgr": (188, 92, 60)},
            ],
        }
        for tag_id, bucket in sorted(by_tag.items())
    ]
    _save_stacked_series_figure(
        path,
        title="Relative Camera Position To Each Calibration Pattern",
        subplots=subplots,
        footer="Per-tag camera position recovered from solvePnP tag detections, expressed in each tag frame.",
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _first_realized_position(row: dict[str, Any]) -> float | None:
    raw = row.get("positions")
    if raw in ("", None):
        return None
    return _float_or_none(str(raw).split("|")[0])


def _first_command_value(row: dict[str, Any]) -> float | None:
    raw_value = _float_or_none(row.get("command_value"))
    if raw_value is not None:
        return raw_value
    for key in ("desired_positions", "effective_positions"):
        raw = row.get(key)
        if raw in ("", None):
            continue
        value = _float_or_none(str(raw).split("|")[0])
        if value is not None:
            return value
    return None


def _matched_position_error_series(
    filter_rows: list[dict[str, Any]],
    gt_rows: list[dict[str, Any]],
    uncertainty_rows: list[dict[str, Any]],
) -> tuple[list[float], list[float], list[float]]:
    gt_samples: list[tuple[float, np.ndarray]] = []
    for row in gt_rows:
        timestamp = _float_or_none(row.get("timestamp_s"))
        position = _position_from_gt_row(row)
        if timestamp is not None and position is not None:
            gt_samples.append((timestamp, position))
    if not gt_samples:
        return [], [], []
    gt_times = np.asarray([item[0] for item in gt_samples], dtype=np.float64)
    radii = {
        float(row["timestamp_s"]): float(row["position_radius_95_m"])
        for row in uncertainty_rows
        if "timestamp_s" in row and "position_radius_95_m" in row
    }
    times: list[float] = []
    errors: list[float] = []
    radius_values: list[float] = []
    for row in filter_rows:
        timestamp = _float_or_none(row.get("timestamp_s"))
        position = _vector_or_none(row.get("position_world_m"))
        if timestamp is None or position is None:
            continue
        gt_index = int(np.argmin(np.abs(gt_times - timestamp)))
        error = float(np.linalg.norm(position - gt_samples[gt_index][1]))
        radius = radii.get(float(timestamp))
        if radius is None:
            continue
        times.append(float(timestamp))
        errors.append(error)
        radius_values.append(float(radius))
    return times, errors, radius_values


def write_isaac_report_figures(run_dir: str | Path, metrics: dict[str, Any]) -> dict[str, str]:
    resolved = Path(run_dir).resolve()
    analysis_dir = resolved / "analysis"
    report_data_dir = analysis_dir / "report_data"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    report_data_dir.mkdir(parents=True, exist_ok=True)

    filter_rows = _read_jsonl_rows(resolved / "estimates" / "filter_state.jsonl")
    uncertainty_rows = _read_jsonl_rows(resolved / "estimates" / "uncertainty.jsonl")
    detection_rows = _read_jsonl_rows(resolved / "raw" / "detections.jsonl")
    command_rows = _read_csv_rows(resolved / "raw" / "commands.csv")
    controller_rows = _read_csv_rows(resolved / "raw" / "controller_diagnostics.csv")
    imu_rows = _read_csv_rows(resolved / "raw" / "imu.csv")
    realized_rows = _read_csv_rows(resolved / "raw" / "realized_joints.csv")
    camera_gt_rows = _read_csv_rows(resolved / "gt" / "camera_gt.csv")
    tag_truth_lookup = _load_tag_truth_lookup(resolved / "gt" / "tag_gt.json")

    metrics_figure = analysis_dir / "isaac_metrics_summary.png"
    uncertainty_figure = analysis_dir / "isaac_uncertainty_timeline.png"
    command_figure = analysis_dir / "isaac_command_timeline.png"

    _save_text_figure(
        metrics_figure,
        title="Isaac Run Summary",
        lines=[
            f"run id: {metrics['run_id']}",
            f"stage: {metrics['manifest']['stage_usd_path']}",
            f"robot preset: {metrics['manifest']['robot_preset']}",
            f"camera frames: {metrics['counts']['camera_frames']}",
            f"imu packets: {metrics['counts']['imu_packets']}",
            f"aux tags: {metrics['counts']['unique_detected_auxiliary_tags']}",
        ],
    )
    _save_line_plot(
        uncertainty_figure,
        title="Position Radius 95% Timeline",
        x_values=[float(row["timestamp_s"]) for row in uncertainty_rows if "timestamp_s" in row],
        y_values=[float(row["position_radius_95_m"]) for row in uncertainty_rows if "position_radius_95_m" in row],
        y_label="m",
    )
    _save_line_plot(
        command_figure,
        title="Command Magnitude Timeline",
        x_values=[float(row["timestamp_s"]) for row in command_rows if "timestamp_s" in row],
        y_values=[abs(value) for row in command_rows if (value := _first_command_value(row)) is not None],
        y_label="abs command",
    )

    filter_positions = [(_float_or_none(row.get("timestamp_s")), _vector_or_none(row.get("position_world_m"))) for row in filter_rows]
    trajectory_times = [timestamp for timestamp, position in filter_positions if timestamp is not None and position is not None]
    trajectory_x = [float(position[0]) for timestamp, position in filter_positions if timestamp is not None and position is not None]
    trajectory_y = [float(position[1]) for timestamp, position in filter_positions if timestamp is not None and position is not None]
    waypoint_points = []
    for waypoint in metrics["manifest"]["controller_config"].get("waypoints", []):
        position = _vector_or_none(waypoint.get("position_world_m")) if isinstance(waypoint, dict) else None
        if position is not None:
            waypoint_points.append((float(position[0]), float(position[1])))

    architecture_path = report_data_dir / "system_architecture.png"
    timing_path = report_data_dir / "timing_timeline.png"
    trajectory_path = report_data_dir / "trajectory_path.png"
    tracking_path = report_data_dir / "path_tracking.png"
    first_frame_path = report_data_dir / "sim_first_frame.png"
    waypoint_stills_path = report_data_dir / "sim_waypoint_arrivals.png"
    position_estimation_path = report_data_dir / "position_estimation_timeline.png"
    pattern_world_positions_path = report_data_dir / "pattern_world_positions_timeline.png"
    imu_measurements_path = report_data_dir / "imu_measurements_timeline.png"
    pattern_relative_camera_path = report_data_dir / "pattern_relative_camera_positions_timeline.png"
    convergence_path = report_data_dir / "smoother_convergence.png"
    residual_histogram_path = report_data_dir / "residual_histogram.png"
    calibration_path = report_data_dir / "uncertainty_calibration.png"
    actuator_path = report_data_dir / "actuator_command_vs_realized.png"

    _save_text_figure(
        architecture_path,
        title="Anchored VIO Runtime Architecture",
        lines=[
            "Isaac runtime -> raw camera / IMU / command logs",
            "AprilTag frontend -> anchored filter -> fixed-lag smoother",
            "Controller closes loop on anchored-frame path estimate",
            f"ROS 2 bridge enabled: {metrics['manifest']['ros2_bridge_used']}",
        ],
    )
    _save_text_figure(
        timing_path,
        title="Sensor And Estimator Timing Summary",
        lines=[
            f"physics rate [Hz]: {metrics['config']['physics_rate_hz']}",
            f"imu rate [Hz]: {metrics['config']['imu_rate_hz']}",
            f"camera rate [Hz]: {metrics['config']['camera_rate_hz']}",
            f"filter rate [Hz]: {metrics['config']['filter_rate_hz']}",
            f"smoother rate [Hz]: {metrics['config']['smoother_rate_hz']}",
            f"controller rate [Hz]: {metrics['config']['controller_rate_hz']}",
        ],
    )
    _save_xy_plot(
        trajectory_path,
        title="Anchored-Frame Trajectory",
        tracks=[(trajectory_x, trajectory_y, (53, 102, 188))],
        point_sets=[(waypoint_points, (188, 92, 60))],
        footer="x-y projection in anchored frame",
    )
    _save_xy_plot(
        tracking_path,
        title="Path Tracking In Anchored Frame",
        tracks=[(trajectory_x, trajectory_y, (31, 160, 92))],
        point_sets=[(waypoint_points, (188, 92, 60))],
        footer=f"completion fraction: {metrics['control']['completion_fraction']}",
    )
    _save_sim_stills(
        resolved,
        output_first_frame_path=first_frame_path,
        output_waypoint_montage_path=waypoint_stills_path,
        controller_rows=controller_rows,
        waypoint_count=len(waypoint_points),
    )
    _save_position_estimation_figure(
        position_estimation_path,
        filter_rows=filter_rows,
        camera_gt_rows=camera_gt_rows,
    )
    _save_pattern_world_positions_figure(
        pattern_world_positions_path,
        filter_rows=filter_rows,
        detection_rows=detection_rows,
        tag_truth_lookup=tag_truth_lookup,
    )
    _save_imu_measurements_figure(
        imu_measurements_path,
        imu_rows=imu_rows,
    )
    _save_pattern_relative_camera_figure(
        pattern_relative_camera_path,
        detection_rows=detection_rows,
    )
    _save_line_plot(
        convergence_path,
        title="Smoother / Uncertainty Convergence",
        x_values=[float(row["timestamp_s"]) for row in uncertainty_rows if "timestamp_s" in row],
        y_values=[float(row["position_radius_95_m"]) for row in uncertainty_rows if "position_radius_95_m" in row],
        y_label="position radius 95% [m]",
    )
    _save_histogram(
        residual_histogram_path,
        title="Innovation Histogram",
        values=[
            float(row.get("innovation_diagnostics", {}).get("last_innovation_norm", 0.0))
            for row in filter_rows
            if isinstance(row.get("innovation_diagnostics"), dict)
        ],
        x_label="innovation norm",
    )
    error_times, position_errors, radius_values = _matched_position_error_series(filter_rows, camera_gt_rows, uncertainty_rows)
    if error_times:
        _save_xy_plot(
            calibration_path,
            title="Uncertainty Calibration",
            tracks=[
                (error_times, position_errors, (188, 92, 60)),
                (error_times, radius_values, (53, 102, 188)),
            ],
            footer="orange = error, blue = reported 95% radius",
        )
    else:
        _save_text_figure(
            calibration_path,
            title="Uncertainty Calibration",
            lines=["Ground-truth pose samples are required to draw the coverage plot."],
        )
    realized_times = [float(row["timestamp_s"]) for row in realized_rows if _float_or_none(row.get("timestamp_s")) is not None and _first_realized_position(row) is not None]
    realized_values = [float(_first_realized_position(row)) for row in realized_rows if _float_or_none(row.get("timestamp_s")) is not None and _first_realized_position(row) is not None]
    command_times = [
        float(row["timestamp_s"])
        for row in command_rows
        if _float_or_none(row.get("timestamp_s")) is not None and _first_command_value(row) is not None
    ]
    command_values = [
        float(_first_command_value(row))
        for row in command_rows
        if _float_or_none(row.get("timestamp_s")) is not None and _first_command_value(row) is not None
    ]
    _save_xy_plot(
        actuator_path,
        title="Actuator Command Versus Realized Motion",
        tracks=[
            (command_times, command_values, (53, 102, 188)),
            (realized_times, realized_values, (188, 92, 60)),
        ],
        footer="time on x-axis, first joint signal on y-axis",
    )
    return {
        "metrics_summary": str(metrics_figure),
        "uncertainty_timeline": str(uncertainty_figure),
        "command_timeline": str(command_figure),
        "system_architecture": str(architecture_path),
        "timing_timeline": str(timing_path),
        "trajectory_path": str(trajectory_path),
        "path_tracking": str(tracking_path),
        "sim_first_frame": str(first_frame_path),
        "sim_waypoint_arrivals": str(waypoint_stills_path),
        "position_estimation_timeline": str(position_estimation_path),
        "pattern_world_positions_timeline": str(pattern_world_positions_path),
        "imu_measurements_timeline": str(imu_measurements_path),
        "pattern_relative_camera_positions_timeline": str(pattern_relative_camera_path),
        "smoother_convergence": str(convergence_path),
        "residual_histogram": str(residual_histogram_path),
        "uncertainty_calibration": str(calibration_path),
        "actuator_command_vs_realized": str(actuator_path),
    }


def write_isaac_estimator_quality_figures(run_dir: str | Path, quality: dict[str, Any]) -> dict[str, str]:
    resolved = Path(run_dir).resolve()
    analysis_dir = resolved / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(quality.get("summary", {}))

    anchor_vs_aux_path = analysis_dir / "anchor_vs_aux_residuals.png"
    smoother_timeline_path = analysis_dir / "smoother_correction_timeline.png"

    _save_text_figure(
        anchor_vs_aux_path,
        title="Anchor vs Auxiliary Residual Quality",
        lines=[
            f"anchor mean / p95 [px]: {summary.get('anchor_mean_reprojection_rmse_px')} / {summary.get('anchor_p95_reprojection_rmse_px')}",
            f"aux mean / p95 [px]: {summary.get('auxiliary_mean_reprojection_rmse_px')} / {summary.get('auxiliary_p95_reprojection_rmse_px')}",
            f"native mean / p95 [px]: {summary.get('native_mean_reprojection_rmse_px')} / {summary.get('native_p95_reprojection_rmse_px')}",
            f"fallback mean / p95 [px]: {summary.get('fallback_mean_reprojection_rmse_px')} / {summary.get('fallback_p95_reprojection_rmse_px')}",
            f"pre-relocalization mean / p95 [px]: {summary.get('pre_relocalization_mean_reprojection_rmse_px')} / {summary.get('pre_relocalization_p95_reprojection_rmse_px')}",
            f"post-relocalization mean / p95 [px]: {summary.get('post_relocalization_mean_reprojection_rmse_px')} / {summary.get('post_relocalization_p95_reprojection_rmse_px')}",
            f"accepted / rejected aux updates: {summary.get('accepted_auxiliary_updates')} / {summary.get('rejected_auxiliary_updates')}",
        ],
    )
    smoother_rows = list(quality.get("smoother_timeline", []))
    _save_line_plot(
        smoother_timeline_path,
        title="Smoother Correction Timeline",
        x_values=[
            float(row["timestamp_s"])
            for row in smoother_rows
            if _float_or_none(row.get("timestamp_s")) is not None and row.get("feedback_correction_norm_m") is not None
        ],
        y_values=[
            float(row["feedback_correction_norm_m"])
            for row in smoother_rows
            if _float_or_none(row.get("timestamp_s")) is not None and row.get("feedback_correction_norm_m") is not None
        ],
        y_label="feedback correction norm [m]",
    )
    return {
        "anchor_vs_aux_residuals": str(anchor_vs_aux_path),
        "smoother_correction_timeline": str(smoother_timeline_path),
    }
