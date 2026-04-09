#!/usr/bin/env python3
"""Render a lightweight local-only media bundle from second-pass suite artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="output/isaac_runs")
    parser.add_argument("--hero-config", default="config/isaac/media/hero_capture.yaml")
    parser.add_argument("--comparison-config", default="config/isaac/media/comparison_capture.yaml")
    parser.add_argument("--dropout-config", default="config/isaac/media/dropout_capture.yaml")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--max-frames", type=int, default=120)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _first_rgb_frames(run_dir: Path, *, max_frames: int) -> list[np.ndarray]:
    rgb_dir = run_dir / "raw" / "rgb"
    if not rgb_dir.exists():
        return []
    frames = []
    for path in sorted(rgb_dir.glob("*.png"))[: max(int(max_frames), 1)]:
        image = cv2.imread(str(path))
        if image is not None:
            frames.append(image)
    return frames


def _load_png(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return cv2.imread(str(path))


def _resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    scale = float(height) / float(max(image.shape[0], 1))
    width = max(int(round(image.shape[1] * scale)), 1)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def _add_text_panel(canvas: np.ndarray, *, title: str, lines: list[str], origin: tuple[int, int]) -> None:
    x0, y0 = origin
    cv2.putText(canvas, title, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2, cv2.LINE_AA)
    y = y0 + 32
    for line in lines:
        cv2.putText(canvas, line[:72], (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (225, 225, 225), 1, cv2.LINE_AA)
        y += 24


def _side_by_side_frame(
    left: np.ndarray,
    right: np.ndarray,
    *,
    title: str,
    left_label: str,
    right_label: str,
    footer_lines: list[str],
) -> np.ndarray:
    target_height = max(left.shape[0], right.shape[0])
    left_resized = _resize_to_height(left, target_height)
    right_resized = _resize_to_height(right, target_height)
    canvas = np.full((target_height + 110, left_resized.shape[1] + right_resized.shape[1] + 60, 3), 20, dtype=np.uint8)
    cv2.putText(canvas, title, (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (244, 244, 244), 2, cv2.LINE_AA)
    canvas[52 : 52 + target_height, 20 : 20 + left_resized.shape[1]] = left_resized
    right_x = 40 + left_resized.shape[1]
    canvas[52 : 52 + target_height, right_x : right_x + right_resized.shape[1]] = right_resized
    cv2.putText(canvas, left_label, (20, target_height + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 180, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, right_label, (right_x, target_height + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 190, 110), 1, cv2.LINE_AA)
    _add_text_panel(canvas, title="Diagnostics", lines=footer_lines, origin=(20, target_height + 104))
    return canvas


def _single_frame_with_inset(
    image: np.ndarray,
    *,
    title: str,
    inset: np.ndarray | None,
    footer_lines: list[str],
) -> np.ndarray:
    base = image.copy()
    canvas = np.full((base.shape[0] + 110, base.shape[1] + 24, 3), 20, dtype=np.uint8)
    cv2.putText(canvas, title, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (244, 244, 244), 2, cv2.LINE_AA)
    canvas[46 : 46 + base.shape[0], 12 : 12 + base.shape[1]] = base
    if inset is not None:
        max_width = max(canvas.shape[1] - 48, 24)
        max_height = max(base.shape[0] - 24, 24)
        scale = min(float(max_width) / float(max(inset.shape[1], 1)), float(max_height) / float(max(inset.shape[0], 1)), 1.0)
        inset_width = max(int(round(inset.shape[1] * scale)), 1)
        inset_height = max(int(round(inset.shape[0] * scale)), 1)
        inset_resized = cv2.resize(inset, (inset_width, inset_height), interpolation=cv2.INTER_LINEAR)
        x0 = max(canvas.shape[1] - inset_resized.shape[1] - 24, 12)
        y0 = 58
        canvas[y0 : y0 + inset_resized.shape[0], x0 : x0 + inset_resized.shape[1]] = inset_resized
        cv2.rectangle(canvas, (x0, y0), (x0 + inset_resized.shape[1], y0 + inset_resized.shape[0]), (230, 230, 230), 1)
    _add_text_panel(canvas, title="Diagnostics", lines=footer_lines, origin=(16, base.shape[0] + 78))
    return canvas


def _write_video(path: Path, frames: list[np.ndarray], *, fps: int) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (frames[0].shape[1], frames[0].shape[0]))
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def _frame_sequence_or_fallback(run_dir: Path, *, fallback_image: np.ndarray | None, max_frames: int) -> list[np.ndarray]:
    frames = _first_rgb_frames(run_dir, max_frames=max_frames)
    if frames:
        return frames
    if fallback_image is None:
        fallback_image = np.full((720, 1280, 3), 235, dtype=np.uint8)
        cv2.putText(fallback_image, "No RGB frames available", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 40), 2, cv2.LINE_AA)
    return [fallback_image.copy() for _ in range(max(1, max_frames // 6))]


def render_media_bundle(output_root: Path, *, fps: int, max_frames: int) -> dict[str, Any]:
    suite_summary_path = output_root / "latest_second_pass_suite" / "analysis" / "suite_summary.json"
    suite_summary = _load_json(suite_summary_path)
    presentation_dir = output_root / "latest_second_pass_suite" / "presentation"
    presentation_dir.mkdir(parents=True, exist_ok=True)
    representative = dict(suite_summary.get("representative_runs", {}))

    nominal_visual_id = representative.get("nominal_full_anchor", {}).get("visual")
    nominal_fused_id = representative.get("nominal_full_anchor", {}).get("fused")
    dropout_visual_id = representative.get("intermittent_anchor", {}).get("visual")
    dropout_fused_id = representative.get("intermittent_anchor", {}).get("fused")
    stress_visual_id = representative.get("servo_stress", {}).get("visual")
    stress_fused_id = representative.get("servo_stress", {}).get("fused")

    nominal_compare = _load_png(output_root / "latest_second_pass_suite" / "analysis" / "nominal_trajectory_compare.png")
    dropout_compare = _load_png(output_root / "latest_second_pass_suite" / "analysis" / "dropout_trajectory_compare.png")
    stress_compare = _load_png(output_root / "latest_second_pass_suite" / "analysis" / "stress_trajectory_compare.png")
    smoother_compare = _load_png(output_root / "latest_second_pass_suite" / "analysis" / "smoother_feedback_compare.png")

    hero_frames = _frame_sequence_or_fallback(output_root / nominal_fused_id, fallback_image=nominal_compare, max_frames=max_frames)
    hero_video_frames = [
        _single_frame_with_inset(
            frame,
            title="Anchored VIO Closed-Loop Path Tracking",
            inset=nominal_compare,
            footer_lines=[
                f"run: {nominal_fused_id}",
                "condition: nominal_full_anchor",
                "mode: fused / closed-loop",
                "bundle: latest_second_pass_suite",
            ],
        )
        for frame in hero_frames
    ]
    _write_video(presentation_dir / "hero_demo.mp4", hero_video_frames, fps=fps)
    cv2.imwrite(str(presentation_dir / "hero_still_main.png"), hero_video_frames[0])
    if nominal_compare is not None:
        cv2.imwrite(str(presentation_dir / "hero_still_compare.png"), nominal_compare)

    def comparison_video(
        left_run_id: str | None,
        right_run_id: str | None,
        output_name: str,
        title: str,
        footer: list[str],
    ) -> None:
        if left_run_id is None or right_run_id is None:
            return
        left_frames = _frame_sequence_or_fallback(output_root / left_run_id, fallback_image=nominal_compare, max_frames=max_frames)
        right_frames = _frame_sequence_or_fallback(output_root / right_run_id, fallback_image=nominal_compare, max_frames=max_frames)
        frame_count = min(len(left_frames), len(right_frames))
        frames = [
            _side_by_side_frame(
                left_frames[index],
                right_frames[index],
                title=title,
                left_label=f"visual: {left_run_id}",
                right_label=f"fused: {right_run_id}",
                footer_lines=footer,
            )
            for index in range(frame_count)
        ]
        _write_video(presentation_dir / output_name, frames, fps=fps)

    comparison_video(
        nominal_visual_id,
        nominal_fused_id,
        "visual_vs_fused_nominal.mp4",
        "Visual vs Fused: Nominal Full Anchor",
        ["condition: nominal_full_anchor", "mode: closed-loop", "seed alignment: representative runs"],
    )
    comparison_video(
        dropout_visual_id,
        dropout_fused_id,
        "visual_vs_fused_dropout.mp4",
        "Visual vs Fused: Intermittent Anchor",
        ["condition: intermittent_anchor", "anchor updates suppressed on schedule", "mode: closed-loop"],
    )
    comparison_video(
        nominal_fused_id,
        stress_fused_id,
        "actuation_stress_demo.mp4",
        "Fused Nominal vs Servo Stress",
        ["left: servo_nominal", "right: servo_stress", "mode: fused / closed-loop"],
    )

    observer_frames = _frame_sequence_or_fallback(output_root / nominal_fused_id, fallback_image=smoother_compare, max_frames=max_frames)
    diagnostics_frames = [
        _single_frame_with_inset(
            frame,
            title="Observer + Phone + Control Diagnostics",
            inset=smoother_compare,
            footer_lines=[
                f"run: {nominal_fused_id}",
                "estimator: fused",
                "controller: closed-loop",
                "diagnostics: residuals / smoother feedback",
            ],
        )
        for frame in observer_frames
    ]
    _write_video(presentation_dir / "observer_phone_diagnostics.mp4", diagnostics_frames, fps=fps)

    manifest = {
        "suite_summary_json": str(suite_summary_path.resolve()),
        "videos": {
            "hero_demo": str((presentation_dir / "hero_demo.mp4").resolve()),
            "visual_vs_fused_nominal": str((presentation_dir / "visual_vs_fused_nominal.mp4").resolve()),
            "visual_vs_fused_dropout": str((presentation_dir / "visual_vs_fused_dropout.mp4").resolve()),
            "actuation_stress_demo": str((presentation_dir / "actuation_stress_demo.mp4").resolve()),
            "observer_phone_diagnostics": str((presentation_dir / "observer_phone_diagnostics.mp4").resolve()),
        },
        "stills": {
            "hero_still_main": str((presentation_dir / "hero_still_main.png").resolve()),
            "hero_still_compare": str((presentation_dir / "hero_still_compare.png").resolve()),
        },
    }
    (presentation_dir / "presentation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    args = parse_args()
    payload = render_media_bundle(Path(args.output_root).resolve(), fps=int(args.fps), max_frames=int(args.max_frames))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
