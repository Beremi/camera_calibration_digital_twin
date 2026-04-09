#!/usr/bin/env python3
"""Render a lightweight local-only media bundle from second-pass suite artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from calib_sim.reporting.isaac_second_pass_suite import DEFAULT_SECOND_PASS_DRAFT_LOCK


REPO_ROOT = Path(__file__).resolve().parents[1]


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="output/isaac_runs")
    parser.add_argument("--lock-path", default=str(DEFAULT_SECOND_PASS_DRAFT_LOCK))
    parser.add_argument("--hero-config", default="config/isaac/media/hero_capture.yaml")
    parser.add_argument("--comparison-config", default="config/isaac/media/comparison_capture.yaml")
    parser.add_argument("--dropout-config", default="config/isaac/media/dropout_capture.yaml")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--max-frames", type=int, default=120)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    target_size = (frames[0].shape[1], frames[0].shape[0])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), target_size)
    try:
        for frame in frames:
            if (frame.shape[1], frame.shape[0]) != target_size:
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
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


def _pick_frame(frames: list[np.ndarray], index: int) -> np.ndarray:
    if not frames:
        return np.full((720, 1280, 3), 235, dtype=np.uint8)
    bounded_index = max(0, min(index, len(frames) - 1))
    return frames[bounded_index]


def _title_card(size: tuple[int, int], *, title: str, bullets: list[str]) -> np.ndarray:
    height, width = size
    canvas = np.full((height, width, 3), (18, 26, 38), dtype=np.uint8)
    cv2.putText(canvas, title, (56, 96), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (244, 244, 244), 2, cv2.LINE_AA)
    y = 162
    for bullet in bullets:
        cv2.putText(canvas, f"- {bullet}", (72, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225, 230, 236), 2, cv2.LINE_AA)
        y += 54
    return canvas


def render_media_bundle(output_root: Path, *, lock_path: Path, fps: int, max_frames: int) -> dict[str, Any]:
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
    scene_overview_still = hero_video_frames[min(len(hero_video_frames) // 4, len(hero_video_frames) - 1)]
    cv2.imwrite(str(presentation_dir / "scene_overview_still.png"), scene_overview_still)
    phone_view_still = _single_frame_with_inset(
        _pick_frame(hero_frames, max(0, len(hero_frames) // 3)),
        title="Phone View With Nominal Path Context",
        inset=nominal_compare,
        footer_lines=[
            f"run: {nominal_fused_id}",
            "view: phone RGB with top-down trajectory inset",
            "condition: nominal_full_anchor",
            "what to notice: stable anchor-driven closed-loop tracking",
        ],
    )
    cv2.imwrite(str(presentation_dir / "phone_view_still.png"), phone_view_still)

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

    dropout_frames = _frame_sequence_or_fallback(output_root / dropout_fused_id, fallback_image=dropout_compare, max_frames=max_frames)
    dropout_still = _single_frame_with_inset(
        _pick_frame(dropout_frames, max(0, len(dropout_frames) // 2)),
        title="Anchor Suppression Interval",
        inset=dropout_compare,
        footer_lines=[
            f"run: {dropout_fused_id}",
            "condition: intermittent_anchor",
            "anchor updates: deterministically suppressed in scheduled windows",
            "what to notice: fused estimate remains controllable through dropout",
        ],
    )
    cv2.imwrite(str(presentation_dir / "dropout_still.png"), dropout_still)

    stress_frames = _frame_sequence_or_fallback(output_root / stress_fused_id, fallback_image=stress_compare, max_frames=max_frames)
    servo_stress_still = _single_frame_with_inset(
        _pick_frame(stress_frames, max(0, len(stress_frames) // 2)),
        title="Servo Stress Exact-Positioning View",
        inset=stress_compare,
        footer_lines=[
            f"run: {stress_fused_id}",
            "condition: servo_stress",
            "overlay: command vs realized end-effector behavior",
            "what to notice: control under lag, deadband, backlash, and dropouts",
        ],
    )
    cv2.imwrite(str(presentation_dir / "servo_stress_still.png"), servo_stress_still)

    teaser_frames: list[np.ndarray] = []
    if hero_video_frames:
        teaser_frames.extend([hero_video_frames[0].copy()] * max(10, fps))
    teaser_frames.extend(
        [
            _title_card(
                (720, 1280),
                title="Anchored Visual vs Visual-Inertial Closed-Loop Tracking",
                bullets=[
                    "Nominal fused is competitive on mean position error",
                    "Intermittent-anchor fused is stable, no longer catastrophic",
                    "Visual still leads on waypoint error in the current suite",
                ],
            )
        ]
        * max(8, fps // 2)
    )
    for source in (nominal_compare, dropout_compare, stress_compare, smoother_compare):
        if source is None:
            continue
        frame = _single_frame_with_inset(
            source,
            title="Second-Pass Draft Highlight",
            inset=None,
            footer_lines=[
                "Visual vs fused comparison",
                "Representative condition figure from latest_second_pass_suite",
            ],
        )
        teaser_frames.extend([frame] * max(10, fps))
    teaser_frames.extend(
        [
            _title_card(
                (720, 1280),
                title="Current Takeaway",
                bullets=[
                    "The second-pass branch repaired the fused dropout failure",
                    "Fused is now reproducible and defensible in the anchored benchmark",
                    "Remaining weakness: visual is still the stronger waypoint baseline",
                ],
            )
        ]
        * max(10, fps)
    )
    _write_video(presentation_dir / "paper_teaser_second_pass.mp4", teaser_frames, fps=fps)

    manifest = {
        "suite_summary_json": str(suite_summary_path.resolve()),
        "representative_runs": representative,
        "videos": {
            "hero_demo": str((presentation_dir / "hero_demo.mp4").resolve()),
            "visual_vs_fused_nominal": str((presentation_dir / "visual_vs_fused_nominal.mp4").resolve()),
            "visual_vs_fused_dropout": str((presentation_dir / "visual_vs_fused_dropout.mp4").resolve()),
            "actuation_stress_demo": str((presentation_dir / "actuation_stress_demo.mp4").resolve()),
            "observer_phone_diagnostics": str((presentation_dir / "observer_phone_diagnostics.mp4").resolve()),
            "paper_teaser_second_pass": str((presentation_dir / "paper_teaser_second_pass.mp4").resolve()),
        },
        "stills": {
            "hero_still_main": str((presentation_dir / "hero_still_main.png").resolve()),
            "hero_still_compare": str((presentation_dir / "hero_still_compare.png").resolve()),
            "scene_overview_still": str((presentation_dir / "scene_overview_still.png").resolve()),
            "phone_view_still": str((presentation_dir / "phone_view_still.png").resolve()),
            "dropout_still": str((presentation_dir / "dropout_still.png").resolve()),
            "servo_stress_still": str((presentation_dir / "servo_stress_still.png").resolve()),
        },
        "captions": {
            "hero_demo": "Overview of the anchored fused closed-loop nominal run with path and diagnostics overlays.",
            "visual_vs_fused_nominal": "Nominal full-anchor split-screen showing that fused is competitive on mean position error while visual still leads on waypoint error.",
            "visual_vs_fused_dropout": "Intermittent-anchor split-screen showing that the promoted fused path no longer diverges catastrophically through scheduled anchor suppression.",
            "actuation_stress_demo": "Comparison of nominal fused behavior and the servo_stress actuation preset, emphasizing stability and calibration rather than a fused win on waypoint error.",
            "observer_phone_diagnostics": "Observer view, phone-view proxy, and smoother/residual diagnostics for the representative fused nominal run.",
            "paper_teaser_second_pass": "Short stitched overview aligned with the refreshed rerun-suite claims.",
        },
    }
    manifest_path = presentation_dir / "presentation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if lock_path.exists():
        lock_payload = _load_json(lock_path)
        draft_selection = lock_payload.setdefault("draft_selection", {})
        representative_run_ids = [
            str(run_id)
            for runs_by_estimator in representative.values()
            for run_id in runs_by_estimator.values()
            if run_id
        ]
        draft_selection["media_source_run_ids"] = representative_run_ids
        draft_selection["presentation_manifest_path"] = _repo_relative(manifest_path)
        draft_selection["media_artifacts_stale"] = False
        build_summary_path = draft_selection.get("publication_build_summary_path")
        if build_summary_path:
            build_summary_file = REPO_ROOT / str(build_summary_path)
            if build_summary_file.exists():
                build_summary = _load_json(build_summary_file)
                if build_summary.get("pdf_exists") and not build_summary.get("placeholders_remaining"):
                    draft_selection["draft_ready"] = True
                    draft_selection["draft_suite_status"] = "review_bundle_ready"
                    draft_selection[
                        "draft_suite_status_reason"
                    ] = (
                        "The refreshed second-pass suite, PDF build, presentation bundle, and "
                        "dashboard have all been regenerated from the promoted fused draft lock."
                    )
        _write_json(lock_path, lock_payload)
    return manifest


def main() -> int:
    args = parse_args()
    payload = render_media_bundle(
        Path(args.output_root).resolve(),
        lock_path=Path(args.lock_path).resolve(),
        fps=int(args.fps),
        max_frames=int(args.max_frames),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
