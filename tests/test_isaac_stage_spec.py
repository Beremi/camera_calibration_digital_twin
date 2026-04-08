"""Pure stage-spec checks for the first-pass Isaac scene."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from calib_sim.isaac.stage_builder import stage_spec_from_config


def _load_scene_config() -> dict:
    path = Path(__file__).resolve().parents[1] / "config" / "isaac" / "scene" / "anchor_room.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_anchor_room_stage_has_one_anchor_and_no_overlapping_tags() -> None:
    spec = stage_spec_from_config(_load_scene_config())

    assert sum(1 for tag in spec.tags if tag.is_anchor) == 1
    assert len(spec.tags) >= 3


def test_stage_spec_rejects_overlapping_tags() -> None:
    scene_config = _load_scene_config()
    tags = [dict(tag) for tag in scene_config["tags"]]
    tags[1]["position_world_m"] = list(tags[0]["position_world_m"])
    scene_config["tags"] = tags

    with pytest.raises(ValueError, match="overlapping tags"):
        stage_spec_from_config(scene_config)
