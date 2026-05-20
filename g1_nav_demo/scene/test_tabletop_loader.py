from __future__ import annotations

import json
import os
import re
import tempfile

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

from g1_nav_demo.scene.tabletop_loader import (
    build_merged_scene,
    is_hazard_item,
    load_manifest,
    load_scenario,
)

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
SCENE_DIR = os.path.join(REPO_ROOT, "g1_nav_demo", "scene")
DEFAULT_MANIFEST = os.path.join(SCENE_DIR, "tabletop_items.json")
ROOM_XML = os.path.join(SCENE_DIR, "g1_nav_room.xml")
HAZARD_DIR = os.path.join(REPO_ROOT, "selected_imgs_videos_demo", "Hazard_detection_selected")


def test_load_manifest_returns_scenarios():
    manifest = load_manifest(DEFAULT_MANIFEST)
    assert manifest["default_scenario"] == "mixed"
    assert set(manifest["scenarios"].keys()) == {"mixed", "all_clear", "high_hazard"}


def test_load_scenario_returns_default_when_none():
    items = load_scenario(DEFAULT_MANIFEST, scenario=None)
    assert any(item["name"] == "flammable_box" for item in items)


def test_load_scenario_returns_requested():
    items = load_scenario(DEFAULT_MANIFEST, scenario="all_clear")
    names = {item["name"] for item in items}
    assert names == {"mug", "book", "laptop"}


def test_load_scenario_unknown_raises():
    with pytest.raises(KeyError):
        load_scenario(DEFAULT_MANIFEST, scenario="does_not_exist")


def test_is_hazard_item_uses_kind():
    assert is_hazard_item({"kind": "hazard_box", "name": "x"}) is True
    assert is_hazard_item({"kind": "mug", "name": "x"}) is False


def test_build_merged_scene_adds_named_geoms(tmp_path):
    out_path = build_merged_scene(
        room_xml_path=ROOM_XML,
        manifest_path=DEFAULT_MANIFEST,
        scenario="mixed",
        hazard_textures_dir=HAZARD_DIR,
        out_dir=str(tmp_path),
    )
    text = open(out_path).read()
    for name in ["flammable_box", "radioactive_box", "mug", "book", "apple"]:
        assert re.search(rf'name="{name}"', text), f"missing geom {name} in merged XML"


def test_build_merged_scene_references_hazard_texture_paths(tmp_path):
    out_path = build_merged_scene(
        room_xml_path=ROOM_XML,
        manifest_path=DEFAULT_MANIFEST,
        scenario="high_hazard",
        hazard_textures_dir=HAZARD_DIR,
        out_dir=str(tmp_path),
    )
    text = open(out_path).read()
    # The loader converts non-PNG textures to PNG (MuJoCo only supports PNG).
    # Check that each hazard item's base name appears as a PNG texture path.
    for stem in ["image_0702", "image_1421", "image_1515"]:
        assert stem in text, f"missing texture stem {stem!r} in merged XML"
    # No raw .jpg paths should be present (all converted to .png).
    assert ".jpg" not in text, "unexpected .jpg reference in merged XML"


def test_build_merged_scene_all_clear_has_no_hazard_textures(tmp_path):
    out_path = build_merged_scene(
        room_xml_path=ROOM_XML,
        manifest_path=DEFAULT_MANIFEST,
        scenario="all_clear",
        hazard_textures_dir=HAZARD_DIR,
        out_dir=str(tmp_path),
    )
    text = open(out_path).read()
    assert "image_" not in text
    assert re.search(r'name="laptop"', text)


def test_merged_scene_loads_in_mujoco(tmp_path):
    import mujoco

    out_path = build_merged_scene(
        room_xml_path=ROOM_XML,
        manifest_path=DEFAULT_MANIFEST,
        scenario="mixed",
        hazard_textures_dir=HAZARD_DIR,
    )
    try:
        model = mujoco.MjModel.from_xml_path(out_path)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "flammable_box") >= 0
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "mug") >= 0
    finally:
        os.remove(out_path)
