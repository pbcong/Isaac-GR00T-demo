from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import pytest

from g1_nav_demo.renderer.video_renderer import VideoRenderer

SCENE_XML = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "scene", "g1_nav_room.xml"
)


@pytest.fixture
def model_and_data():
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_snapshot_returns_png_bytes(model_and_data, tmp_path):
    model, data = model_and_data
    renderer = VideoRenderer(model, output_path=str(tmp_path / "ignored.mp4"))
    try:
        png = renderer.snapshot("birdseye", data)
    finally:
        renderer.close()

    assert isinstance(png, bytes)
    assert png[:8] == b"\x89PNG\r\n\x1a\n", "Result is not a PNG file"


def test_snapshot_unknown_camera_raises(model_and_data, tmp_path):
    model, data = model_and_data
    renderer = VideoRenderer(model, output_path=str(tmp_path / "ignored.mp4"))
    try:
        with pytest.raises(ValueError, match="camera"):
            renderer.snapshot("not_a_real_camera", data)
    finally:
        renderer.close()


def test_snapshot_respects_dimensions(model_and_data, tmp_path):
    import io

    from PIL import Image

    model, data = model_and_data
    renderer = VideoRenderer(model, output_path=str(tmp_path / "ignored.mp4"))
    try:
        png = renderer.snapshot("birdseye", data, width=320, height=240)
    finally:
        renderer.close()
    img = Image.open(io.BytesIO(png))
    assert img.size == (320, 240)


import numpy as np


def test_hazard_banner_draws_red_top_strip(model_and_data, tmp_path):
    model, data = model_and_data
    renderer = VideoRenderer(model, output_path=str(tmp_path / "ignored.mp4"))
    try:
        renderer.hazard_banner = "HAZARD DETECTED: flammable_box"
        torso_id = __import__("mujoco").mj_name2id(
            model, __import__("mujoco").mjtObj.mjOBJ_BODY, "torso_link"
        )
        frame = renderer.render_frame(
            data, command="inspect the table", distance=0.0,
            update_head_camera=True, head_body_id=torso_id,
            goal_waypoints=None, current_wp_idx=0,
        )
    finally:
        renderer.close()

    # Top strip should be dominated by red pixels.
    top = frame[:60, :, :]
    mean_color = top.reshape(-1, 3).mean(axis=0)
    assert mean_color[0] > 120, f"red channel too low: {mean_color}"
    assert mean_color[0] > mean_color[1] * 1.5
    assert mean_color[0] > mean_color[2] * 1.5


def test_no_banner_when_attribute_none(model_and_data, tmp_path):
    model, data = model_and_data
    renderer = VideoRenderer(model, output_path=str(tmp_path / "ignored.mp4"))
    try:
        assert renderer.hazard_banner is None
        torso_id = __import__("mujoco").mj_name2id(
            model, __import__("mujoco").mjtObj.mjOBJ_BODY, "torso_link"
        )
        frame = renderer.render_frame(
            data, command="go to the table", distance=0.0,
            update_head_camera=True, head_body_id=torso_id,
        )
    finally:
        renderer.close()

    top = frame[:60, :, :]
    mean_color = top.reshape(-1, 3).mean(axis=0)
    # Without a banner, red should NOT dominate green and blue at the top.
    assert not (mean_color[0] > mean_color[1] * 1.5 and mean_color[0] > mean_color[2] * 1.5)


def test_obstacle_banner_default_is_none(model_and_data, tmp_path):
    model, data = model_and_data
    renderer = VideoRenderer(model, output_path=str(tmp_path / "ignored.mp4"))
    try:
        assert renderer.obstacle_banner is None
    finally:
        renderer.close()


def test_obstacle_banner_draws_yellow_top_strip(model_and_data, tmp_path):
    model, data = model_and_data
    renderer = VideoRenderer(model, output_path=str(tmp_path / "ignored.mp4"))
    try:
        renderer.obstacle_banner = "OBSTACLE DETECTED — WAITING"
        torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        frame = renderer.render_frame(
            data, command="go to table", distance=0.0,
            update_head_camera=True, head_body_id=torso_id,
        )
    finally:
        renderer.close()
    top = frame[:60, :, :]
    mean = top.reshape(-1, 3).mean(axis=0)
    # Golden yellow: high red, high green, low blue
    assert mean[0] > 150, f"red too low: {mean}"
    assert mean[1] > 100, f"green too low: {mean}"
    assert mean[2] < 80, f"blue too high (not yellow): {mean}"
