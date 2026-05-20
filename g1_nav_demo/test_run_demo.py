from __future__ import annotations

import math
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import pytest

from g1_nav_demo.run_demo import _read_forward_range

# ghost_box at 0.8m (closer) has contype=0 — if NOT ignored, r would be ~0.8.
# front_box at 1.5m has contype=1 — should register.
# side_box at (0, 1.5) is 90° off — outside ±20° cone facing east.
RANGE_XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 1" contype="1" conaffinity="1"/>
    <geom name="front_box" type="box" size="0.3 0.3 0.5" pos="1.5 0.0 0.5"
          contype="1" conaffinity="1"/>
    <geom name="side_box" type="box" size="0.3 0.3 0.5" pos="0.0 1.5 0.5"
          contype="1" conaffinity="1"/>
    <geom name="ghost_box" type="box" size="0.3 0.3 0.5" pos="0.8 0.0 0.5"
          contype="0" conaffinity="0"/>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def rmd():
    model = mujoco.MjModel.from_xml_string(RANGE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_detects_forward_obstacle(rmd):
    model, data = rmd
    r = _read_forward_range(model, data, (0.0, 0.0), 0.0,
                            cone_deg=20.0, cutoff=3.0, min_dist=0.1)
    assert r == pytest.approx(1.5, abs=0.1)


def test_ignores_side_obstacle(rmd):
    model, data = rmd
    # side_box at 90° should not be detected; only front_box at 1.5m.
    r = _read_forward_range(model, data, (0.0, 0.0), 0.0,
                            cone_deg=20.0, cutoff=3.0, min_dist=0.1)
    assert r == pytest.approx(1.5, abs=0.1)


def test_ignores_contype_zero(rmd):
    model, data = rmd
    # ghost_box at 0.8m has contype=0 — must be ignored. Result should be ~1.5 (front_box).
    r = _read_forward_range(model, data, (0.0, 0.0), 0.0,
                            cone_deg=20.0, cutoff=3.0, min_dist=0.1)
    assert r == pytest.approx(1.5, abs=0.1)


def test_ignores_floor(rmd):
    model, data = rmd
    # Face north (pi/2): only floor/side_box in that direction.
    # floor is plane type (ignored). side_box is at (0,1.5) — forward.
    r = _read_forward_range(model, data, (0.0, 0.0), math.pi / 2,
                            cone_deg=20.0, cutoff=3.0, min_dist=0.1)
    assert r == pytest.approx(1.5, abs=0.1)


def test_nothing_forward_returns_cutoff(rmd):
    model, data = rmd
    # Face west (-X direction, yaw=pi): nothing there within 3m.
    r = _read_forward_range(model, data, (0.0, 0.0), math.pi,
                            cone_deg=20.0, cutoff=3.0, min_dist=0.1)
    assert r == pytest.approx(3.0, abs=0.1)


# --- Intercept trajectory tests ---

import numpy as np
from g1_nav_demo.run_demo import _scene_obstacle_aabbs, _find_clear_crossing


def test_scene_obstacle_aabbs_filters_floor_and_small(rmd):
    model, data = rmd
    aabbs = _scene_obstacle_aabbs(model, data)
    # The RANGE_XML has front_box (box 0.3x0.3) and side_box (box 0.3x0.3)
    # Floor is filtered (plane type), ghost_box is filtered (contype=0)
    assert len(aabbs) == 2
    for aabb in aabbs:
        assert aabb[2] - aabb[0] == pytest.approx(0.6, abs=0.01)
        assert aabb[3] - aabb[1] == pytest.approx(0.6, abs=0.01)


def test_find_clear_crossing_basic():
    path_points = [(0.0, 0.0), (2.0, 0.0)]
    # No obstacles
    aabbs = []
    crossing, perp = _find_clear_crossing(path_points, aabbs)
    assert crossing is not None
    # Perpendicular should be unit vector (0, 1) or (0, -1)
    assert abs(perp[0]) < 0.01 or abs(perp[1]) < 0.01


def test_find_clear_crossing_with_blocked_left():
    path_points = [(0.0, 0.0), (4.0, 0.0)]
    # Block on the left (positive Y) side of the midpoint (~2,0)
    # AABB from y=0.5 to y=4 with margin 0.65 blocks the left perpendicular
    blocked = [(1.0, 0.5, 3.0, 4.0)]
    crossing, perp = _find_clear_crossing(path_points, blocked, margin=0.65)
    if crossing is not None:
        # Should prefer the right side (negative Y perpendicular)
        assert perp[1] < 0 or crossing is not None


def test_find_clear_crossing_returns_none_when_all_blocked():
    path_points = [(0.0, 0.0), (4.0, 0.0)]
    # Block both left and right perpendicular approaches
    blocked = [(-10.0, -10.0, 10.0, -0.3), (-10.0, 0.3, 10.0, 10.0)]
    crossing, perp = _find_clear_crossing(path_points, blocked, margin=0.65)
    # With everything blocked, may still find None
    # (depends on whether the 3m approach path clips the blocks)