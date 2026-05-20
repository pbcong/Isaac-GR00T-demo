from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import pytest

ROOM_XML = os.path.join(os.path.dirname(__file__), "g1_nav_room.xml")


def test_static_obstacles_present():
    model = mujoco.MjModel.from_xml_path(ROOM_XML)
    geom_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        for i in range(model.ngeom)
    }
    assert "crate" in geom_names
    assert "barrel" in geom_names
    assert "pillar" in geom_names


def test_static_obstacle_positions():
    model = mujoco.MjModel.from_xml_path(ROOM_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    for name, ex, ey in [("crate", 2.0, 0.0), ("barrel", 0.5, -1.5), ("pillar", -0.5, 2.0)]:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert data.geom_xpos[gid][0] == pytest.approx(ex, abs=0.05), name
        assert data.geom_xpos[gid][1] == pytest.approx(ey, abs=0.05), name