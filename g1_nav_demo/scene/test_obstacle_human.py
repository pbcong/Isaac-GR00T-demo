from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import pytest

from g1_nav_demo.scene.obstacle_human import ObstacleHuman

MINIMAL_XML = """
<mujoco>
  <worldbody>
    <body name="moving_human" pos="0 0 0.9">
      <freejoint name="human_freejoint"/>
      <geom name="human_body" type="cylinder" size="0.35 0.9"
            contype="1" conaffinity="1"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def md():
    model = mujoco.MjModel.from_xml_string(MINIMAL_XML)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    return model, data


def test_initial_position_written(md):
    model, data = md
    ObstacleHuman(model, data, start_xy=(1.5, -2.5))
    assert data.qpos[0] == pytest.approx(1.5, abs=1e-6)
    assert data.qpos[1] == pytest.approx(-2.5, abs=1e-6)
    assert data.qpos[2] == pytest.approx(0.9, abs=1e-6)


def test_step_advances_position(md):
    model, data = md
    h = ObstacleHuman(model, data, start_xy=(0.0, 0.0),
                      direction_xy=(0.0, 1.0), speed=1.0)
    h.step(dt=1.0)
    assert data.qpos[1] == pytest.approx(1.0, abs=1e-5)


def test_step_caps_at_travel_dist(md):
    model, data = md
    h = ObstacleHuman(model, data, start_xy=(0.0, 0.0),
                      direction_xy=(1.0, 0.0), speed=1.0, travel_dist=2.0)
    h.step(dt=100.0)
    assert h.is_done
    assert data.qpos[0] == pytest.approx(2.0, abs=1e-5)


def test_not_done_initially(md):
    model, data = md
    assert not ObstacleHuman(model, data).is_done


def test_done_after_full_travel(md):
    model, data = md
    h = ObstacleHuman(model, data, speed=1.0, travel_dist=1.0)
    h.step(dt=1.0)
    assert h.is_done


def test_no_movement_after_done(md):
    model, data = md
    h = ObstacleHuman(model, data, start_xy=(0.0, 0.0),
                      direction_xy=(1.0, 0.0), speed=1.0, travel_dist=1.0)
    h.step(dt=1.0)
    h.step(dt=1.0)  # should be a no-op
    assert data.qpos[0] == pytest.approx(1.0, abs=1e-5)


def test_invalid_body_raises(md):
    model, data = md
    with pytest.raises(ValueError, match="not found"):
        ObstacleHuman(model, data, body_name="nonexistent")


def test_unnormalised_direction_is_corrected(md):
    model, data = md
    h = ObstacleHuman(model, data, start_xy=(0.0, 0.0),
                      direction_xy=(3.0, 0.0), speed=1.0, travel_dist=1.0)
    h.step(dt=1.0)
    assert data.qpos[0] == pytest.approx(1.0, abs=1e-5)
    assert data.qpos[1] == pytest.approx(0.0, abs=1e-5)