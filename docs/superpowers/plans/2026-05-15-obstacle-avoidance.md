# Obstacle Avoidance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add static floor obstacles and a reactive rule-based dynamic-human avoidance system to the G1 nav demo.

**Architecture:** Three layers added on top of existing code without modifying VLM, GoalPlanner, or AgentLoop: (1) physical obstacles in the scene XML and VLM SCENE_PROMPT, (2) a standalone `AvoidanceStateMachine` (NAVIGATING→STOPPED→REROUTING) driven by a module-level forward-rangefinder function, (3) an `ObstacleHuman` kinematic body stepped inside `run_to_goal_with_renderer` each sim step.

**Tech Stack:** MuJoCo 3.x, Python 3.10+, pytest, NumPy

---

## File Map

**Create:**
- `g1_nav_demo/scene/test_static_obstacles.py` — verify crate/barrel/pillar exist in model
- `g1_nav_demo/scene/obstacle_human.py` — `ObstacleHuman`: kinematic body walker
- `g1_nav_demo/scene/test_obstacle_human.py` — unit tests for ObstacleHuman
- `g1_nav_demo/avoidance.py` — `AvoidanceStateMachine`
- `g1_nav_demo/test_avoidance.py` — unit tests for AvoidanceStateMachine
- `g1_nav_demo/test_run_demo.py` — unit test for `_read_forward_range`

**Modify:**
- `g1_nav_demo/scene/g1_nav_room.xml` — add crate, barrel, pillar, moving_human bodies
- `g1_nav_demo/vlm/goal_parser.py` — add crate/barrel/pillar bounding boxes to `SCENE_PROMPT`
- `g1_nav_demo/renderer/video_renderer.py` — add `obstacle_banner` attribute + yellow overlay
- `g1_nav_demo/renderer/test_video_renderer.py` — test obstacle_banner renders yellow
- `g1_nav_demo/run_demo.py` — `_read_forward_range()`, updated `run_to_goal_with_renderer`, `_init_obstacle_human`, CLI flags

---

## Task 1: Static Obstacles — Scene XML and SCENE_PROMPT

**Files:**
- Modify: `g1_nav_demo/scene/g1_nav_room.xml`
- Modify: `g1_nav_demo/vlm/goal_parser.py`
- Create: `g1_nav_demo/scene/test_static_obstacles.py`

- [ ] **Write the failing tests**

Create `g1_nav_demo/scene/test_static_obstacles.py`:

```python
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
```

- [ ] **Run test — expect FAIL**

```
pytest g1_nav_demo/scene/test_static_obstacles.py -v
```
Expected: FAILED — geom names not found.

- [ ] **Add static obstacles to g1_nav_room.xml**

In `g1_nav_demo/scene/g1_nav_room.xml`, add before `</worldbody>`:

```xml
    <body name="crate" pos="2.0 0.0 0">
      <geom name="crate" type="box" size="0.3 0.3 0.5" pos="0 0 0.5"
            rgba="0.55 0.4 0.2 1" contype="1" conaffinity="1"/>
    </body>

    <body name="barrel" pos="0.5 -1.5 0">
      <geom name="barrel" type="cylinder" size="0.2 0.5" pos="0 0 0.5"
            rgba="0.3 0.3 0.6 1" contype="1" conaffinity="1"/>
    </body>

    <body name="pillar" pos="-0.5 2.0 0">
      <geom name="pillar" type="cylinder" size="0.15 0.75" pos="0 0 0.75"
            rgba="0.7 0.7 0.7 1" contype="1" conaffinity="1"/>
    </body>
```

- [ ] **Run test — expect PASS**

```
pytest g1_nav_demo/scene/test_static_obstacles.py -v
```
Expected: 2 tests PASS.

- [ ] **Update SCENE_PROMPT in goal_parser.py**

In `g1_nav_demo/vlm/goal_parser.py`, in the `SCENE_PROMPT` string, find the line:

```
  - bookshelf: center (-2.0, 1.0), half-extents (0.25, 0.5) → region (-2.25, 0.5)–(-1.75, 1.5)
```

Add three lines immediately after it:

```
  - crate:     center (2.0, 0.0),  half-extents (0.3, 0.3)   → region (1.7, -0.3)–(2.3, 0.3)
  - barrel:    center (0.5, -1.5), half-extents (0.2, 0.2)   → region (0.3, -1.7)–(0.7, -1.3)
  - pillar:    center (-0.5, 2.0), half-extents (0.15, 0.15) → region (-0.65, 1.85)–(-0.35, 2.15)
```

- [ ] **Commit**

```bash
git add g1_nav_demo/scene/g1_nav_room.xml \
        g1_nav_demo/vlm/goal_parser.py \
        g1_nav_demo/scene/test_static_obstacles.py
git commit -m "feat: add static floor obstacles (crate, barrel, pillar) to scene and VLM prompt"
```

---

## Task 2: Moving Human Body and ObstacleHuman Class

**Files:**
- Modify: `g1_nav_demo/scene/g1_nav_room.xml`
- Create: `g1_nav_demo/scene/obstacle_human.py`
- Create: `g1_nav_demo/scene/test_obstacle_human.py`

- [ ] **Write the failing tests**

Create `g1_nav_demo/scene/test_obstacle_human.py`:

```python
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
```

- [ ] **Run test — expect FAIL**

```
pytest g1_nav_demo/scene/test_obstacle_human.py -v
```
Expected: ImportError — module not found.

- [ ] **Create obstacle_human.py**

Create `g1_nav_demo/scene/obstacle_human.py`:

```python
from __future__ import annotations

import math

import mujoco
import numpy as np


class ObstacleHuman:
    """Kinematic body that walks a straight line in the MuJoCo simulation."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        body_name: str = "moving_human",
        start_xy: tuple[float, float] = (1.5, -2.5),
        direction_xy: tuple[float, float] = (0.0, 1.0),
        speed: float = 0.8,
        travel_dist: float = 5.0,
    ) -> None:
        self._model = model
        self._data = data

        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Body {body_name!r} not found in model")

        jnt_adr = int(model.body_jntadr[body_id])
        if jnt_adr < 0:
            raise ValueError(f"Body {body_name!r} has no joint")
        self._qpos_adr = int(model.jnt_qposadr[jnt_adr])

        dx, dy = direction_xy
        norm = math.hypot(dx, dy)
        if norm < 1e-9:
            raise ValueError("direction_xy must be non-zero")
        self._dir = np.array([dx / norm, dy / norm], dtype=np.float64)

        self._start = np.array(start_xy, dtype=np.float64)
        self._speed = float(speed)
        self._travel_dist = float(travel_dist)
        self._traveled = 0.0

        self._set_pos(self._start)

    def _set_pos(self, xy: np.ndarray) -> None:
        q = self._data.qpos
        q[self._qpos_adr + 0] = float(xy[0])
        q[self._qpos_adr + 1] = float(xy[1])
        q[self._qpos_adr + 2] = 0.9   # centre height keeps feet on floor
        q[self._qpos_adr + 3] = 1.0   # quaternion w
        q[self._qpos_adr + 4] = 0.0
        q[self._qpos_adr + 5] = 0.0
        q[self._qpos_adr + 6] = 0.0
        mujoco.mj_kinematics(self._model, self._data)

    def step(self, dt: float) -> None:
        if self.is_done:
            return
        self._traveled = min(self._traveled + self._speed * dt, self._travel_dist)
        self._set_pos(self._start + self._dir * self._traveled)

    @property
    def is_done(self) -> bool:
        return self._traveled >= self._travel_dist
```

- [ ] **Run test — expect PASS**

```
pytest g1_nav_demo/scene/test_obstacle_human.py -v
```
Expected: all 8 tests PASS.

- [ ] **Add moving_human freejoint body to g1_nav_room.xml**

In `g1_nav_demo/scene/g1_nav_room.xml`, add after the pillar body (still inside `</worldbody>`):

```xml
    <body name="moving_human" pos="1.5 -2.5 0.9">
      <freejoint name="human_freejoint"/>
      <geom name="human_body" type="cylinder" size="0.35 0.9"
            rgba="0.7 0.5 0.3 1" contype="1" conaffinity="1"/>
    </body>
```

- [ ] **Verify existing tests still pass**

```
pytest g1_nav_demo/ -v --ignore=g1_nav_demo/models -x
```
Expected: all tests PASS (freejoint body doesn't break anything).

- [ ] **Commit**

```bash
git add g1_nav_demo/scene/g1_nav_room.xml \
        g1_nav_demo/scene/obstacle_human.py \
        g1_nav_demo/scene/test_obstacle_human.py
git commit -m "feat: add ObstacleHuman class and moving_human body to scene"
```

---

## Task 3: VideoRenderer obstacle_banner

**Files:**
- Modify: `g1_nav_demo/renderer/video_renderer.py`
- Modify: `g1_nav_demo/renderer/test_video_renderer.py`

- [ ] **Write the failing tests**

Add to `g1_nav_demo/renderer/test_video_renderer.py`:

```python
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
```

- [ ] **Run test — expect FAIL**

```
pytest g1_nav_demo/renderer/test_video_renderer.py::test_obstacle_banner_default_is_none \
       g1_nav_demo/renderer/test_video_renderer.py::test_obstacle_banner_draws_yellow_top_strip -v
```
Expected: FAILED — `VideoRenderer` has no attribute `obstacle_banner`.

- [ ] **Add obstacle_banner to VideoRenderer**

In `g1_nav_demo/renderer/video_renderer.py`:

After `self.safe_banner: str | None = None` (line 58), add:

```python
        self.obstacle_banner: str | None = None
```

In `render_frame`, replace:

```python
        if self.hazard_banner:
            combined = self._overlay_banner(combined, self.hazard_banner, (0, 0, 220))
        elif self.safe_banner:
            combined = self._overlay_banner(combined, self.safe_banner, (34, 139, 34))
```

with:

```python
        if self.hazard_banner:
            combined = self._overlay_banner(combined, self.hazard_banner, (0, 0, 220))
        elif self.safe_banner:
            combined = self._overlay_banner(combined, self.safe_banner, (34, 139, 34))
        elif self.obstacle_banner:
            combined = self._overlay_banner(combined, self.obstacle_banner, (0, 200, 255))
```

`(0, 200, 255)` is BGR for golden yellow — renders as RGB (255, 200, 0).

- [ ] **Run all renderer tests — expect PASS**

```
pytest g1_nav_demo/renderer/test_video_renderer.py -v
```
Expected: all tests PASS including the two new ones.

- [ ] **Commit**

```bash
git add g1_nav_demo/renderer/video_renderer.py \
        g1_nav_demo/renderer/test_video_renderer.py
git commit -m "feat: add obstacle_banner (yellow overlay) to VideoRenderer"
```

---

## Task 4: AvoidanceStateMachine

**Files:**
- Create: `g1_nav_demo/avoidance.py`
- Create: `g1_nav_demo/test_avoidance.py`

- [ ] **Write the failing tests**

Create `g1_nav_demo/test_avoidance.py`:

```python
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from g1_nav_demo.avoidance import AvoidanceStateMachine
from g1_nav_demo.planner.goal_planner import GoalPlanner


@pytest.fixture
def asm():
    # timeout_steps=3 keeps tests short
    return AvoidanceStateMachine(
        stop_dist=1.0, clear_dist=1.2, timeout_steps=3, detour_dist=1.0
    )


def _planner(waypoints):
    p = GoalPlanner()
    p.set_waypoints(waypoints)
    return p


def test_starts_navigating_no_banner(asm):
    assert asm.is_navigating
    assert asm.banner is None


def test_clear_range_stays_navigating(asm):
    asm.step(2.0, (0.0, 0.0), MagicMock(), None)
    assert asm.is_navigating


def test_close_range_transitions_stopped(asm):
    asm.step(0.5, (0.0, 0.0), MagicMock(), None)
    assert not asm.is_navigating
    assert asm.banner is not None and "OBSTACLE" in asm.banner


def test_cleared_obstacle_resumes(asm):
    asm.step(0.5, (0.0, 0.0), MagicMock(), None)  # → STOPPED
    asm.step(1.5, (0.0, 0.0), MagicMock(), None)  # → NAVIGATING
    assert asm.is_navigating
    assert asm.banner is None


def test_timeout_triggers_reroute_then_navigating(asm):
    # 1 step to STOP + 3 blocked steps hit timeout + 1 REROUTING step → NAVIGATING
    planner = _planner([(2.0, 0.0)])
    for _ in range(5):
        asm.step(0.5, (0.0, 0.0), planner, None)
    assert asm.is_navigating


def test_reroute_inserts_left_perp_waypoint(asm):
    # Robot at (0,0), next_wp=(2,0).
    # forward=(1,0), perp-left=(0,1). detour=(0,1).
    planner = _planner([(2.0, 0.0)])
    for _ in range(5):
        asm.step(0.5, (0.0, 0.0), planner, None)
    wp = planner.current_waypoint
    assert wp[0] == pytest.approx(0.0, abs=0.01)
    assert wp[1] == pytest.approx(1.0, abs=0.01)


def test_reroute_appends_remaining_original_waypoints(asm):
    planner = _planner([(2.0, 0.0), (3.0, 1.0)])
    for _ in range(5):
        asm.step(0.5, (0.0, 0.0), planner, None)
    # new list: [detour, (2,0), (3,1)]
    assert len(planner._waypoints) == 3
    assert planner._waypoints[1] == (2.0, 0.0)
    assert planner._waypoints[2] == (3.0, 1.0)


def test_reset_returns_to_navigating(asm):
    asm.step(0.5, (0.0, 0.0), MagicMock(), None)  # → STOPPED
    asm.reset()
    assert asm.is_navigating
    assert asm.banner is None
```

- [ ] **Run test — expect FAIL**

```
pytest g1_nav_demo/test_avoidance.py -v
```
Expected: ImportError — module not found.

- [ ] **Create avoidance.py**

Create `g1_nav_demo/avoidance.py`:

```python
from __future__ import annotations

import math
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from g1_nav_demo.planner.goal_planner import GoalPlanner


class _State(Enum):
    NAVIGATING = auto()
    STOPPED = auto()
    REROUTING = auto()


class AvoidanceStateMachine:
    """Rule-based stop → wait → geometric-reroute obstacle avoidance."""

    def __init__(
        self,
        stop_dist: float = 1.0,
        clear_dist: float = 1.2,
        timeout_steps: int = 75,
        detour_dist: float = 1.0,
    ) -> None:
        self.stop_dist = stop_dist
        self.clear_dist = clear_dist
        self.timeout_steps = timeout_steps
        self.detour_dist = detour_dist
        self._state = _State.NAVIGATING
        self._blocked_steps = 0
        self.banner: str | None = None

    @property
    def is_navigating(self) -> bool:
        return self._state == _State.NAVIGATING

    def step(
        self,
        range_reading: float,
        current_pos: tuple[float, float],
        goal_planner: "GoalPlanner",
        face_yaw: float | None,
    ) -> None:
        if self._state == _State.NAVIGATING:
            self.banner = None
            if range_reading < self.stop_dist:
                self._state = _State.STOPPED
                self._blocked_steps = 0
                self.banner = "OBSTACLE DETECTED — WAITING"

        elif self._state == _State.STOPPED:
            self._blocked_steps += 1
            if range_reading >= self.clear_dist:
                self._state = _State.NAVIGATING
                self.banner = None
            elif self._blocked_steps >= self.timeout_steps:
                self._state = _State.REROUTING

        elif self._state == _State.REROUTING:
            self.banner = "REROUTING..."
            self._do_reroute(current_pos, goal_planner, face_yaw)
            self._state = _State.NAVIGATING
            self._blocked_steps = 0

    def _do_reroute(
        self,
        current_pos: tuple[float, float],
        goal_planner: "GoalPlanner",
        face_yaw: float | None,
    ) -> None:
        next_wp = goal_planner.current_waypoint
        if next_wp is None:
            return
        dx = next_wp[0] - current_pos[0]
        dy = next_wp[1] - current_pos[1]
        dist = math.hypot(dx, dy)
        forward = np.array([dx / dist, dy / dist]) if dist > 1e-3 else np.array([1.0, 0.0])
        perp = np.array([-forward[1], forward[0]])  # 90° left
        detour = np.array(current_pos, dtype=np.float64) + perp * self.detour_dist
        remaining = list(goal_planner._waypoints[goal_planner._current_wp_idx:])
        goal_planner.set_waypoints(
            [(float(detour[0]), float(detour[1]))] + remaining,
            face_yaw=face_yaw,
        )

    def reset(self) -> None:
        self._state = _State.NAVIGATING
        self._blocked_steps = 0
        self.banner = None
```

- [ ] **Run tests — expect PASS**

```
pytest g1_nav_demo/test_avoidance.py -v
```
Expected: all 8 tests PASS.

- [ ] **Commit**

```bash
git add g1_nav_demo/avoidance.py g1_nav_demo/test_avoidance.py
git commit -m "feat: add AvoidanceStateMachine with stop/wait/reroute behaviour"
```

---

## Task 5: Wire Everything — run_demo.py

**Files:**
- Modify: `g1_nav_demo/run_demo.py`
- Create: `g1_nav_demo/test_run_demo.py`

- [ ] **Write the failing rangefinder tests**

Create `g1_nav_demo/test_run_demo.py`:

```python
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
```

- [ ] **Run test — expect FAIL**

```
pytest g1_nav_demo/test_run_demo.py -v
```
Expected: ImportError — `_read_forward_range` not defined.

- [ ] **Add _read_forward_range to run_demo.py**

In `g1_nav_demo/run_demo.py`, add after the import block (before the `LEG_JOINT_NAMES` list):

```python
def _read_forward_range(
    model: "mujoco.MjModel",
    data: "mujoco.MjData",
    robot_xy: tuple[float, float],
    current_yaw: float,
    cone_deg: float = 20.0,
    cutoff: float = 3.0,
    min_dist: float = 0.5,
) -> float:
    """Return distance to nearest collidable geom in the forward cone.

    Skips floor (plane geoms) and contype-0 geoms (tabletop items).
    min_dist filters out the robot's own limb geoms which are always nearby.
    """
    robot = np.array(robot_xy, dtype=np.float64)
    cos_y = math.cos(current_yaw)
    sin_y = math.sin(current_yaw)
    forward = np.array([cos_y, sin_y])
    cos_half_cone = math.cos(math.radians(cone_deg))
    best = cutoff

    for i in range(model.ngeom):
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        if model.geom_contype[i] == 0:
            continue
        geom_xy = data.geom_xpos[i][:2]
        to_geom = geom_xy - robot
        dist = float(np.linalg.norm(to_geom))
        if dist < min_dist or dist > cutoff:
            continue
        if float(np.dot(to_geom / dist, forward)) >= cos_half_cone:
            best = min(best, dist)

    return best
```

- [ ] **Run rangefinder tests — expect PASS**

```
pytest g1_nav_demo/test_run_demo.py -v
```
Expected: all 5 tests PASS.

- [ ] **Add avoidance parameters to NavigationSession.__init__**

In `NavigationSession.__init__`, add four parameters after `torso_body_id`:

```python
        avoidance_stop_dist: float = 1.0,
        avoidance_clear_dist: float = 1.2,
        avoidance_timeout_steps: int = 75,
        avoidance_detour_dist: float = 1.0,
```

In the `__init__` body, after `self.torso_body_id = torso_body_id`, add:

```python
        self.avoidance_stop_dist = avoidance_stop_dist
        self.avoidance_clear_dist = avoidance_clear_dist
        self.avoidance_timeout_steps = avoidance_timeout_steps
        self.avoidance_detour_dist = avoidance_detour_dist
        self.obstacle_human: "ObstacleHuman | None" = None
```

Add `_init_obstacle_human` as a method on `NavigationSession`:

```python
    def _init_obstacle_human(self) -> None:
        from g1_nav_demo.scene.obstacle_human import ObstacleHuman
        body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "moving_human"
        )
        if body_id < 0:
            logger.warning("moving_human body not in model; obstacle disabled")
            return
        self.obstacle_human = ObstacleHuman(
            self.model, self.data, body_name="moving_human"
        )
        logger.info("ObstacleHuman initialised (body_id=%d)", body_id)
```

- [ ] **Replace run_to_goal_with_renderer**

Replace the existing `run_to_goal_with_renderer` method on `NavigationSession` with:

```python
    def run_to_goal_with_renderer(
        self, goal: "Goal", command: str, video_renderer: "VideoRenderer",
        face_yaw_override: float | None = None,
    ) -> bool:
        from g1_nav_demo.avoidance import AvoidanceStateMachine

        face_yaw = (
            face_yaw_override if face_yaw_override is not None
            else self._compute_face_yaw(goal)
        )
        self.goal_planner.set_waypoints(goal.waypoints, face_yaw=face_yaw)

        avoidance = AvoidanceStateMachine(
            stop_dist=self.avoidance_stop_dist,
            clear_dist=self.avoidance_clear_dist,
            timeout_steps=self.avoidance_timeout_steps,
            detour_dist=self.avoidance_detour_dist,
        )

        target_positions = self.default_angles.copy()
        velocity_command = np.zeros(3, dtype=np.float32)
        reached = False
        plan_result = None
        steps_per_render = max(1, self.sim_fps // self.render_fps)

        for step in range(self.max_steps):
            if step % self.decimation == 0:
                current_pos = self.current_position()
                current_yaw = self.current_yaw()

                range_val = _read_forward_range(
                    self.model, self.data, current_pos, current_yaw
                )
                avoidance.step(range_val, current_pos, self.goal_planner, face_yaw)
                video_renderer.obstacle_banner = avoidance.banner

                if avoidance.is_navigating:
                    plan_result = self.goal_planner.compute_command(
                        current_pos, current_yaw
                    )
                    if plan_result.reached:
                        logger.info(
                            "Reached goal at step %d (distance=%.3f)",
                            step, plan_result.distance,
                        )
                        reached = True
                        break
                    velocity_command = np.array(
                        [plan_result.vx, plan_result.vy, plan_result.vyaw],
                        dtype=np.float32,
                    )
                else:
                    velocity_command = np.zeros(3, dtype=np.float32)

                dof_pos = np.array(
                    self.data.qpos[self.leg_qpos_adr], dtype=np.float32
                )
                dof_vel = np.array(
                    self.data.qvel[self.leg_dof_adr], dtype=np.float32
                )
                angular_velocity = np.array(
                    [self.data.qvel[3], self.data.qvel[4], self.data.qvel[5]],
                    dtype=np.float32,
                )
                quaternion = np.array(
                    [self.data.qpos[3], self.data.qpos[4],
                     self.data.qpos[5], self.data.qpos[6]],
                    dtype=np.float32,
                )
                projected_gravity = G1WalkPolicy.compute_projected_gravity(quaternion)
                target_positions = self.walk_policy.get_action(
                    projected_gravity=projected_gravity,
                    velocity_command=velocity_command,
                    dof_pos=dof_pos, dof_vel=dof_vel,
                    angular_velocity=angular_velocity,
                )

            dof_pos = np.array(self.data.qpos[self.leg_qpos_adr], dtype=np.float32)
            dof_vel = np.array(self.data.qvel[self.leg_dof_adr], dtype=np.float32)
            torques = self.kps * (target_positions - dof_pos) - self.kds * dof_vel
            torques = np.clip(torques, self.ctrl_range[:, 0], self.ctrl_range[:, 1])
            self.data.ctrl[self.leg_actuator_ids] = torques

            upper_pos = np.array(self.data.qpos[self.upper_qpos_adr], dtype=np.float32)
            upper_vel = np.array(self.data.qvel[self.upper_dof_adr], dtype=np.float32)
            upper_torques = (
                self.upper_kps * (self.upper_default_pos - upper_pos)
                - self.upper_kds * upper_vel
            )
            upper_torques = np.clip(
                upper_torques,
                self.upper_ctrl_range[:, 0], self.upper_ctrl_range[:, 1],
            )
            self.data.ctrl[self.upper_actuator_ids] = upper_torques

            if self.obstacle_human is not None:
                self.obstacle_human.step(self.model.opt.timestep)

            mujoco.mj_step(self.model, self.data)

            if step % steps_per_render == 0 and plan_result is not None:
                n_wps = len(self.goal_planner._waypoints)
                wp_idx = self.goal_planner._current_wp_idx
                frame = video_renderer.render_frame(
                    self.data,
                    command=f"{command} [wp {min(wp_idx + 1, n_wps)}/{n_wps}]",
                    distance=plan_result.distance,
                    update_head_camera=True,
                    head_body_id=self.torso_body_id,
                    goal_waypoints=list(self.goal_planner._waypoints),
                    current_wp_idx=wp_idx,
                )
                video_renderer.write_frame(frame)

        if not reached:
            logger.warning("Did not reach goal within %d steps", self.max_steps)
        return reached
```

- [ ] **Add CLI flags to main()**

In `main()`, after the `--tabletop-manifest` argument, add:

```python
    parser.add_argument("--moving-obstacle", action="store_true",
                        help="Enable kinematic human obstacle crossing the robot path")
    parser.add_argument("--obstacle-stop-dist", type=float, default=1.0,
                        help="Range (m) at which robot stops for obstacle (default: 1.0)")
    parser.add_argument("--obstacle-clear-dist", type=float, default=1.2,
                        help="Range (m) at which robot resumes after obstacle clears (default: 1.2)")
    parser.add_argument("--obstacle-timeout", type=int, default=75,
                        help="Control steps blocked before reroute (~1.5 s at 50 Hz, default: 75)")
    parser.add_argument("--obstacle-detour-dist", type=float, default=1.0,
                        help="Perpendicular detour offset (m) for reroute waypoint (default: 1.0)")
```

- [ ] **Update _init_simulation to pass params and init human**

In `_init_simulation`, update the `NavigationSession(...)` constructor call to include:

```python
    session = NavigationSession(
        model=model,
        data=data,
        walk_policy=walk_policy,
        goal_planner=goal_planner,
        vlm_bridge=vlm_bridge,
        sim_fps=args.sim_fps,
        render_fps=args.render_fps,
        max_steps=args.max_steps,
        torso_body_id=torso_body_id,
        avoidance_stop_dist=args.obstacle_stop_dist,
        avoidance_clear_dist=args.obstacle_clear_dist,
        avoidance_timeout_steps=args.obstacle_timeout,
        avoidance_detour_dist=args.obstacle_detour_dist,
    )
```

After the `NavigationSession` is constructed (before the `AgentLoop` line), add:

```python
    if args.moving_obstacle:
        session._init_obstacle_human()
```

- [ ] **Run the full test suite**

```
pytest g1_nav_demo/ -v --ignore=g1_nav_demo/models -x
```
Expected: all tests PASS.

- [ ] **Commit**

```bash
git add g1_nav_demo/run_demo.py g1_nav_demo/test_run_demo.py
git commit -m "feat: wire forward rangefinder, AvoidanceStateMachine, and ObstacleHuman into NavigationSession"
```

---

## Self-Review Notes

- `AvoidanceStateMachine.banner` is set to `None` at the start of every NAVIGATING step — so "REROUTING..." persists for exactly one control-step then clears automatically.
- `_read_forward_range` uses `min_dist=0.5` by default to filter robot limb geoms; tests use `min_dist=0.1` to detect obstacles as close as 0.8 m.
- After rerouting, `goal_planner.set_waypoints` resets `_current_wp_idx` to 0, so the detour becomes the immediate next waypoint.
- `obstacle_human` is `None` when `--moving-obstacle` is not passed, and the sim loop skips its step unconditionally.
- All existing tests that load `g1_nav_room.xml` continue to pass — new bodies don't change camera names or DOF ordering of the robot.
