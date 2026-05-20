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
        stand_steps: int = 0,
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
        self._stand_steps = int(stand_steps)

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
        if self._stand_steps > 0:
            self._stand_steps -= 1
            return
        self._traveled = min(self._traveled + self._speed * dt, self._travel_dist)
        self._set_pos(self._start + self._dir * self._traveled)

    @property
    def is_done(self) -> bool:
        return self._traveled >= self._travel_dist