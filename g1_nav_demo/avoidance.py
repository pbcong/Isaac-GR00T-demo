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
    """Rule-based stop -> wait -> geometric-reroute obstacle avoidance."""

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
        perp = np.array([-forward[1], forward[0]])  # 90 degrees left
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