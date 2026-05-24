from __future__ import annotations

from enum import Enum, auto


class _State(Enum):
    NAVIGATING = auto()
    STOPPED = auto()
    REROUTING = auto()


class AvoidanceStateMachine:
    """Rule-based stop -> wait -> reroute obstacle avoidance."""

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
        face_yaw: float | None,
    ) -> bool:
        """Advance state machine. Returns True when a reroute should be triggered."""
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
            self._state = _State.NAVIGATING
            self._blocked_steps = 0
            return True

        return False

    def reset(self) -> None:
        self._state = _State.NAVIGATING
        self._blocked_steps = 0
        self.banner = None
