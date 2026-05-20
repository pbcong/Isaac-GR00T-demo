from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class PlanResult:
    vx: float
    vy: float
    vyaw: float
    reached: bool
    distance: float


class GoalPlanner:
    def __init__(
        self,
        max_vx: float = 1.0,
        max_vy: float = 0.5,
        max_vyaw: float = 1.5,
        kp_x: float = 1.0,
        kp_y: float = 0.5,
        kp_yaw: float = 2.0,
        goal_threshold: float = 0.2,
        waypoint_threshold: float = 0.4,
        angle_threshold: float = 0.15,
        slow_distance: float = 1.0,
        face_yaw_threshold: float = 0.4,
    ) -> None:
        self.max_vx = max_vx
        self.max_vy = max_vy
        self.max_vyaw = max_vyaw
        self.kp_x = kp_x
        self.kp_y = kp_y
        self.kp_yaw = kp_yaw
        self.goal_threshold = goal_threshold
        self.waypoint_threshold = waypoint_threshold
        self.angle_threshold = angle_threshold
        self.slow_distance = slow_distance
        self.face_yaw_threshold = face_yaw_threshold
        self._waypoints: list[tuple[float, float]] = []
        self._current_wp_idx: int = 0
        self._face_yaw: float | None = None
        self._orient_steps: int = 0

    def set_waypoints(
        self,
        waypoints: list[tuple[float, float]],
        face_yaw: float | None = None,
    ) -> None:
        self._waypoints = list(waypoints)
        self._current_wp_idx = 0
        self._face_yaw = face_yaw

    @property
    def current_waypoint(self) -> tuple[float, float] | None:
        if self._current_wp_idx < len(self._waypoints):
            return self._waypoints[self._current_wp_idx]
        return None

    @property
    def is_final_waypoint(self) -> bool:
        return self._current_wp_idx >= len(self._waypoints) - 1

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def compute_command(
        self,
        current_pos: tuple[float, float],
        current_yaw: float,
    ) -> PlanResult:
        if not self._waypoints:
            return PlanResult(vx=0.0, vy=0.0, vyaw=0.0, reached=True, distance=0.0)

        wp = self.current_waypoint
        if wp is None:
            return PlanResult(vx=0.0, vy=0.0, vyaw=0.0, reached=True, distance=0.0)

        dx = wp[0] - current_pos[0]
        dy = wp[1] - current_pos[1]
        distance = math.hypot(dx, dy)

        is_final = self.is_final_waypoint
        threshold = self.goal_threshold if is_final else self.waypoint_threshold

        if is_final and distance < self.goal_threshold:
            if self._face_yaw is not None:
                yaw_err = self._wrap_angle(self._face_yaw - current_yaw)
                if abs(yaw_err) < self.face_yaw_threshold:
                    return PlanResult(vx=0.0, vy=0.0, vyaw=0.0, reached=True, distance=distance)
                vyaw_face = self.kp_yaw * yaw_err
                vyaw_face = max(-self.max_vyaw, min(self.max_vyaw, vyaw_face))
                return PlanResult(vx=0.0, vy=0.0, vyaw=vyaw_face, reached=False, distance=distance)
            return PlanResult(vx=0.0, vy=0.0, vyaw=0.0, reached=True, distance=distance)

        if not is_final and distance < self.waypoint_threshold:
            self._current_wp_idx += 1
            wp = self.current_waypoint
            if wp is None:
                return PlanResult(vx=0.0, vy=0.0, vyaw=0.0, reached=True, distance=distance)
            dx = wp[0] - current_pos[0]
            dy = wp[1] - current_pos[1]
            distance = math.hypot(dx, dy)

        cos_y = math.cos(current_yaw)
        sin_y = math.sin(current_yaw)
        dx_body = dx * cos_y + dy * sin_y
        dy_body = -dx * sin_y + dy * cos_y

        desired_yaw = math.atan2(dy, dx)
        yaw_error = self._wrap_angle(desired_yaw - current_yaw)

        speed_scale = min(1.0, distance / self.slow_distance)

        if is_final and distance < self.goal_threshold * 2:
            speed_scale *= 0.5

        yaw_scale = min(1.0, distance / self.slow_distance) if distance > self.goal_threshold else 1.0

        vx = self.kp_x * dx_body * speed_scale
        vy = self.kp_y * dy_body * speed_scale
        vyaw = self.kp_yaw * yaw_error * yaw_scale

        if abs(yaw_error) > 4 * self.angle_threshold:
            vx *= 0.2
            vy *= 0.2

        vx = max(-self.max_vx, min(self.max_vx, vx))
        vy = max(-self.max_vy, min(self.max_vy, vy))
        vyaw = max(-self.max_vyaw, min(self.max_vyaw, vyaw))

        reached = False
        return PlanResult(vx=vx, vy=vy, vyaw=vyaw, reached=reached, distance=distance)