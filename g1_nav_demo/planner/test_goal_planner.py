from __future__ import annotations

import math

from g1_nav_demo.planner.goal_planner import GoalPlanner, PlanResult


def _planner_with_wps(planner: GoalPlanner, wps: list) -> GoalPlanner:
    planner.set_waypoints(wps)
    return planner


def test_goal_reached() -> None:
    planner = GoalPlanner()
    planner.set_waypoints([(0.0, 0.0)])
    result = planner.compute_command((0.0, 0.0), 0.0)
    assert result.reached is True
    assert result.vx == 0.0
    assert result.vy == 0.0
    assert result.vyaw == 0.0


def test_goal_ahead() -> None:
    planner = GoalPlanner()
    planner.set_waypoints([(2.0, 0.0)])
    result = planner.compute_command((0.0, 0.0), 0.0)
    assert result.reached is False
    assert result.vx > 0


def test_goal_behind() -> None:
    planner = GoalPlanner()
    planner.set_waypoints([(-2.0, 0.0)])
    result = planner.compute_command((0.0, 0.0), 0.0)
    assert result.reached is False
    assert abs(result.vyaw) > 1.0


def test_no_waypoints() -> None:
    planner = GoalPlanner()
    result = planner.compute_command((0.0, 0.0), 0.0)
    assert result.reached is True


def test_wrap_angle() -> None:
    assert abs(GoalPlanner._wrap_angle(0.0)) < 1e-9
    assert abs(GoalPlanner._wrap_angle(math.pi) - math.pi) < 1e-9
    assert abs(GoalPlanner._wrap_angle(-math.pi) + math.pi) < 1e-9
    assert abs(GoalPlanner._wrap_angle(3 * math.pi) - math.pi) < 1e-9


def test_slow_near_goal() -> None:
    planner_close = GoalPlanner(slow_distance=1.0)
    planner_close.set_waypoints([(0.5, 0.0)])
    close = planner_close.compute_command((0.0, 0.0), 0.0)

    planner_far = GoalPlanner(slow_distance=1.0)
    planner_far.set_waypoints([(3.0, 0.0)])
    far = planner_far.compute_command((0.0, 0.0), 0.0)

    assert abs(close.vx) < abs(far.vx)


def test_clamping() -> None:
    planner = GoalPlanner(max_vx=0.5)
    planner.set_waypoints([(100.0, 0.0)])
    result = planner.compute_command((0.0, 0.0), 0.0)
    assert result.vx <= 0.5


def test_angle_priority() -> None:
    planner = GoalPlanner(angle_threshold=0.15)
    planner.set_waypoints([(-2.0, 0.0)])
    result = planner.compute_command((0.0, 0.0), 0.0)
    assert abs(result.vyaw) > 4 * 0.15


def test_waypoint_advancement() -> None:
    planner = GoalPlanner(waypoint_threshold=0.5, goal_threshold=0.5)
    planner.set_waypoints([(1.0, 0.0), (3.0, 2.0)])
    result = planner.compute_command((0.8, 0.0), 0.0)
    assert result.reached is False
    assert planner._current_wp_idx == 1


def test_waypoint_final_reached() -> None:
    planner = GoalPlanner(waypoint_threshold=0.4, goal_threshold=0.5)
    planner.set_waypoints([(3.0, 2.0)])
    result = planner.compute_command((3.0, 2.1), 0.0)
    assert result.reached is True
