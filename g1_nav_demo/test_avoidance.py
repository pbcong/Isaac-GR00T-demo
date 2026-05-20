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
    asm.step(0.5, (0.0, 0.0), MagicMock(), None)  # -> STOPPED
    asm.step(1.5, (0.0, 0.0), MagicMock(), None)  # -> NAVIGATING
    assert asm.is_navigating
    assert asm.banner is None


def test_timeout_triggers_reroute_then_navigating(asm):
    # 1 step to STOP + 3 blocked steps hit timeout + 1 REROUTING step -> NAVIGATING
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
    asm.step(0.5, (0.0, 0.0), MagicMock(), None)  # -> STOPPED
    asm.reset()
    assert asm.is_navigating
    assert asm.banner is None