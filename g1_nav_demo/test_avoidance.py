from __future__ import annotations

import pytest

from g1_nav_demo.avoidance import AvoidanceStateMachine


@pytest.fixture
def asm():
    # timeout_steps=3 keeps tests short
    return AvoidanceStateMachine(
        stop_dist=1.0, clear_dist=1.2, timeout_steps=3, detour_dist=1.0
    )


def test_starts_navigating_no_banner(asm):
    assert asm.is_navigating
    assert asm.banner is None


def test_clear_range_stays_navigating(asm):
    asm.step(2.0, (0.0, 0.0), None)
    assert asm.is_navigating


def test_close_range_transitions_stopped(asm):
    asm.step(0.5, (0.0, 0.0), None)
    assert not asm.is_navigating
    assert asm.banner is not None and "OBSTACLE" in asm.banner


def test_cleared_obstacle_resumes(asm):
    asm.step(0.5, (0.0, 0.0), None)  # -> STOPPED
    asm.step(1.5, (0.0, 0.0), None)  # -> NAVIGATING
    assert asm.is_navigating
    assert asm.banner is None


def test_timeout_triggers_reroute_signal(asm):
    # 1 step to STOP + 3 blocked steps hit timeout + 1 REROUTING step returns True
    results = [asm.step(0.5, (0.0, 0.0), None) for _ in range(5)]
    assert results[-1] is True


def test_after_reroute_signal_state_is_navigating(asm):
    for _ in range(5):
        asm.step(0.5, (0.0, 0.0), None)
    assert asm.is_navigating


def test_step_returns_false_normally(asm):
    assert asm.step(2.0, (0.0, 0.0), None) is False


def test_reset_returns_to_navigating(asm):
    asm.step(0.5, (0.0, 0.0), None)  # -> STOPPED
    asm.reset()
    assert asm.is_navigating
    assert asm.banner is None
