from g1_nav_demo.vlm.goal_parser import Goal, VLMBridge


def _parse(text):
    return VLMBridge()._extract_goal(text)


def test_extract_valid_json():
    goal = _parse('{"target_name": "table", "waypoints": [[1.0, 0.5], [2.3, 2.0]]}')
    assert goal is not None
    assert goal.target_name == "table"
    assert goal.waypoints == [(1.0, 0.5), (2.3, 2.0)]


def test_extract_strips_markdown():
    goal = _parse('```json\n{"target_name": "chair", "waypoints": [[1.0, 2.5]]}\n```')
    assert goal is not None
    assert goal.target_name == "chair"


def test_extract_invalid_json():
    assert _parse("I cannot find the target.") is None


def test_extract_missing_waypoints():
    assert _parse('{"target_name": "table"}') is None


def test_extract_empty_waypoints():
    assert _parse('{"target_name": "table", "waypoints": []}') is None


def test_goal_properties():
    goal = Goal(target_name="table", waypoints=[(1.0, 0.5), (2.3, 2.0)])
    assert goal.x == 2.3
    assert goal.y == 2.0


def test_extract_strips_thinking_block():
    text = (
        "<think>\nI see a table at roughly (3,2). "
        "Let me plan around it...\n</think>\n"
        '{"target_name": "table", "waypoints": [[1.5, 0.5], [2.3, 2.0]]}'
    )
    goal = _parse(text)
    assert goal is not None
    assert goal.target_name == "table"
    assert len(goal.waypoints) == 2


def test_vlm_bridge_returns_none_on_connection_error():
    bridge = VLMBridge(model_name="any/model", api_base="http://localhost:19999/v1")
    result = bridge.parse("go to the table")
    assert result is None


def test_extract_inspect_true():
    goal = _parse('{"target_name": "table", "waypoints": [[1.0, 0.5]], "inspect": true}')
    assert goal is not None
    assert goal.inspect is True


def test_extract_inspect_default_false():
    goal = _parse('{"target_name": "table", "waypoints": [[1.0, 0.5]]}')
    assert goal is not None
    assert goal.inspect is False


def test_extract_inspect_explicit_false():
    goal = _parse(
        '{"target_name": "table", "waypoints": [[1.0, 0.5]], "inspect": false}'
    )
    assert goal is not None
    assert goal.inspect is False
