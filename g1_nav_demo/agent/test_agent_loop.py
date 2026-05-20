from __future__ import annotations

import base64
import json
from unittest.mock import MagicMock

import pytest

from g1_nav_demo.agent.agent_loop import (
    _NO_TOOL_CALL_RETRIES,
    TOOL_SCHEMAS,
    AgentLoop,
    _image_tool_result,
    _text_tool_result,
    _write_report_json,
)


def test_tool_schemas_have_expected_names():
    names = {t["function"]["name"] for t in TOOL_SCHEMAS}
    assert names == {"navigate", "look", "report"}


def test_tool_schemas_all_have_description():
    for schema in TOOL_SCHEMAS:
        assert schema["function"]["description"], f"{schema['function']['name']} missing description"


def test_text_tool_result_format():
    msg = _text_tool_result("call-1", "hello")
    assert msg == {"role": "tool", "tool_call_id": "call-1", "content": "hello"}


def test_image_tool_result_format():
    msg = _image_tool_result("call-2", "abc123==")
    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "call-2"
    assert msg["content"][0]["type"] == "image_url"
    assert "abc123==" in msg["content"][0]["image_url"]["url"]


def test_write_report_json_writes_all_fields(tmp_path):
    path = str(tmp_path / "report.json")
    result = {
        "verdict": "safe",
        "findings": [{"name": "mug", "hazardous": False, "reason": "ceramic"}],
        "message": "All clear",
    }
    _write_report_json(path, "inspect the table", result)
    data = json.loads(open(path).read())
    assert data["verdict"] == "safe"
    assert data["command"] == "inspect the table"
    assert data["message"] == "All clear"
    assert data["findings"][0]["name"] == "mug"


def test_write_report_json_creates_parent_dirs(tmp_path):
    path = str(tmp_path / "nested" / "dir" / "report.json")
    _write_report_json(path, "cmd", {"verdict": "complete", "findings": [], "message": "done"})
    assert json.loads(open(path).read())["verdict"] == "complete"


def _make_loop():
    session = MagicMock()
    return AgentLoop(session=session, model_name="test-model", api_key="k"), session


def test_handle_navigate_success():
    loop, session = _make_loop()
    session.parse_goal.return_value = MagicMock()
    session.run_to_goal_with_renderer.return_value = True
    session.current_position.return_value = (2.4, 1.2)

    outcome = loop._handle_navigate("Go to the table", "cmd", MagicMock())

    assert outcome["reached"] is True
    assert outcome["position"] == [2.4, 1.2]
    session.parse_goal.assert_called_once_with("Go to the table")


def test_handle_navigate_parse_failure():
    loop, session = _make_loop()
    session.parse_goal.return_value = None

    outcome = loop._handle_navigate("nowhere", "cmd", MagicMock())

    assert outcome["reached"] is False
    assert "reason" in outcome


def test_handle_navigate_nav_failure():
    loop, session = _make_loop()
    session.parse_goal.return_value = MagicMock()
    session.run_to_goal_with_renderer.return_value = False
    session.current_position.return_value = (0.0, 0.0)

    outcome = loop._handle_navigate("Go far away", "cmd", MagicMock())

    assert outcome["reached"] is False


def test_handle_look_idles_and_snapshots():
    loop, session = _make_loop()
    renderer = MagicMock()
    renderer.snapshot.return_value = b"png-bytes"

    loop._snap_prefix = "/tmp/test"
    loop._look_count = 0
    loop._handle_look(renderer)

    session.idle.assert_called_once_with(duration_steps=250, video_renderer=renderer)
    renderer.snapshot.assert_called_once_with(
        "head_onboard", session.data, width=1280, height=960
    )


def test_handle_look_returns_base64():
    loop, session = _make_loop()
    renderer = MagicMock()
    renderer.snapshot.return_value = b"fake-png"

    loop._snap_prefix = "/tmp/test"
    loop._look_count = 0
    result = loop._handle_look(renderer)

    assert result == base64.b64encode(b"fake-png").decode()


def test_handle_report_writes_json(tmp_path):
    loop, session = _make_loop()
    renderer = MagicMock()
    renderer.safe_banner = None
    result = {"verdict": "safe", "findings": [], "message": "All clear"}
    path = str(tmp_path / "report.json")

    loop._handle_report(result, "inspect table", path, renderer)

    data = json.loads(open(path).read())
    assert data["verdict"] == "safe"
    assert data["command"] == "inspect table"


def test_handle_report_sets_hazard_banner_then_clears(tmp_path):
    loop, session = _make_loop()
    banner_log: list = []

    class TrackedRenderer(MagicMock):
        def __setattr__(self, name, value):
            if name == "hazard_banner":
                banner_log.append(value)
            super().__setattr__(name, value)

    renderer = TrackedRenderer()
    result = {
        "verdict": "hazardous",
        "findings": [{"name": "radioactive box", "hazardous": True, "reason": "trefoil"}],
        "message": "Hazard found",
    }
    loop._handle_report(result, "cmd", str(tmp_path / "r.json"), renderer)

    assert any(isinstance(v, str) and "radioactive box" in v for v in banner_log)
    assert banner_log[-1] is None


def test_handle_report_sets_safe_banner_then_clears(tmp_path):
    loop, session = _make_loop()
    banner_log: list = []

    class TrackedRenderer(MagicMock):
        def __setattr__(self, name, value):
            if name == "safe_banner":
                banner_log.append(value)
            super().__setattr__(name, value)

    renderer = TrackedRenderer()
    result = {"verdict": "complete", "findings": [], "message": "Done"}
    loop._handle_report(result, "cmd", str(tmp_path / "r.json"), renderer)

    assert any(isinstance(v, str) and "COMPLETE" in v for v in banner_log)
    assert banner_log[-1] is None


def _make_tool_call(name: str, args: dict, call_id: str):
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(args)
    return tc


def test_run_turn_navigate_then_report(tmp_path):
    loop, session = _make_loop()
    session.parse_goal.return_value = MagicMock()
    session.run_to_goal_with_renderer.return_value = True
    session.current_position.return_value = (2.4, 1.2)

    nav_msg = MagicMock()
    nav_msg.tool_calls = [_make_tool_call("navigate", {"instruction": "Go to table"}, "tc-1")]

    rep_msg = MagicMock()
    rep_msg.tool_calls = [
        _make_tool_call("report", {"verdict": "complete", "message": "Reached table"}, "tc-2")
    ]

    client_mock = MagicMock()
    client_mock.chat.completions.create.side_effect = [
        MagicMock(choices=[MagicMock(message=nav_msg)]),
        MagicMock(choices=[MagicMock(message=rep_msg)]),
    ]
    loop._client = client_mock

    renderer = MagicMock()
    renderer.safe_banner = None
    result = loop.run_turn("Go to table", renderer, str(tmp_path / "report.json"))

    assert result["verdict"] == "complete"
    assert result["message"] == "Reached table"
    assert client_mock.chat.completions.create.call_count == 2


def test_run_turn_stops_when_no_tool_calls(tmp_path):
    loop, _ = _make_loop()
    no_tools_msg = MagicMock()
    no_tools_msg.tool_calls = None

    client_mock = MagicMock()
    client_mock.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=no_tools_msg)]
    )
    loop._client = client_mock

    result = loop.run_turn("do something", MagicMock(), str(tmp_path / "r.json"))

    assert result["verdict"] == "failed"
    assert client_mock.chat.completions.create.call_count == 1 + _NO_TOOL_CALL_RETRIES


def test_run_turn_look_sends_image_as_user_message(tmp_path):
    loop, session = _make_loop()
    session.idle = MagicMock()
    session.data = MagicMock()
    look_renderer = MagicMock()
    look_renderer.snapshot.return_value = b"\x89PNG\r\nfake-image"
    look_renderer.safe_banner = None

    look_msg = MagicMock()
    look_msg.tool_calls = [_make_tool_call("look", {}, "tc-look")]

    rep_msg = MagicMock()
    rep_msg.tool_calls = [
        _make_tool_call("report", {"verdict": "safe", "message": "All clear"}, "tc-rep"),
    ]

    client_mock = MagicMock()
    client_mock.chat.completions.create.side_effect = [
        MagicMock(choices=[MagicMock(message=look_msg)]),
        MagicMock(choices=[MagicMock(message=look_msg)]),
        MagicMock(choices=[MagicMock(message=look_msg)]),
        MagicMock(choices=[MagicMock(message=rep_msg)]),
    ]
    loop._client = client_mock

    result = loop.run_turn("inspect table", look_renderer, str(tmp_path / "report.json"))

    assert result["verdict"] == "safe"

    call_args_list = client_mock.chat.completions.create.call_args_list
    assert len(call_args_list) == 4

    messages_last_call = call_args_list[3].kwargs["messages"]

    has_user_image = False
    for msg in messages_last_call:
        if msg.get("role") == "user" and isinstance(msg.get("content"), list):
            for block in msg["content"]:
                if block.get("type") == "image_url":
                    has_user_image = True

    assert has_user_image, "Expected a user message with image_url content after look()"


def test_run_turn_safe_report_rejected_with_few_looks(tmp_path):
    loop, session = _make_loop()
    session.parse_goal.return_value = MagicMock()
    session.run_to_goal_with_renderer.return_value = True
    session.current_position.return_value = (1.0, 2.0)
    session.data = MagicMock()
    renderer = MagicMock()
    renderer.snapshot.return_value = b"\x89PNG\r\nfake-image"
    renderer.safe_banner = None

    look_msg = MagicMock()
    look_msg.tool_calls = [_make_tool_call("look", {}, "tc-look")]

    premature_safe_msg = MagicMock()
    premature_safe_msg.tool_calls = [
        _make_tool_call("report", {"verdict": "safe", "message": "All clear"}, "tc-early"),
    ]

    nav_msg = MagicMock()
    nav_msg.tool_calls = [_make_tool_call("navigate", {"instruction": "Go to back side"}, "tc-nav")]

    safe_enough_msg = MagicMock()
    safe_enough_msg.tool_calls = [
        _make_tool_call("report", {"verdict": "safe", "message": "Inspected all sides"}, "tc-final"),
    ]

    client_mock = MagicMock()
    client_mock.chat.completions.create.side_effect = [
        MagicMock(choices=[MagicMock(message=look_msg)]),
        MagicMock(choices=[MagicMock(message=premature_safe_msg)]),
        MagicMock(choices=[MagicMock(message=nav_msg)]),
        MagicMock(choices=[MagicMock(message=look_msg)]),
        MagicMock(choices=[MagicMock(message=look_msg)]),
        MagicMock(choices=[MagicMock(message=safe_enough_msg)]),
    ]
    loop._client = client_mock

    result = loop.run_turn("inspect the box", renderer, str(tmp_path / "report.json"))

    assert result["verdict"] == "safe"
    assert result["message"] == "Inspected all sides"
    assert client_mock.chat.completions.create.call_count == 6