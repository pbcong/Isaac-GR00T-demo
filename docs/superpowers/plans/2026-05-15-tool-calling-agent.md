# Tool-Calling Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hardcoded `parse_goal → run_to_goal → inspect_target_agentic` orchestration with a pure-Python LLM tool-calling loop (`AgentLoop`) that exposes three tools — `navigate`, `look`, `report` — and lets the model chain them freely.

**Architecture:** A new `AgentLoop` class holds a reference to `NavigationSession` and a lazy OpenAI client. `run_turn(command, video_renderer, report_json_path)` builds a conversation, calls the LLM in a loop dispatching tool calls until `report()` is called or `MAX_TURNS` is hit. `NavigationSession` stays unchanged as the low-level executor; `InspectionBridge` and the old agentic loop are deleted.

**Tech Stack:** Python 3.11+, `openai` SDK (already in deps), `pytest` + `unittest.mock` for tests.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `g1_nav_demo/agent/__init__.py` | Create | Empty package marker |
| `g1_nav_demo/agent/agent_loop.py` | Create | `AgentLoop`, `TOOL_SCHEMAS`, `SYSTEM_PROMPT`, helper fns |
| `g1_nav_demo/agent/test_agent_loop.py` | Create | Unit tests (all mocked, no MuJoCo) |
| `g1_nav_demo/run_demo.py` | Modify | Remove old inspection methods; wire `AgentLoop` into run functions |
| `g1_nav_demo/vlm/inspection.py` | Delete | Superseded by agent |
| `g1_nav_demo/vlm/test_inspection.py` | Delete | Tests deleted code |
| `g1_nav_demo/test_run_demo.py` | Delete | Tests deleted `inspect_target*` methods |

---

## Task 1: Create agent package with helpers and tool schemas

**Files:**
- Create: `g1_nav_demo/agent/__init__.py`
- Create: `g1_nav_demo/agent/agent_loop.py` (helpers + schemas only, no `AgentLoop` class yet)
- Create: `g1_nav_demo/agent/test_agent_loop.py`

- [ ] **Step 1: Write the failing tests**

```python
# g1_nav_demo/agent/test_agent_loop.py
from __future__ import annotations

import base64
import json
from unittest.mock import MagicMock

import pytest

from g1_nav_demo.agent.agent_loop import (
    TOOL_SCHEMAS,
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest g1_nav_demo/agent/test_agent_loop.py -v
```

Expected: `ModuleNotFoundError: No module named 'g1_nav_demo.agent'`

- [ ] **Step 3: Create the package files**

```python
# g1_nav_demo/agent/__init__.py
# (empty)
```

```python
# g1_nav_demo/agent/agent_loop.py
from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

MAX_TURNS = 20
_OPENROUTER_BASE = "https://openrouter.ai/api/v1"

SYSTEM_PROMPT = """\
You are a safety-inspection robot operating in a room. You have three tools:

- navigate(instruction): Move the robot to a location. Describe where in natural language.
- look(): Take a head-camera snapshot. Returns an image of what the robot sees right now.
- report(verdict, findings, message): End the mission turn and report what you found.

INSPECTION STRATEGY:
- To inspect an object: navigate to it, look at it from multiple angles, then report.
- Hazard indicators: red/orange box with diamond symbols, radioactive trefoil, flame, skull = HAZARDOUS.
- Seeing a face obliquely does NOT count as inspecting it — stand directly in front of each face.
- Declare "hazardous" immediately on confirmed symbol. Declare "safe" only after all accessible faces seen.
- Non-inspection missions (e.g. "go to the table"): navigate and call report(verdict="complete").
- If navigation fails, report(verdict="failed").

Always end the mission by calling report().
"""

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "navigate",
            "description": (
                "Navigate the robot to a location described in natural language. "
                "Handles path planning and execution internally. "
                "Use this to approach objects or reposition for a better view."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": (
                            "Natural language navigation instruction, "
                            "e.g. 'Go to the front of the table and face it'"
                        ),
                    }
                },
                "required": ["instruction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "look",
            "description": (
                "Take a snapshot from the robot's head camera. "
                "Returns an image of what the robot currently sees. "
                "Use this to observe objects or check for hazards."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "report",
            "description": (
                "End the current mission turn and report findings. "
                "Call this when you have enough information to conclude the mission."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "verdict": {
                        "type": "string",
                        "enum": ["safe", "hazardous", "complete", "failed"],
                        "description": "Mission outcome",
                    },
                    "findings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "hazardous": {"type": "boolean"},
                                "reason": {"type": "string"},
                            },
                            "required": ["name", "hazardous"],
                        },
                        "description": "Items observed (empty list ok for non-inspection missions)",
                    },
                    "message": {
                        "type": "string",
                        "description": "Human-readable summary of what was done and found",
                    },
                },
                "required": ["verdict", "message"],
            },
        },
    },
]


def _text_tool_result(tool_call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def _image_tool_result(tool_call_id: str, image_b64: str) -> dict:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
            }
        ],
    }


def _write_report_json(path: str, command: str, result: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = {
        "command": command,
        "verdict": result["verdict"],
        "message": result["message"],
        "findings": result.get("findings", []),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest g1_nav_demo/agent/test_agent_loop.py -v
```

Expected: 5 passed.

---

## Task 2: Implement navigate and look handlers

**Files:**
- Modify: `g1_nav_demo/agent/agent_loop.py` (add `AgentLoop` class with `_handle_navigate`, `_handle_look`)
- Modify: `g1_nav_demo/agent/test_agent_loop.py` (add handler tests)

- [ ] **Step 1: Add handler tests**

Add `AgentLoop` to the existing import block at the top of `g1_nav_demo/agent/test_agent_loop.py`:

```python
from g1_nav_demo.agent.agent_loop import (
    TOOL_SCHEMAS,
    AgentLoop,
    _image_tool_result,
    _text_tool_result,
    _write_report_json,
)
```

Then append these test functions at the end of the file:

```python


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
```

- [ ] **Step 2: Run tests to confirm new ones fail**

```bash
python -m pytest g1_nav_demo/agent/test_agent_loop.py -v
```

Expected: `ImportError: cannot import name 'AgentLoop'` — all collection fails until the class is added in the next step.

- [ ] **Step 3: Add AgentLoop class with navigate and look handlers**

Append to `g1_nav_demo/agent/agent_loop.py`:

```python
import base64
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from g1_nav_demo.run_demo import NavigationSession
    from g1_nav_demo.renderer.video_renderer import VideoRenderer


class AgentLoop:
    def __init__(
        self,
        session: "NavigationSession",
        model_name: str = "x-ai/grok-4.3",
        api_base: str = _OPENROUTER_BASE,
        api_key: str | None = None,
    ) -> None:
        self.session = session
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self._client = None
        self._snap_prefix: str = "look"
        self._look_count: int = 0

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.api_base, api_key=self.api_key)
        return self._client

    def _handle_navigate(
        self,
        instruction: str,
        original_command: str,
        video_renderer: "VideoRenderer",
    ) -> dict:
        goal = self.session.parse_goal(instruction)
        if goal is None:
            logger.warning("navigate: could not plan path for %r", instruction)
            return {"reached": False, "reason": "Could not plan path for instruction"}
        reached = self.session.run_to_goal_with_renderer(goal, instruction, video_renderer)
        pos = self.session.current_position()
        return {"reached": reached, "position": list(pos)}

    def _handle_look(self, video_renderer: "VideoRenderer") -> str:
        self.session.idle(duration_steps=250, video_renderer=video_renderer)
        png_bytes = video_renderer.snapshot(
            "head_onboard", self.session.data, width=1280, height=960
        )
        self._look_count += 1
        snap_path = f"{self._snap_prefix}_look_{self._look_count:02d}.png"
        try:
            with open(snap_path, "wb") as f:
                f.write(png_bytes)
            logger.debug("Saved look snapshot → %s", snap_path)
        except OSError as e:
            logger.warning("Could not save look snapshot: %s", e)
        return base64.b64encode(png_bytes).decode()
```

- [ ] **Step 4: Run tests to confirm all pass**

```bash
python -m pytest g1_nav_demo/agent/test_agent_loop.py -v
```

Expected: 10 passed (5 helper tests + 5 handler tests).

---

## Task 3: Implement report handler and run_turn loop

**Files:**
- Modify: `g1_nav_demo/agent/agent_loop.py` (add `_handle_report`, `run_turn`)
- Modify: `g1_nav_demo/agent/test_agent_loop.py` (add report + run_turn tests)
- Commit

- [ ] **Step 1: Add report and run_turn tests**

Append to `g1_nav_demo/agent/test_agent_loop.py`:

```python
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
    assert client_mock.chat.completions.create.call_count == 1
```

- [ ] **Step 2: Run tests to confirm new ones fail**

```bash
python -m pytest g1_nav_demo/agent/test_agent_loop.py -v
```

Expected: 10 pass, 6 fail with `AttributeError`.

- [ ] **Step 3: Add `_handle_report` and `run_turn` to AgentLoop in `agent_loop.py`**

Add these two methods inside the `AgentLoop` class:

```python
    def run_turn(
        self,
        command: str,
        video_renderer: "VideoRenderer",
        report_json_path: str,
    ) -> dict:
        self._snap_prefix = report_json_path.rsplit(".", 1)[0]
        self._look_count = 0

        history = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": command},
        ]
        result = {
            "verdict": "failed",
            "findings": [],
            "message": "Agent did not call report()",
        }

        for _ in range(MAX_TURNS):
            response = self._get_client().chat.completions.create(
                model=self.model_name,
                messages=history,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                extra_body={"reasoning": {"enabled": True}},
            )
            msg = response.choices[0].message
            history.append(msg)

            if not msg.tool_calls:
                logger.warning("Agent returned no tool calls; ending turn")
                break

            done = False
            tool_results = []
            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)

                if name == "navigate":
                    outcome = self._handle_navigate(args["instruction"], command, video_renderer)
                    tool_results.append(_text_tool_result(tc.id, json.dumps(outcome)))

                elif name == "look":
                    image_b64 = self._handle_look(video_renderer)
                    tool_results.append(_image_tool_result(tc.id, image_b64))

                elif name == "report":
                    result = {
                        "verdict": args["verdict"],
                        "findings": args.get("findings", []),
                        "message": args["message"],
                    }
                    self._handle_report(result, command, report_json_path, video_renderer)
                    tool_results.append(_text_tool_result(tc.id, "Turn complete."))
                    done = True

            history.extend(tool_results)
            if done:
                break

        return result

    def _handle_report(
        self,
        result: dict,
        command: str,
        json_path: str,
        video_renderer: "VideoRenderer",
    ) -> None:
        _write_report_json(json_path, command, result)
        verdict = result["verdict"]
        if verdict == "hazardous":
            names = ", ".join(
                f["name"] for f in result.get("findings", []) if f.get("hazardous")
            )
            video_renderer.hazard_banner = f"HAZARD DETECTED: {names}"
            self.session.idle(
                duration_steps=1500, video_renderer=video_renderer, command=command
            )
            video_renderer.hazard_banner = None
        elif verdict in ("safe", "complete"):
            video_renderer.safe_banner = f"MISSION COMPLETE: {verdict.upper()}"
            self.session.idle(
                duration_steps=1500, video_renderer=video_renderer, command=command
            )
            video_renderer.safe_banner = None
```

- [ ] **Step 4: Run tests to confirm all 16 pass**

```bash
python -m pytest g1_nav_demo/agent/test_agent_loop.py -v
```

Expected: 16 passed.

- [ ] **Step 5: Commit**

```bash
git add g1_nav_demo/agent/__init__.py g1_nav_demo/agent/agent_loop.py g1_nav_demo/agent/test_agent_loop.py
git commit -m "Add AgentLoop with navigate/look/report tool-calling loop"
```

---

## Task 4: Wire AgentLoop into run_demo.py and delete superseded code

**Files:**
- Modify: `g1_nav_demo/run_demo.py`
- Delete: `g1_nav_demo/vlm/inspection.py`
- Delete: `g1_nav_demo/vlm/test_inspection.py`
- Delete: `g1_nav_demo/test_run_demo.py`

- [ ] **Step 1: Update imports at the top of `run_demo.py`**

Remove these two lines:
```python
from g1_nav_demo.vlm.inspection import InspectionBridge, InspectionResult
```

Add this line after the existing imports:
```python
from g1_nav_demo.agent.agent_loop import AgentLoop
```

- [ ] **Step 2: Remove `INSPECTABLE_TARGETS` and `_write_inspection_json` from `run_demo.py`**

Delete this constant (around line 85):
```python
INSPECTABLE_TARGETS = {"table"}
```

Delete this function (around line 92):
```python
def _write_inspection_json(
    path: str, command: str, target: str, result: "InspectionResult"
) -> None:
    ...
```

- [ ] **Step 3: Update `NavigationSession.__init__` — remove `inspection_bridge` parameter**

Replace the `__init__` signature and body (remove `inspection_bridge` parameter and the `self.inspection_bridge` line):

```python
    def __init__(
        self,
        model,
        data,
        walk_policy: G1WalkPolicy,
        goal_planner: GoalPlanner,
        vlm_bridge: VLMBridge,
        sim_fps: int = 500,
        render_fps: int = 30,
        max_steps: int = 5000,
        torso_body_id: int = 0,
    ) -> None:
        self.model = model
        self.data = data
        self.walk_policy = walk_policy
        self.goal_planner = goal_planner
        self.vlm_bridge = vlm_bridge
        self.sim_fps = sim_fps
        self.render_fps = render_fps
        self.max_steps = max_steps
        self.torso_body_id = torso_body_id

        self._build_index_mappings()

        self.kps = G1WalkPolicy.KPS
        self.kds = G1WalkPolicy.KDS
        self.default_angles = G1WalkPolicy.DEFAULT_ANGLES
        self.decimation = max(1, int(round(G1WalkPolicy.CONTROL_DT / model.opt.timestep)))
```

- [ ] **Step 4: Remove the three inspection methods and class constant from `NavigationSession`**

Delete the following from `NavigationSession` entirely:
- `MAX_INSPECTION_REPOSITIONS = 4` class attribute
- `inspect_target()` method
- `_build_result_from_action()` method
- `inspect_target_agentic()` method

- [ ] **Step 5: Update `_init_simulation` to return `(session, agent_loop)`**

Replace the end of `_init_simulation` (after `walk_policy.reset()`):

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
    )
    agent_loop = AgentLoop(session=session, model_name=args.vlm_model)
    return session, agent_loop
```

- [ ] **Step 6: Rewrite `_run_single_turn` to use AgentLoop**

Replace the entire `_run_single_turn` function:

```python
def _run_single_turn(args, session: NavigationSession, agent_loop: AgentLoop) -> None:
    report_json = args.output.rsplit(".", 1)[0] + "_report.json"
    video_renderer = VideoRenderer(
        session.model, output_path=args.output, fps=session.render_fps,
        width=1280, height=480,
    )
    try:
        result = agent_loop.run_turn(args.command, video_renderer, report_json)
        logger.info("Turn complete: %s — %s", result["verdict"], result["message"])
    finally:
        video_renderer.close()
```

- [ ] **Step 7: Rewrite `_run_multiturn` to use AgentLoop**

Replace the entire `_run_multiturn` function:

```python
def _run_multiturn(args, session: NavigationSession, agent_loop: AgentLoop) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    turn = 0
    first_command = args.command

    while True:
        if turn == 0 and first_command:
            command = first_command
            print(f"[Turn 1] Command: {command}")
        else:
            try:
                command = input("Command (or 'quit'): ").strip()
            except EOFError:
                break
            if not command:
                continue
            if command.lower() in ("quit", "exit", "q"):
                break

        turn += 1
        video_path = os.path.join(args.output_dir, f"turn_{turn:03d}.mp4")
        report_json = os.path.join(args.output_dir, f"turn_{turn:03d}_report.json")

        video_renderer = VideoRenderer(
            session.model, output_path=video_path, fps=session.render_fps,
            width=1280, height=480,
        )
        try:
            result = agent_loop.run_turn(command, video_renderer, report_json)
            pos = session.current_position()
            print(f"  Turn {turn}: {result['verdict']} — {result['message']}")
            print(f"  Position: ({pos[0]:.2f}, {pos[1]:.2f})")
        finally:
            video_renderer.close()

    print(f"Session ended. {turn} turn(s) completed.")
```

- [ ] **Step 8: Update `main()` to unpack `(session, agent_loop)` and pass agent_loop**

Replace in `main()`:
```python
    session = _init_simulation(args)

    if args.multiturn:
        _run_multiturn(args, session)
    else:
        _run_single_turn(args, session)
```

With:
```python
    session, agent_loop = _init_simulation(args)

    if args.multiturn:
        _run_multiturn(args, session, agent_loop)
    else:
        _run_single_turn(args, session, agent_loop)
```

- [ ] **Step 9: Delete the three superseded files**

```bash
rm g1_nav_demo/vlm/inspection.py
rm g1_nav_demo/vlm/test_inspection.py
rm g1_nav_demo/test_run_demo.py
```

- [ ] **Step 10: Run the surviving tests to verify nothing is broken**

```bash
python -m pytest g1_nav_demo/ -v
```

Expected: all tests in `agent/`, `planner/`, `renderer/`, `scene/`, `vlm/` pass. `test_inspection.py` and `test_run_demo.py` are gone.

- [ ] **Step 11: Verify the CLI still starts (no import errors)**

```bash
python -m g1_nav_demo.run_demo --help
```

Expected: usage string printed, no `ImportError` or `AttributeError`.

- [ ] **Step 12: Commit**

```bash
git add g1_nav_demo/run_demo.py
git rm g1_nav_demo/vlm/inspection.py g1_nav_demo/vlm/test_inspection.py g1_nav_demo/test_run_demo.py
git commit -m "Wire AgentLoop into run_demo; remove InspectionBridge and old agentic loop"
```
