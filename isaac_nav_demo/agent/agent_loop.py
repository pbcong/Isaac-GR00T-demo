from __future__ import annotations

import base64
import json
import logging
import os
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

MAX_TURNS = 20
_NO_TOOL_CALL_RETRIES = 2
_OPENROUTER_BASE = "https://openrouter.ai/api/v1"

SYSTEM_PROMPT = """\
You are a safety-inspection robot operating in a warehouse. You have three tools:

- navigate(instruction): Move the robot to a location or face a direction. Describe where \
in natural language, e.g. "Go to the back of the crate and face it".
- look(): Take a head-camera snapshot. Returns an image of what the robot sees right now.
- report(verdict, findings, message): End the mission and report findings. YOU MUST ALWAYS \
call this to finish — never end your turn without calling report().

MISSION TYPES:

A) NAVIGATION-ONLY missions (e.g. "go to the table", "move to the shelf"):
   → Navigate to the destination, then call report(verdict="complete").
   → No look() or inspection needed.

B) INSPECTION missions (e.g. "inspect the box", "check if the container is safe"):
   → You must be thorough. A hazard could be on any side, so you need to check every \
accessible side before declaring "safe".
   → Declaring "safe" after only checking some sides is wrong.
   → If you see a confirmed hazard symbol, declare "hazardous" immediately — no need to \
check remaining sides.
   → Hazard indicators: red/orange box with diamond symbols, radioactive trefoil, flame, \
skull, or similar warning marks.
   → If navigation fails repeatedly, report(verdict="failed").

CRITICAL: Every mission MUST end with a report() call. Do not produce a text reply — \
always use a tool call. If you are unsure what to do next, call report() with the best \
verdict you can justify.
"""

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "navigate",
            "description": (
                "Navigate the robot to a location or face a direction described in "
                "natural language. Handles path planning and execution internally. "
                "Use this to approach objects, or to reposition so a different "
                "face/side of an object is visible — e.g. 'Go to the left side of "
                "the crate and face it' to inspect an unseen side."
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

if TYPE_CHECKING:
    from isaac_nav_demo.sim_session import IsaacNavigationSession as NavigationSession
    from isaac_nav_demo.renderer.video_renderer import VideoRenderer


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
        self._no_tool_retries: int = 0

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
        max_reroutes = 3
        for attempt in range(max_reroutes + 1):
            reached, blocked = self.session.run_to_goal_with_renderer(goal, instruction, video_renderer)
            if reached or not blocked:
                break
            if attempt < max_reroutes:
                logger.info("Obstacle timeout — rerouting (attempt %d/%d)", attempt + 1, max_reroutes)
                goal = self.session.compute_detour_goal(goal)
        pos = self.session.current_position()
        return {"reached": reached, "position": list(pos)}

    def _handle_look(self, video_renderer: "VideoRenderer") -> str:
        self.session.idle(duration_steps=250, video_renderer=video_renderer)
        png_bytes = video_renderer.snapshot(width=1280, height=960)
        self._look_count += 1
        snap_path = f"{self._snap_prefix}_look_{self._look_count:02d}.png"
        try:
            with open(snap_path, "wb") as f:
                f.write(png_bytes)
            logger.debug("Saved look snapshot → %s", snap_path)
        except OSError as e:
            logger.warning("Could not save look snapshot: %s", e)
        return base64.b64encode(png_bytes).decode()

    def run_turn(
        self,
        command: str,
        video_renderer: "VideoRenderer",
        report_json_path: str,
    ) -> dict:
        self._snap_prefix = report_json_path.rsplit(".", 1)[0]
        self._look_count = 0
        self._no_tool_retries = 0

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
            )
            msg = response.choices[0].message
            msg_dict = msg.model_dump(exclude_none=True)
            if "content" not in msg_dict and msg.content is None:
                msg_dict["content"] = None
            msg_dict["role"] = "assistant"
            history.append(msg_dict)

            if not msg.tool_calls:
                self._no_tool_retries += 1
                if self._no_tool_retries > _NO_TOOL_CALL_RETRIES:
                    logger.warning("Agent returned no tool calls after retries; ending turn")
                    break
                logger.warning("Agent returned no tool calls; nudging to call report()")
                history.append({
                    "role": "user",
                    "content": "You must call report() to end the mission. Please call it now.",
                })
                continue

            done = False
            tool_results = []
            captured_images: list[str] = []
            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)

                if name == "navigate":
                    outcome = self._handle_navigate(args["instruction"], command, video_renderer)
                    tool_results.append(_text_tool_result(tc.id, json.dumps(outcome)))

                elif name == "look":
                    image_b64 = self._handle_look(video_renderer)
                    captured_images.append(image_b64)
                    tool_results.append(_text_tool_result(tc.id, f"Image captured (look #{self._look_count})."))

                elif name == "report":
                    verdict = args["verdict"]
                    if verdict == "safe" and self._look_count < 3:
                        tool_results.append(_text_tool_result(
                            tc.id,
                            f"You have only looked {self._look_count} time(s) before "
                            f"declaring '{verdict}'. Consider whether you have actually "
                            f"inspected all accessible sides. If not, navigate to an "
                            f"unseen side and look() again before reporting.",
                        ))
                        continue

                    result = {
                        "verdict": args["verdict"],
                        "findings": args.get("findings", []),
                        "message": args["message"],
                    }
                    self._handle_report(result, command, report_json_path, video_renderer)
                    tool_results.append(_text_tool_result(tc.id, "Turn complete."))
                    done = True

            history.extend(tool_results)

            if captured_images:
                user_content: list[dict] = [
                    {"type": "text", "text": "Here is the view from the robot's head camera:"},
                ]
                for img_b64 in captured_images:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    })
                history.append({"role": "user", "content": user_content})

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