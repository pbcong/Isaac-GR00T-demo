from __future__ import annotations

import base64
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

logger = logging.getLogger(__name__)


@dataclass
class Goal:
    target_name: str
    waypoints: list[tuple[float, float]] = field(default_factory=list)
    target_pos: tuple[float, float] | None = None  # world-frame center of the target object
    face_direction: str | None = None
    inspect: bool = False

    @property
    def x(self) -> float:
        return self.waypoints[-1][0] if self.waypoints else 0.0

    @property
    def y(self) -> float:
        return self.waypoints[-1][1] if self.waypoints else 0.0


SCENE_PROMPT = """\
You are a path-planning assistant for a humanoid robot.

IMAGE: top-down room render from a camera 12 m above.
  Grid lines at 0.5 m intervals; yellow numbers at 1 m intervals.
  +Y = north, +X = east.
  No robot marker is drawn — the robot's current position and heading \
are given in the text prompt below.

OBSTACLES (bounding boxes in world coordinates):
  Each obstacle occupies the rectangular region from (x_min, y_min) to \
(x_max, y_max). THE TARGET OBJECT IS ITSELF AN OBSTACLE — you must go \
AROUND it to reach the requested side, never through it.

  Known obstacles (verify against the image but use these coordinates):
  - table:     center (3.0, 2.0), half-extents (0.6, 0.4)  → region (2.4, 1.6)–(3.6, 2.4)
  - chair:     center (1.0, 3.0), half-extents (0.25, 0.25) → region (0.75, 2.75)–(1.25, 3.25)
  - door:      center (5.0, 0.0), half-extents (0.05, 0.5)  → region (4.95, -0.5)–(5.05, 0.5)
  - bookshelf: center (-2.0, 1.0), half-extents (0.25, 0.5) → region (-2.25, 0.5)–(-1.75, 1.5)
  - crate:     center (2.0, 0.0),  half-extents (0.3, 0.3)   → region (1.7, -0.3)–(2.3, 0.3)
  - barrel:    center (0.5, -1.5), half-extents (0.2, 0.2)   → region (0.3, -1.7)–(0.7, -1.3)
  - pillar:    center (-0.5, 2.0), half-extents (0.15, 0.15) → region (-0.65, 1.85)–(-0.35, 2.15)

FACES:
  "front"  = -Y face (smallest Y boundary)
  "back"   = +Y face (largest Y boundary)
  "left"   = -X face (smallest X boundary)
  "right"  = +X face (largest X boundary)

RULES:
  1. APPROACH (last waypoint): place it 0.8 m from the nearest clear face of
     the target, on the side the command specifies (see FACES above).
     The last waypoint MUST NOT be inside any obstacle.

  2. NEVER place a waypoint inside any obstacle or within 0.5 m of its
     boundary. Use the bounding-box coordinates above to check.

  3. MENTALLY TRACE every straight-line segment between consecutive waypoints.
     If any segment crosses or touches an obstacle bounding box (+0.5 m \
margin), that path is INVALID. Insert detour waypoints.

  4. DETOUR STRATEGY: route around the nearest corner of the obstacle's
     bounding box. Place a waypoint at least 0.5 m outside the box corner,
     then continue toward the goal.

  5. COUNT: as little waypoints as possible, but as many as necessary to follow the rules.

  6. FIRST WAYPOINT: start from the robot's current position given in the \
text prompt. The first waypoint should be the first navigational point \
from the robot's current location.

  7. FACE DIRECTION: the robot should ARRIVE at the last waypoint FACING \
the target. Set the approach so the robot's forward direction points \
toward the nearest face of the target object.

  8. INSPECTION INTENT: Set "inspect" to true ONLY if the command
     explicitly asks to inspect, check, examine, scan, or look at the
     target's contents. "Go to the table" → false.
     "Inspect the table" → true. "Go to the table and check it" → true.

  9. TARGET CENTER: Set "target_pos" to the [x, y] center of the target
     object using the bounding-box centers from the OBSTACLES section above.

Output ONLY this JSON on one line. No markdown. No text before or after.
{"target_name": "<name>", "target_pos": [x, y], "waypoints": [[x1,y1], ..., [xN,yN]], "face_direction": "<front|back|left|right>", "inspect": <true|false>}
"""


_OPENROUTER_BASE = "https://openrouter.ai/api/v1"

_THINK_OPEN = r"<tool_call>"
_THINK_CLOSE = r""


def _yaw_to_direction(yaw: float) -> str:
    import math
    deg = math.degrees(yaw) % 360
    if deg < 22.5 or deg >= 337.5:
        return "east (+X)"
    elif deg < 67.5:
        return "northeast"
    elif deg < 112.5:
        return "north (+Y)"
    elif deg < 157.5:
        return "northwest"
    elif deg < 202.5:
        return "west (-X)"
    elif deg < 247.5:
        return "southwest"
    elif deg < 292.5:
        return "south (-Y)"
    else:
        return "southeast"


class VLMBridge:
    def __init__(
        self,
        model_name: str = "x-ai/grok-4.3",
        api_base: str = _OPENROUTER_BASE,
        api_key: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.api_base, api_key=self.api_key)
        return self._client

    def _render_scene(self, mj_model, mj_data) -> bytes | None:
        try:
            from g1_nav_demo.vlm.scene_map import render_mujoco_frame
            return render_mujoco_frame(mj_model, mj_data)
        except Exception as e:
            logger.error("Scene render failed: %s", e)
            return None

    def parse(
        self,
        command: str,
        mj_model=None,
        mj_data=None,
        robot_pos: tuple[float, float] | None = None,
        robot_yaw: float | None = None,
        debug_image_path: str | None = None,
    ) -> Optional[Goal]:
        png_bytes = None
        if mj_model is not None and mj_data is not None:
            png_bytes = self._render_scene(mj_model, mj_data)

        if png_bytes is None:
            logger.error("No scene image available — cannot plan path")
            return None

        if debug_image_path:
            try:
                with open(debug_image_path, "wb") as f:
                    f.write(png_bytes)
                logger.info("VLM scene image saved to %s", debug_image_path)
            except Exception as e:
                logger.warning("Could not save debug image: %s", e)

        img_b64 = base64.b64encode(png_bytes).decode()

        if robot_pos is not None and robot_yaw is not None:
            direction = _yaw_to_direction(robot_yaw)
            user_text = (
                f"Robot is currently at ({robot_pos[0]:.1f}, {robot_pos[1]:.1f}), "
                f"facing {direction}. {command}"
            )
        elif robot_pos is not None:
            user_text = (
                f"Robot is currently at ({robot_pos[0]:.1f}, {robot_pos[1]:.1f}). {command}"
            )
        else:
            user_text = command

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SCENE_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                            },
                            {"type": "text", "text": user_text},
                        ],
                    },
                ],
                max_tokens=600,
                temperature=0.2,
                extra_body={"reasoning": {"enabled": True}},
            )
            msg = response.choices[0].message
            text = msg.content or ""
            goal = self._extract_goal(text)
            if goal is None:
                logger.warning("VLM response could not be parsed:\n%s", text)
            else:
                logger.info(
                    "VLM goal: %s -> %s (face: %s)",
                    goal.target_name, goal.waypoints, goal.face_direction,
                )
            return goal
        except Exception as e:
            logger.error("VLM call failed: %s", e)
            return None

    def _extract_goal(self, text: str) -> Optional[Goal]:
        think_pat = _THINK_OPEN + r".*?" + _THINK_CLOSE
        text = re.sub(think_pat, "", text, flags=re.DOTALL)
        text = re.sub(r"```(?:json)?\s*", "", text)
        for m in re.finditer(r'\{"target_name"[^}]*\}', text, re.DOTALL):
            try:
                obj = json.loads(m.group())
                name = obj.get("target_name")
                wps = obj.get("waypoints")
                face = obj.get("face_direction")
                inspect = obj.get("inspect", False)
                tp = obj.get("target_pos")
                if isinstance(name, str) and isinstance(wps, list) and len(wps) > 0:
                    waypoints = []
                    for wp in wps:
                        if isinstance(wp, (list, tuple)) and len(wp) == 2:
                            waypoints.append((float(wp[0]), float(wp[1])))
                    target_pos = None
                    if isinstance(tp, (list, tuple)) and len(tp) == 2:
                        try:
                            target_pos = (float(tp[0]), float(tp[1]))
                        except (TypeError, ValueError):
                            pass
                    if waypoints:
                        return Goal(
                            target_name=name,
                            waypoints=waypoints,
                            target_pos=target_pos,
                            face_direction=face if isinstance(face, str) else None,
                            inspect=bool(inspect),
                        )
            except (json.JSONDecodeError, ValueError):
                continue
        return None
