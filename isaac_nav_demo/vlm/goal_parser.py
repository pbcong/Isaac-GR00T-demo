"""VLM path-planning bridge for the Isaac Sim warehouse demo.

Identical in structure to the MuJoCo version but:
  - SCENE_PROMPT is built dynamically from the live obstacle map
  - Scene image comes from Isaac's overhead camera (scene_map.render_isaac_frame)
  - Obstacle coordinates match the 24 m × 12 m warehouse layout
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

if TYPE_CHECKING:
    from omni.isaac.core import World
    from omni.isaac.sensor import Camera

logger = logging.getLogger(__name__)

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Goal:
    target_name: str
    waypoints: list[tuple[float, float]] = field(default_factory=list)
    target_pos: tuple[float, float] | None = None
    face_direction: str | None = None
    inspect: bool = False

    @property
    def x(self) -> float:
        return self.waypoints[-1][0] if self.waypoints else 0.0

    @property
    def y(self) -> float:
        return self.waypoints[-1][1] if self.waypoints else 0.0


# ---------------------------------------------------------------------------
# SCENE_PROMPT builder
# ---------------------------------------------------------------------------

_SCENE_PROMPT_TEMPLATE = """\
You are a path-planning assistant for a humanoid robot in a warehouse.

IMAGE: top-down warehouse render from a camera 25 m above.
  Grid lines at 1 m intervals; yellow numbers at 2 m intervals.
  +Y = north, +X = east (toward fire hazard zone).
  No robot marker is drawn — the robot's current position and heading
  are given in the text prompt below.

WAREHOUSE LAYOUT:
  The warehouse is approximately 24 m (X) × 12 m (Y).
  Two shelf rows run east-west along Y = +4 m and Y = -4 m.
  A central aisle runs east along Y ≈ 0.
  A burning barrel (fire hazard) is located near the far end at X ≈ 17.

OBSTACLES (bounding boxes in world coordinates):
  Each obstacle occupies the rectangular region from (x_min, y_min) to
  (x_max, y_max).  THE TARGET OBJECT IS ITSELF AN OBSTACLE — you must go
  AROUND it, never through it.

{obstacles}

FACES:
  "front"  = -Y face   "back"  = +Y face
  "left"   = -X face   "right" = +X face

RULES:
  1. APPROACH (last waypoint): 0.8 m from the nearest clear face of the target.
  2. NEVER place a waypoint inside any obstacle or within 0.5 m of its boundary.
  3. TRACE every segment between waypoints — if it crosses an obstacle (+0.5 m
     margin) insert detour waypoints around the nearest box corner.
  4. Keep waypoints in the central aisle (−3 < Y < 3) to avoid shelf rows.
  5. COUNT: as few waypoints as necessary, but enough to be collision-free.
  6. FIRST WAYPOINT: start from the robot's current position given below.
  7. FACE DIRECTION: arrive facing the target's nearest clear face.
  8. INSPECTION INTENT: "inspect" = true only if the command explicitly asks
     to inspect, check, examine, scan, or look at the target.
  9. TARGET CENTER: set "target_pos" to the [x, y] centre of the target.

Output ONLY this JSON on one line. No markdown. No text before or after.
{{"target_name": "<name>", "target_pos": [x, y], "waypoints": [[x1,y1], ..., [xN,yN]], "face_direction": "<front|back|left|right>", "inspect": <true|false>}}
"""


def build_scene_prompt(obstacle_map: dict[str, tuple[float, float, float, float]]) -> str:
    """Build SCENE_PROMPT with live obstacle coordinates from the scene."""
    lines: list[str] = []
    for name, (cx, cy, hx, hy) in obstacle_map.items():
        lines.append(
            f"  - {name}: center ({cx:.1f}, {cy:.1f}), half-extents ({hx:.2f}, {hy:.2f})"
            f"  → region ({cx-hx:.2f}, {cy-hy:.2f})–({cx+hx:.2f}, {cy+hy:.2f})"
        )
    return _SCENE_PROMPT_TEMPLATE.format(obstacles="\n".join(lines))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# VLMBridge
# ---------------------------------------------------------------------------

class VLMBridge:
    def __init__(
        self,
        obstacle_map: dict[str, tuple[float, float, float, float]],
        model_name: str = "x-ai/grok-4.3",
        api_base: str = _OPENROUTER_BASE,
        api_key: str | None = None,
    ) -> None:
        self.obstacle_map = obstacle_map
        self.model_name   = model_name
        self.api_base     = api_base
        self.api_key      = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self._client      = None
        self._scene_prompt: str | None = None

    @property
    def scene_prompt(self) -> str:
        if self._scene_prompt is None:
            self._scene_prompt = build_scene_prompt(self.obstacle_map)
        return self._scene_prompt

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.api_base, api_key=self.api_key)
        return self._client

    def _render_scene(
        self,
        world: "World",
        overhead_cam: "Camera",
        debug_image_path: str | None = None,
    ) -> bytes | None:
        try:
            from isaac_nav_demo.vlm.scene_map import render_isaac_frame
            png = render_isaac_frame(world, overhead_cam)
            if debug_image_path:
                try:
                    with open(debug_image_path, "wb") as f:
                        f.write(png)
                    logger.info("VLM scene image saved to %s", debug_image_path)
                except OSError as e:
                    logger.warning("Could not save debug image: %s", e)
            return png
        except Exception as e:
            logger.error("Scene render failed: %s", e)
            return None

    def parse(
        self,
        command: str,
        world: "World | None" = None,
        overhead_cam: "Camera | None" = None,
        robot_pos: tuple[float, float] | None = None,
        robot_yaw: float | None = None,
        debug_image_path: str | None = None,
    ) -> Optional[Goal]:
        png_bytes = None
        if world is not None and overhead_cam is not None:
            png_bytes = self._render_scene(world, overhead_cam, debug_image_path)

        if png_bytes is None:
            logger.error("No scene image available — cannot plan path")
            return None

        img_b64 = base64.b64encode(png_bytes).decode()

        if robot_pos is not None and robot_yaw is not None:
            direction = _yaw_to_direction(robot_yaw)
            user_text = (
                f"Robot is at ({robot_pos[0]:.1f}, {robot_pos[1]:.1f}), "
                f"facing {direction}. {command}"
            )
        elif robot_pos is not None:
            user_text = f"Robot is at ({robot_pos[0]:.1f}, {robot_pos[1]:.1f}). {command}"
        else:
            user_text = command

        try:
            response = self._get_client().chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.scene_prompt},
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
            text = response.choices[0].message.content or ""
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
        text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL)
        text = re.sub(r"```(?:json)?\s*", "", text)
        for m in re.finditer(r'\{"target_name"[^}]*\}', text, re.DOTALL):
            try:
                obj      = json.loads(m.group())
                name     = obj.get("target_name")
                wps      = obj.get("waypoints")
                face     = obj.get("face_direction")
                inspect  = obj.get("inspect", False)
                tp       = obj.get("target_pos")
                if isinstance(name, str) and isinstance(wps, list) and wps:
                    waypoints = [
                        (float(wp[0]), float(wp[1]))
                        for wp in wps
                        if isinstance(wp, (list, tuple)) and len(wp) == 2
                    ]
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
