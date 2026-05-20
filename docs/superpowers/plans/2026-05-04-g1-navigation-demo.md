# G1 Navigation Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a demo where a user types a natural language command, a Qwen VLM parses it into a goal coordinate, and a Unitree G1 humanoid walks to that goal in MuJoCo simulation, rendered as MP4 video.

**Architecture:** Simple pipeline: User text → Qwen3.5-9B VLM parses text+scene description into (x, y) goal → Proportional controller computes velocity commands → Pre-trained RL walking policy (from unitree_rl_gym) drives G1 in MuJoCo → Offscreen rendering captures dual-camera MP4 video with command overlay.

**Tech Stack:** MuJoCo, unitree_rl_gym (IsaacGym-trained RL policy + MuJoCo deploy), Qwen3.5-9B (transformers), PyAV (video encoding)

---

## File Structure

```
g1_nav_demo/
├── run_demo.py                  # Main entry point: CLI → VLM → sim → video
├── scene/
│   └── g1_nav_room.xml          # MuJoCo scene XML (G1 + room + furniture)
├── walk_policy/
│   ├── __init__.py              # Package init, exports G1WalkPolicy
│   └── g1_walk_policy.py        # Wraps unitree_rl_gym's pre-trained policy for MuJoCo
├── planner/
│   ├── __init__.py              # Package init, exports GoalPlanner
│   └── goal_planner.py          # Proportional goal-reaching controller
├── vlm/
│   ├── __init__.py              # Package init, exports VLMBridge
│   └── goal_parser.py           # Qwen3.5-9B VLM goal parser + keyword fallback
├── renderer/
│   ├── __init__.py              # Package init, exports VideoRenderer
│   └── video_renderer.py        # Dual-camera MuJoCo rendering + PyAV MP4 output
├── requirements.txt             # Python dependencies
└── README.md                    # Setup and usage instructions
```

---

### Task 1: Scene Setup — MuJoCo XML with G1 and Room

**Files:**
- Create: `g1_nav_demo/scene/g1_nav_room.xml`

- [ ] **Step 1: Clone unitree_mujoco to get G1 model assets**

Run:
```bash
git clone https://github.com/unitreerobotics/unitree_mujoco.git /tmp/unitree_mujoco
```

Verify:
```bash
ls /tmp/unitree_mujoco/unitree_robots/g1/
# Should show g1_23dof.xml, g1_29dof.xml, meshes/, etc.
```

- [ ] **Step 2: Clone unitree_rl_gym to get the pre-trained walking policy**

Run:
```bash
git clone https://github.com/unitreerobotics/unitree_rl_gym.git /tmp/unitree_rl_gym
```

Verify:
```bash
ls /tmp/unitree_rl_gym/deploy/pre_train/g1/
# Should show motion.pt (the pre-trained G1 walking policy)
```

- [ ] **Step 3: Create the MuJoCo scene XML**

Create `g1_nav_demo/scene/g1_nav_room.xml` by adapting the unitree_mujoco G1 scene. This XML includes:
- The G1 29-DOF model (using `g1_29dof.xml` as base)
- A flat floor with a grid texture
- A table geom at (3, 2) — box geometry, wooden color
- A chair geom at (1, 3) — box geometry, gray color
- A door marker at (5, 0) — thin tall box, brown color
- Lights for rendering

```xml
<mujoco model="g1_nav_room">
  <include file="../path/to/g1_29dof.xml"/>

  <worldbody>
    <!-- Floor -->
    <geom name="floor" type="plane" size="10 10 0.1" rgba="0.9 0.9 0.9 1"/>

    <!-- Table at (3, 2) -->
    <geom name="table_top" type="box" size="0.8 0.5 0.04" pos="3 2 0.75" rgba="0.6 0.4 0.2 1"/>
    <geom name="table_leg1" type="cylinder" size="0.03 0.375" pos="2.5 1.7 0.375" rgba="0.5 0.3 0.15 1"/>
    <geom name="table_leg2" type="cylinder" size="0.03 0.375" pos="3.5 1.7 0.375" rgba="0.5 0.3 0.15 1"/>
    <geom name="table_leg3" type="cylinder" size="0.03 0.375" pos="2.5 2.3 0.375" rgba="0.5 0.3 0.15 1"/>
    <geom name="table_leg4" type="cylinder" size="0.03 0.375" pos="3.5 2.3 0.375" rgba="0.5 0.3 0.15 1"/>

    <!-- Chair at (1, 3) -->
    <geom name="chair_seat" type="box" size="0.3 0.3 0.04" pos="1 3 0.45" rgba="0.5 0.5 0.5 1"/>
    <geom name="chair_back" type="box" size="0.3 0.04 0.3" pos="1 2.7 0.65" rgba="0.5 0.5 0.5 1"/>
    <geom name="chair_leg1" type="cylinder" size="0.02 0.225" pos="0.7 2.7 0.225" rgba="0.4 0.4 0.4 1"/>
    <geom name="chair_leg2" type="cylinder" size="0.02 0.225" pos="1.3 2.7 0.225" rgba="0.4 0.4 0.4 1"/>
    <geom name="chair_leg3" type="cylinder" size="0.02 0.225" pos="0.7 3.3 0.225" rgba="0.4 0.4 0.4 1"/>
    <geom name="chair_leg4" type="cylinder" size="0.02 0.225" pos="1.3 3.3 0.225" rgba="0.4 0.4 0.4 1"/>

    <!-- Door marker at (5, 0) -->
    <geom name="door" type="box" size="0.05 0.8 1.1" pos="5 0 0.55" rgba="0.55 0.35 0.2 1"/>

    <!-- Target marker (movable, controlled by planner) -->
    <geom name="target_marker" type="sphere" size="0.1" pos="3 2 0.05" rgba="1 0 0 0.6" contype="0" conaffinity="0"/>
  </worldbody>

  <!-- Cameras -->
  <camera name="birdseye" pos="3 -5 8" xyaxes="1 0 0 0 0.5 1" fovy="60"/>
  <camera name="ego" pos="-0.1 0 1.3" quat="0.9998 0 0.02 0" fovy="90"/>
</mujoco>
```

The exact include path and structure will need adjustment based on the unitree_mujoco directory layout. The key point is reusing the G1 model assets and adding furniture geoms.

- [ ] **Step 4: Verify the scene loads in MuJoCo**

```bash
cd g1_nav_demo && python -c "
import mujoco
model = mujoco.MjModel.from_xml_path('scene/g1_nav_room.xml')
data = mujoco.MjData(model)
mujoco.mj_step(model, data)
print(f'Model loaded: {model.nq} DOF, {model.ngeom} geoms')
"
```

Expected: Model loads without errors, prints DOF count and geom count.

- [ ] **Step 5: Commit**

```bash
git add g1_nav_demo/scene/g1_nav_room.xml
git commit -m "feat: add MuJoCo scene XML with G1 robot and room furniture"
```

---

### Task 2: Walking Policy Wrapper

**Files:**
- Create: `g1_nav_demo/walk_policy/__init__.py`
- Create: `g1_nav_demo/walk_policy/g1_walk_policy.py`

The walking policy wraps the unitree_rl_gym pre-trained policy. Based on the deploy code in `unitree_rl_gym/deploy/deploy_mujoco/`, the policy takes proprioceptive observations and outputs joint position targets.

- [ ] **Step 1: Study the unitree_rl_gym deploy code**

Read the deployment scripts in `/tmp/unitree_rl_gym/deploy/deploy_mujoco/` to understand:
- The observation space (what proprioceptive data the policy needs)
- The action space (what joint positions the policy outputs)
- The expected control frequency
- The G1 joint ordering

Key files: `g1.yaml` config, any `play.py` or `run.py` scripts in deploy.

- [ ] **Step 2: Create the walk_policy package**

Create `g1_nav_demo/walk_policy/__init__.py`:
```python
from g1_walk_policy import G1WalkPolicy

__all__ = ["G1WalkPolicy"]
```

Create `g1_nav_demo/walk_policy/g1_walk_policy.py`:
```python
import numpy as np
import torch
import mujoco


class G1WalkPolicy:
    """Wraps the unitree_rl_gym pre-trained G1 walking policy for MuJoCo deployment.

    Takes velocity commands (vx, vy, vyaw) and proprioceptive observations,
    outputs joint position targets for the G1 humanoid.

    The policy network is a simple MLP that processes:
    - Projected gravity (3,)
    - Velocity commands (3,) - [vx, vy, vyaw]
    - Joint positions (29,)
    - Joint velocities (29,)
    - Last actions (29,)

    And outputs:
    - Joint position targets (29,) in normalized space
    """

    def __init__(self, policy_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.policy = torch.jit.load(policy_path, map_location=self.device)
        self.policy.eval()

        self.default_angles = np.zeros(29)  # Will be set from G1 config
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 1.0
        self.action_scale = 0.25
        self.num_actions = 29
        self.num_obs = 93  # Will be adjusted based on actual config

        self.last_action = np.zeros(self.num_actions, dtype=np.float32)

    def reset(self):
        """Reset policy state."""
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)

    def get_action(
        self,
        projected_gravity: np.ndarray,  # (3,)
        velocity_command: np.ndarray,   # (3,) [vx, vy, vyaw]
        dof_pos: np.ndarray,             # (29,)
        dof_vel: np.ndarray,             # (29,)
    ) -> np.ndarray:
        """Compute joint position targets from observations and velocity commands.

        Args:
            projected_gravity: Gravity vector projected to body frame (3,)
            velocity_command: Desired velocities [vx, vy, vyaw] (3,)
            dof_pos: Current joint positions (29,)
            dof_vel: Current joint velocities (29,)

        Returns:
            Joint position targets (29,) in rad
        """
        obs = np.concatenate([
            projected_gravity,                    # (3,)
            velocity_command * 1.0,               # (3,) scaled commands
            (dof_pos - self.default_angles) * self.dof_pos_scale,  # (29,)
            dof_vel * self.dof_vel_scale,         # (29,)
            self.last_action,                      # (29,)
        ]).astype(np.float32)

        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_tensor = self.policy(obs_tensor)

        action = action_tensor.squeeze(0).cpu().numpy()
        self.last_action = action.copy()

        target_positions = self.default_angles + action * self.action_scale
        return target_positions
```

**Note:** The exact observation dimensions, scaling factors, and default angles must be calibrated based on the actual `g1.yaml` config from unitree_rl_gym. The values above are approximations that will need adjustment during integration testing.

- [ ] **Step 3: Test that the policy checkpoint loads**

```python
import torch
policy = torch.jit.load("/tmp/unitree_rl_gym/deploy/pre_train/g1/motion.pt", map_location="cpu")
# Verify it's a TorchScript model
print(type(policy))
```

- [ ] **Step 4: Commit**

```bash
git add g1_nav_demo/walk_policy/
git commit -m "feat: add G1 walking policy wrapper"
```

---

### Task 3: Goal Planner

**Files:**
- Create: `g1_nav_demo/planner/__init__.py`
- Create: `g1_nav_demo/planner/goal_planner.py`

- [ ] **Step 1: Create the planner package**

Create `g1_nav_demo/planner/__init__.py`:
```python
from goal_planner import GoalPlanner, PlanResult

__all__ = ["GoalPlanner", "PlanResult"]
```

Create `g1_nav_demo/planner/goal_planner.py`:
```python
from dataclasses import dataclass
import numpy as np
from typing import Optional


@dataclass
class PlanResult:
    """Result from one step of the goal planner."""
    vx: float          # Forward velocity command (m/s)
    vy: float          # Lateral velocity command (m/s)
    vyaw: float        # Yaw velocity command (rad/s)
    reached: bool      # Whether the goal has been reached
    distance: float    # Current distance to goal


class GoalPlanner:
    """Proportional controller that drives a walking robot toward a goal position.

    Computes velocity commands (vx, vy, vyaw) based on the difference between
    current robot position/orientation and a target goal position.

    The planner uses proportional control with clamping:
    - vx is proportional to the forward component of the goal vector
    - vy is proportional to the lateral component of the goal vector
    - vyaw is proportional to the angular error to face the goal
    """

    def __init__(
        self,
        max_vx: float = 1.0,
        max_vy: float = 0.5,
        max_vyaw: float = 1.5,
        kp_x: float = 1.0,
        kp_y: float = 0.5,
        kp_yaw: float = 2.0,
        goal_threshold: float = 0.3,
        angle_threshold: float = 0.15,
        slow_distance: float = 1.0,
    ):
        self.max_vx = max_vx
        self.max_vy = max_vy
        self.max_vyaw = max_vyaw
        self.kp_x = kp_x
        self.kp_y = kp_y
        self.kp_yaw = kp_yaw
        self.goal_threshold = goal_threshold
        self.angle_threshold = angle_threshold
        self.slow_distance = slow_distance

    def compute_command(
        self,
        current_pos: np.ndarray,     # (2,) [x, y] in world frame
        current_yaw: float,           # Radians, world frame
        goal_pos: np.ndarray,         # (2,) [x, y] in world frame
    ) -> PlanResult:
        """Compute velocity commands to reach the goal.

        Args:
            current_pos: Robot position [x, y] in world frame (m)
            current_yaw: Robot heading in world frame (rad)
            goal_pos: Target position [x, y] in world frame (m)

        Returns:
            PlanResult with velocity commands and status
        """
        diff = goal_pos - current_pos
        distance = np.linalg.norm(diff)

        if distance < self.goal_threshold:
            return PlanResult(vx=0.0, vy=0.0, vyaw=0.0, reached=True, distance=distance)

        # Transform goal vector to robot body frame
        cos_yaw = np.cos(-current_yaw)
        sin_yaw = np.sin(-current_yaw)
        goal_body_x = cos_yaw * diff[0] - sin_yaw * diff[1]
        goal_body_y = sin_yaw * diff[0] + cos_yaw * diff[1]

        # Desired heading angle
        desired_yaw = np.arctan2(diff[1], diff[0])
        yaw_error = self._wrap_angle(desired_yaw - current_yaw)

        # Slow down near goal
        speed_scale = min(1.0, distance / self.slow_distance)

        # Proportional control in body frame
        vx = np.clip(self.kp_x * goal_body_x * speed_scale, -self.max_vx, self.max_vx)
        vy = np.clip(self.kp_y * goal_body_y * speed_scale, -self.max_vy, self.max_vy)
        vyaw = np.clip(self.kp_yaw * yaw_error, -self.max_vyaw, self.max_vyaw)

        # If yaw error is large, prioritize turning over forward motion
        if abs(yaw_error) > self.angle_threshold * 4:
            vx *= 0.3
            vy *= 0.3

        return PlanResult(vx=vx, vy=vy, vyaw=vyaw, reached=False, distance=distance)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi
```

- [ ] **Step 2: Write a quick unit test**

Create `g1_nav_demo/planner/test_goal_planner.py`:
```python
import numpy as np
from goal_planner import GoalPlanner, PlanResult


def test_goal_reached():
    planner = GoalPlanner()
    result = planner.compute_command(
        current_pos=np.array([3.0, 2.0]),
        current_yaw=0.0,
        goal_pos=np.array([3.0, 2.0]),
    )
    assert result.reached is True
    assert result.vx == 0.0


def test_goal_ahead():
    planner = GoalPlanner()
    result = planner.compute_command(
        current_pos=np.array([0.0, 0.0]),
        current_yaw=0.0,
        goal_pos=np.array([2.0, 0.0]),
    )
    assert result.reached is False
    assert result.vx > 0  # Should move forward
    assert abs(result.vy) < 0.1  # Small lateral component


def test_goal_behind():
    planner = GoalPlanner()
    result = planner.compute_command(
        current_pos=np.array([0.0, 0.0]),
        current_yaw=0.0,
        goal_pos=np.array([-2.0, 0.0]),
    )
    assert result.reached is False
    assert abs(result.vyaw) > 0.5  # Should turn significantly


if __name__ == "__main__":
    test_goal_reached()
    test_goal_ahead()
    test_goal_behind()
    print("All tests passed!")
```

- [ ] **Step 3: Run the tests**

```bash
cd g1_nav_demo && python -m planner.test_goal_planner
```

Expected: "All tests passed!"

- [ ] **Step 4: Commit**

```bash
git add g1_nav_demo/planner/
git commit -m "feat: add proportional goal planner with unit tests"
```

---

### Task 4: VLM Goal Parser

**Files:**
- Create: `g1_nav_demo/vlm/__init__.py`
- Create: `g1_nav_demo/vlm/goal_parser.py`

- [ ] **Step 1: Create the VLM package**

Create `g1_nav_demo/vlm/__init__.py`:
```python
from goal_parser import VLMBridge, KeywordParser

__all__ = ["VLMBridge", "KeywordParser"]
```

Create `g1_nav_demo/vlm/goal_parser.py`:
```python
import json
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np


SCENE_DESCRIPTION = """You are controlling a Unitree G1 humanoid robot in a room.

Scene layout (coordinates in meters, origin at room center):
- Table: position (3.0, 2.0) - a brown wooden table
- Chair: position (1.0, 3.0) - a gray office chair
- Door: position (5.0, 0.0) - a brown door (exit)
- Bookshelf: position (-2.0, 1.0) - a tall wooden bookshelf on the left wall
- Couch: position (0.0, -3.0) - a couch near the south wall

The robot starts at position (0.0, 0.0) facing the positive X direction.

Given a navigation command, output ONLY a JSON object with:
- "target_name": name of the target object
- "x": target x coordinate (float)
- "y": target y coordinate (float)

Example for "go to the table":
{"target_name": "table", "x": 3.0, "y": 2.0}"""


@dataclass
class Goal:
    """Parsed navigation goal."""
    target_name: str
    x: float
    y: float


class KeywordParser:
    """Fallback keyword-based goal parser. No GPU required.

    Matches known object names in the command text to predefined positions.
    """

    SCENE_OBJECTS = {
        "table": (3.0, 2.0),
        "chair": (1.0, 3.0),
        "door": (5.0, 0.0),
        "bookshelf": (-2.0, 1.0),
        "couch": (0.0, -3.0),
        "sofa": (0.0, -3.0),  # alias
        "exit": (5.0, 0.0),   # alias for door
    }

    def parse(self, command: str) -> Optional[Goal]:
        """Parse a command by keyword matching.

        Args:
            command: Natural language navigation command

        Returns:
            Goal if a known object is found, None otherwise
        """
        command_lower = command.lower().strip()

        for name, (x, y) in self.SCENE_OBJECTS.items():
            if name in command_lower:
                return Goal(target_name=name, x=x, y=y)

        return None


class VLMBridge:
    """Qwen3.5-9B VLM-based goal parser.

    Uses the Qwen3.5-9B model to parse natural language commands into navigation
    goals. Falls back to KeywordParser if VLM is unavailable or fails.

    Requires ~24GB GPU memory for bfloat16 inference.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-9B",
        device: str = "cuda",
        fallback_to_keywords: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.fallback_to_keywords = fallback_to_keywords
        self.keyword_parser = KeywordParser()
        self.model = None
        self.processor = None
        self._initialized = False

    def _init_model(self):
        """Lazy-load the VLM model on first use."""
        if self._initialized:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoProcessor

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto",
            )
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self._initialized = True
            print(f"Loaded VLM: {self.model_name}")
        except Exception as e:
            print(f"Failed to load VLM: {e}")
            if self.fallback_to_keywords:
                print("Falling back to keyword parser")
            else:
                raise

    def parse(self, command: str) -> Optional[Goal]:
        """Parse a natural language command into a navigation goal.

        Args:
            command: Natural language navigation command (e.g., "go to the table")

        Returns:
            Goal with target name and coordinates, or None if parsing fails
        """
        self._init_model()

        if self.model is not None:
            goal = self._parse_with_vlm(command)
            if goal is not None:
                return goal

        if self.fallback_to_keywords:
            return self.keyword_parser.parse(command)

        return None

    def _parse_with_vlm(self, command: str) -> Optional[Goal]:
        """Attempt to parse command using the VLM."""
        try:
            from qwen_vl_utils import process_vision_info

            messages = [
                {
                    "role": "system",
                    "content": SCENE_DESCRIPTION,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Command: {command}",
                        },
                    ],
                },
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, max_new_tokens=256, temperature=0.1
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            return self._extract_goal_from_text(output_text)

        except Exception as e:
            print(f"VLM parsing failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    @staticmethod
    def _extract_goal_from_text(text: str) -> Optional[Goal]:
        """Extract a Goal from VLM output text.

        Tries to find a JSON object in the text and parse it.
        """
        # Try to find JSON in the response
        json_match = re.search(r'\{[^{}]*"target_name"[^{}]*\}', text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return Goal(
                    target_name=data["target_name"],
                    x=float(data["x"]),
                    y=float(data["y"]),
                )
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

        # Try any JSON object with x and y
        json_match = re.search(r'\{[^{}]*\}', text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                if "x" in data and "y" in data:
                    return Goal(
                        target_name=data.get("target_name", "unknown"),
                        x=float(data["x"]),
                        y=float(data["y"]),
                    )
            except (json.JSONDecodeError, ValueError):
                pass

        return None
```

- [ ] **Step 2: Test the keyword parser (no GPU required)**

Create `g1_nav_demo/vlm/test_goal_parser.py`:
```python
from goal_parser import KeywordParser, VLMBridge


def test_keyword_parser():
    parser = KeywordParser()

    # Test known objects
    goal = parser.parse("go to the table")
    assert goal is not None
    assert goal.target_name == "table"
    assert goal.x == 3.0
    assert goal.y == 2.0

    goal = parser.parse("walk to the door")
    assert goal is not None
    assert goal.target_name == "door"
    assert goal.x == 5.0

    # Test alias
    goal = parser.parse("head to the sofa")
    assert goal is not None
    assert goal.target_name == "couch" or goal.target_name == "sofa"

    # Test unknown
    goal = parser.parse("fly to the moon")
    assert goal is None

    print("All keyword parser tests passed!")


if __name__ == "__main__":
    test_keyword_parser()
```

- [ ] **Step 3: Run the keyword parser tests**

```bash
cd g1_nav_demo && python -m vlm.test_goal_parser
```

Expected: "All keyword parser tests passed!"

- [ ] **Step 4: Commit**

```bash
git add g1_nav_demo/vlm/
git commit -m "feat: add VLM goal parser with keyword fallback"
```

---

### Task 5: Video Renderer

**Files:**
- Create: `g1_nav_demo/renderer/__init__.py`
- Create: `g1_nav_demo/renderer/video_renderer.py`

- [ ] **Step 1: Create the renderer package**

Create `g1_nav_demo/renderer/__init__.py`:
```python
from video_renderer import VideoRenderer

__all__ = ["VideoRenderer"]
```

Create `g1_nav_demo/renderer/video_renderer.py`:
```python
from pathlib import Path
import numpy as np
import mujoco
import av
import cv2


class VideoRenderer:
    """Renders MuJoCo simulation from dual cameras and writes MP4 video.

    Provides a bird's-eye view and an ego (first-person) view, stacked
    horizontally, with the navigation command overlaid as text.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        output_path: str = "demo_output.mp4",
        fps: int = 30,
        width: int = 1280,
        height: int = 480,
        birdseye_camera: str = "birdseye",
        ego_camera: str = "ego",
        crf: int = 22,
    ):
        self.model = model
        self.output_path = Path(output_path)
        self.fps = fps
        self.width = width
        self.height = height
        self.birdseye_camera = birdseye_camera
        self.ego_camera = ego_camera
        self.crf = crf

        # Per-view dimensions (stacked horizontally)
        self.view_width = width // 2
        self.view_height = height

        # Renderer
        self.renderer = mujoco.Renderer(model, height=self.view_height, width=self.view_width)

        # Video writer (initialized lazily)
        self.container = None
        self.stream = None

    def _init_video_writer(self, first_frame: np.ndarray):
        """Initialize the PyAV video writer."""
        self.container = av.open(str(self.output_path), mode="w")
        self.stream = self.container.add_stream("h264", rate=self.fps)
        self.stream.width = self.width
        self.stream.height = self.height
        self.stream.pix_fmt = "yuv420p"
        self.stream.codec_context.options = {
            "crf": str(self.crf),
            "profile:v": "high",
        }

    @staticmethod
    def _overlay_text(
        frame: np.ndarray,
        text: str,
        position: tuple[int, int] = (10, 30),
        font_scale: float = 1.0,
        color: tuple[int, int, int] = (255, 255, 255),
        thickness: int = 2,
    ) -> np.ndarray:
        """Overlay text on a frame using OpenCV."""
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Add dark background behind text for readability
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(
            frame_bgr,
            (position[0] - 5, position[1] - th - 5),
            (position[0] + tw + 5, position[1] + 5),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            frame_bgr,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (color[2], color[1], color[0]),  # BGR
            thickness,
        )
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return frame_rgb

    def render_frame(
        self,
        data: mujoco.MjData,
        command: str = "",
        distance: float | None = None,
        update_ego_camera: bool = True,
        ego_body_id: int | None = None,
    ) -> np.ndarray:
        """Render a single frame from both cameras and combine.

        Args:
            data: MuJoCo simulation data
            command: Navigation command to overlay
            distance: Distance to goal (shown in overlay)
            update_ego_camera: Whether to update ego camera position
            ego_body_id: Body ID to attach ego camera to

        Returns:
            Combined RGB frame as numpy array (H, W, 3)
        """
        # Update ego camera position if needed
        if update_ego_camera and ego_body_id is not None:
            self._update_ego_camera(data, ego_body_id)

        # Render bird's-eye view
        self.renderer.update_scene(data, camera=self.birdseye_camera)
        birdseye_frame = self.renderer.render()

        # Render ego view
        self.renderer.update_scene(data, camera=self.ego_camera)
        ego_frame = self.renderer.render()

        # Stack horizontally
        combined = np.concatenate([birdseye_frame, ego_frame], axis=1)

        # Overlay command text
        if command:
            overlay_text = command
            if distance is not None:
                overlay_text += f" | Distance: {distance:.2f}m"
            combined = self._overlay_text(combined, overlay_text)

        return combined

    def _update_ego_camera(self, data: mujoco.MjData, body_id: int):
        """Position the ego camera relative to the robot body."""
        body_pos = data.xpos[body_id].copy()
        body_quat = data.xquat[body_id].copy()

        # Camera offset from body: slightly behind and above
        cam_offset = np.array([-0.2, 0.0, 1.3])

        # Rotate offset by body quaternion
        offset_rotated = self._rotate_vector_by_quat(cam_offset, body_quat)
        cam_pos = body_pos + offset_rotated

        # Update camera position in model
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.ego_camera)
        self.model.cam_pos[cam_id] = cam_pos
        # Camera looks forward from the robot's perspective
        self.model.cam_quat[cam_id] = body_quat

    @staticmethod
    def _rotate_vector_by_quat(vec: np.ndarray, quat: np.ndarray) -> np.ndarray:
        """Rotate a vector by a quaternion (w, x, y, z format)."""
        w, x, y, z = quat
        # Rotation matrix from quaternion
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
        ])
        return R @ vec

    def write_frame(self, frame: np.ndarray):
        """Write a single frame to the video file."""
        if self.container is None:
            self._init_video_writer(frame)

        video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        for packet in self.stream.encode(video_frame):
            self.container.mux(packet)

    def close(self):
        """Flush and close the video file."""
        if self.container is not None and self.stream is not None:
            for packet in self.stream.encode():
                self.container.mux(packet)
            self.container.close()
            print(f"Video saved to: {self.output_path}")
```

- [ ] **Step 2: Commit**

```bash
git add g1_nav_demo/renderer/
git commit -m "feat: add dual-camera video renderer with text overlay"
```

---

### Task 6: Main Demo Entry Point

**Files:**
- Create: `g1_nav_demo/run_demo.py`
- Create: `g1_nav_demo/requirements.txt`

- [ ] **Step 1: Create the main demo script**

Create `g1_nav_demo/run_demo.py`:
```python
#!/usr/bin/env python3
"""G1 Navigation Demo: Natural language command → VLM → Goal → Walk → Video.

Usage:
    # With VLM (requires GPU with ~24GB VRAM):
    python run_demo.py --command "go to the table" --use-vlm

    # With keyword parser (no GPU required for parsing):
    python run_demo.py --command "go to the table"

    # Interactive mode:
    python run_demo.py --interactive
"""

import argparse
import time
import numpy as np
import mujoco

# Local imports
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from walk_policy.g1_walk_policy import G1WalkPolicy
from planner.goal_planner import GoalPlanner
from vlm.goal_parser import VLMBridge, KeywordParser, Goal
from renderer.video_renderer import VideoRenderer


# Scene object positions (must match MuJoCo XML)
SCENE_OBJECTS = {
    "table": np.array([3.0, 2.0]),
    "chair": np.array([1.0, 3.0]),
    "door": np.array([5.0, 0.0]),
    "bookshelf": np.array([-2.0, 1.0]),
    "couch": np.array([0.0, -3.0]),
}


def parse_args():
    parser = argparse.ArgumentParser(description="G1 Navigation Demo")
    parser.add_argument("--command", type=str, default="go to the table",
                        help="Natural language navigation command")
    parser.add_argument("--scene-xml", type=str, default="scene/g1_nav_room.xml",
                        help="Path to MuJoCo scene XML")
    parser.add_argument("--policy-path", type=str,
                        default="/tmp/unitree_rl_gym/deploy/pre_train/g1/motion.pt",
                        help="Path to walking policy checkpoint")
    parser.add_argument("--output", type=str, default="demo_output.mp4",
                        help="Output video path")
    parser.add_argument("--use-vlm", action="store_true",
                        help="Use Qwen VLM for parsing (requires GPU)")
    parser.add_argument("--vlm-model", type=str, default="Qwen/Qwen3.5-9B",
                        help="VLM model name")
    parser.add_argument("--interactive", action="store=True",
                        help="Interactive mode: type commands one at a time")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Maximum simulation steps")
    parser.add_argument("--sim-fps", type=int, default=50,
                        help="Simulation control frequency (Hz)")
    parser.add_argument("--render-fps", type=int, default=30,
                        help="Video output frame rate")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for walking policy and VLM")
    return parser.parse_args()


def run_single_command(
    command: str,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    walk_policy: G1WalkPolicy,
    planner: GoalPlanner,
    renderer: VideoRenderer,
    goal: Goal,
    max_steps: int,
    sim_fps: int,
    render_fps: int,
    ego_body_id: int,
):
    """Run one navigation command and record video."""
    walk_policy.reset()

    # Move target marker to goal position
    target_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_marker")
    if target_geom_id >= 0:
        model.geom_pos[target_geom_id] = [goal.x, goal.y, 0.05]

    render_every = max(1, sim_fps // render_fps)
    frame_count = 0

    for step in range(max_steps):
        # Get robot state
        robot_pos = data.qpos[:2].copy()  # x, y from free joint
        robot_quat = data.qpose[3:7].copy()  # quaternion
        robot_yaw = np.arctan2(
            2.0 * (robot_quat[0] * robot_quat[3] + robot_quat[1] * robot_quat[2]),
            1.0 - 2.0 * (robot_quat[2]**2 + robot_quat[3]**2)
        )

        # Compute velocity command from planner
        plan = planner.compute_command(
            current_pos=robot_pos,
            current_yaw=robot_yaw,
            goal_pos=np.array([goal.x, goal.y]),
        )

        if plan.reached:
            print(f"Reached goal '{goal.target_name}' at ({goal.x}, {goal.y})!")
            # Step a few more times to stabilize
            for _ in range(20):
                mujoco.mj_step(model, data)
            break

        # Get walking policy action
        velocity_command = np.array([plan.vx, plan.vy, plan.vyaw])
        projected_gravity = np.array([0.0, 0.0, -1.0])  # Flat ground
        dof_pos = data.qpos[7:36].copy()  # 29 DOF positions after free joint
        dof_vel = data.qvel[6:35].copy()   # 29 DOF velocities

        target_positions = walk_policy.get_action(
            projected_gravity=projected_gravity,
            velocity_command=velocity_command,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
        )

        # Apply actions to simulation
        data.ctrl[:] = target_positions

        # Step simulation
        mujoco.mj_step(model, data)

        # Render frame
        if step % render_every == 0:
            frame = renderer.render_frame(
                data, command=command, distance=plan.distance,
                update_ego_camera=True, ego_body_id=ego_body_id,
            )
            renderer.write_frame(frame)
            frame_count += 1

    else:
        print(f"Did not reach goal within {max_steps} steps")

    return frame_count


def main():
    args = parse_args()

    # Load MuJoCo model
    print(f"Loading scene: {args.scene_xml}")
    model = mujoco.MjModel.from_xml_path(args.scene_xml)
    data = mujoco.MjData(model)

    # Load walking policy
    print(f"Loading walking policy: {args.policy_path}")
    walk_policy = G1WalkPolicy(policy_path=args.policy_path, device=args.device)

    # Create planner
    planner = GoalPlanner()

    # Create renderer
    renderer = VideoRenderer(
        model=model,
        output_path=args.output,
        fps=args.render_fps,
    )

    # Find the robot body ID for ego camera tracking
    ego_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    if ego_body_id < 0:
        # Fallback to first body after worldbody
        ego_body_id = 1

    # Parse command
    if args.use_vlm:
        print(f"Loading VLM: {args.vlm_model}")
        parser = VLMBridge(model_name=args.vlm_model, device=args.device)
    else:
        parser = KeywordParser()

    # Initialize simulation
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # Get goal from command
    print(f"\nCommand: '{args.command}'")
    goal = parser.parse(args.command)

    if goal is None:
        print(f"Could not parse command: '{args.command}'")
        print(f"Known objects: {list(SCENE_OBJECTS.keys())}")
        return

    print(f"Goal: {goal.target_name} at ({goal.x}, {goal.y})")

    # Run the navigation
    frame_count = run_single_command(
        command=args.command,
        model=model,
        data=data,
        walk_policy=walk_policy,
        planner=planner,
        renderer=renderer,
        goal=goal,
        max_steps=args.max_steps,
        sim_fps=args.sim_fps,
        render_fps=args.render_fps,
        ego_body_id=ego_body_id,
    )

    # Save video
    renderer.close()
    print(f"Recorded {frame_count} frames")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create requirements.txt**

Create `g1_nav_demo/requirements.txt`:
```
mujoco>=3.0.0
numpy>=1.24.0
torch>=2.0.0
transformers>=4.57.0
qwen-vl-utils[decord]==0.0.8
accelerate>=0.20.0
av>=12.0.0
opencv-python>=4.8.0
Pillow>=10.0.0
```

- [ ] **Step 3: Commit**

```bash
git add g1_nav_demo/run_demo.py g1_nav_demo/requirements.txt
git commit -m "feat: add main demo entry point and requirements"
```

---

### Task 7: Integration Testing and Calibration

**Files:**
- Modify: `g1_nav_demo/run_demo.py` (fix issues found during testing)
- Modify: `g1_nav_demo/walk_policy/g1_walk_policy.py` (calibrate observation/action spaces)
- Modify: `g1_nav_demo/scene/g1_nav_room.xml` (fix rendering / physics issues)

This task is intentionally open-ended because the exact issues depend on what's discovered during testing. The key steps are:

- [ ] **Step 1: Clone dependency repos and verify assets exist**

```bash
cd /tmp
git clone https://github.com/unitreerobotics/unitree_mujoco.git
git clone https://github.com/unitreerobotics/unitree_rl_gym.git
ls unitree_mujoco/unitree_robots/g1/g1_29dof.xml
ls unitree_rl_gym/deploy/pre_train/g1/motion.pt
```

- [ ] **Step 2: Test MuJoCo scene loading**

```bash
cd g1_nav_demo && python -c "
import mujoco
model = mujoco.MjModel.from_xml_path('scene/g1_nav_room.xml')
data = mujoco.MjData(model)
mujoco.mj_step(model, data)
print('Scene loaded successfully')
print(f'  DOF: {model.nq}')
print(f'  Geoms: {model.ngeom}')
print(f'  Bodies: {model.nbody}')
"
```

Fix any XML errors (include paths, missing assets, etc.)

- [ ] **Step 3: Test keyword parser end-to-end (no GPU required)**

```bash
cd g1_nav_demo && python -c "
from vlm.goal_parser import KeywordParser
parser = KeywordParser()
goal = parser.parse('go to the table')
print(f'Parsed: {goal}')
"
```

Expected output: `Parsed: Goal(target_name='table', x=3.0, y=2.0)`

- [ ] **Step 4: Test walking policy loading**

```bash
cd g1_nav_demo && python -c "
import torch
policy = torch.jit.load('/tmp/unitree_rl_gym/deploy/pre_train/g1/motion.pt', map_location='cpu')
print(f'Policy loaded: {type(policy)}')
# Test forward pass
dummy_obs = torch.randn(1, 93)  # Adjust size based on actual config
output = policy(dummy_obs)
print(f'Output shape: {output.shape}')
"
```

If the observation size is wrong, adjust `self.num_obs` in `g1_walk_policy.py`.

- [ ] **Step 5: Test full pipeline with keyword parser (no walking, just rendering)**

Create a minimal test that:
1. Loads MuJoCo scene
2. Creates a GoalPlanner
3. Creates a VideoRenderer
4. Runs sim for 100 steps with zero velocity commands
5. Saves a short MP4

Debug any rendering issues, camera positions, etc.

- [ ] **Step 6: Calibrate walking policy**

Read the `g1.yaml` config from `unitree_rl_gym` to get:
- Exact observation dimensions and ordering
- Default joint angles
- Scaling factors (`dof_pos_scale`, `dof_vel_scale`, `action_scale`)
- Control frequency
- Clip actions range

Update `g1_walk_policy.py` with the correct values.

- [ ] **Step 7: Full integration test with keyword commands**

```bash
cd g1_nav_demo && python run_demo.py --command "go to the table" --policy-path /tmp/unitree_rl_gym/deploy/pre_train/g1/motion.pt --output test_output.mp4
```

Verify:
- Video file is created
- Robot approaches the table
- Robot stops near the target
- Command text is overlaid on video

- [ ] **Step 8: If VLM is available, test VLM parsing**

```bash
cd g1_nav_demo && python run_demo.py --command "navigate to the wooden surface" --use-vlm --output vlm_test.mp4
```

Verify the VLM correctly maps "wooden surface" to the table.

- [ ] **Step 9: Commit all fixes**

```bash
git add g1_nav_demo/
git commit -m "fix: calibrate walking policy and fix integration issues"
```

---

### Task 8: README and Final Polish

**Files:**
- Create: `g1_nav_demo/README.md`

- [ ] **Step 1: Create the README**

Create `g1_nav_demo/README.md` with setup instructions, usage examples, and troubleshooting tips.

- [ ] **Step 2: Commit**

```bash
git add g1_nav_demo/README.md
git commit -m "docs: add README for G1 navigation demo"
```

---

## Spec Coverage Review

| Spec Requirement | Task |
|---|---|
| User types natural language command | Task 6 (run_demo.py CLI) |
| Qwen VLM parses into goal coordinates | Task 4 (vlm/goal_parser.py) |
| Fallback keyword parser | Task 4 (KeywordParser) |
| Proportional goal-reaching planner | Task 3 (planner/goal_planner.py) |
| RL walking controller | Task 2 (walk_policy/g1_walk_policy.py) |
| MuJoCo simulation with room + furniture | Task 1 (scene/g1_nav_room.xml) |
| Unitree G1 humanoid | Task 1 (uses unitree_mujoco assets) |
| Dual camera rendering | Task 5 (renderer/video_renderer.py) |
| MP4 video output | Task 5 (PyAV encoding) |
| Text overlay with command + distance | Task 5 (_overlay_text) |
| Target goal marker in scene | Task 1 (red sphere geom) |
| Simple room with table | Task 1 |
| Integration test | Task 7 |