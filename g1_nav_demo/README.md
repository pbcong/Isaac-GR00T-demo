# G1 Navigation Demo

A Unitree G1 humanoid robot navigates to furniture in a MuJoCo room based on natural-language commands. A VLM (via OpenRouter) parses the command using a top-down grid map, then a proportional planner drives the RL walking policy. An LLM tool-calling agent orchestrates navigation, observation, and reporting through three tools: `navigate`, `look`, and `report`.

```
"inspect the table"
    → AgentLoop receives command
    → LLM decides: navigate("go to the table")  →  VLMBridge → GoalPlanner → WalkPolicy → MuJoCo
    → LLM decides: look()                       →  idle(250) → head camera snapshot (1280×960)
    → LLM decides: report("hazardous", [...])    →  JSON report + banner render
```

---

## Quick Start

### 1 — Set up environment

```bash
cd Isaac-GR00T
source .venv/bin/activate
```

Add your OpenRouter API key to `.env`:
```
OPENROUTER_API_KEY=sk-or-...
```

### 2 — Single-turn demo

```bash
bash g1_nav_demo/run_vlm_demo.sh "go to the table" demo_output.mp4
```

### 3 — Multi-turn interactive demo

```bash
bash g1_nav_demo/run_vlm_demo.sh --multiturn demo_output/
# Then type commands:
#   go to the chair
#   inspect the table
#   quit
```

---

## Tested Scenarios

| Command | Destination | Approach position | Steps to reach |
|---|---|---|---|
| `go to the table` | table (3.0, 2.0) | (3.0, 1.0) south face | ~2980 |
| `walk to the chair` | chair (1.0, 3.0) | (1.0, 2.2) south face | ~2220 |
| `head to the door` | door (5.0, 0.0) | (4.2, 0.0) west face | ~3760 |
| `go to the bookshelf` | bookshelf (−2.0, 1.0) | (−1.0, 1.0) east face | ~3220 |
| `go to the couch` | couch (0.0, −3.0) | (0.0, −2.0) north face | |

---

## CLI Reference

| Option | Default | Description |
|---|---|---|
| `--command` | required (single-turn) | Natural-language navigation command |
| `--scene-xml` | auto | MuJoCo scene XML |
| `--policy-path` | required | Path to walking policy `.pt` checkpoint |
| `--output` | `demo_output.mp4` | Output video file (single-turn) |
| `--vlm-model` | `x-ai/grok-4.3` | Model name for OpenRouter API |
| `--max-steps` | 20000 | Max simulation steps per turn |
| `--sim-fps` | 500 | Physics frequency (Hz) |
| `--render-fps` | 30 | Video frame rate |
| `--device` | `cuda` | Torch device |
| `--multiturn` | off | Enable interactive multi-turn mode |
| `--output-dir` | `demo_output` | Directory for per-turn videos (multiturn) |
| `--tabletop-scenario` | from manifest | Tabletop scenario name |
| `--hazard-textures-dir` | auto | Directory containing hazard placard images |
| `--tabletop-manifest` | `scene/tabletop_items.json` | Alternate manifest path |

---

## Architecture — Tool-Calling Agent

The orchestrator is an `AgentLoop` that receives a human command and autonomously chains three tools to accomplish the mission. The low-level machinery (VLMBridge, GoalPlanner, WalkPolicy, MuJoCo) stays unchanged as internals.

```
Human command (per turn)
        │
        ▼
   AgentLoop.run_turn()              g1_nav_demo/agent/agent_loop.py
        │
        ├── navigate(instruction)  → VLMBridge.parse() → GoalPlanner → WalkPolicy → MuJoCo
        ├── look()                  → idle(250) + VideoRenderer.snapshot("head_onboard")
        └── report(verdict, ...)    → write JSON + banner render → break loop

   MuJoCo sim state persists across turns.
   Conversation history resets each turn.
```

### Tools

| Tool | Input | What it does | Returns |
|---|---|---|---|
| `navigate(instruction)` | Natural language, e.g. `"Go to the front of the table"` | VLMBridge parses command → GoalPlanner → WalkPolicy → MuJoCo sim | `{"reached": bool, "position": [x, y]}` |
| `look()` | None | Robot idles 250 steps to settle physics, then captures a 1280×960 head-camera PNG | Image inline as base64 in tool result |
| `report(verdict, findings, message)` | `verdict`: safe/hazardous/complete/failed; `findings`: list of items; `message`: summary | Writes JSON report, triggers banner render, signals loop exit | Ends the turn |

### AgentLoop

The core loop in `g1_nav_demo/agent/agent_loop.py`:

```python
history = [system_prompt, user_command]

for _ in range(MAX_TURNS=20):
    response = LLM(history, tools, tool_choice="auto")
    history.append(response.message)

    if no tool_calls: break

    for each tool_call:
        result = dispatch(name, args)
        history.append(tool_result(tool_call.id, result))

    if report() was called: break

return result dict
```

- **MAX_TURNS = 20** — safety cap to prevent runaway API cost.
- If the model returns no tool calls, the loop ends with `verdict="failed"`.
- `report()` always ends the turn cleanly.

### System Prompt

```
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
```

---

## How It Works — Full Pipeline Walkthrough

### Stage 1: VLM Goal Parsing (`vlm/goal_parser.py`)

**Input:** A natural-language command like `"go to the table"`.

**What happens:**

1. **Render the scene map** (`vlm/scene_map.py`): MuJoCo renders a top-down birdseye view (camera at 1.5, 0, 12 — fixed at room center, always shows the full room). A PIL overlay draws a coordinate grid (0.5 m lines, 1.0 m labels) with no robot marker.

2. **Build the VLM prompt**: The birdseye PNG is base64-encoded and sent to the VLM API (OpenRouter) along with:
   - A **system prompt** (`SCENE_PROMPT`) describing the grid, obstacle rules, approach-position rules, and detour strategy. It asks the VLM to output JSON.
   - A **user message** containing the image + text like: `"Robot is currently at (0.0, 0.0), facing east (+X). go to the table"`. In multi-turn mode, the robot's actual current position and heading are included.

3. **VLM response**: The model returns JSON:
   ```json
   {"target_name": "table", "waypoints": [[0,0], [1.5,0.5], [3.0,1.0]]}
   ```
   This is parsed into a `Goal` dataclass with `target_name: str` and `waypoints: list[tuple[float,float]]`.

### Stage 2: Waypoint Following (`planner/goal_planner.py`)

Proportional controller that tracks waypoints and outputs `(vx, vy, vyaw)` velocity commands at 50 Hz. The robot slows when close and turns before walking.

### Stage 3: RL Walking Policy (`walk_policy/g1_walk_policy.py`)

TorchScript JIT policy maps 47-dim observation (angular velocity, projected gravity, velocity command, joint positions/velocities, gait phase) to 12 joint targets via PD control.

### Stage 4: PD Torque Control + MuJoCo Simulation

Each 500 Hz sim step: PD torques applied to leg joints + upper body holds default pose + `mj_step()`.

### Stage 5: Video Rendering (`renderer/video_renderer.py`)

Dual-camera renderer (overhead + chase cam) with waypoint overlay and banner support.

---

## Multi-Turn Mode

When launched with `--multiturn`:

1. The program prompts for a command via stdin.
2. `AgentLoop.run_turn()` processes the command — the LLM autonomously chains `navigate`, `look`, and `report` tool calls.
3. The program displays the verdict and position.
4. The program prompts for the next command. Typing `quit`/`exit`/`q` ends the session.
5. Each turn produces a video (`turn_001.mp4`, `turn_002.mp4`, ...) and a JSON report (`turn_001_report.json`, etc.).

The **MuJoCo simulation state persists** between turns — joint angles, position, and velocity carry over. The walking policy's internal state (last action, gait phase) also persists.

---

## Hazard Inspection

The LLM agent handles inspection autonomously. When given a command like `"inspect the table"`:

1. The agent calls `navigate("Go to the front of the table and face it")` — walks to the target.
2. The agent calls `look()` — takes a 1280×960 head-camera snapshot.
3. The agent examines the image and may call `navigate` again to reposition for a different angle, then `look` again.
4. When confident, the agent calls `report()` with `verdict="hazardous"` or `verdict="safe"`.

The report JSON is written to disk:
```json
{
  "command": "inspect the table",
  "verdict": "hazardous",
  "message": "Found radioactive placard on the box",
  "findings": [
    {"name": "red box", "hazardous": true, "reason": "radioactive trefoil symbol"}
  ]
}
```

If `verdict="hazardous"`, a red "HAZARD DETECTED" banner is overlaid on the video. For `safe` or `complete`, a green "MISSION COMPLETE" banner appears.

### Scenarios

Three scenarios ship in `g1_nav_demo/scene/tabletop_items.json`:

| Scenario | Items |
|---|---|
| `easy_hazard` | Flammable Liquid placard + mug + book |
| `hard_hazard` | Radioactive placard + mug + book + apple |
| `safe_box` | mug + book + laptop only |

Select with `--tabletop-scenario`.

---

## Data Flow Summary

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  User says   │     │  AgentLoop   │     │              │     │  Walk Policy  │
│ "inspect the │────▶│  LLM chooses │────▶│  Navigate →  │────▶│  47-dim obs   │
│   table"     │     │  tool calls  │     │  VLM → Plan  │     │  → 12 joints  │
└─────────────┘     │  autonomously│     └──────────────┘     └──────────────┘
                    └──────┬───────┘                            │
                           │                             ┌───────▼────────┐
                    ┌──────▼───────┐                     │  PD Control     │
                    │  Look/Report  │                     │  MuJoCo 500 Hz  │
                    │  tool results │                     └────────────────┘
                    └──────────────┘
```

---

## File Layout

```
g1_nav_demo/
├── start_vlm_server.sh        # Start local vLLM server (alternative to OpenRouter)
├── run_vlm_demo.sh            # Run demo with VLM (--multiturn for interactive)
├── run_demo.py                # Main entry point (NavigationSession + _run_single_turn / _run_multiturn)
├── README.md                  # This file
├── HOWTO.md                   # Setup & running guide
├── requirements.txt           # pip-installable dependencies
├── agent/
│   ├── __init__.py             # Package marker
│   ├── agent_loop.py           # AgentLoop, TOOL_SCHEMAS, SYSTEM_PROMPT
│   └── test_agent_loop.py      # Unit tests (16 tests, all mocked)
├── walk_policy/
│   ├── motion.pt              # Pre-trained G1 RL walking policy
│   └── g1_walk_policy.py       # Policy wrapper (obs → action → joint targets)
├── models/                     # VLM weights (for local server mode)
├── vlm/
│   ├── goal_parser.py          # VLMBridge (OpenRouter client) + Goal(inspect=)
│   ├── scene_map.py             # Renders birdseye grid-only PNG for VLM
│   └── test_goal_parser.py      # Unit tests for goal parser
├── planner/
│   └── goal_planner.py          # Proportional waypoint controller
├── renderer/
│   ├── video_renderer.py        # Dual-camera renderer + banner + snapshot()
│   └── test_video_renderer.py
├── scripts/
│   └── render_head_cam.py      # Standalone head-camera preview for tuning
└── scene/
    ├── g1_nav_room.xml          # MuJoCo scene (room + furniture + cameras)
    ├── g1_29dof.xml             # G1 robot model (includes head_onboard camera)
    ├── tabletop_items.json      # Hazard scenario manifest
    ├── tabletop_loader.py       # Merges scenario items into scene XML
    └── test_tabletop_loader.py
```

---

## Known Limitations

- No geometric obstacle avoidance — the planner follows VLM waypoints directly
- Walking policy may drift sideways slightly; longer paths have more accumulated error
- VLM call adds ~1-2 s latency per turn
- Approach positions are VLM-determined; no dynamic re-planning if the robot drifts off course
- The LLM agent is limited to the three tools (`navigate`, `look`, `report`) — new tools can be added by extending `TOOL_SCHEMAS` and the dispatch table in `run_turn()`
- Tabletop items are static geoms (no physics interaction); if the demo later needs items to be knockable, they need free-joint bodies