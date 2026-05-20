# G1 Humanoid Navigation Demo Design

## Summary

A demo where a user types a natural language command (e.g., "go to the table"), a Qwen VLM parses it into navigation coordinates, and a Unitree G1 humanoid robot walks to the target in MuJoCo simulation, rendered as MP4 video.

## Architecture

```
User Input                    VLM Parsing                 Navigation                   Rendering
─────────                    ───────────                 ──────────                   ─────────
"go to the table" ──► Qwen3.5-9B VLM ──► (goal_x, goal_y) ──► Goal Planner ──► RL Walk Policy ──► MuJoCo Sim ──► MP4 Video
                             │                                                            │
                             └─ Scene description:                                         └─ Dual camera:
                                "table at (3,2),                                             - 3rd person view
                                 door at (5,0),                                               - ego view
                                 chair at (1,3)"
```

## Components

### 1. Scene (MuJoCo XML)
- Simple rectangular room (~6m x 6m)
- Floor with texture/grid
- Furniture objects with collision geometry:
  - Table at position (3, 2)
  - Chair at position (1, 3)
  - Door/exit at position (5, 0)
- Objects have semantic labels used by the VLM prompt

### 2. VLM Goal Parser
- Model: Qwen3.5-9B (or closest available VLM variant)
- Input: User's natural language command + scene description
- Output: JSON with `(target_x, target_y, target_heading_degrees)` or the name of the target object
- Prompt template provides the scene layout so the model can map object names to coordinates
- Fallback: simple keyword matching for known objects if VLM unavailable

### 3. Goal-Reaching Planner
- Simple proportional controller that outputs velocity commands (vx, vy, ωz)
- Takes current robot pose (from MuJoCo sim) and target pose
- Outputs velocity commands fed to the walking policy
- Stops when within threshold distance of target
- Avoids obstacles by checking MuJoCo collision sensors (optional: simple waypoint navigation)

### 4. RL Walking Controller
- Uses an existing Unitree G1 walking policy from either:
  - Unitree's mujoco-sim (official SDK)
  - IsaacLab trained RL policy
  - The gr00t_wbc package's walking controller
- Input: velocity commands (vx, vy, ωz) from planner
- Output: joint position targets for MuJoCo sim
- Runs at ~50Hz control loop

### 5. Video Rendering
- MuJoCo offscreen rendering ( EGL for headless)
- Dual camera views: bird's eye + ego view
- VideoRecordingWrapper-inspired MP4 output using PyAV
- Overlay: language command text rendered on video frame
- Output: single MP4 file per episode

## Data Flow

1. User types command: "go to the table"
2. VLM receives prompt with scene description and command, returns `{"target": "table", "x": 3.0, "y": 2.0, "heading": 0}`
3. Simulation initializes with G1 at (0, 0) facing forward
4. Goal planner computes velocity commands based on (current_pos → target_pos)
5. Walking policy executes velocity commands, updates MuJoCo sim
6. At each step, render both camera views, overlay command text
7. When robot reaches target (distance < threshold), stop and save MP4

## Files to Create

```
g1_nav_demo/
├── README.md                    # Setup and run instructions
├── scene/
│   └── g1_room.xml             # MuJoCo scene XML (room + furniture + G1)
├── walk_policy/
│   ├── __init__.py
│   └── g1_walk_policy.py       # RL walking policy wrapper
├── planner/
│   ├── __init__.py
│   └── goal_planner.py         # Proportional goal-reaching controller
├── vlm/
│   ├── __init__.py
│   └── goal_parser.py           # Qwen VLM goal parser + fallback keyword parser
├── run_demo.py                  # Main entry point: CLI → VLM → sim → video
└── requirements.txt             # Dependencies (transformers, mujoco, etc.)
```

## Dependencies

- `mujoco` - Simulation
- `transformers` + `torch` - Qwen VLM inference
- `av` (PyAV) - Video encoding
- `numpy` - Math
- Unitree G1 MuJoCo model assets (from Unitree's mujoco-sim repo)

## Limitations

- No dynamic obstacle avoidance (static scene only)
- VLM parsing depends on model availability and quality
- Walking policy quality depends on the source (Unitree SDK vs custom trained)
- Simple proportional controller for goal reaching (no path planning around obstacles)