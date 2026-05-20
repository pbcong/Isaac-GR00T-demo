# G1 Navigation Demo — Setup & How to Run

---

## Environment Setup

### Prerequisites

| Requirement | Version |
|---|---|
| OS | Linux x86_64 |
| NVIDIA driver | ≥ 550 (CUDA 12.4+) |
| Python | 3.10.x (exactly) |
| uv | any recent version |

Install `uv` if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1 — Clone the repo

```bash
git clone <repo-url>
cd Isaac-GR00T
```

### 2 — Create the venv (Python 3.10)

```bash
uv venv .venv --python 3.10
source .venv/bin/activate
```

### 3 — Install the base project

```bash
uv pip install -e .
```

### 4 — Install demo-specific packages

```bash
uv pip install \
    "torch==2.10.0+cu128" \
    "torchvision==0.25.0+cu128" \
    --extra-index-url https://download.pytorch.org/whl/cu128

uv pip install \
    vllm==0.19.1 \
    mujoco==3.8.0 \
    av==16.1.0 \
    openai>=1.0.0 \
    accelerate==1.12.0 \
    "transformers>=4.57.0" \
    "qwen-vl-utils[decord]==0.0.8" \
    "Pillow>=10.0.0" \
    "opencv-python>=4.8.0"
```

### 5 — Set your OpenRouter API key

Create a `.env` file in the project root:

```
OPENROUTER_API_KEY=sk-or-...
```

The agent uses OpenRouter to call the LLM (default model: `x-ai/grok-4.3`). The tool-calling loop sends the system prompt, user command, and tool results to the API on each iteration.

### 6 — Download Qwen3.5-9B model weights (for local VLM server mode only)

If you want to run a local goal-parsing VLM instead of OpenRouter:

```bash
export HF_HOME=$PWD/g1_nav_demo/models
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.5-9B')
"
```

Expected path after download:
```
g1_nav_demo/models/hub/models--Qwen--Qwen3.5-9B/snapshots/<hash>/
```

This needs ~18 GB of disk space and a Hugging Face account token if the model is gated:

```bash
huggingface-cli login   # only needed once
```

### 7 — Verify GPU memory

```bash
nvidia-smi
```

---

## Running the Demo

### Single-turn (one command, then exit)

```bash
cd Isaac-GR00T
source .venv/bin/activate

MUJOCO_GL=egl python g1_nav_demo/run_demo.py \
    --command "go to the table" \
    --policy-path g1_nav_demo/walk_policy/motion.pt \
    --output demo_output.mp4 \
    --device cuda \
    --max-steps 20000
```

This runs `AgentLoop.run_turn()` once: the LLM receives `"go to the table"` and autonomously chains `navigate`, `look`, and `report` tool calls. Output:
- `demo_output.mp4` — video with waypoint overlay and any banner
- `demo_output_report.json` — JSON verdict and findings

### Multi-turn interactive

```bash
cd Isaac-GR00T
source .venv/bin/activate

MUJOCO_GL=egl python g1_nav_demo/run_demo.py \
    --multiturn \
    --policy-path g1_nav_demo/walk_policy/motion.pt \
    --output-dir demo_output \
    --device cuda \
    --max-steps 20000

# Then type commands:
#   inspect the table       ← agent navigates, looks, and reports
#   go to the chair         ← agent navigates and reports
#   quit
```

Each turn produces:
- `demo_output/turn_001.mp4` — video of that turn
- `demo_output/turn_001_report.json` — JSON report

### Using the convenience script

```bash
# Single-turn
bash g1_nav_demo/run_vlm_demo.sh "go to the table" demo_output.mp4

# Multi-turn
bash g1_nav_demo/run_vlm_demo.sh --multiturn demo_output/
```

The script sets `MUJOCO_GL=egl`, loads `.env`, and uses GPU 3 by default.

### Local VLM server (alternative to OpenRouter for goal parsing)

Two terminals are needed:

**Terminal 1 — start the server:**
```bash
bash g1_nav_demo/start_vlm_server.sh
```

**Terminal 2 — run the demo:**
```bash
cd Isaac-GR00T
source .venv/bin/activate
MUJOCO_GL=egl python g1_nav_demo/run_demo.py \
    --command "go to the table" \
    --policy-path g1_nav_demo/walk_policy/motion.pt \
    --output demo_output.mp4 \
    --device cuda
```

> **Note:** The local vLLM server is only used for goal parsing (`VLMBridge`). The agent's tool-calling LLM always uses OpenRouter. If you want a fully local setup, change `api_base` and `model_name` when constructing `AgentLoop`.

---

## How the Agent Works

### Per-turn flow

```
User command ──→ AgentLoop.run_turn(command, video_renderer, report_json_path)
                     │
                     ├── Build conversation: [system_prompt, user_command]
                     │
                     └── Loop (up to MAX_TURNS=20):
                           │
                           ├── Call LLM with tools + history
                           │
                           ├── For each tool_call:
                           │     ├── navigate(instruction)
                           │     │     → session.parse_goal(instruction)
                           │     │     → session.run_to_goal_with_renderer(goal, instruction, renderer)
                           │     │     → return {"reached": bool, "position": [x, y]}
                           │     │
                           │     ├── look()
                           │     │     → session.idle(250 steps)
                           │     │     → renderer.snapshot("head_onboard", data, width=1280, height=960)
                           │     │     → save PNG to disk, return base64 image
                           │     │
                           │     └── report(verdict, findings, message)
                           │           → write JSON report
                           │           → set banner (hazard or complete)
                           │           → session.idle(1500 steps) for banner video
                           │           → clear banner
                           │           → break loop
                           │
                           └── If no tool_calls: break with verdict="failed"
```

### Model configuration

The default model is `x-ai/grok-4.3` via OpenRouter. To use a different model:

```python
from g1_nav_demo.agent.agent_loop import AgentLoop

loop = AgentLoop(session=session, model_name="anthropic/claude-sonnet-4")
```

Or set the `OPENROUTER_API_KEY` environment variable and pass a different model via `--vlm-model`.

---

## Available Commands

| Command | What happens |
|---|---|
| `go to the table` | Agent navigates to the table and reports `verdict="complete"` |
| `walk to the chair` | Agent navigates to the chair and reports |
| `inspect the table` | Agent navigates, looks, possibly repositions, and reports `verdict="hazardous"` or `"safe"` |
| `check the table for hazards` | Same as inspect — agent looks and reports |
| `head to the door` | Agent navigates to the door and reports |

The LLM decides how many `navigate` and `look` calls to chain based on the command. Simple navigation commands (e.g. "go to the table") typically result in: `navigate → report(complete)`. Inspection commands result in: `navigate → look → [navigate → look]* → report(hazardous/safe)`.

---

## Inspection Details

### Scenarios

Three scenarios ship in `g1_nav_demo/scene/tabletop_items.json`:

| Scenario | Items |
|---|---|
| `easy_hazard` | Flammable Liquid placard + mug + book |
| `hard_hazard` | Radioactive placard + mug + book + apple |
| `safe_box` | mug + book + laptop only |

Select with `--tabletop-scenario`:

```bash
python g1_nav_demo/run_demo.py \
    --command "inspect the table" \
    --policy-path g1_nav_demo/walk_policy/motion.pt \
    --output demo_hazard.mp4 \
    --tabletop-scenario hard_hazard
```

### Report JSON format

Each turn writes a `_report.json` file:

```json
{
  "command": "inspect the table",
  "verdict": "hazardous",
  "message": "Found a radioactive placard on the red box",
  "findings": [
    {"name": "red box", "hazardous": true, "reason": "radioactive trefoil symbol"},
    {"name": "mug", "hazardous": false, "reason": "ceramic cup"}
  ]
}
```

### Head camera snapshots

Each `look()` call saves a PNG: `demo_output_look_01.png`, `demo_output_look_02.png`, etc. These are the same images the LLM sees — useful for debugging what the agent observed.

### Tuning the on-board camera

If the head-camera view is poorly framed:

```bash
python g1_nav_demo/scripts/render_head_cam.py --out preview.png
```

Then tweak the `<camera name="head_onboard" ...>` line in `g1_nav_demo/scene/g1_29dof.xml`.

---

## GPU Memory Notes

For the local vLLM Qwen3.5-9B server (goal parsing only):

| Setting | Value | Why |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | `2,3` | Two idle GPUs (change to match yours) |
| `--tensor-parallel-size` | `2` | Splits ~18 GB model across 2 GPUs (~9 GB each) |
| `--gpu-memory-utilization` | `0.36` | 0.36 × 40 GB = 14.4 GB requested per GPU |
| `--max-model-len` | `16384` | Reduces KV-cache allocation; nav commands are short |

The tool-calling LLM (grok-4.3 via OpenRouter) requires no local GPU.

---

## CLI Reference

| Option | Default | Description |
|---|---|---|
| `--command` | required (single-turn) | Natural-language command |
| `--scene-xml` | auto | MuJoCo scene XML |
| `--policy-path` | required | Path to walking policy `.pt` checkpoint |
| `--output` | `demo_output.mp4` | Output video file (single-turn) |
| `--vlm-model` | `x-ai/grok-4.3` | Model name for agent LLM and goal parser |
| `--max-steps` | 20000 | Max simulation steps per navigated segment |
| `--sim-fps` | 500 | Physics frequency (Hz) |
| `--render-fps` | 30 | Video frame rate |
| `--device` | `cuda` | Torch device |
| `--multiturn` | off | Enable interactive multi-turn mode |
| `--output-dir` | `demo_output` | Directory for per-turn videos (multiturn) |
| `--tabletop-scenario` | from manifest | Which scenario to load |
| `--hazard-textures-dir` | auto | Folder of hazard placard JPGs |
| `--tabletop-manifest` | `scene/tabletop_items.json` | Alternate manifest path |

---

## File Layout

```
g1_nav_demo/
├── start_vlm_server.sh        # Start local vLLM server for goal parsing
├── run_vlm_demo.sh            # Convenience wrapper for running the demo
├── HOWTO.md                   # This file — setup & running guide
├── run_demo.py                # Main entry point (NavigationSession, _init_simulation, _run_single_turn, _run_multiturn)
├── requirements.txt           # pip-installable dependencies
├── agent/
│   ├── __init__.py             # Package marker
│   ├── agent_loop.py           # AgentLoop, TOOL_SCHEMAS, SYSTEM_PROMPT, helpers
│   └── test_agent_loop.py      # 16 unit tests (all mocked, no MuJoCo needed)
├── walk_policy/
│   ├── motion.pt               # Pre-trained RL walking policy
│   └── g1_walk_policy.py
├── models/                     # Qwen3.5-9B weights (local server mode, set via HF_HOME)
│   └── hub/models--Qwen--Qwen3.5-9B/
├── vlm/
│   ├── goal_parser.py          # VLMBridge (OpenRouter client) + KeywordParser + Goal(inspect=)
│   ├── scene_map.py             # Generates top-down map image sent to VLM
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

## Extension Points

- **New tools:** Add a schema entry to `TOOL_SCHEMAS` in `agent_loop.py` and a handler branch in `run_turn()` — the loop picks them up automatically.
- **Cross-turn memory:** After `report()`, distill a summary and prepend it to the next turn's system prompt.
- **Different LLM:** Change `model_name` when constructing `AgentLoop`, or set a different `api_base` for a non-OpenRouter endpoint.
- **LangGraph migration:** The while-loop maps directly onto a LangGraph node cycle if checkpointing or visualization is ever needed.