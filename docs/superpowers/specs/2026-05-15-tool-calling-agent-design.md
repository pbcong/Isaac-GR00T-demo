# Tool-Calling Agent Design

**Date:** 2026-05-15
**Branch:** feature/g1-hazard-inspection
**Status:** Approved

## Problem

The current G1 navigation demo orchestrates behavior through hardcoded Python:
`parse_goal → run_to_goal → inspect_target_agentic`. The inspection agent can only
choose between two JSON actions (`reposition`, `declare`) that Python manually
interprets. Navigation and inspection are tightly coupled, and there is no way for
the agent to compose actions freely or respond to failures.

## Goal

Replace the hardcoded orchestration with a single LLM tool-calling agent that receives
a mission command and autonomously chains three tools to accomplish it. The low-level
machinery (VLMBridge, GoalPlanner, WalkPolicy, MuJoCo) stays unchanged as internals.

## Architecture

```
Human command (per turn)
        │
        ▼
   AgentLoop.run_turn()          new — g1_nav_demo/agent/agent_loop.py
        │
        ├── navigate(instruction) → VLMBridge.parse() → GoalPlanner → WalkPolicy → MuJoCo
        ├── look()               → idle + VideoRenderer.snapshot("head_onboard")
        └── report(verdict, ...) → write JSON + banner render → break loop

   MuJoCo sim state persists across turns.
   Conversation history resets each turn.
```

## Tools

### `navigate(instruction: str)`
- **Input:** natural language, e.g. `"Go to the front of the table and face it"`
- **Internals:** `VLMBridge.parse(instruction)` → `run_to_goal_with_renderer()`; top-down scene image used internally, never exposed to the agent
- **Returns:** `{"reached": bool, "position": [x, y]}`
- **On failure:** agent sees `{"reached": false}` and can retry or call `report(verdict="failed")`

### `look()`
- **Input:** none
- **Internals:** `idle(250 steps)` to settle physics, then `VideoRenderer.snapshot("head_onboard", width=1280, height=960)`
- **Returns:** image inline in tool result as `content: [{type: image_url, data: ...}]`; PNG saved to disk with auto-incrementing name for debugging
- **Note:** agent sees actual pixels — no intermediate text description

### `report(verdict, findings, message)`
- **Input:**
  - `verdict`: `"safe" | "hazardous" | "complete" | "failed"`
  - `findings`: list of `{name: str, hazardous: bool, reason: str}` (empty ok for non-inspection)
  - `message`: human-readable summary
- **Internals:** writes JSON report, triggers hazard/safe banner + idle render, signals loop exit
- **Effect:** ends the current turn

## AgentLoop

**File:** `g1_nav_demo/agent/agent_loop.py`

```
history = [system_prompt, user_command]

for _ in range(MAX_TURNS=20):
    response = LLM(history, tools, tool_choice="auto")
    history.append(response.message)

    if no tool_calls: break          # model confused / gave up

    for each tool_call:
        result = dispatch(name, args)
        history.append(tool_result(tool_call.id, result))

    if report() was called: break    # clean exit

return result dict
```

**MAX_TURNS = 20** — safety cap to prevent runaway API cost.

## System Prompt

```
You are a safety-inspection robot in a room. Navigate, observe, and report.

INSPECTION STRATEGY:
- Approach the target, look at it from multiple angles before concluding.
- Red/orange box with diamond hazard symbols, radioactive trefoil, flame, or skull = HAZARDOUS.
- Seeing a face obliquely does NOT count — stand directly in front of each face.
- Declare hazardous immediately on confirmed symbol. Declare safe only after all accessible faces seen.
- Non-inspection missions: navigate and call report(verdict="complete").

Always end with report().
```

## Session Structure

- **Multi-turn:** human provides one command per turn; agent runs tool loop until `report()`; MuJoCo sim state (robot position, world) persists between turns
- **Single-turn:** same as multi-turn with one command then exit
- **Top-down scene:** only used internally by `navigate` handler via VLMBridge — the agent never sees it directly

## File Changes

| File | Change |
|------|--------|
| `g1_nav_demo/agent/__init__.py` | new (empty) |
| `g1_nav_demo/agent/agent_loop.py` | new — `AgentLoop`, `TOOL_SCHEMAS`, `SYSTEM_PROMPT` |
| `g1_nav_demo/run_demo.py` | `_run_multiturn` and `_run_single_turn` call `agent_loop.run_turn()`; `NavigationSession.inspect_target_agentic()` and `inspect_target()` removed |
| `g1_nav_demo/vlm/inspection.py` | `InspectionBridge` and `AGENTIC_INSPECTION_PROMPT` removed (superseded) |

`NavigationSession` otherwise untouched — still the low-level executor.

## What Stays Unchanged

- `VLMBridge` — still handles top-down scene parsing inside `navigate` tool
- `GoalPlanner` — pure PD math controller, unchanged
- `G1WalkPolicy` — neural net policy, unchanged
- `VideoRenderer` — renders frames and snapshots, unchanged
- `NavigationSession.run_to_goal_with_renderer()` — unchanged
- `NavigationSession.parse_goal()` — unchanged

## Extension Points

- **New tools:** add a schema entry + handler function — the loop picks them up automatically
- **Cross-turn memory:** after `report()`, distill a summary and prepend to next turn's system prompt
- **Parallel sub-agents:** navigation could become its own agent with its own tool loop
- **LangGraph migration:** the while-loop maps directly onto a LangGraph node cycle if checkpointing or visualization is ever needed
