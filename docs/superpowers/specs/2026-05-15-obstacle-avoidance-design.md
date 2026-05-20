# Obstacle Avoidance Design

**Date:** 2026-05-15  
**Branch:** feature/g1-hazard-inspection  
**Scope:** Reactive obstacle avoidance for static + dynamic obstacles in the G1 nav demo

---

## Goal

Add obstacle avoidance to the G1 humanoid navigation demo to test robustness and safety. The robot must detect and react to obstacles using only onboard sensor information — no ground-truth position data from the simulator. Two scenarios are covered:

1. **Static obstacles** — three new physical obstacles added to the room (crate, barrel, pillar). Their bounding boxes are added to the VLM's `SCENE_PROMPT` so it plans around them. They are also detected by the forward rangefinder as a safety net.
2. **Dynamic obstacles (moving human)** — a kinematic cylinder body crosses the robot's path mid-navigation. The robot detects it via a simulated forward rangefinder and reacts with a rule-based state machine.

---

## Architecture

Two independent additions layered on the existing system. The VLM, agent loop, and `GoalPlanner` are not modified.

```
NavigationSession.run_to_goal_with_renderer()
  ├── [existing] GoalPlanner.compute_command()   → nominal velocity
  ├── [new]      _read_forward_range()           → obstacle distance
  ├── [new]      AvoidanceStateMachine.step()    → override velocity if blocked
  └── [new]      ObstacleHuman.step()            → update human position in sim
```

---

## Components

### 1. `g1_nav_demo/scene/obstacle_human.py` (new)

`ObstacleHuman` — manages a kinematic body walking a fixed straight line.

**Interface:**
```python
class ObstacleHuman:
    def __init__(self, model, data, body_name: str,
                 start_xy: tuple[float, float],
                 direction_xy: tuple[float, float],
                 speed: float = 0.8,
                 travel_dist: float = 5.0): ...

    def step(self, dt: float) -> None:
        """Advance position by speed * dt, write to data.qpos, call mj_kinematics."""

    @property
    def is_done(self) -> bool:
        """True when the human has traveled travel_dist and stopped."""
```

**Default configuration:**
- Start: `(1.5, -2.5)` — below the robot's typical path to the table
- Direction: `(0, 1)` — walking north (+Y)
- Speed: `0.8 m/s`
- Travel distance: `5.0 m` — stops at `(1.5, 2.5)`, well past the crossing point
- Crossing point: ~`(1.5, 0.5)` — intersects the robot's route from origin to table

The human body is a cylinder: radius 0.35 m, half-height 0.9 m. Center z = 0.9 m (feet on floor).

---

### 2. Scene XML — `g1_nav_room.xml` (modified)

#### Static obstacles (new)

Three new floor-level obstacles added directly in `<worldbody>` with `contype="1" conaffinity="1"`. They appear in the bird's-eye render automatically and are detected by the forward rangefinder.

| Name | Shape | Center (x, y) | Size (half-extents) | Purpose |
|------|-------|---------------|---------------------|---------|
| `crate` | box | (2.0, 0.0) | (0.3, 0.3, 0.5) | Blocks direct path from origin to table |
| `barrel` | cylinder | (0.5, -1.5) | r=0.2, h=0.5 | Narrows southern corridor |
| `pillar` | cylinder | (-0.5, 2.0) | r=0.15, h=0.75 | Narrows path between bookshelf and chair |

```xml
<body name="crate" pos="2.0 0.0 0">
  <geom name="crate" type="box" size="0.3 0.3 0.5" pos="0 0 0.5"
        rgba="0.55 0.4 0.2 1" contype="1" conaffinity="1"/>
</body>

<body name="barrel" pos="0.5 -1.5 0">
  <geom name="barrel" type="cylinder" size="0.2 0.5" pos="0 0 0.5"
        rgba="0.3 0.3 0.6 1" contype="1" conaffinity="1"/>
</body>

<body name="pillar" pos="-0.5 2.0 0">
  <geom name="pillar" type="cylinder" size="0.15 0.75" pos="0 0 0.75"
        rgba="0.7 0.7 0.7 1" contype="1" conaffinity="1"/>
</body>
```

These also need corresponding entries added to the `SCENE_PROMPT` in `goal_parser.py` so the VLM plans routes around them:

```
- crate:  center (2.0, 0.0), half-extents (0.3, 0.3)  → region (1.7, -0.3)–(2.3, 0.3)
- barrel: center (0.5, -1.5), half-extents (0.2, 0.2)  → region (0.3, -1.7)–(0.7, -1.3)
- pillar: center (-0.5, 2.0), half-extents (0.15, 0.15) → region (-0.65, 1.85)–(-0.35, 2.15)
```

#### Dynamic obstacle (moving human)

Add a freejoint human body. Initial position placed off to the side at start location:

```xml
<body name="moving_human" pos="1.5 -2.5 0.9">
  <freejoint name="human_freejoint"/>
  <geom name="human_body" type="cylinder" size="0.35 0.9"
        rgba="0.7 0.5 0.3 1" contype="1" conaffinity="1"/>
</body>
```

The freejoint gives 7 DOFs `[x, y, z, qw, qx, qy, qz]`. `ObstacleHuman.step()` writes to `data.qpos[human_qpos_adr : human_qpos_adr+7]` and calls `mujoco.mj_kinematics(model, data)`.

---

### 3. `tabletop_loader.py` (modified)

Add support for `"kind": "moving_human"` in the scenario manifest so the human can be configured per scenario. Emit the freejoint body XML into `build_merged_scene` alongside table items (injected into `<worldbody>`, not inside the table body).

`build_merged_scene` continues to return only the merged XML path (no API change). After loading the model, `NavigationSession._init_obstacle_human()` scans `model.body` names for `"moving_human"` to find the qpos address. If the body is absent (scenario has no `moving_human` entry), `obstacle_human` is set to `None` — fully backward compatible.

---

### 4. `NavigationSession` — `run_demo.py` (modified)

#### `_read_forward_range(cone_deg=20.0, cutoff=3.0) -> float`

Simulated forward rangefinder. No XML sensor required — equivalent to what a physical rangefinder/lidar would measure.

```
for each geom i in model:
    skip if model.geom_type[i] == mjGEOM_PLANE     # floor
    skip if model.geom_contype[i] == 0             # tabletop items (contype=0 in XML)
    geom_pos_2d = data.geom_xpos[i][:2]
    to_geom = geom_pos_2d - robot_pos_2d
    dist = |to_geom|
    if dist < 0.1 or dist > cutoff: skip           # too close (own body) or too far
    angle = acos(dot(forward_unit, to_geom / dist))
    if angle < cone_deg: min_dist = min(min_dist, dist)
return min_dist
```

The moving human body has `contype=1`, so it is detected. Furniture (table, chair) also has `contype=1` and is detected — this is correct: the robot should also slow/stop if it approaches furniture unexpectedly.

Called every `decimation` steps (robot control rate, ~50 Hz at 500 Hz sim).

#### Avoidance state machine

Embedded in `run_to_goal_with_renderer`. Three states: `NAVIGATING`, `STOPPED`, `REROUTING`.

```
NAVIGATING
  nominal velocity from GoalPlanner
  range < stop_dist (default 1.0 m)  →  STOPPED
    zero velocity, blocked_steps = 0
    video_renderer.obstacle_banner = "OBSTACLE DETECTED — WAITING"

STOPPED
  zero velocity, blocked_steps++   # incremented every control step (~50 Hz)
  range > clear_dist (default 1.2 m)  →  NAVIGATING   [obstacle cleared]
    video_renderer.obstacle_banner = None
  blocked_steps >= timeout_steps (default 75 ≈ 1.5 s at ~50 Hz)  →  REROUTING

REROUTING
  1. current_pos = session.current_position()
  2. next_wp = goal_planner.current_waypoint
  3. forward = unit vector from current_pos to next_wp
  4. perp = rotate forward 90° left = (-forward.y, forward.x)
  5. detour = current_pos + perp * detour_dist (default 1.0 m)
  6. new_waypoints = [detour] + remaining_waypoints[current_wp_idx:]
  7. goal_planner.set_waypoints(new_waypoints, face_yaw=original_face_yaw)
  video_renderer.obstacle_banner = "REROUTING..."
  →  NAVIGATING
```

Hysteresis gap (stop at 1.0 m, resume at 1.2 m) prevents chattering at the threshold.

#### Human step in sim loop

In the main sim loop inside `run_to_goal_with_renderer`, after `mujoco.mj_step`:
```python
if self.obstacle_human is not None:
    self.obstacle_human.step(self.model.opt.timestep)
```

`obstacle_human` is set on `NavigationSession` when the scene includes a `moving_human` body; `None` otherwise.

---

### 5. `VideoRenderer` (modified)

Add `obstacle_banner: str | None = None` attribute alongside the existing `hazard_banner` and `safe_banner`. Rendered as a **yellow** overlay (distinct from red hazard banner) in `render_frame`. Text: whatever is set — e.g., `"OBSTACLE DETECTED — WAITING"` or `"REROUTING..."`.

---

### 6. CLI flags — `run_demo.py` (modified)

| Flag | Default | Description |
|------|---------|-------------|
| `--moving-obstacle` | off | Enable the kinematic human obstacle |
| `--obstacle-stop-dist` | `1.0` | Range (m) at which robot stops |
| `--obstacle-clear-dist` | `1.2` | Range (m) at which robot resumes |
| `--obstacle-timeout` | `75` | Control steps blocked before rerouting (≈1.5 s at ~50 Hz control rate) |
| `--obstacle-detour-dist` | `1.0` | Perpendicular offset (m) for detour waypoint |

---

## Data Flow

```
sim step N:
  1. ObstacleHuman.step(dt)              → update human qpos
  2. mujoco.mj_step()                    → physics
  3. [every decimation steps]
       range = _read_forward_range()
       state_machine.step(range, goal_planner)
       velocity_cmd = state_machine.current_velocity  (zero if STOPPED/REROUTING)
       walk_policy.get_action(velocity_cmd, ...)
  4. [every steps_per_render steps]
       video_renderer.render_frame(obstacle_banner=state_machine.banner)
```

---

## Testing

- **Unit: `ObstacleHuman`** — verify position advances correctly, `is_done` triggers at correct distance, qpos write is correct shape.
- **Unit: `_read_forward_range`** — place a geom directly in front of the robot at known distance, verify return value; place one behind, verify not detected; verify floor geom and contype=0 geoms are ignored.
- **Unit: state machine** — drive range values through NAVIGATING→STOPPED→REROUTING→NAVIGATING, assert correct velocity outputs and waypoint mutations.
- **Scene: static obstacles** — verify crate, barrel, pillar appear in the bird's-eye render and that the VLM plans waypoints that avoid them.
- **Integration: demo run** — run with `--moving-obstacle`, verify MP4 shows yellow banner and rerouted path in video.

---

## Out of Scope

- Multi-direction rangefinder (left/right disambiguation for detour direction — always turns left for simplicity)
- Replanning via VLM/agent (entirely rule-based)
- Multiple simultaneous moving obstacles
- Obstacle avoidance during the `idle()` phase (robot is stationary, no collision risk)
