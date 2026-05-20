# G1 Navigation Demo — Hazard Inspection (Feature A)

**Status:** Design approved 2026-05-14. Awaiting implementation plan.

## Goal

Extend the existing G1 navigation demo so that when a natural-language command
asks the robot to *inspect* a target (e.g. "inspect the table", "go to the
table and check it"), the robot — after reaching the target — uses its head
camera and a VLM to enumerate items on the tabletop and raise an alarm if any
are hazardous. The alarm shows as a red banner in the recorded video, a JSON
file on disk, and console log lines.

Inspection intent is decided **by the navigation VLM itself** (no regex / no
keyword list) — it returns an extra boolean field alongside the waypoints.

## Non-goals

- Obstacle avoidance during navigation (that is Feature B, a separate spec).
- Physical interaction with tabletop items (no grasping, no knocking over).
- Inspection of targets other than the table for MVP. The set of inspectable
  targets is a single-element list, easily extended later.
- Audio in the output video.
- Changing the alarm behavior at runtime (no `--on-hazard` retreat mode);
  robot stays put and idles after the banner appears.

---

## Architecture

```
User: "go to the table and inspect it"
        │
        ▼
┌──────────────────────────┐
│ VLM Goal Parser          │  birdseye PNG + text
│ (existing, extended)     │  → JSON {target, waypoints, face, inspect}
└──────────┬───────────────┘
           │  Goal(inspect=True)
           ▼
┌──────────────────────────┐
│ NavigationSession        │  walks waypoints (unchanged)
│ run_to_goal()            │
└──────────┬───────────────┘
           │  reached=True
           ▼  if goal.inspect and target ∈ INSPECTABLE_TARGETS
┌──────────────────────────┐
│ NavigationSession        │  short idle to settle pose
│ inspect_target() (NEW)   │  → render head-cam PNG
│                          │  → call InspectionBridge
│                          │  → write turn_NNN_inspection.json
│                          │  → if alarm: set renderer.hazard_banner
│                          │  → idle ~3 s so banner is recorded
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ VideoRenderer            │  red HAZARD banner overlay
│ (extended)               │
└──────────────────────────┘
```

Touchpoints (five): `scene/` (head camera + tabletop items), `vlm/goal_parser.py`
(intent field), `vlm/inspection.py` (new), `run_demo.py` (hook + CLI), and
`renderer/video_renderer.py` (banner + camera snapshot helper).

---

## Scene & tabletop items

### Head camera fix

The existing `head` camera in `g1_nav_room.xml` is **not** an on-board head
camera — it is used by `VideoRenderer` as the 3rd-person chase view (the
renderer overwrites `cam_pos` / `cam_quat` every frame). It must remain
unchanged or the existing chase view breaks.

For inspection we add a **new** camera `head_onboard` attached to the robot's
torso body in `scene/g1_29dof.xml` (the `head_link` is a geom on the torso,
not its own body):

- Position: forward and above the torso origin so it sits beyond the head
  geometry — starting value `pos="0.10 0 0.45"` (torso-frame).
- Orientation: `xyaxes` set so the camera looks forward and tilts ~20°
  downward, so a G1 standing 0.8 m from the table sees the tabletop in
  frame. Starting value `xyaxes="0 -1 0 0.34 0 -0.94"` (≈20° down).
- The existing `<camera name="head">` in `g1_nav_room.xml` is left alone.

The exact offsets and tilt must be validated empirically: the first
implementation step renders a single frame from `head_onboard` after
`mj_resetData` + `mj_forward` (robot in default standing pose, in front of
the table) and saves it to disk for visual inspection. Tune until the
tabletop is centered, unobstructed, and roughly horizon-level.

### Tabletop item kinds

Five kinds total, kept small on purpose:

| kind | MuJoCo geom | Default size (m) | Visual |
|---|---|---|---|
| `hazard_box` | textured `box` | 0.10 × 0.10 × 0.10 cube | Hazard placard image as texture on visible faces |
| `mug`        | `cylinder` | r 0.04, h 0.10 | White |
| `book`       | `box`      | 0.15 × 0.10 × 0.025 | Solid brown |
| `apple`      | `sphere`   | r 0.04 | Red |
| `laptop`     | `box`      | 0.20 × 0.15 × 0.02 | Dark grey |

Items are added as **static geoms** (no free joint) attached to the table
body. They will not fall or be knocked. Keeping items static makes the demo
deterministic and avoids depending on contact physics tuning.

### Item manifest

`g1_nav_demo/scene/tabletop_items.json`:

```json
{
  "table_top_z": 0.75,
  "default_scenario": "mixed",
  "scenarios": {
    "mixed": [
      {"name": "flammable_box",   "kind": "hazard_box",
       "texture": "image_0116.jpg", "pos_xy": [0.25, 0.15]},
      {"name": "radioactive_box", "kind": "hazard_box",
       "texture": "image_0304.jpg", "pos_xy": [-0.3, -0.1]},
      {"name": "mug",   "kind": "mug",   "pos_xy": [-0.15, 0.2]},
      {"name": "book",  "kind": "book",  "pos_xy": [0.3, -0.2]},
      {"name": "apple", "kind": "apple", "pos_xy": [0.0, 0.0]}
    ],
    "all_clear": [
      {"name": "mug",    "kind": "mug",    "pos_xy": [0.2, 0.1]},
      {"name": "book",   "kind": "book",   "pos_xy": [-0.2, -0.15]},
      {"name": "laptop", "kind": "laptop", "pos_xy": [0.0, 0.1]}
    ],
    "high_hazard": [
      {"name": "explosives_box",  "kind": "hazard_box",
       "texture": "image_0702.jpg", "pos_xy": [0.2, 0.1]},
      {"name": "infectious_box",  "kind": "hazard_box",
       "texture": "image_1421.jpg", "pos_xy": [-0.2, -0.1]},
      {"name": "combustible_box", "kind": "hazard_box",
       "texture": "image_1515.jpg", "pos_xy": [0.0, 0.2]}
    ]
  }
}
```

`pos_xy` is in the table's local frame; the loader translates by the table
body's world position. The hazard-vs-not-hazard label is **derived from the
`kind` field** (`kind == "hazard_box"` → hazardous). The label is used only
by tests and never shown to the VLM.

### Scene loader

`g1_nav_demo/scene/tabletop_loader.py` — a small Python module called *before*
`mujoco.MjModel.from_xml_path()` in `run_demo._init_simulation`:

1. Read `g1_nav_room.xml`.
2. Read `tabletop_items.json` and pick the requested scenario.
3. For each item, build a MuJoCo XML snippet:
   - `hazard_box` → adds a `<texture file="..."/>` + `<material name="..."
     texture="..."/>` to `<asset>`, and a `<geom type="box" material="..."
     ...>` inside `<body name="table">`.
   - Other kinds → simpler `<geom>` with `rgba`.
4. Write the merged XML to a temp file and return its path. The caller
   passes that path to `MjModel.from_xml_path`.

This avoids mutating the source `g1_nav_room.xml`. Loader is unit-testable
without MuJoCo (it produces a string).

Hazard textures resolve against a configurable `--hazard-textures-dir`
(default: `selected_imgs_videos_demo/Hazard_detection_selected` at repo root).

---

## VLM changes

### Goal parser (`vlm/goal_parser.py`)

- `Goal` dataclass gains `inspect: bool = False`.
- `SCENE_PROMPT` gains:
  ```
  INSPECTION INTENT:
    Set "inspect" to true ONLY if the command explicitly asks to
    inspect, check, examine, scan, or look at the target's contents.
    "Go to the table" → false. "Inspect the table" → true.
    "Go to the table and check it" → true.
  ```
- Output JSON gains `"inspect": <bool>`.
- `_extract_goal` reads the new field; defaults to `False` if absent.

### Inspection bridge (NEW `vlm/inspection.py`)

```python
@dataclass
class InspectionItem:
    name: str
    hazardous: bool
    reason: str

@dataclass
class InspectionResult:
    items: list[InspectionItem]
    alarm: bool
    raw_response: str

class InspectionBridge:
    def __init__(self, model_name: str = ..., api_base: str = ..., api_key: str | None = None): ...
    def inspect(self, head_cam_png: bytes, target_name: str) -> InspectionResult | None: ...
```

Prompt:
```
You are a safety inspector. The image is from a humanoid robot's head camera,
looking down at a tabletop. List every distinct item visible and decide whether
it is hazardous. An item is hazardous if it bears a UN-style hazard placard
(Class 1–9, e.g. EXPLOSIVES, FLAMMABLE, RADIOACTIVE, INFECTIOUS SUBSTANCE,
SPONTANEOUSLY COMBUSTIBLE) or is itself a weapon, open flame, exposed
chemical, or sharp instrument.

Output ONLY this JSON on one line. No markdown.
{"items": [{"name": "...", "hazardous": true|false, "reason": "..."}], "alarm": true|false}

alarm MUST be true iff any item.hazardous is true.
```

`alarm` is re-validated client-side: if the VLM sets `alarm: false` but any
item has `hazardous: true`, we override to `alarm: true` and log a warning.
This guards against inconsistent VLM output.

On VLM failure (network error, unparseable response) `inspect` returns
`None`. The caller treats `None` as "inspection unavailable" and does not
raise the banner.

---

## Session & renderer integration

### `run_demo.py`

- Module-level constant: `INSPECTABLE_TARGETS = {"table"}`.
- `NavigationSession.__init__` gains a parameter `inspection_bridge:
  InspectionBridge | None = None`.
- New method:
  ```python
  def inspect_target(
      self,
      goal: Goal,
      command: str,
      video_renderer: VideoRenderer,
      inspection_json_path: str,
  ) -> InspectionResult | None:
      if not goal.inspect: return None
      if goal.target_name not in INSPECTABLE_TARGETS: return None
      if self.inspection_bridge is None: return None
      self.idle(duration_steps=250)        # ~0.5 s pose settle
      head_png = video_renderer.snapshot("head")
      result = self.inspection_bridge.inspect(head_png, goal.target_name)
      if result is None: return None
      _write_inspection_json(inspection_json_path, result)
      if result.alarm:
          names = ", ".join(i.name for i in result.items if i.hazardous)
          logger.warning("HAZARD DETECTED at %s: %s", goal.target_name, names)
          video_renderer.hazard_banner = f"HAZARD DETECTED: {names}"
      else:
          logger.info("Inspection clear at %s", goal.target_name)
      self.idle(duration_steps=1500)       # ~3 s so banner shows in video
      video_renderer.hazard_banner = None  # clear before next turn
      return result
  ```
- `run_to_goal` is **not** changed; the caller (`_run_single_turn` /
  `_run_multiturn`) invokes `inspect_target` after `run_to_goal` returns
  `reached=True`. The inspection JSON path is derived from the video path
  by replacing the extension:
  - multi-turn: `demo_output/turn_001_table.mp4` → `demo_output/turn_001_inspection.json`
  - single-turn: `demo_output.mp4` → `demo_output_inspection.json`
- `_init_simulation` constructs the `InspectionBridge` (same OpenRouter
  config as the existing `VLMBridge`) and passes it into `NavigationSession`.
- CLI flags:
  - `--tabletop-scenario {mixed,all_clear,high_hazard}` (default: from JSON
    `default_scenario`)
  - `--hazard-textures-dir <path>` (default: see scene loader)

### `renderer/video_renderer.py`

- `VideoRenderer` gains `self.hazard_banner: str | None = None`.
- `render_frame()` checks `self.hazard_banner` at the end and, if not `None`,
  draws a red filled rectangle across the top of the combined frame (full
  width × 60 px) with white sans-serif text `"⚠ HAZARD DETECTED: ..."`.
  Uses Pillow `ImageDraw` — no new dependency.
- New method `snapshot(camera_name: str, data: mujoco.MjData) -> bytes`:
  resolves the camera id by name (so `head_onboard` works), renders one
  MuJoCo camera to an offscreen buffer, returns PNG bytes. Builds a fresh
  `mujoco.Renderer` at 640×480 (enough for the VLM) and closes it after
  use, to avoid interfering with the active panel renderers.

---

## Data flow summary (inspection turn)

```
"inspect the table"
   → VLM goal parser
   → Goal(target_name="table", waypoints=[...], inspect=True)
   → run_to_goal()  (existing pipeline, unchanged)
   → reached=True
   → inspect_target()
       → idle 0.5 s
       → snapshot("head")  →  head_cam.png (in memory)
       → InspectionBridge.inspect(head_cam.png, "table")
       → InspectionResult{items=[...], alarm=True/False}
       → write turn_NNN_inspection.json
       → if alarm: set renderer.hazard_banner
       → idle 3 s (banner recorded into video)
       → clear banner
   → next turn / exit
```

---

## Testing

| Test | File | What it asserts |
|---|---|---|
| Intent field, inspect command | `vlm/test_goal_parser.py` (extend) | Mocked VLM response with `"inspect": true` produces `Goal.inspect == True`. |
| Intent field, plain navigation | same | Mocked response without `inspect` field defaults to `False`. |
| Loader produces correct geoms | `scene/test_tabletop_loader.py` (NEW) | For each scenario, merged XML contains the expected geom names. |
| Loader resolves textures | same | `hazard_box` items produce `<texture file="...">` referencing the configured dir. |
| Inspection result parsing | `vlm/test_inspection.py` (NEW) | Given a mocked VLM response string, `InspectionResult.alarm` matches expected. |
| Alarm consistency override | same | Response with `items[*].hazardous=true` but `alarm=false` → final `alarm=True`. |
| Banner overlay | `renderer/test_video_renderer.py` (NEW or extend) | When `hazard_banner` is set, top row of the rendered frame contains red pixels. |
| Manual smoke test | `HOWTO.md` | `inspect the table` with `mixed` scenario produces a video with the banner and a JSON file showing `alarm: true`. |

No network calls in unit tests — VLM responses are mocked at the
`OpenAI.chat.completions.create` boundary, same as existing tests.

---

## Files changed / added

```
g1_nav_demo/
├── scene/
│   ├── g1_29dof.xml             # MODIFIED: head camera relocated + tilted
│   ├── g1_nav_room.xml          # MODIFIED: remove static head camera
│   ├── tabletop_items.json      # NEW: scenario manifests
│   ├── tabletop_loader.py       # NEW: merges items into scene XML at load time
│   └── test_tabletop_loader.py  # NEW
├── vlm/
│   ├── goal_parser.py           # MODIFIED: Goal.inspect, prompt update
│   ├── inspection.py            # NEW: InspectionBridge + InspectionResult
│   ├── test_goal_parser.py      # MODIFIED
│   └── test_inspection.py       # NEW
├── renderer/
│   ├── video_renderer.py        # MODIFIED: hazard_banner overlay, snapshot()
│   └── test_video_renderer.py   # NEW (or extend if exists)
├── run_demo.py                  # MODIFIED: inspect_target hook + CLI flags
├── HOWTO.md                     # MODIFIED: document inspect commands + scenarios
└── README.md                    # MODIFIED: brief inspection section
```

---

## Risks & open items

- **Head-cam framing depends on visual validation.** First implementation
  step renders a single still from the new head-cam configuration to disk
  for eyeballing before any inspection wiring goes in.
- **VLM accuracy on real-world placards is unverified.** The 8 example
  images are photographed off-angle with busy backgrounds. If the VLM
  fails to detect them reliably when applied as box textures, fallback
  is to enrich the inspection prompt with an explicit class list and
  examples. This is a tuning task, not a design change.
- **Static items vs. physics.** If the demo later needs items to be
  knocked / interacted with, switch them from static geoms to bodies with
  free joints; the manifest schema already accommodates this (add an
  optional `physics: true` field).
