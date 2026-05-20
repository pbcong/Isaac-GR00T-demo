# G1 Hazard Inspection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the G1 navigation demo so commands containing inspection intent ("inspect the table") trigger a head-camera VLM hazard check after arrival, raising a red banner on the video and writing a JSON report if any hazardous items are detected.

**Architecture:** Add a new on-board `head_onboard` camera (the existing `head` camera is the chase view and must remain untouched). Extend the goal parser to return an `inspect` boolean. Add scenario-driven tabletop items via a JSON manifest and a Python loader that splices geoms into a temp XML before MuJoCo loads it. Add a second VLM client (`InspectionBridge`) called by `NavigationSession.inspect_target()` after arrival, which writes a JSON report and sets a `hazard_banner` on the renderer.

**Tech Stack:** MuJoCo 3.8, Pillow + OpenCV (already in the demo), OpenAI SDK against OpenRouter, pytest (existing test layout), PyAV for video.

**Spec:** `docs/superpowers/specs/2026-05-14-g1-hazard-inspection-design.md`

---

## Task 1: Add `VideoRenderer.snapshot()` helper

Adds a method that renders a single MuJoCo camera by name and returns PNG bytes. Used later by `inspect_target` to feed the VLM. Does not change any existing behavior.

**Files:**
- Modify: `g1_nav_demo/renderer/video_renderer.py`
- Test: `g1_nav_demo/renderer/test_video_renderer.py` (NEW)

- [ ] **Step 1: Write the failing test**

Create `g1_nav_demo/renderer/test_video_renderer.py`:

```python
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import pytest

from g1_nav_demo.renderer.video_renderer import VideoRenderer

SCENE_XML = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "scene", "g1_nav_room.xml"
)


@pytest.fixture
def model_and_data():
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_snapshot_returns_png_bytes(model_and_data, tmp_path):
    model, data = model_and_data
    renderer = VideoRenderer(model, output_path=str(tmp_path / "ignored.mp4"))
    try:
        png = renderer.snapshot("birdseye", data)
    finally:
        renderer.close()

    assert isinstance(png, bytes)
    assert png[:8] == b"\x89PNG\r\n\x1a\n", "Result is not a PNG file"


def test_snapshot_unknown_camera_raises(model_and_data, tmp_path):
    model, data = model_and_data
    renderer = VideoRenderer(model, output_path=str(tmp_path / "ignored.mp4"))
    try:
        with pytest.raises(ValueError, match="camera"):
            renderer.snapshot("not_a_real_camera", data)
    finally:
        renderer.close()
```

- [ ] **Step 2: Run the test and verify it fails**

```bash
cd /mnt/raid10/erwin/testing/astar/Isaac-GR00T
.venv/bin/pytest g1_nav_demo/renderer/test_video_renderer.py -v
```

Expected: FAIL — `AttributeError: 'VideoRenderer' object has no attribute 'snapshot'`.

- [ ] **Step 3: Implement `snapshot()`**

Edit `g1_nav_demo/renderer/video_renderer.py`. Add this method to `VideoRenderer`, placed after `close()` and before the end of the class:

```python
    def snapshot(
        self,
        camera_name: str,
        data: mujoco.MjData,
        width: int = 640,
        height: int = 480,
    ) -> bytes:
        """Render a single MuJoCo camera and return PNG-encoded bytes.

        Creates a fresh Renderer at the requested size and closes it
        immediately, so this does not interfere with the active panel
        renderers used by render_frame().
        """
        import io

        from PIL import Image

        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id < 0:
            raise ValueError(f"Unknown camera: {camera_name!r}")

        renderer = mujoco.Renderer(self.model, height=height, width=width)
        try:
            renderer.update_scene(data, camera=cam_id)
            rgb = renderer.render()
        finally:
            renderer.close()

        buf = io.BytesIO()
        Image.fromarray(rgb.astype(np.uint8)).save(buf, format="PNG")
        return buf.getvalue()
```

- [ ] **Step 4: Run the test and verify it passes**

```bash
.venv/bin/pytest g1_nav_demo/renderer/test_video_renderer.py -v
```

Expected: PASS for both `test_snapshot_returns_png_bytes` and `test_snapshot_unknown_camera_raises`.

- [ ] **Step 5: Commit**

```bash
git add g1_nav_demo/renderer/video_renderer.py g1_nav_demo/renderer/test_video_renderer.py
git commit -m "Add VideoRenderer.snapshot() for single-camera PNG capture"
```

---

## Task 2: Tabletop item manifest + scene loader

Adds a JSON manifest of tabletop items and a Python loader that produces a merged scene XML. Pure string-manipulation logic — no MuJoCo dependency in the loader tests.

**Files:**
- Create: `g1_nav_demo/scene/tabletop_items.json`
- Create: `g1_nav_demo/scene/tabletop_loader.py`
- Create: `g1_nav_demo/scene/test_tabletop_loader.py`
- Create: `g1_nav_demo/scene/__init__.py` (if not already present)

- [ ] **Step 1: Confirm `scene/__init__.py` exists; create if missing**

```bash
ls g1_nav_demo/scene/__init__.py 2>/dev/null || touch g1_nav_demo/scene/__init__.py
```

- [ ] **Step 2: Create the manifest file**

Create `g1_nav_demo/scene/tabletop_items.json` with this exact content:

```json
{
  "table_top_z": 0.75,
  "default_scenario": "mixed",
  "scenarios": {
    "mixed": [
      {"name": "flammable_box", "kind": "hazard_box",
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
      {"name": "explosives_box", "kind": "hazard_box",
       "texture": "image_0702.jpg", "pos_xy": [0.2, 0.1]},
      {"name": "infectious_box", "kind": "hazard_box",
       "texture": "image_1421.jpg", "pos_xy": [-0.2, -0.1]},
      {"name": "combustible_box", "kind": "hazard_box",
       "texture": "image_1515.jpg", "pos_xy": [0.0, 0.2]}
    ]
  }
}
```

- [ ] **Step 3: Write the failing tests**

Create `g1_nav_demo/scene/test_tabletop_loader.py`:

```python
from __future__ import annotations

import json
import os
import re
import tempfile

import pytest

from g1_nav_demo.scene.tabletop_loader import (
    build_merged_scene,
    is_hazard_item,
    load_manifest,
    load_scenario,
)

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
SCENE_DIR = os.path.join(REPO_ROOT, "g1_nav_demo", "scene")
DEFAULT_MANIFEST = os.path.join(SCENE_DIR, "tabletop_items.json")
ROOM_XML = os.path.join(SCENE_DIR, "g1_nav_room.xml")
HAZARD_DIR = os.path.join(REPO_ROOT, "selected_imgs_videos_demo", "Hazard_detection_selected")


def test_load_manifest_returns_scenarios():
    manifest = load_manifest(DEFAULT_MANIFEST)
    assert manifest["default_scenario"] == "mixed"
    assert set(manifest["scenarios"].keys()) == {"mixed", "all_clear", "high_hazard"}


def test_load_scenario_returns_default_when_none():
    items = load_scenario(DEFAULT_MANIFEST, scenario=None)
    assert any(item["name"] == "flammable_box" for item in items)


def test_load_scenario_returns_requested():
    items = load_scenario(DEFAULT_MANIFEST, scenario="all_clear")
    names = {item["name"] for item in items}
    assert names == {"mug", "book", "laptop"}


def test_load_scenario_unknown_raises():
    with pytest.raises(KeyError):
        load_scenario(DEFAULT_MANIFEST, scenario="does_not_exist")


def test_is_hazard_item_uses_kind():
    assert is_hazard_item({"kind": "hazard_box", "name": "x"}) is True
    assert is_hazard_item({"kind": "mug", "name": "x"}) is False


def test_build_merged_scene_adds_named_geoms(tmp_path):
    out_path = build_merged_scene(
        room_xml_path=ROOM_XML,
        manifest_path=DEFAULT_MANIFEST,
        scenario="mixed",
        hazard_textures_dir=HAZARD_DIR,
        out_dir=str(tmp_path),
    )
    text = open(out_path).read()
    for name in ["flammable_box", "radioactive_box", "mug", "book", "apple"]:
        assert re.search(rf'name="{name}"', text), f"missing geom {name} in merged XML"


def test_build_merged_scene_references_hazard_texture_paths(tmp_path):
    out_path = build_merged_scene(
        room_xml_path=ROOM_XML,
        manifest_path=DEFAULT_MANIFEST,
        scenario="high_hazard",
        hazard_textures_dir=HAZARD_DIR,
        out_dir=str(tmp_path),
    )
    text = open(out_path).read()
    for filename in ["image_0702.jpg", "image_1421.jpg", "image_1515.jpg"]:
        expected = os.path.join(HAZARD_DIR, filename)
        assert expected in text, f"missing texture path for {filename}"


def test_build_merged_scene_all_clear_has_no_hazard_textures(tmp_path):
    out_path = build_merged_scene(
        room_xml_path=ROOM_XML,
        manifest_path=DEFAULT_MANIFEST,
        scenario="all_clear",
        hazard_textures_dir=HAZARD_DIR,
        out_dir=str(tmp_path),
    )
    text = open(out_path).read()
    assert "image_" not in text
    assert re.search(r'name="laptop"', text)
```

- [ ] **Step 4: Run the tests and verify they fail**

```bash
.venv/bin/pytest g1_nav_demo/scene/test_tabletop_loader.py -v
```

Expected: collection error — `ModuleNotFoundError: No module named 'g1_nav_demo.scene.tabletop_loader'`.

- [ ] **Step 5: Implement the loader**

Create `g1_nav_demo/scene/tabletop_loader.py`:

```python
from __future__ import annotations

import json
import os
import tempfile
from typing import Iterable

NORMAL_KIND_GEOMS: dict[str, dict] = {
    "mug":    {"type": "cylinder", "size": "0.04 0.05",       "rgba": "0.95 0.95 0.95 1", "dz": 0.05},
    "book":   {"type": "box",      "size": "0.075 0.05 0.0125","rgba": "0.45 0.3 0.2 1",  "dz": 0.0125},
    "apple":  {"type": "sphere",   "size": "0.04",            "rgba": "0.85 0.1 0.1 1",  "dz": 0.04},
    "laptop": {"type": "box",      "size": "0.1 0.075 0.01",  "rgba": "0.2 0.2 0.2 1",   "dz": 0.01},
}

HAZARD_BOX_HALF_SIZE: float = 0.05  # geom is a cube of side 0.10 m
HAZARD_BOX_DZ: float = HAZARD_BOX_HALF_SIZE  # bottom face sits on table top


def load_manifest(manifest_path: str) -> dict:
    with open(manifest_path) as f:
        return json.load(f)


def load_scenario(manifest_path: str, scenario: str | None) -> list[dict]:
    manifest = load_manifest(manifest_path)
    name = scenario or manifest.get("default_scenario", "mixed")
    if name not in manifest["scenarios"]:
        raise KeyError(f"Unknown scenario {name!r}; have {list(manifest['scenarios'])}")
    return manifest["scenarios"][name]


def is_hazard_item(item: dict) -> bool:
    return item.get("kind") == "hazard_box"


def _build_hazard_box_xml(
    item: dict, idx: int, table_top_z: float, hazard_textures_dir: str
) -> tuple[str, str]:
    """Return (asset_snippet, geom_snippet) for a hazard_box item."""
    name = item["name"]
    tex_filename = item["texture"]
    tex_path = os.path.join(hazard_textures_dir, tex_filename)
    px, py = item["pos_xy"]
    tex_id = f"tex_{name}"
    mat_id = f"mat_{name}"

    asset = (
        f'    <texture name="{tex_id}" type="2d" file="{tex_path}"/>\n'
        f'    <material name="{mat_id}" texture="{tex_id}" texuniform="false"/>\n'
    )
    geom = (
        f'      <geom name="{name}" type="box" '
        f'size="{HAZARD_BOX_HALF_SIZE} {HAZARD_BOX_HALF_SIZE} {HAZARD_BOX_HALF_SIZE}" '
        f'pos="{px} {py} {table_top_z + HAZARD_BOX_DZ + 0.005}" '
        f'material="{mat_id}" contype="0" conaffinity="0"/>\n'
    )
    return asset, geom


def _build_normal_geom_xml(item: dict, table_top_z: float) -> str:
    kind = item["kind"]
    if kind not in NORMAL_KIND_GEOMS:
        raise KeyError(f"Unknown item kind {kind!r}")
    spec = NORMAL_KIND_GEOMS[kind]
    name = item["name"]
    px, py = item["pos_xy"]
    pz = table_top_z + spec["dz"] + 0.005  # 5 mm clearance above table top
    return (
        f'      <geom name="{name}" type="{spec["type"]}" size="{spec["size"]}" '
        f'pos="{px} {py} {pz}" rgba="{spec["rgba"]}" '
        f'contype="0" conaffinity="0"/>\n'
    )


def build_merged_scene(
    room_xml_path: str,
    manifest_path: str,
    scenario: str | None,
    hazard_textures_dir: str,
    out_dir: str | None = None,
) -> str:
    """Read room XML, splice in tabletop items per scenario, write to a temp file.

    Returns the path of the merged XML file.
    """
    items = load_scenario(manifest_path, scenario)
    manifest = load_manifest(manifest_path)
    table_top_z = float(manifest["table_top_z"])

    with open(room_xml_path) as f:
        original_xml = f.read()

    asset_snippets: list[str] = []
    geom_snippets: list[str] = []
    for idx, item in enumerate(items):
        if is_hazard_item(item):
            asset, geom = _build_hazard_box_xml(item, idx, table_top_z, hazard_textures_dir)
            asset_snippets.append(asset)
            geom_snippets.append(geom)
        else:
            geom_snippets.append(_build_normal_geom_xml(item, table_top_z))

    asset_block = "".join(asset_snippets)
    geom_block = "".join(geom_snippets)

    # Splice asset entries into the <asset>...</asset> block.
    if asset_block:
        if "</asset>" not in original_xml:
            raise RuntimeError("Could not find </asset> in room XML")
        merged_xml = original_xml.replace("</asset>", asset_block + "  </asset>", 1)
    else:
        merged_xml = original_xml

    # Splice geom entries into the <body name="table" ...>...</body> block.
    # The original XML has the table body's closing </body> after the last leg geom.
    # Strategy: locate '<body name="table"' then find the first '</body>' after it.
    table_open = merged_xml.find('<body name="table"')
    if table_open < 0:
        raise RuntimeError('Could not find <body name="table" ...> in room XML')
    table_close = merged_xml.find("</body>", table_open)
    if table_close < 0:
        raise RuntimeError("Could not find table body closing tag")
    merged_xml = (
        merged_xml[:table_close] + geom_block + "    " + merged_xml[table_close:]
    )

    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix="g1_nav_merged_")
    else:
        os.makedirs(out_dir, exist_ok=True)
    # Write next to the original so relative <include file="g1_29dof.xml"/> still resolves.
    out_path = os.path.join(out_dir, "g1_nav_room_merged.xml")
    # If out_dir is not the original scene dir, the include path won't work; in that case
    # write into the original scene dir with a unique name.
    scene_dir = os.path.dirname(os.path.abspath(room_xml_path))
    if os.path.abspath(out_dir) != scene_dir:
        out_path = tempfile.mktemp(prefix="g1_nav_room_merged_", suffix=".xml", dir=scene_dir)
    with open(out_path, "w") as f:
        f.write(merged_xml)
    return out_path
```

- [ ] **Step 6: Run the tests and verify they pass**

```bash
.venv/bin/pytest g1_nav_demo/scene/test_tabletop_loader.py -v
```

Expected: PASS for all 7 tests.

- [ ] **Step 7: Commit**

```bash
git add g1_nav_demo/scene/__init__.py g1_nav_demo/scene/tabletop_items.json \
        g1_nav_demo/scene/tabletop_loader.py g1_nav_demo/scene/test_tabletop_loader.py
git commit -m "Add tabletop item manifest and scene loader for hazard scenarios"
```

---

## Task 3: Wire loader into `_init_simulation` + CLI flags

Hooks `build_merged_scene` into the simulation entry point so the demo loads the requested scenario. Adds CLI flags. Includes a smoke test that the merged scene actually loads in MuJoCo.

**Files:**
- Modify: `g1_nav_demo/run_demo.py:362-422,479-525`
- Test: `g1_nav_demo/scene/test_tabletop_loader.py` (add one integration test)

- [ ] **Step 1: Add the MuJoCo smoke test**

Append to `g1_nav_demo/scene/test_tabletop_loader.py`:

```python
import os as _os
_os.environ.setdefault("MUJOCO_GL", "egl")


def test_merged_scene_loads_in_mujoco(tmp_path):
    import mujoco

    out_path = build_merged_scene(
        room_xml_path=ROOM_XML,
        manifest_path=DEFAULT_MANIFEST,
        scenario="mixed",
        hazard_textures_dir=HAZARD_DIR,
    )
    try:
        model = mujoco.MjModel.from_xml_path(out_path)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "flammable_box") >= 0
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "mug") >= 0
    finally:
        _os.remove(out_path)
```

- [ ] **Step 2: Run the new test and verify it fails for the right reason**

```bash
.venv/bin/pytest g1_nav_demo/scene/test_tabletop_loader.py::test_merged_scene_loads_in_mujoco -v
```

Expected: passes IF the loader handles include-path correctly. If it fails on `<include file="g1_29dof.xml"/>` not resolving, treat the loader's "write to scene_dir" branch as the fix.

If the test passes already (because Step 5 of Task 2 wrote into the scene dir), record this in the next step and move on.

- [ ] **Step 3: Add CLI flags to `run_demo.py`**

In `g1_nav_demo/run_demo.py`, locate the `parser.add_argument("--output-dir", ...)` line in `main()` and add immediately after it:

```python
    parser.add_argument("--tabletop-scenario", type=str, default=None,
                        help="Tabletop scenario name from tabletop_items.json")
    parser.add_argument("--hazard-textures-dir", type=str, default=None,
                        help="Directory containing hazard placard images")
    parser.add_argument("--tabletop-manifest", type=str, default=None,
                        help="Path to tabletop_items.json (default: scene/tabletop_items.json)")
```

- [ ] **Step 4: Default the new flags and merge the scene in `_init_simulation`**

In `g1_nav_demo/run_demo.py`, replace the existing `_init_simulation` function's first lines (currently `model = mujoco.MjModel.from_xml_path(args.scene_xml)`) with this expanded block:

```python
def _init_simulation(args) -> tuple:
    from g1_nav_demo.scene.tabletop_loader import build_merged_scene

    scene_dir = os.path.dirname(os.path.abspath(args.scene_xml))
    repo_root = os.path.abspath(os.path.join(scene_dir, "..", ".."))

    manifest_path = args.tabletop_manifest or os.path.join(scene_dir, "tabletop_items.json")
    hazard_dir = args.hazard_textures_dir or os.path.join(
        repo_root, "selected_imgs_videos_demo", "Hazard_detection_selected"
    )
    merged_xml = build_merged_scene(
        room_xml_path=args.scene_xml,
        manifest_path=manifest_path,
        scenario=args.tabletop_scenario,
        hazard_textures_dir=hazard_dir,
    )
    model = mujoco.MjModel.from_xml_path(merged_xml)
    model.opt.timestep = 1.0 / args.sim_fps
    data = mujoco.MjData(model)
```

(Leave the rest of `_init_simulation` unchanged — the rest of the function uses `model` and `data` exactly as before.)

- [ ] **Step 5: Smoke-test the demo loads with the new scenario**

```bash
.venv/bin/python -c "
import os, sys
os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, '.')
import argparse
from g1_nav_demo.run_demo import _init_simulation
args = argparse.Namespace(
    scene_xml='g1_nav_demo/scene/g1_nav_room.xml',
    policy_path='g1_nav_demo/walk_policy/motion.pt',
    vlm_model='anthropic/claude-sonnet-latest',
    sim_fps=500, render_fps=30, max_steps=100, device='cpu',
    tabletop_scenario='mixed', hazard_textures_dir=None, tabletop_manifest=None,
)
session = _init_simulation(args)
print('OK scene loaded; mug geom id:',
      __import__('mujoco').mj_name2id(session.model, __import__('mujoco').mjtObj.mjOBJ_GEOM, 'mug'))
"
```

Expected: `OK scene loaded; mug geom id: <positive int>`. (Ignore any stderr from torch/MuJoCo init noise.)

- [ ] **Step 6: Commit**

```bash
git add g1_nav_demo/run_demo.py g1_nav_demo/scene/test_tabletop_loader.py
git commit -m "Wire tabletop scenario loader into demo init + CLI flags"
```

---

## Task 4: Add `head_onboard` camera + tuning script

Adds a new camera attached to the torso (the only body containing the head geom). Adds a small standalone script to render a single frame from it so the operator can eyeball the framing and tune offsets if needed.

**Files:**
- Modify: `g1_nav_demo/scene/g1_29dof.xml` (add camera inside the torso body)
- Create: `g1_nav_demo/scripts/render_head_cam.py`

- [ ] **Step 1: Find the exact torso body opening line in `g1_29dof.xml`**

```bash
grep -n 'name="torso_link"' g1_nav_demo/scene/g1_29dof.xml
```

Expected: a line like `<body name="torso_link" pos="..." quat="..."> `. Note the line number for context.

- [ ] **Step 2: Insert the new camera just after the torso body's opening tag**

Edit `g1_nav_demo/scene/g1_29dof.xml`. Immediately after the opening `<body name="torso_link" ...>` tag (and BEFORE the first child element inside), insert this line:

```xml
            <camera name="head_onboard" pos="0.10 0 0.45" xyaxes="0 -1 0 0.34 0 -0.94" fovy="70"/>
```

The `xyaxes` value `0 -1 0  0.34 0 -0.94` makes the camera look forward (along +X) and tilts it ≈20° downward.

- [ ] **Step 3: Create the tuning script**

Create `g1_nav_demo/scripts/__init__.py`:

```bash
mkdir -p g1_nav_demo/scripts
touch g1_nav_demo/scripts/__init__.py
```

Create `g1_nav_demo/scripts/render_head_cam.py`:

```python
#!/usr/bin/env python3
"""Render a single frame from the on-board head camera and save it as PNG.

Used to eyeball the head_onboard camera position/orientation. Place the
robot in front of the table by reusing the navigation init code, then
render and save.
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import mujoco
import numpy as np

from g1_nav_demo.run_demo import (
    LEG_JOINT_NAMES,
    UPPER_ACTUATOR_DEFAULTS,
    UPPER_JOINT_NAMES,
)
from g1_nav_demo.scene.tabletop_loader import build_merged_scene
from g1_nav_demo.walk_policy.g1_walk_policy import G1WalkPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-xml", default="g1_nav_demo/scene/g1_nav_room.xml")
    parser.add_argument("--scenario", default="mixed")
    parser.add_argument("--robot-pos", type=float, nargs=2, default=[3.0, 1.0],
                        help="Robot XY in front of the table")
    parser.add_argument("--robot-yaw-deg", type=float, default=90.0,
                        help="Robot heading (90 = facing +Y, i.e. toward the table)")
    parser.add_argument("--out", default="head_cam_preview.png")
    args = parser.parse_args()

    scene_dir = os.path.dirname(os.path.abspath(args.scene_xml))
    repo_root = os.path.abspath(os.path.join(scene_dir, "..", ".."))
    manifest = os.path.join(scene_dir, "tabletop_items.json")
    hazard_dir = os.path.join(repo_root, "selected_imgs_videos_demo", "Hazard_detection_selected")

    merged = build_merged_scene(
        room_xml_path=args.scene_xml,
        manifest_path=manifest,
        scenario=args.scenario,
        hazard_textures_dir=hazard_dir,
    )
    model = mujoco.MjModel.from_xml_path(merged)
    data = mujoco.MjData(model)

    # Place the robot in front of the table, facing toward it.
    data.qpos[0] = args.robot_pos[0]
    data.qpos[1] = args.robot_pos[1]
    data.qpos[2] = 0.793

    import math
    yaw = math.radians(args.robot_yaw_deg)
    data.qpos[3] = math.cos(yaw / 2.0)  # w
    data.qpos[4] = 0.0
    data.qpos[5] = 0.0
    data.qpos[6] = math.sin(yaw / 2.0)  # z

    default_angles = G1WalkPolicy.DEFAULT_ANGLES
    for i, joint_name in enumerate(LEG_JOINT_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        data.qpos[model.jnt_qposadr[jid]] = default_angles[i]
    for joint_name in UPPER_JOINT_NAMES:
        act_name = joint_name.replace("_joint", "")
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        data.qpos[model.jnt_qposadr[jid]] = UPPER_ACTUATOR_DEFAULTS[act_name][0]

    mujoco.mj_forward(model, data)

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_onboard")
    if cam_id < 0:
        raise SystemExit("Camera 'head_onboard' not found in scene XML")
    renderer = mujoco.Renderer(model, height=480, width=640)
    renderer.update_scene(data, camera=cam_id)
    rgb = renderer.render()
    renderer.close()

    from PIL import Image
    Image.fromarray(rgb.astype(np.uint8)).save(args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tuning script and inspect the output**

```bash
.venv/bin/python g1_nav_demo/scripts/render_head_cam.py --out head_cam_preview.png
```

Expected: `Saved head_cam_preview.png`. Open the image and verify the tabletop is visible, the hazard boxes are in frame, and the robot's head/torso geometry is not occluding the view.

If the framing is poor:
- Tabletop too low → increase `xyaxes` z-component magnitude (more downtilt).
- Tabletop cut off at edges → bump `fovy` from 70 to 90 in the XML camera line.
- Head geometry visible → increase the `pos` X-component (move camera further forward), e.g. from `0.10` to `0.15`.

Adjust the camera line in `g1_29dof.xml`, re-run, repeat until satisfied. Save the final values.

- [ ] **Step 5: Run all existing tests to confirm nothing broke**

```bash
.venv/bin/pytest g1_nav_demo/ -v
```

Expected: all pass (the XML addition is invisible to existing tests).

- [ ] **Step 6: Commit**

```bash
git add g1_nav_demo/scene/g1_29dof.xml g1_nav_demo/scripts/__init__.py \
        g1_nav_demo/scripts/render_head_cam.py
git commit -m "Add head_onboard camera and tuning script for inspection view"
```

---

## Task 5: `VideoRenderer.hazard_banner` overlay

Adds a top-banner overlay drawn when `hazard_banner` is set. Style matches the existing OpenCV-based overlays (consistent with `_overlay_text`).

**Files:**
- Modify: `g1_nav_demo/renderer/video_renderer.py`
- Test: `g1_nav_demo/renderer/test_video_renderer.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `g1_nav_demo/renderer/test_video_renderer.py`:

```python
import numpy as np


def test_hazard_banner_draws_red_top_strip(model_and_data, tmp_path):
    model, data = model_and_data
    renderer = VideoRenderer(model, output_path=str(tmp_path / "ignored.mp4"))
    try:
        renderer.hazard_banner = "HAZARD DETECTED: flammable_box"
        torso_id = __import__("mujoco").mj_name2id(
            model, __import__("mujoco").mjtObj.mjOBJ_BODY, "torso_link"
        )
        frame = renderer.render_frame(
            data, command="inspect the table", distance=0.0,
            update_head_camera=True, head_body_id=torso_id,
            goal_waypoints=None, current_wp_idx=0,
        )
    finally:
        renderer.close()

    # Top strip should be dominated by red pixels.
    top = frame[:60, :, :]
    mean_color = top.reshape(-1, 3).mean(axis=0)
    assert mean_color[0] > 120, f"red channel too low: {mean_color}"
    assert mean_color[0] > mean_color[1] * 1.5
    assert mean_color[0] > mean_color[2] * 1.5


def test_no_banner_when_attribute_none(model_and_data, tmp_path):
    model, data = model_and_data
    renderer = VideoRenderer(model, output_path=str(tmp_path / "ignored.mp4"))
    try:
        assert renderer.hazard_banner is None
        torso_id = __import__("mujoco").mj_name2id(
            model, __import__("mujoco").mjtObj.mjOBJ_BODY, "torso_link"
        )
        frame = renderer.render_frame(
            data, command="go to the table", distance=0.0,
            update_head_camera=True, head_body_id=torso_id,
        )
    finally:
        renderer.close()

    top = frame[:60, :, :]
    mean_color = top.reshape(-1, 3).mean(axis=0)
    # Without a banner, red should NOT dominate green and blue at the top.
    assert not (mean_color[0] > mean_color[1] * 1.5 and mean_color[0] > mean_color[2] * 1.5)
```

- [ ] **Step 2: Run the tests and verify they fail**

```bash
.venv/bin/pytest g1_nav_demo/renderer/test_video_renderer.py::test_hazard_banner_draws_red_top_strip -v
```

Expected: FAIL — `AttributeError: 'VideoRenderer' object has no attribute 'hazard_banner'`.

- [ ] **Step 3: Add `hazard_banner` attribute and overlay logic**

Edit `g1_nav_demo/renderer/video_renderer.py`. In `__init__`, after `self._closed = False`, add:

```python
        self.hazard_banner: str | None = None
```

Add this method to the `VideoRenderer` class, placed near the other overlay helpers (after `_overlay_label`):

```python
    def _overlay_hazard_banner(self, frame: np.ndarray, text: str) -> np.ndarray:
        import cv2

        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        h, w, _ = bgr.shape
        cv2.rectangle(bgr, (0, 0), (w, 60), (0, 0, 220), -1)  # BGR red
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.9
        thick = 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        x = max(10, (w - tw) // 2)
        y = 40
        cv2.putText(bgr, text, (x, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
```

In `render_frame`, just before the final `return combined`, insert:

```python
        if self.hazard_banner:
            combined = self._overlay_hazard_banner(combined, self.hazard_banner)
```

- [ ] **Step 4: Run the tests and verify they pass**

```bash
.venv/bin/pytest g1_nav_demo/renderer/test_video_renderer.py -v
```

Expected: PASS for all renderer tests.

- [ ] **Step 5: Commit**

```bash
git add g1_nav_demo/renderer/video_renderer.py g1_nav_demo/renderer/test_video_renderer.py
git commit -m "Add hazard_banner red overlay to VideoRenderer"
```

---

## Task 6: `Goal.inspect` field + parser prompt rule

Extends the existing goal parser to surface inspection intent. The VLM itself decides; no regex on the user's text.

**Files:**
- Modify: `g1_nav_demo/vlm/goal_parser.py`
- Modify: `g1_nav_demo/vlm/test_goal_parser.py`

- [ ] **Step 1: Add the failing tests**

Append to `g1_nav_demo/vlm/test_goal_parser.py`:

```python
def test_extract_inspect_true():
    goal = _parse('{"target_name": "table", "waypoints": [[1.0, 0.5]], "inspect": true}')
    assert goal is not None
    assert goal.inspect is True


def test_extract_inspect_default_false():
    goal = _parse('{"target_name": "table", "waypoints": [[1.0, 0.5]]}')
    assert goal is not None
    assert goal.inspect is False


def test_extract_inspect_explicit_false():
    goal = _parse(
        '{"target_name": "table", "waypoints": [[1.0, 0.5]], "inspect": false}'
    )
    assert goal is not None
    assert goal.inspect is False
```

- [ ] **Step 2: Run the new tests and verify they fail**

```bash
.venv/bin/pytest g1_nav_demo/vlm/test_goal_parser.py::test_extract_inspect_true -v
```

Expected: FAIL — `AttributeError: 'Goal' object has no attribute 'inspect'`.

- [ ] **Step 3: Add `inspect` to `Goal` and the parser**

In `g1_nav_demo/vlm/goal_parser.py`, edit the `Goal` dataclass:

```python
@dataclass
class Goal:
    target_name: str
    waypoints: list[tuple[float, float]] = field(default_factory=list)
    face_direction: str | None = None
    inspect: bool = False

    @property
    def x(self) -> float:
        return self.waypoints[-1][0] if self.waypoints else 0.0

    @property
    def y(self) -> float:
        return self.waypoints[-1][1] if self.waypoints else 0.0
```

In `_extract_goal`, change the regex to allow nested brackets (the current regex `[^}]*` already works — but extend the JSON read to pull `inspect`). Replace the body of the for-loop with:

```python
        for m in re.finditer(r'\{"target_name"[^}]*\}', text, re.DOTALL):
            try:
                obj = json.loads(m.group())
                name = obj.get("target_name")
                wps = obj.get("waypoints")
                face = obj.get("face_direction")
                inspect = obj.get("inspect", False)
                if isinstance(name, str) and isinstance(wps, list) and len(wps) > 0:
                    waypoints = []
                    for wp in wps:
                        if isinstance(wp, (list, tuple)) and len(wp) == 2:
                            waypoints.append((float(wp[0]), float(wp[1])))
                    if waypoints:
                        return Goal(
                            target_name=name,
                            waypoints=waypoints,
                            face_direction=face if isinstance(face, str) else None,
                            inspect=bool(inspect),
                        )
            except (json.JSONDecodeError, ValueError):
                continue
        return None
```

- [ ] **Step 4: Update `SCENE_PROMPT` to teach the VLM the new field**

Still in `g1_nav_demo/vlm/goal_parser.py`, find the `SCENE_PROMPT` string. Just before the final `Output ONLY this JSON ...` line, add this rule:

```
  8. INSPECTION INTENT: Set "inspect" to true ONLY if the command
     explicitly asks to inspect, check, examine, scan, or look at the
     target's contents. "Go to the table" → false.
     "Inspect the table" → true. "Go to the table and check it" → true.
```

And replace the example output line:

```
Output ONLY this JSON on one line. No markdown. No text before or after.
{"target_name": "<name>", "waypoints": [[x1,y1], ..., [xN,yN]], "face_direction": "<front|back|left|right>", "inspect": <true|false>}
```

- [ ] **Step 5: Run all parser tests and verify they pass**

```bash
.venv/bin/pytest g1_nav_demo/vlm/test_goal_parser.py -v
```

Expected: PASS for all parser tests (existing + 3 new).

- [ ] **Step 6: Commit**

```bash
git add g1_nav_demo/vlm/goal_parser.py g1_nav_demo/vlm/test_goal_parser.py
git commit -m "Add inspect intent field to Goal and goal parser"
```

---

## Task 7: `InspectionBridge` and `InspectionResult`

Adds the second VLM client used for hazard inspection. Mocks the OpenAI client in tests; no network.

**Files:**
- Create: `g1_nav_demo/vlm/inspection.py`
- Create: `g1_nav_demo/vlm/test_inspection.py`

- [ ] **Step 1: Write the failing tests**

Create `g1_nav_demo/vlm/test_inspection.py`:

```python
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from g1_nav_demo.vlm.inspection import (
    InspectionBridge,
    InspectionItem,
    InspectionResult,
)


def _bridge():
    return InspectionBridge(model_name="any/model")


def test_extract_simple_clear():
    text = '{"items": [{"name": "mug", "hazardous": false, "reason": "ceramic cup"}], "alarm": false}'
    result = _bridge()._extract_result(text)
    assert result is not None
    assert result.alarm is False
    assert len(result.items) == 1
    assert result.items[0].name == "mug"


def test_extract_simple_alarm():
    text = (
        '{"items": ['
        ' {"name": "knife", "hazardous": true, "reason": "sharp blade"},'
        ' {"name": "mug", "hazardous": false, "reason": "ceramic"}'
        '], "alarm": true}'
    )
    result = _bridge()._extract_result(text)
    assert result is not None
    assert result.alarm is True
    assert {i.name for i in result.items if i.hazardous} == {"knife"}


def test_extract_overrides_alarm_when_inconsistent():
    """VLM said alarm:false but listed a hazardous item -> must be alarm:true."""
    text = (
        '{"items": [{"name": "flammable_box", "hazardous": true, "reason": "placard"}],'
        ' "alarm": false}'
    )
    result = _bridge()._extract_result(text)
    assert result is not None
    assert result.alarm is True


def test_extract_strips_markdown_fence():
    text = '```json\n{"items": [], "alarm": false}\n```'
    result = _bridge()._extract_result(text)
    assert result is not None
    assert result.items == []
    assert result.alarm is False


def test_extract_garbage_returns_none():
    assert _bridge()._extract_result("I cannot tell what is on the table.") is None


def test_extract_missing_items_returns_none():
    assert _bridge()._extract_result('{"alarm": true}') is None


def test_inspect_returns_none_on_api_failure():
    bridge = InspectionBridge(model_name="any/model", api_base="http://localhost:19999/v1")
    result = bridge.inspect(b"\x89PNG\r\n", "table")
    assert result is None


def test_inspect_calls_openai_with_image(monkeypatch):
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content='{"items": [], "alarm": false}'))]
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    bridge = InspectionBridge(model_name="any/model")
    monkeypatch.setattr(bridge, "_get_client", lambda: fake_client)

    result = bridge.inspect(b"\x89PNG\r\nfake-image-bytes", "table")
    assert result is not None
    assert result.alarm is False

    call_kwargs = fake_client.chat.completions.create.call_args.kwargs
    user_content = call_kwargs["messages"][1]["content"]
    image_entry = next(c for c in user_content if c["type"] == "image_url")
    assert image_entry["image_url"]["url"].startswith("data:image/png;base64,")
```

- [ ] **Step 2: Run the tests and verify collection fails**

```bash
.venv/bin/pytest g1_nav_demo/vlm/test_inspection.py -v
```

Expected: collection error — `ModuleNotFoundError: No module named 'g1_nav_demo.vlm.inspection'`.

- [ ] **Step 3: Implement `inspection.py`**

Create `g1_nav_demo/vlm/inspection.py`:

```python
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
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")
except ImportError:
    pass

logger = logging.getLogger(__name__)


@dataclass
class InspectionItem:
    name: str
    hazardous: bool
    reason: str = ""


@dataclass
class InspectionResult:
    items: list[InspectionItem] = field(default_factory=list)
    alarm: bool = False
    raw_response: str = ""


INSPECTION_PROMPT = """\
You are a safety inspector. The image is from a humanoid robot's head camera,
looking down at a tabletop. List every distinct item visible and decide whether
it is hazardous. An item is hazardous if it bears a UN-style hazard placard
(Class 1-9, e.g. EXPLOSIVES, FLAMMABLE LIQUID, FLAMMABLE SOLID, SPONTANEOUSLY
COMBUSTIBLE, RADIOACTIVE, INFECTIOUS SUBSTANCE) or is itself a weapon, open
flame, exposed chemical, or sharp instrument.

Output ONLY this JSON on one line. No markdown. No text before or after.
{"items": [{"name": "...", "hazardous": true|false, "reason": "..."}], "alarm": true|false}

alarm MUST be true iff any item.hazardous is true.
"""


_OPENROUTER_BASE = "https://openrouter.ai/api/v1"


class InspectionBridge:
    def __init__(
        self,
        model_name: str = "anthropic/claude-sonnet-latest",
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

    def inspect(self, head_cam_png: bytes, target_name: str) -> Optional[InspectionResult]:
        img_b64 = base64.b64encode(head_cam_png).decode()
        user_text = f"Inspect what is on the {target_name} for any hazardous items."

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": INSPECTION_PROMPT},
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
                max_tokens=800,
                temperature=0.1,
            )
            text = response.choices[0].message.content or ""
        except Exception as e:
            logger.error("Inspection VLM call failed: %s", e)
            return None

        result = self._extract_result(text)
        if result is None:
            logger.warning("Inspection response could not be parsed:\n%s", text)
        return result

    def _extract_result(self, text: str) -> Optional[InspectionResult]:
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
        match = re.search(r'\{[^{}]*"items"[^{}]*\[.*?\][^{}]*\}', text, re.DOTALL)
        if not match:
            return None
        try:
            obj = json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            return None

        items_raw = obj.get("items")
        if not isinstance(items_raw, list):
            return None

        items: list[InspectionItem] = []
        for entry in items_raw:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            haz = entry.get("hazardous")
            if not isinstance(name, str) or not isinstance(haz, bool):
                continue
            reason = entry.get("reason") if isinstance(entry.get("reason"), str) else ""
            items.append(InspectionItem(name=name, hazardous=haz, reason=reason))

        alarm_raw = obj.get("alarm", False)
        alarm = bool(alarm_raw)
        any_hazard = any(i.hazardous for i in items)
        if any_hazard and not alarm:
            logger.warning(
                "VLM marked items hazardous but set alarm=false; overriding to true."
            )
            alarm = True

        return InspectionResult(items=items, alarm=alarm, raw_response=text)
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
.venv/bin/pytest g1_nav_demo/vlm/test_inspection.py -v
```

Expected: PASS for all 8 tests.

- [ ] **Step 5: Commit**

```bash
git add g1_nav_demo/vlm/inspection.py g1_nav_demo/vlm/test_inspection.py
git commit -m "Add InspectionBridge with structured per-item hazard output"
```

---

## Task 8: `NavigationSession.inspect_target` + JSON output

Adds the post-arrival inspection hook. Writes the report JSON, sets the banner, and idles long enough for the banner to be recorded.

**Files:**
- Modify: `g1_nav_demo/run_demo.py`
- Test: `g1_nav_demo/test_run_demo.py` (NEW)

- [ ] **Step 1: Write failing tests for `inspect_target`**

Create `g1_nav_demo/test_run_demo.py`:

```python
from __future__ import annotations

import json
import os
from unittest.mock import MagicMock

os.environ.setdefault("MUJOCO_GL", "egl")

from g1_nav_demo.run_demo import INSPECTABLE_TARGETS
from g1_nav_demo.vlm.goal_parser import Goal
from g1_nav_demo.vlm.inspection import InspectionItem, InspectionResult


def _stub_session():
    """Build a NavigationSession-like object without MuJoCo init.

    We test inspect_target() in isolation by stubbing model/data/walk_policy.
    """
    from g1_nav_demo.run_demo import NavigationSession

    session = NavigationSession.__new__(NavigationSession)
    session.inspection_bridge = MagicMock()
    session.idle = MagicMock()
    return session


def test_inspect_target_skips_when_not_requested(tmp_path):
    session = _stub_session()
    goal = Goal(target_name="table", waypoints=[(3.0, 1.0)], inspect=False)
    video_renderer = MagicMock()
    result = session.inspect_target(
        goal=goal, command="go to the table", video_renderer=video_renderer,
        inspection_json_path=str(tmp_path / "out.json"),
    )
    assert result is None
    session.inspection_bridge.inspect.assert_not_called()


def test_inspect_target_skips_when_target_not_inspectable(tmp_path):
    session = _stub_session()
    goal = Goal(target_name="door", waypoints=[(4.2, 0.0)], inspect=True)
    video_renderer = MagicMock()
    result = session.inspect_target(
        goal=goal, command="inspect the door", video_renderer=video_renderer,
        inspection_json_path=str(tmp_path / "out.json"),
    )
    assert result is None
    assert "table" in INSPECTABLE_TARGETS


def test_inspect_target_writes_json_and_no_banner_when_clear(tmp_path):
    session = _stub_session()
    session.inspection_bridge.inspect.return_value = InspectionResult(
        items=[InspectionItem(name="mug", hazardous=False, reason="cup")],
        alarm=False, raw_response="{}",
    )
    goal = Goal(target_name="table", waypoints=[(3.0, 1.0)], inspect=True)
    video_renderer = MagicMock()
    video_renderer.snapshot.return_value = b"\x89PNG"
    video_renderer.hazard_banner = None
    out = str(tmp_path / "turn_001_inspection.json")

    result = session.inspect_target(
        goal=goal, command="inspect the table", video_renderer=video_renderer,
        inspection_json_path=out,
    )

    assert result is not None
    assert result.alarm is False
    assert os.path.exists(out)
    payload = json.loads(open(out).read())
    assert payload["alarm"] is False
    assert payload["items"][0]["name"] == "mug"
    assert video_renderer.hazard_banner is None


def test_inspect_target_sets_banner_when_alarm(tmp_path):
    session = _stub_session()
    session.inspection_bridge.inspect.return_value = InspectionResult(
        items=[
            InspectionItem(name="flammable_box", hazardous=True, reason="placard"),
            InspectionItem(name="mug", hazardous=False, reason="cup"),
        ],
        alarm=True, raw_response="{}",
    )
    goal = Goal(target_name="table", waypoints=[(3.0, 1.0)], inspect=True)

    # Track every value assigned to `hazard_banner` so we can check both the
    # "HAZARD DETECTED" set and the final clear-to-None happened in order.
    banner_log: list[str | None] = []

    class RecorderMock(MagicMock):
        def __setattr__(self, name, value):
            if name == "hazard_banner":
                banner_log.append(value)
            super().__setattr__(name, value)

    video_renderer = RecorderMock()
    video_renderer.snapshot.return_value = b"\x89PNG"
    out = str(tmp_path / "turn_002_inspection.json")

    result = session.inspect_target(
        goal=goal, command="inspect the table", video_renderer=video_renderer,
        inspection_json_path=out,
    )

    assert result is not None
    assert result.alarm is True
    assert any(isinstance(v, str) and v.startswith("HAZARD DETECTED") for v in banner_log), \
        f"Expected HAZARD DETECTED banner to be set, log was {banner_log}"
    assert banner_log[-1] is None, "Banner should be cleared at end of inspect_target"
    # idle called twice: 0.5 s settle + 3 s banner-record
    assert session.idle.call_count >= 2
```

- [ ] **Step 2: Run the new tests and verify the import fails**

```bash
.venv/bin/pytest g1_nav_demo/test_run_demo.py -v
```

Expected: collection error — `ImportError: cannot import name 'INSPECTABLE_TARGETS'` (or similar).

- [ ] **Step 3: Add `INSPECTABLE_TARGETS`, `inspect_target`, and JSON helper to `run_demo.py`**

In `g1_nav_demo/run_demo.py`:

(a) At the top of the file, add imports near the existing ones:

```python
import json as _json
from g1_nav_demo.vlm.inspection import InspectionBridge, InspectionResult
```

(b) Add a module-level constant just below the existing `UPPER_JOINT_NAMES` list:

```python
INSPECTABLE_TARGETS = {"table"}
```

(c) Update `NavigationSession.__init__` to accept an `inspection_bridge` parameter. Change its signature and body — replace the existing `def __init__(self, ..., torso_body_id: int = 0,)` and the corresponding body assignments to include:

```python
    def __init__(
        self,
        model,
        data,
        walk_policy: G1WalkPolicy,
        goal_planner: GoalPlanner,
        vlm_bridge: VLMBridge,
        sim_fps: int = 500,
        render_fps: int = 30,
        max_steps: int = 5000,
        torso_body_id: int = 0,
        inspection_bridge: InspectionBridge | None = None,
    ) -> None:
        self.model = model
        self.data = data
        self.walk_policy = walk_policy
        self.goal_planner = goal_planner
        self.vlm_bridge = vlm_bridge
        self.inspection_bridge = inspection_bridge
        self.sim_fps = sim_fps
        self.render_fps = render_fps
        self.max_steps = max_steps
        self.torso_body_id = torso_body_id

        self._build_index_mappings()

        self.kps = G1WalkPolicy.KPS
        self.kds = G1WalkPolicy.KDS
        self.default_angles = G1WalkPolicy.DEFAULT_ANGLES
        self.decimation = max(1, int(round(G1WalkPolicy.CONTROL_DT / model.opt.timestep)))
```

(d) Add the `inspect_target` method to `NavigationSession`, placed after `idle`:

```python
    def inspect_target(
        self,
        goal: Goal,
        command: str,
        video_renderer,
        inspection_json_path: str,
    ) -> InspectionResult | None:
        if not goal.inspect:
            return None
        if goal.target_name not in INSPECTABLE_TARGETS:
            logger.info("Inspection requested for %s but not in INSPECTABLE_TARGETS",
                        goal.target_name)
            return None
        if self.inspection_bridge is None:
            logger.warning("Inspection requested but no InspectionBridge configured")
            return None

        self.idle(duration_steps=250)
        head_png = video_renderer.snapshot("head_onboard", self.data)
        result = self.inspection_bridge.inspect(head_png, goal.target_name)
        if result is None:
            logger.error("Inspection VLM returned no parseable result")
            return None

        _write_inspection_json(inspection_json_path, command, goal.target_name, result)

        if result.alarm:
            hazard_names = ", ".join(i.name for i in result.items if i.hazardous)
            logger.warning("HAZARD DETECTED at %s: %s", goal.target_name, hazard_names)
            video_renderer.hazard_banner = f"HAZARD DETECTED: {hazard_names}"
        else:
            logger.info("Inspection clear at %s", goal.target_name)

        self.idle(duration_steps=1500)
        video_renderer.hazard_banner = None
        return result
```

(e) Add the JSON helper function at module level, near `quat_to_yaw`:

```python
def _write_inspection_json(
    path: str, command: str, target: str, result: "InspectionResult"
) -> None:
    payload = {
        "command": command,
        "target": target,
        "alarm": result.alarm,
        "items": [
            {"name": i.name, "hazardous": i.hazardous, "reason": i.reason}
            for i in result.items
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        _json.dump(payload, f, indent=2)
```

(f) Update `_init_simulation` to construct an `InspectionBridge` and pass it. Inside `_init_simulation`, locate `vlm_bridge = VLMBridge(model_name=args.vlm_model)` and immediately after it add:

```python
    inspection_bridge = InspectionBridge(model_name=args.vlm_model)
```

Then update the `session = NavigationSession(...)` call near the end of `_init_simulation` to include `inspection_bridge=inspection_bridge` as an additional keyword argument.

- [ ] **Step 4: Run the new tests and verify they pass**

```bash
.venv/bin/pytest g1_nav_demo/test_run_demo.py -v
```

Expected: PASS for all four tests.

- [ ] **Step 5: Run the full suite to confirm nothing else broke**

```bash
.venv/bin/pytest g1_nav_demo/ -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add g1_nav_demo/run_demo.py g1_nav_demo/test_run_demo.py
git commit -m "Add NavigationSession.inspect_target hook with JSON report and banner"
```

---

## Task 9: Wire `inspect_target` into single-turn and multi-turn loops

Triggers `inspect_target` after `run_to_goal` returns `reached=True` in both modes, with the correct JSON output paths.

**Files:**
- Modify: `g1_nav_demo/run_demo.py` (`_run_single_turn`, `_run_multiturn`)

- [ ] **Step 1: Update `_run_single_turn`**

Replace the body of `_run_single_turn` in `g1_nav_demo/run_demo.py` with:

```python
def _run_single_turn(args, session: NavigationSession) -> None:
    debug_img = args.output.rsplit(".", 1)[0] + "_vlm_scene"
    goal = session.parse_goal(args.command, debug_prefix=debug_img)
    if goal is None:
        logger.error("Could not parse command: %s", args.command)
        sys.exit(1)

    video_renderer = VideoRenderer(
        session.model, output_path=args.output, fps=session.render_fps,
        width=1280, height=480,
    )
    try:
        reached = session.run_to_goal_with_renderer(goal, args.command, video_renderer)
        if not reached:
            logger.warning("Did not reach goal within %d steps", args.max_steps)
            return
        inspection_json = args.output.rsplit(".", 1)[0] + "_inspection.json"
        session.inspect_target(
            goal=goal, command=args.command,
            video_renderer=video_renderer,
            inspection_json_path=inspection_json,
        )
        # Render one final frame so any banner state and final pose make it into the video.
        frame = video_renderer.render_frame(
            session.data, command=args.command, distance=0.0,
            update_head_camera=True, head_body_id=session.torso_body_id,
            goal_waypoints=goal.waypoints,
            current_wp_idx=session.goal_planner._current_wp_idx,
        )
        video_renderer.write_frame(frame)
    finally:
        video_renderer.close()
```

- [ ] **Step 2: Refactor `NavigationSession.run_to_goal` to accept an existing renderer**

The existing `run_to_goal` constructs its own `VideoRenderer`. We need it to share the renderer across navigation + inspection so the banner appears in the same video file. In `g1_nav_demo/run_demo.py`, rename the current `run_to_goal` to `run_to_goal_with_renderer` and change its signature so it takes a renderer instead of a video path. Replace `run_to_goal` with this thin wrapper:

```python
    def run_to_goal(self, goal: Goal, command: str, video_path: str) -> bool:
        video_renderer = VideoRenderer(
            self.model, output_path=video_path, fps=self.render_fps,
            width=1280, height=480,
        )
        try:
            return self.run_to_goal_with_renderer(goal, command, video_renderer)
        finally:
            video_renderer.close()

    def run_to_goal_with_renderer(
        self, goal: Goal, command: str, video_renderer: "VideoRenderer"
    ) -> bool:
        face_yaw = self._compute_face_yaw(goal)
        self.goal_planner.set_waypoints(goal.waypoints, face_yaw=face_yaw)
        target_positions = self.default_angles.copy()
        reached = False
        plan_result = None

        steps_per_render = max(1, self.sim_fps // self.render_fps)

        for step in range(self.max_steps):
            if step % self.decimation == 0:
                current_pos = self.current_position()
                current_yaw = self.current_yaw()
                plan_result = self.goal_planner.compute_command(current_pos, current_yaw)
                if plan_result.reached:
                    logger.info(
                        "Reached goal at step %d (distance=%.3f)",
                        step, plan_result.distance,
                    )
                    reached = True
                    break

                velocity_command = np.array(
                    [plan_result.vx, plan_result.vy, plan_result.vyaw], dtype=np.float32
                )
                dof_pos = np.array(self.data.qpos[self.leg_qpos_adr], dtype=np.float32)
                dof_vel = np.array(self.data.qvel[self.leg_dof_adr], dtype=np.float32)
                angular_velocity = np.array(
                    [self.data.qvel[3], self.data.qvel[4], self.data.qvel[5]], dtype=np.float32,
                )
                quaternion = np.array(
                    [self.data.qpos[3], self.data.qpos[4], self.data.qpos[5], self.data.qpos[6]],
                    dtype=np.float32,
                )
                projected_gravity = G1WalkPolicy.compute_projected_gravity(quaternion)
                target_positions = self.walk_policy.get_action(
                    projected_gravity=projected_gravity,
                    velocity_command=velocity_command,
                    dof_pos=dof_pos, dof_vel=dof_vel,
                    angular_velocity=angular_velocity,
                )

            dof_pos = np.array(self.data.qpos[self.leg_qpos_adr], dtype=np.float32)
            dof_vel = np.array(self.data.qvel[self.leg_dof_adr], dtype=np.float32)
            torques = self.kps * (target_positions - dof_pos) - self.kds * dof_vel
            torques = np.clip(torques, self.ctrl_range[:, 0], self.ctrl_range[:, 1])
            self.data.ctrl[self.leg_actuator_ids] = torques

            upper_pos = np.array(self.data.qpos[self.upper_qpos_adr], dtype=np.float32)
            upper_vel = np.array(self.data.qvel[self.upper_dof_adr], dtype=np.float32)
            upper_torques = (
                self.upper_kps * (self.upper_default_pos - upper_pos) - self.upper_kds * upper_vel
            )
            upper_torques = np.clip(
                upper_torques, self.upper_ctrl_range[:, 0], self.upper_ctrl_range[:, 1]
            )
            self.data.ctrl[self.upper_actuator_ids] = upper_torques

            mujoco.mj_step(self.model, self.data)

            if step % steps_per_render == 0 and plan_result is not None:
                n_wps = len(goal.waypoints)
                wp_idx = self.goal_planner._current_wp_idx
                frame = video_renderer.render_frame(
                    self.data,
                    command=f"{command} [wp {min(wp_idx + 1, n_wps)}/{n_wps}]",
                    distance=plan_result.distance,
                    update_head_camera=True,
                    head_body_id=self.torso_body_id,
                    goal_waypoints=goal.waypoints,
                    current_wp_idx=wp_idx,
                )
                video_renderer.write_frame(frame)

        if not reached:
            logger.warning("Did not reach goal within %d steps", self.max_steps)
        return reached
```

Note: `idle()` must also write frames so the banner shows up in the video. Modify `idle()` so that it optionally accepts a `video_renderer` and writes frames at `render_fps`. Replace the current `idle()`:

```python
    def idle(
        self,
        duration_steps: int = 500,
        video_renderer: "VideoRenderer | None" = None,
        command: str = "",
        goal_waypoints: list | None = None,
    ) -> None:
        zero_cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        target_positions = self.default_angles.copy()
        steps_per_render = max(1, self.sim_fps // self.render_fps)

        for step in range(duration_steps):
            if step % self.decimation == 0:
                dof_pos = np.array(self.data.qpos[self.leg_qpos_adr], dtype=np.float32)
                dof_vel = np.array(self.data.qvel[self.leg_dof_adr], dtype=np.float32)
                angular_velocity = np.array(
                    [self.data.qvel[3], self.data.qvel[4], self.data.qvel[5]], dtype=np.float32,
                )
                quaternion = np.array(
                    [self.data.qpos[3], self.data.qpos[4], self.data.qpos[5], self.data.qpos[6]],
                    dtype=np.float32,
                )
                projected_gravity = G1WalkPolicy.compute_projected_gravity(quaternion)
                target_positions = self.walk_policy.get_action(
                    projected_gravity=projected_gravity,
                    velocity_command=zero_cmd,
                    dof_pos=dof_pos, dof_vel=dof_vel,
                    angular_velocity=angular_velocity,
                )

            dof_pos = np.array(self.data.qpos[self.leg_qpos_adr], dtype=np.float32)
            dof_vel = np.array(self.data.qvel[self.leg_dof_adr], dtype=np.float32)
            torques = self.kps * (target_positions - dof_pos) - self.kds * dof_vel
            torques = np.clip(torques, self.ctrl_range[:, 0], self.ctrl_range[:, 1])
            self.data.ctrl[self.leg_actuator_ids] = torques

            upper_pos = np.array(self.data.qpos[self.upper_qpos_adr], dtype=np.float32)
            upper_vel = np.array(self.data.qvel[self.upper_dof_adr], dtype=np.float32)
            upper_torques = (
                self.upper_kps * (self.upper_default_pos - upper_pos) - self.upper_kds * upper_vel
            )
            upper_torques = np.clip(
                upper_torques, self.upper_ctrl_range[:, 0], self.upper_ctrl_range[:, 1]
            )
            self.data.ctrl[self.upper_actuator_ids] = upper_torques

            mujoco.mj_step(self.model, self.data)

            if video_renderer is not None and step % steps_per_render == 0:
                frame = video_renderer.render_frame(
                    self.data, command=command, distance=0.0,
                    update_head_camera=True, head_body_id=self.torso_body_id,
                    goal_waypoints=goal_waypoints, current_wp_idx=0,
                )
                video_renderer.write_frame(frame)
```

Then update `inspect_target` (from Task 8) so it passes the renderer through to `idle`:

```python
        self.idle(duration_steps=250, video_renderer=video_renderer, command=command)
        head_png = video_renderer.snapshot("head_onboard", self.data)
        result = self.inspection_bridge.inspect(head_png, goal.target_name)
        ...
        self.idle(duration_steps=1500, video_renderer=video_renderer, command=command)
```

- [ ] **Step 3: Update `_run_multiturn` to call `inspect_target`**

Replace the `_run_multiturn` body with:

```python
def _run_multiturn(args, session: NavigationSession) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    turn = 0
    first_command = args.command

    while True:
        if turn == 0 and first_command:
            command = first_command
            print(f"[Turn 1] Command: {command}")
        else:
            try:
                command = input("Command (or 'quit'): ").strip()
            except EOFError:
                break
            if not command:
                continue
            if command.lower() in ("quit", "exit", "q"):
                break

        turn += 1
        debug_prefix = os.path.join(args.output_dir, f"turn_{turn:03d}")
        goal = session.parse_goal(command, debug_prefix=debug_prefix)
        if goal is None:
            print(f"  Could not parse command: {command}")
            turn -= 1
            continue

        video_path = os.path.join(
            args.output_dir, f"turn_{turn:03d}_{goal.target_name}.mp4"
        )
        video_renderer = VideoRenderer(
            session.model, output_path=video_path, fps=session.render_fps,
            width=1280, height=480,
        )
        try:
            reached = session.run_to_goal_with_renderer(goal, command, video_renderer)
            if reached:
                inspection_json = os.path.join(
                    args.output_dir, f"turn_{turn:03d}_inspection.json"
                )
                session.inspect_target(
                    goal=goal, command=command,
                    video_renderer=video_renderer,
                    inspection_json_path=inspection_json,
                )
                # Brief settling idle after inspection (or if no inspection happened)
                if not goal.inspect:
                    session.idle(duration_steps=500, video_renderer=video_renderer, command=command)
                pos = session.current_position()
                print(f"  Reached {goal.target_name}! Position: ({pos[0]:.2f}, {pos[1]:.2f})")
            else:
                pos = session.current_position()
                print(f"  Failed to reach {goal.target_name}. Position: ({pos[0]:.2f}, {pos[1]:.2f})")
        finally:
            video_renderer.close()

    print(f"Session ended. {turn} turn(s) completed. Videos in {args.output_dir}/")
```

- [ ] **Step 4: Run all tests to confirm nothing broke**

```bash
.venv/bin/pytest g1_nav_demo/ -v
```

Expected: all pass. (The integration tests don't exercise this code path; unit tests in `test_run_demo.py` still pass because they directly call `inspect_target`.)

- [ ] **Step 5: Manual end-to-end smoke test**

With the VLM server (OpenRouter or local) reachable and a valid `OPENROUTER_API_KEY` set:

```bash
.venv/bin/python g1_nav_demo/run_demo.py \
    --command "inspect the table" \
    --policy-path g1_nav_demo/walk_policy/motion.pt \
    --output demo_inspect.mp4 \
    --tabletop-scenario mixed \
    --max-steps 5000
```

Expected outputs:
- `demo_inspect.mp4` plays with the robot walking to the table, a brief idle, then a red "HAZARD DETECTED: ..." banner visible for ~3 s.
- `demo_inspect_inspection.json` exists and contains `"alarm": true` with the hazard item names.
- Console contains a `HAZARD DETECTED at table: ...` warning line.

- [ ] **Step 6: Commit**

```bash
git add g1_nav_demo/run_demo.py
git commit -m "Wire inspect_target into single-turn and multi-turn loops with shared renderer"
```

---

## Task 10: Documentation

Updates HOWTO and README to describe inspection commands, scenarios, and the new CLI flags.

**Files:**
- Modify: `g1_nav_demo/HOWTO.md`
- Modify: `g1_nav_demo/README.md`

- [ ] **Step 1: Append to `g1_nav_demo/HOWTO.md`**

Add this section at the end of `g1_nav_demo/HOWTO.md`:

```markdown
---

## Hazard Inspection Mode

### What it does

If the command contains an inspection verb (`inspect`, `check`, `examine`,
`scan`, `look at`), the robot — after walking to the target — uses its
**on-board head camera** to take one photo of the tabletop, sends it to the
VLM as a safety inspection, and:

- Writes `<output_prefix>_inspection.json` with per-item findings.
- If any item is hazardous, overlays a red banner on the recorded video for
  ~3 seconds and prints `HAZARD DETECTED at table: <names>` to the console.

The VLM itself decides intent — no keyword regex on your command text.
"Go to the table" does NOT inspect; "Go to the table and check it" does.

### Scenarios

Three scenarios ship in `g1_nav_demo/scene/tabletop_items.json`:

| Scenario | Items |
|---|---|
| `mixed` (default) | 2 hazard placards (Flammable Liquid, Radioactive) + mug + book + apple |
| `all_clear`       | mug + book + laptop only |
| `high_hazard`     | Explosives + Infectious Substance + Spontaneously Combustible |

Select with `--tabletop-scenario`.

### Example

```bash
bash g1_nav_demo/run_vlm_demo.sh --multiturn demo_output/
# Then in the prompt:
#   go to the table         # walks, no inspection
#   inspect the table       # walks then inspects, banner if hazardous
#   quit
```

To start in `all_clear` mode:

```bash
.venv/bin/python g1_nav_demo/run_demo.py \
    --command "inspect the table" \
    --policy-path g1_nav_demo/walk_policy/motion.pt \
    --output demo_clear.mp4 \
    --tabletop-scenario all_clear
```

### Tuning the on-board camera

If the inspection view is poorly framed (head/torso in shot, table cut off),
run the standalone preview:

```bash
.venv/bin/python g1_nav_demo/scripts/render_head_cam.py --out preview.png
```

Then tweak the `<camera name="head_onboard" ...>` line in
`g1_nav_demo/scene/g1_29dof.xml`:

- Push camera further forward → increase pos X (e.g. `0.10` → `0.15`).
- More downward tilt → make the second triple in `xyaxes` more negative-Z
  (e.g. `0.34 0 -0.94` → `0.50 0 -0.87`).
- Wider field of view → bump `fovy` from `70` to `90`.

### New CLI flags

| Flag | Default | Description |
|---|---|---|
| `--tabletop-scenario` | from manifest `default_scenario` | Which scenario to load |
| `--hazard-textures-dir` | repo `selected_imgs_videos_demo/Hazard_detection_selected` | Folder of placard JPGs |
| `--tabletop-manifest` | `scene/tabletop_items.json` | Alternate manifest path |
```

- [ ] **Step 2: Add a short section to `g1_nav_demo/README.md`**

In `g1_nav_demo/README.md`, after the `## How It Works — Full Pipeline Walkthrough` section's last stage (Stage 5), add a new stage:

```markdown
### Stage 6: Hazard Inspection (optional, intent-triggered)

**Input:** A natural-language command containing an inspection verb
(e.g. "inspect the table"). The VLM goal parser sets `inspect: true`
in its JSON output.

**What happens:**

1. After `run_to_goal` returns `reached=True`, `NavigationSession.inspect_target`
   is called (only for targets in `INSPECTABLE_TARGETS = {"table"}` for MVP).
2. The robot stands idle for ~0.5 s so its pose settles.
3. `VideoRenderer.snapshot("head_onboard", data)` renders one 640×480 PNG
   from the on-board head camera (a new camera attached to the torso body,
   forward of the head geometry, tilted ~20° down).
4. `InspectionBridge.inspect(png, "table")` sends the image + a structured
   prompt to the same VLM (different prompt, list of items + per-item
   `hazardous` flag). It returns an `InspectionResult` dataclass.
5. The result is written to `<turn_prefix>_inspection.json` and, if
   `alarm=True`, a red banner is set on the renderer.
6. The robot idles ~3 s with the banner overlay so it appears in the video.

**Output:** A JSON report on disk and (if alarm) a red banner in the video.

Scene tabletop items are defined in `scene/tabletop_items.json`; the
`scene/tabletop_loader.py` module splices them into the scene XML before
MuJoCo loads it. Hazard images come from `selected_imgs_videos_demo/Hazard_detection_selected/`.
```

- [ ] **Step 3: Commit**

```bash
git add g1_nav_demo/HOWTO.md g1_nav_demo/README.md
git commit -m "Document hazard inspection mode in HOWTO and README"
```

---

## Self-review notes (filled during plan authoring)

- **Spec coverage:** every section of the spec maps to a task (head-cam fix → Task 4; tabletop loader/manifest → Task 2-3; renderer banner + snapshot → Tasks 1, 5; goal parser change → Task 6; `InspectionBridge` → Task 7; session hook + JSON → Task 8; CLI wiring → Tasks 3 and 9; docs → Task 10).
- **Placeholder scan:** all steps include actual code or commands; no "TBD" / "similar to" placeholders.
- **Type consistency:** `InspectionItem.reason` defaults to `""`, used the same way in tests and serialization. `inspect_target` keyword arguments match between the implementation (Task 8) and the call sites (Task 9). `snapshot()` signature `(camera_name, data, ...)` is consistent across Tasks 1, 8, 9.
- **Risk callouts honored:** head-cam framing is empirical, owned by Task 4's tuning step; VLM accuracy on real placards is acknowledged as a tuning concern in the spec, not blocked here.

---

## Execution

After all tasks complete, the end-to-end behavior described in the spec's
"Data flow summary (inspection turn)" section will be operational. Run the
Task 9 Step 5 manual smoke test as the acceptance check.
