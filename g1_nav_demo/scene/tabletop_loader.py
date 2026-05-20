from __future__ import annotations

import json
import os
import tempfile

NORMAL_KIND_GEOMS: dict[str, dict] = {
    "mug":    {"type": "cylinder", "size": "0.04 0.05",       "rgba": "0.95 0.95 0.95 1", "dz": 0.05},
    "book":   {"type": "box",      "size": "0.075 0.05 0.0125","rgba": "0.45 0.3 0.2 1",  "dz": 0.0125},
    "apple":  {"type": "sphere",   "size": "0.04",            "rgba": "0.85 0.1 0.1 1",  "dz": 0.04},
    "laptop": {"type": "box",      "size": "0.1 0.075 0.01",  "rgba": "0.2 0.2 0.2 1",   "dz": 0.01},
}

HAZARD_BOX_HALF_XY: float = 0.35
HAZARD_BOX_HALF_Z: float = 0.25
HAZARD_BOX_DZ: float = HAZARD_BOX_HALF_Z

# Placard sized to fill nearly the whole side face while preserving the
# image's aspect ratio.  Thickness/offset push the placard slightly off the
# box surface to avoid z-fighting and read as a physical sign.
PLACARD_FILL: float = 0.95          # fraction of the face dimension to fill
PLACARD_THICK: float = 0.004
PLACARD_OFFSET: float = 0.004


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


def _ensure_png(tex_path: str, scene_dir: str) -> str:
    """Return a path to a PNG version of *tex_path*.

    MuJoCo only loads PNG textures natively.  If *tex_path* is already a PNG
    (case-insensitive), it is returned unchanged.  Otherwise the image is
    converted and cached as a PNG inside *scene_dir* so that the merged XML
    file (which lives in *scene_dir*) can reference it with a relative-safe
    absolute path.
    """
    if tex_path.lower().endswith(".png"):
        return tex_path

    base = os.path.splitext(os.path.basename(tex_path))[0]
    out_png = os.path.join(scene_dir, f"_hazard_tex_{base}.png")
    if not os.path.exists(out_png):
        try:
            from PIL import Image  # type: ignore[import]
        except ImportError:
            raise RuntimeError(
                "Pillow is required to convert non-PNG hazard textures. "
                "Install it with: pip install Pillow"
            )
        with Image.open(tex_path) as img:
            img.save(out_png, format="PNG")
    return out_png


def _placard_half_extents(tex_path: str) -> tuple[float, float]:
    """Return (half_w, half_h) for a placard, preserving the texture aspect.

    Fills as much of a side face as ``PLACARD_FILL`` allows along the tighter
    axis, so portrait and square placards both display undistorted.
    """
    try:
        from PIL import Image  # type: ignore[import]
        with Image.open(tex_path) as img:
            iw, ih = img.size
    except Exception:
        iw, ih = 1, 1
    aspect = iw / ih  # width / height

    max_half_w = HAZARD_BOX_HALF_XY * PLACARD_FILL
    max_half_h = HAZARD_BOX_HALF_Z * PLACARD_FILL
    # Try fitting to max height first
    half_h = max_half_h
    half_w = half_h * aspect
    if half_w > max_half_w:
        half_w = max_half_w
        half_h = half_w / aspect
    return half_w, half_h


def _build_hazard_box_xml(
    item: dict, table_top_z: float, hazard_textures_dir: str, scene_dir: str
) -> tuple[str, str]:
    """Return (asset_snippet, geom_snippet) for a hazard_box item.

    Creates a large red box with an aspect-correct placard on each of the
    four side faces so the hazard symbol is clearly readable from any
    approach angle.

    Placard geoms use type="plane" with explicit quaternions so the texture
    maps to the full face (MuJoCo's 2D projection is top-down in local XY,
    which only samples a thin edge when applied to vertical box faces).

    Quaternion conventions (MuJoCo format w x y z):
      front (+Y normal): 180° around (0, 1/√2, 1/√2)  → "0 0 .7071 .7071"
      back  (-Y normal): 90°  around world X            → ".7071 .7071 0 0"
      right (+X normal): 120° around (1,1,1)/√3         → ".5 .5 .5 .5"
      left  (-X normal): 120° around (1,-1,-1)/√3       → ".5 .5 -.5 -.5"
    All produce local_y = world +Z (v-axis vertical, image upright).
    """
    name = item["name"]
    px, py = item["pos_xy"]
    tex_id = f"tex_{name}"
    mat_id = f"mat_{name}"

    _ALL_SIDES = {"front", "back", "left", "right"}
    placard_sides = set(item.get("placard_sides", list(_ALL_SIDES)))
    unknown = placard_sides - _ALL_SIDES
    if unknown:
        raise ValueError(f"Unknown placard_sides {unknown}; valid: {_ALL_SIDES}")

    if placard_sides:
        tex_filename = item["texture"]
        tex_path = os.path.join(hazard_textures_dir, tex_filename)
        tex_path = _ensure_png(tex_path, scene_dir)
        asset = (
            f'    <texture name="{tex_id}" type="2d" file="{tex_path}"/>\n'
            f'    <material name="{mat_id}" texture="{tex_id}" texuniform="false" '
            f'emission="0.3" specular="0.05" shininess="0.05" reflectance="0"/>\n'
        )
    else:
        asset = ""

    box_z = table_top_z + HAZARD_BOX_DZ + 0.005
    placard_z = box_z
    pw, ph = _placard_half_extents(tex_path) if placard_sides else (0.0, 0.0)
    off = PLACARD_OFFSET

    def _plane(pname: str, pos: str, quat: str) -> str:
        return (
            f'      <geom name="{pname}" type="plane" '
            f'size="{pw} {ph} 0.001" pos="{pos}" quat="{quat}" '
            f'material="{mat_id}" contype="0" conaffinity="0"/>\n'
        )

    geom = (
        f'      <geom name="{name}" type="box" '
        f'size="{HAZARD_BOX_HALF_XY} {HAZARD_BOX_HALF_XY} {HAZARD_BOX_HALF_Z}" '
        f'pos="{px} {py} {box_z}" rgba="0.85 0.08 0.08 1" contype="0" conaffinity="0"/>\n'
    )
    if "front" in placard_sides:
        geom += _plane(f"{name}_placard_front",
                       f"{px} {py + HAZARD_BOX_HALF_XY + off} {placard_z}",
                       "0 0 0.7071 0.7071")
    if "back" in placard_sides:
        geom += _plane(f"{name}_placard_back",
                       f"{px} {py - HAZARD_BOX_HALF_XY - off} {placard_z}",
                       "0.7071 0.7071 0 0")
    if "right" in placard_sides:
        geom += _plane(f"{name}_placard_right",
                       f"{px + HAZARD_BOX_HALF_XY + off} {py} {placard_z}",
                       "0.5 0.5 0.5 0.5")
    if "left" in placard_sides:
        geom += _plane(f"{name}_placard_left",
                       f"{px - HAZARD_BOX_HALF_XY - off} {py} {placard_z}",
                       "0.5 0.5 -0.5 -0.5")
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
    manifest = load_manifest(manifest_path)
    name = scenario or manifest.get("default_scenario", "mixed")
    if name not in manifest["scenarios"]:
        raise KeyError(f"Unknown scenario {name!r}; have {list(manifest['scenarios'])}")
    items = manifest["scenarios"][name]
    table_top_z = float(manifest["table_top_z"])

    with open(room_xml_path) as f:
        original_xml = f.read()

    # scene_dir is needed early so _build_hazard_box_xml can convert JPG → PNG
    scene_dir = os.path.dirname(os.path.abspath(room_xml_path))

    asset_snippets: list[str] = []
    geom_snippets: list[str] = []
    for item in items:
        if is_hazard_item(item):
            asset, geom = _build_hazard_box_xml(item, table_top_z, hazard_textures_dir, scene_dir)
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

    # Write next to the original so relative <include file="g1_29dof.xml"/> still resolves.
    if out_dir is not None and os.path.abspath(out_dir) == scene_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(scene_dir, "g1_nav_room_merged.xml")
    with open(out_path, "w") as f:
        f.write(merged_xml)
    return out_path
