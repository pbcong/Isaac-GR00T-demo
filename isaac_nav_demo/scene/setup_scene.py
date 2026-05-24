"""Builds the Isaac Sim warehouse scene.

Loads the NVIDIA Simple Warehouse USD, places extra props (pallets, barrel),
positions the G1 robot at the entrance, and returns the obstacle map used
by the VLM path planner.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from omni.isaac.core import World

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Asset paths
# ---------------------------------------------------------------------------

# NVIDIA built-in warehouse (comes with Isaac Sim install)
_WAREHOUSE_USD = (
    "omniverse://localhost/NVIDIA/Assets/Isaac/4.5/Isaac/Environments/"
    "Simple_Warehouse/warehouse_with_forklifts.usd"
)

# Unitree G1 – clone https://github.com/unitreerobotics/unitree_sim_isaaclab
# then point this env var to the repo root, or set the path directly below.
import os
_UNITREE_ROOT = os.environ.get(
    "UNITREE_ISAACLAB_ROOT",
    os.path.join(os.path.dirname(__file__), "..", "..", "unitree_sim_isaaclab"),
)
_G1_USD = os.path.join(_UNITREE_ROOT, "Unitree", "G1", "g1.usd")

# Warehouse prop assets (these ship with the Isaac Sim warehouse package)
_PROP_BASE = (
    "omniverse://localhost/NVIDIA/Assets/Isaac/4.5/Isaac/Props/Warehouse/"
)
_PALETTE_USD = _PROP_BASE + "Pallets/SM_PaletteA.usd"
_BARREL_USD   = _PROP_BASE + "Barrel/SM_BarrelA_02.usd"

# ---------------------------------------------------------------------------
# Scene layout
# ---------------------------------------------------------------------------
#
#  Warehouse floor ~24 m × 12 m.  Origin = entrance centre.
#
#  Y
#  ^   ╔══════════════════════════════════════════════╗
#  4   ║ [shelf A1]──────────[shelf A2]──────────────║
#  3   ║                                              ║
#  2   ║  [P1]              [forklift F1]             ║
#  1   ║                                   [P3]       ║
#  0   ║→ ROBOT       [forklift F2]     🔥 BARREL    ║
# -1   ║                                   [P4]       ║
# -2   ║  [P2]                                        ║
# -3   ║                                              ║
# -4   ║ [shelf B1]──────────[shelf B2]──────────────║
#      ╚══════════════════════════════════════════════╝
#      0    2    4    6    8   10   12   14   16   18  → X

# (prim_path_suffix, usd, pos_xyz)
_EXTRA_PROPS: list[tuple[str, str, tuple]] = [
    ("pallet_P1", _PALETTE_USD, (3.0,  1.5, 0.0)),
    ("pallet_P2", _PALETTE_USD, (3.0, -1.5, 0.0)),
    ("pallet_P3", _PALETTE_USD, (15.0,  1.5, 0.0)),
    ("pallet_P4", _PALETTE_USD, (15.0, -1.5, 0.0)),
    ("fire_barrel", _BARREL_USD, (17.0, 0.0, 0.0)),
]

# Robot start pose
_ROBOT_START_POS = np.array([0.5, 0.0, 0.0])
_ROBOT_START_YAW_DEG = 0.0   # facing +X (into the warehouse)

# Obstacle map: {name: (cx, cy, hx, hy)} — world-frame axis-aligned boxes
# (centre x, centre y, half-extent x, half-extent y)
# Used by VLMBridge to build the SCENE_PROMPT and by GoalPlanner for collision.
OBSTACLE_MAP: dict[str, tuple[float, float, float, float]] = {
    "shelf_A1":    ( 5.0,  4.0,  4.0, 0.4),
    "shelf_A2":    (13.0,  4.0,  4.0, 0.4),
    "shelf_B1":    ( 5.0, -4.0,  4.0, 0.4),
    "shelf_B2":    (13.0, -4.0,  4.0, 0.4),
    "forklift_F1": ( 9.0,  2.0,  0.65, 1.3),
    "forklift_F2": (11.0, -1.5,  0.65, 1.3),
    "pallet_P1":   ( 3.0,  1.5,  0.5, 0.4),
    "pallet_P2":   ( 3.0, -1.5,  0.5, 0.4),
    "pallet_P3":   (15.0,  1.5,  0.5, 0.4),
    "pallet_P4":   (15.0, -1.5,  0.5, 0.4),
    "fire_barrel": (17.0,  0.0,  0.3, 0.3),
}

# Position of fire in world frame (used by fire_emitter)
FIRE_WORLD_POS: tuple[float, float, float] = (17.0, 0.0, 0.5)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_translate(stage, prim_path: str, pos: tuple) -> None:
    from pxr import UsdGeom, Gf
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        logger.warning("Prim not found for translation: %s", prim_path)
        return
    xformable = UsdGeom.Xformable(prim)
    ops = xformable.GetOrderedXformOps()
    translate_op = next(
        (op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeTranslate),
        None,
    )
    if translate_op is None:
        translate_op = xformable.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))


def _set_yaw(stage, prim_path: str, yaw_deg: float) -> None:
    from pxr import UsdGeom, Gf
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    xformable = UsdGeom.Xformable(prim)
    ops = xformable.GetOrderedXformOps()
    rot_op = next(
        (op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ),
        None,
    )
    if rot_op is None:
        rot_op = xformable.AddRotateXYZOp()
    rot_op.Set(Gf.Vec3f(0.0, 0.0, float(yaw_deg)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_warehouse(world: "World") -> dict[str, tuple[float, float, float, float]]:
    """Load warehouse scene into *world* and return the obstacle map.

    Call this before ``world.reset()``.  Returns OBSTACLE_MAP so the caller
    can pass it directly to VLMBridge.
    """
    from omni.isaac.core.utils.stage import add_reference_to_stage

    stage = world.stage

    # --- Base warehouse environment ---
    logger.info("Loading warehouse USD: %s", _WAREHOUSE_USD)
    add_reference_to_stage(usd_path=_WAREHOUSE_USD, prim_path="/World/Warehouse")

    # --- Extra props ---
    for suffix, usd, pos in _EXTRA_PROPS:
        prim_path = f"/World/Props/{suffix}"
        logger.info("Adding prop %s at %s", suffix, pos)
        add_reference_to_stage(usd_path=usd, prim_path=prim_path)
        _set_translate(stage, prim_path, pos)

    # --- G1 robot ---
    logger.info("Loading G1 robot USD: %s", _G1_USD)
    add_reference_to_stage(usd_path=_G1_USD, prim_path="/World/G1")
    _set_translate(stage, "/World/G1", tuple(_ROBOT_START_POS))
    _set_yaw(stage, "/World/G1", _ROBOT_START_YAW_DEG)

    return OBSTACLE_MAP
