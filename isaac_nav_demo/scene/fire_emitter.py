"""Adds a fire VFX emitter to the warehouse scene.

Two strategies, tried in order:
  A) Load a pre-built NVIDIA fire VFX USD asset (requires Omniverse connection).
  B) Create a PhysXFlow (omni.physx.flow) emitter programmatically (local fallback).

The fire renders in RTX / RayTracedLighting mode and appears in all Isaac
cameras, making it visible to the robot's onboard camera and detectable by
the VLM agent.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Pre-built burning-barrel fire VFX from the NVIDIA Omniverse asset catalog.
# Download the "Fire & Smoke" sample pack from:
#   https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html
# and point FIRE_ASSET_USD at the extracted .usd, OR leave it pointing at the
# nucleus server path below (requires an Omniverse Nucleus connection).
_FIRE_ASSET_USD = (
    "omniverse://localhost/NVIDIA/Assets/Effects/2023_1/Effects/Fire/burning_barrel.usd"
)

_FIRE_PRIM_PATH = "/World/FireHazard"


def add_fire(stage, pos: tuple[float, float, float] = (17.0, 0.0, 0.5)) -> None:
    """Place fire VFX at *pos* in world coordinates.

    Tries the pre-built NVIDIA asset first; falls back to a programmatic
    PhysXFlow emitter if the asset is unavailable.
    """
    try:
        _add_fire_asset(stage, pos)
        logger.info("Fire VFX loaded from asset: %s", _FIRE_ASSET_USD)
    except Exception as exc:
        logger.warning("Fire asset load failed (%s); falling back to PhysXFlow", exc)
        _add_flow_emitter(stage, pos)


# ---------------------------------------------------------------------------
# Strategy A: pre-built asset
# ---------------------------------------------------------------------------

def _add_fire_asset(stage, pos: tuple) -> None:
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from pxr import UsdGeom, Gf

    add_reference_to_stage(usd_path=_FIRE_ASSET_USD, prim_path=_FIRE_PRIM_PATH)

    prim = stage.GetPrimAtPath(_FIRE_PRIM_PATH)
    if not prim.IsValid():
        raise RuntimeError(f"Prim not valid after add_reference_to_stage: {_FIRE_PRIM_PATH}")

    xformable = UsdGeom.Xformable(prim)
    ops = xformable.GetOrderedXformOps()
    translate_op = next(
        (op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeTranslate),
        None,
    )
    if translate_op is None:
        translate_op = xformable.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))


# ---------------------------------------------------------------------------
# Strategy B: programmatic PhysXFlow emitter
# ---------------------------------------------------------------------------

def _add_flow_emitter(stage, pos: tuple) -> None:
    """Create a minimal PhysXFlow fire emitter at *pos*.

    PhysXFlow simulates volumetric combustion (fuel + temperature) on the
    GPU and renders via RTX.  The parameters below produce a visible ~1.5 m
    flame column suitable for detection by the robot's head camera.

    Reference: https://docs.omniverse.nvidia.com/extensions/latest/ext_simulation.html
    """
    import omni.kit.commands
    from pxr import Sdf, UsdGeom, Gf

    # Create a sphere geometry to act as the emitter volume
    emitter_path = "/World/FireFlow/Emitter"
    sphere_path  = "/World/FireFlow/EmitterSphere"

    omni.kit.commands.execute(
        "CreatePrimWithDefaultXform",
        prim_type="Sphere",
        prim_path=sphere_path,
        attributes={"radius": 0.2},
    )
    sphere_prim = stage.GetPrimAtPath(sphere_path)
    UsdGeom.Xformable(sphere_prim).AddTranslateOp().Set(
        Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2]))
    )

    # Attach PhysxFlow attributes to drive fire simulation
    try:
        from omni.physxflow import PhysxFlowSchema  # type: ignore[import]

        flow_api = PhysxFlowSchema.PhysxFlowEmitterAPI.Apply(sphere_prim)
        flow_api.CreateFuelAttr().Set(1.0)            # fuel density
        flow_api.CreateTemperatureAttr().Set(3000.0)  # kelvin — orange/yellow flame
        flow_api.CreateVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 2.0))  # upward

        logger.info("PhysXFlow emitter created at %s", pos)
    except (ImportError, Exception) as exc:
        logger.warning(
            "PhysxFlowSchema not available (%s). "
            "Fire will appear as a coloured sphere placeholder.",
            exc,
        )
        _add_fire_placeholder(stage, sphere_prim)


def _add_fire_placeholder(stage, sphere_prim) -> None:
    """Last-resort visual: bright orange emissive sphere standing in for fire.

    Visible in all render modes; used when neither the VFX asset nor
    PhysXFlow is available (e.g., CPU-only CI machines).
    """
    from pxr import UsdShade, Gf, Sdf

    mat_path = "/World/FireFlow/FireMat"
    material = UsdShade.Material.Define(stage, mat_path)
    shader   = UsdShade.Shader.Define(stage, mat_path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(1.0, 0.3, 0.0)   # orange glow
    )
    shader.CreateInput("emissiveIntensity", Sdf.ValueTypeNames.Float).Set(8.0)

    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(), "surface"
    )
    UsdShade.MaterialBindingAPI(sphere_prim).Bind(material)
    logger.info("Fire placeholder sphere created")
