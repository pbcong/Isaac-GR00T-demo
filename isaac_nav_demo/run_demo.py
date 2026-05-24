#!/usr/bin/env python3
"""Isaac Sim warehouse fire-inspection demo entry point.

Usage (single-turn):
    python -m isaac_nav_demo.run_demo \
        --command "Smoke detected in sector B7. Navigate to the fire source and inspect it." \
        --policy-path /path/to/motion.pt \
        --output demo_warehouse_fire.mp4

Usage (multi-turn interactive):
    python -m isaac_nav_demo.run_demo \
        --policy-path /path/to/motion.pt \
        --multiturn \
        --output-dir demo_output_warehouse

Environment variables:
    OPENROUTER_API_KEY     — LLM API key
    UNITREE_ISAACLAB_ROOT  — path to unitree_sim_isaaclab repo (for G1 USD)
"""
from __future__ import annotations

# Isaac Sim MUST be imported first — before any omni.* or pxr.* imports.
from isaacsim import SimulationApp

_APP_CFG = {
    "headless": True,
    "renderer": "RayTracedLighting",   # shows fire; swap to "PathTracing" for best quality
    "width": 1280,
    "height": 720,
}
simulation_app = SimulationApp(_APP_CFG)

# ---- all other imports after SimulationApp is live ----
import argparse
import json
import logging
import os
import sys

import numpy as np
from omni.isaac.core import World
from omni.isaac.sensor import Camera
import omni.isaac.core.utils.rotations as rot_utils

from isaac_nav_demo.scene.setup_scene import build_warehouse, FIRE_WORLD_POS
from isaac_nav_demo.scene.fire_emitter import add_fire
from isaac_nav_demo.sim_session import IsaacNavigationSession
from isaac_nav_demo.planner.goal_planner import GoalPlanner
from isaac_nav_demo.vlm.goal_parser import VLMBridge
from isaac_nav_demo.walk_policy.g1_walk_policy import G1WalkPolicy
from isaac_nav_demo.agent.agent_loop import AgentLoop
from isaac_nav_demo.renderer.video_renderer import VideoRenderer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Overhead VLM camera (separate from the video renderer cameras)
# ---------------------------------------------------------------------------

def _make_overhead_cam(world: World) -> Camera:
    cam = Camera(
        prim_path="/World/Cameras/OverheadVLM",
        position=np.array([10.0, 0.0, 25.0]),
        orientation=rot_utils.euler_angles_to_quats(
            np.array([-90.0, 0.0, 0.0]), degrees=True
        ),
        resolution=(640, 640),
        frequency=1,
    )
    cam.initialize()
    return cam


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

def _init_simulation(args) -> tuple[IsaacNavigationSession, AgentLoop]:
    world = World(stage_units_in_meters=1.0)

    # Build scene and get obstacle map
    obstacle_map = build_warehouse(world)

    # Add fire VFX on top of the barrel
    add_fire(world.stage, pos=FIRE_WORLD_POS)

    # Overhead camera for VLM scene images
    overhead_cam = _make_overhead_cam(world)

    # Reset world (compiles physics, sets initial poses)
    world.reset()

    walk_policy  = G1WalkPolicy(args.policy_path, device=args.device)
    goal_planner = GoalPlanner()
    vlm_bridge   = VLMBridge(obstacle_map=obstacle_map, model_name=args.vlm_model)

    session = IsaacNavigationSession(
        world=world,
        walk_policy=walk_policy,
        goal_planner=goal_planner,
        vlm_bridge=vlm_bridge,
        obstacle_map=obstacle_map,
        overhead_cam=overhead_cam,
        sim_fps=args.sim_fps,
        render_fps=args.render_fps,
        max_steps=args.max_steps,
        avoidance_stop_dist=args.obstacle_stop_dist,
        avoidance_clear_dist=args.obstacle_clear_dist,
        avoidance_timeout_steps=args.obstacle_timeout,
        avoidance_detour_dist=args.obstacle_detour_dist,
    )
    session.initialize()

    agent_loop = AgentLoop(session=session, model_name=args.vlm_model)
    return session, agent_loop


# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------

def _run_single_turn(args, session: IsaacNavigationSession, agent_loop: AgentLoop) -> None:
    report_json = args.output.rsplit(".", 1)[0] + "_report.json"
    video_renderer = VideoRenderer(
        session.world,
        output_path=args.output,
        fps=session.render_fps,
        width=1280,
        height=480,
    )
    try:
        result = agent_loop.run_turn(args.command, video_renderer, report_json)
        logger.info("Turn complete: %s — %s", result["verdict"], result["message"])
    finally:
        video_renderer.close()


def _run_multiturn(args, session: IsaacNavigationSession, agent_loop: AgentLoop) -> None:
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
        video_path  = os.path.join(args.output_dir, f"turn_{turn:03d}.mp4")
        report_json = os.path.join(args.output_dir, f"turn_{turn:03d}_report.json")

        video_renderer = VideoRenderer(
            session.world,
            output_path=video_path,
            fps=session.render_fps,
            width=1280,
            height=480,
        )
        try:
            result = agent_loop.run_turn(command, video_renderer, report_json)
            pos = session.current_position()
            print(f"  Turn {turn}: {result['verdict']} — {result['message']}")
            print(f"  Position: ({pos[0]:.2f}, {pos[1]:.2f})")
        finally:
            video_renderer.close()

    print(f"Session ended. {turn} turn(s) completed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Isaac Sim Warehouse Fire Inspection Demo")
    parser.add_argument("--command", type=str, default=None,
        help='Mission command, e.g. "Inspect the fire source near barrel B7"')
    parser.add_argument("--policy-path", type=str, required=True,
        help="Path to G1 walking policy JIT checkpoint (.pt)")
    parser.add_argument("--output", type=str, default="demo_warehouse_fire.mp4",
        help="Output video file (single-turn mode)")
    parser.add_argument("--vlm-model", type=str, default="x-ai/grok-4.3",
        help="LLM model name for agent + VLM bridge")
    parser.add_argument("--max-steps", type=int, default=15000,
        help="Max simulation steps per navigation turn")
    parser.add_argument("--sim-fps", type=int, default=200,
        help="Physics simulation frequency (Hz)")
    parser.add_argument("--render-fps", type=int, default=30,
        help="Video render frequency (Hz)")
    parser.add_argument("--device", type=str, default="cuda",
        help="Torch device for policy inference")
    parser.add_argument("--multiturn", action="store_true",
        help="Interactive multi-turn mode")
    parser.add_argument("--output-dir", type=str, default="demo_output_warehouse",
        help="Output directory for multi-turn videos")
    parser.add_argument("--obstacle-stop-dist",  type=float, default=1.0)
    parser.add_argument("--obstacle-clear-dist", type=float, default=1.2)
    parser.add_argument("--obstacle-timeout",    type=int,   default=600)
    parser.add_argument("--obstacle-detour-dist",type=float, default=1.0)
    args = parser.parse_args()

    if not args.multiturn and args.command is None:
        parser.error("--command is required in single-turn mode")

    session, agent_loop = _init_simulation(args)

    if args.multiturn:
        _run_multiturn(args, session, agent_loop)
    else:
        _run_single_turn(args, session, agent_loop)

    simulation_app.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
