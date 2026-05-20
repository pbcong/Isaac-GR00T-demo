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
