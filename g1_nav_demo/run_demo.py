#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json as _json
import logging
import math
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import mujoco
import numpy as np

from g1_nav_demo.planner.goal_planner import GoalPlanner
from g1_nav_demo.renderer.video_renderer import VideoRenderer
from g1_nav_demo.vlm.goal_parser import VLMBridge, Goal
from g1_nav_demo.agent.agent_loop import AgentLoop
from g1_nav_demo.walk_policy.g1_walk_policy import G1WalkPolicy

logger = logging.getLogger(__name__)

def _read_forward_range(
    model: "mujoco.MjModel",
    data: "mujoco.MjData",
    robot_xy: tuple[float, float],
    current_yaw: float,
    cone_deg: float = 20.0,
    cutoff: float = 3.0,
    min_dist: float = 0.5,
) -> float:
    """Return distance to nearest collidable geom in the forward cone.

    Skips floor (plane geoms) and contype-0 geoms (tabletop items).
    min_dist filters out the robot's own limb geoms which are always nearby.
    """
    robot = np.array(robot_xy, dtype=np.float64)
    cos_y = math.cos(current_yaw)
    sin_y = math.sin(current_yaw)
    forward = np.array([cos_y, sin_y])
    cos_half_cone = math.cos(math.radians(cone_deg))
    best = cutoff

    for i in range(model.ngeom):
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        if model.geom_contype[i] == 0:
            continue
        geom_xy = data.geom_xpos[i][:2]
        to_geom = geom_xy - robot
        dist = float(np.linalg.norm(to_geom))
        if dist < min_dist or dist > cutoff:
            continue
        if float(np.dot(to_geom / dist, forward)) >= cos_half_cone:
            best = min(best, dist)

    return best



LEG_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]

LEG_ACTUATOR_NAMES = [
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
]

UPPER_ACTUATOR_DEFAULTS = {
    "waist_yaw":        (0.0,  200.0, 10.0),
    "waist_roll":       (0.0,  200.0, 10.0),
    "waist_pitch":      (0.0,  200.0, 10.0),
    "left_shoulder_pitch":  (0.25,  40.0, 4.0),
    "left_shoulder_roll":   (0.3,   40.0, 4.0),
    "left_shoulder_yaw":    (0.0,   40.0, 4.0),
    "left_elbow":           (0.9,   40.0, 4.0),
    "left_wrist_roll":      (0.0,   20.0, 2.0),
    "left_wrist_pitch":     (0.0,   20.0, 2.0),
    "left_wrist_yaw":       (0.0,   20.0, 2.0),
    "right_shoulder_pitch": (-0.25, 40.0, 4.0),
    "right_shoulder_roll":  (-0.3,  40.0, 4.0),
    "right_shoulder_yaw":   (0.0,   40.0, 4.0),
    "right_elbow":          (0.9,   40.0, 4.0),
    "right_wrist_roll":     (0.0,   20.0, 2.0),
    "right_wrist_pitch":    (0.0,   20.0, 2.0),
    "right_wrist_yaw":      (0.0,   20.0, 2.0),
}

UPPER_JOINT_NAMES = [
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]




def quat_to_yaw(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))





class NavigationSession:
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
        avoidance_stop_dist: float = 1.0,
        avoidance_clear_dist: float = 1.2,
        avoidance_timeout_steps: int = 75,
        avoidance_detour_dist: float = 1.0,
    ) -> None:
        self.model = model
        self.data = data
        self.walk_policy = walk_policy
        self.goal_planner = goal_planner
        self.vlm_bridge = vlm_bridge
        self.sim_fps = sim_fps
        self.render_fps = render_fps
        self.max_steps = max_steps
        self.torso_body_id = torso_body_id
        self.avoidance_stop_dist = avoidance_stop_dist
        self.avoidance_clear_dist = avoidance_clear_dist
        self.avoidance_timeout_steps = avoidance_timeout_steps
        self.avoidance_detour_dist = avoidance_detour_dist
        self.obstacle_human: "ObstacleHuman | None" = None

        self._build_index_mappings()

        self.kps = G1WalkPolicy.KPS
        self.kds = G1WalkPolicy.KDS
        self.default_angles = G1WalkPolicy.DEFAULT_ANGLES
        self.decimation = max(1, int(round(G1WalkPolicy.CONTROL_DT / model.opt.timestep)))

    def _build_index_mappings(self) -> None:
        model = self.model

        leg_qpos_adr = []
        leg_dof_adr = []
        leg_actuator_ids = []
        for joint_name, act_name in zip(LEG_JOINT_NAMES, LEG_ACTUATOR_NAMES):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            leg_qpos_adr.append(model.jnt_qposadr[jid])
            leg_dof_adr.append(model.jnt_dofadr[jid])
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
            leg_actuator_ids.append(aid)
        self.leg_qpos_adr = np.array(leg_qpos_adr, dtype=np.intp)
        self.leg_dof_adr = np.array(leg_dof_adr, dtype=np.intp)
        self.leg_actuator_ids = np.array(leg_actuator_ids, dtype=np.intp)
        self.ctrl_range = model.actuator_ctrlrange[self.leg_actuator_ids]

        upper_qpos_adr = []
        upper_dof_adr = []
        upper_actuator_ids = []
        upper_default_pos = []
        upper_kps = []
        upper_kds = []
        for joint_name in UPPER_JOINT_NAMES:
            act_name = joint_name.replace("_joint", "")
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            upper_qpos_adr.append(model.jnt_qposadr[jid])
            upper_dof_adr.append(model.jnt_dofadr[jid])
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
            upper_actuator_ids.append(aid)
            default, kp, kd = UPPER_ACTUATOR_DEFAULTS[act_name]
            upper_default_pos.append(default)
            upper_kps.append(kp)
            upper_kds.append(kd)
        self.upper_qpos_adr = np.array(upper_qpos_adr, dtype=np.intp)
        self.upper_dof_adr = np.array(upper_dof_adr, dtype=np.intp)
        self.upper_actuator_ids = np.array(upper_actuator_ids, dtype=np.intp)
        self.upper_default_pos = np.array(upper_default_pos, dtype=np.float32)
        self.upper_kps = np.array(upper_kps, dtype=np.float32)
        self.upper_kds = np.array(upper_kds, dtype=np.float32)
        self.upper_ctrl_range = model.actuator_ctrlrange[self.upper_actuator_ids]

    def _init_obstacle_human(self) -> None:
        from g1_nav_demo.scene.obstacle_human import ObstacleHuman
        body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "moving_human"
        )
        if body_id < 0:
            logger.warning("moving_human body not in model; obstacle disabled")
            return
        self.obstacle_human = ObstacleHuman(
            self.model, self.data, body_name="moving_human"
        )
        logger.info("ObstacleHuman initialised (body_id=%d)", body_id)

    def _configure_obstacle_intercept(self, goal: Goal) -> None:
        if self.obstacle_human is None or not goal.waypoints:
            return

        from g1_nav_demo.scene.obstacle_human import ObstacleHuman

        robot_pos = np.array(self.current_position(), dtype=np.float64)
        wp = np.array(goal.waypoints[0], dtype=np.float64)
        approach = wp - robot_pos
        norm = float(np.linalg.norm(approach))
        if norm < 0.1:
            return
        perp = np.array([-approach[1], approach[0]]) / norm  # 90° left of approach

        stand_steps = int(5.0 / self.model.opt.timestep)
        self.obstacle_human = ObstacleHuman(
            self.model, self.data, body_name="moving_human",
            start_xy=(float(wp[0]), float(wp[1])),
            direction_xy=(float(perp[0]), float(perp[1])),
            speed=0.8, travel_dist=3.0,
            stand_steps=stand_steps,
        )
        logger.info(
            "ObstacleHuman at waypoint (%.1f, %.1f); blocking %.0f s then moving",
            float(wp[0]), float(wp[1]), stand_steps * self.model.opt.timestep,
        )

    def current_position(self) -> tuple[float, float]:
        return (float(self.data.qpos[0]), float(self.data.qpos[1]))

    def current_yaw(self) -> float:
        quaternion = np.array(
            [self.data.qpos[3], self.data.qpos[4], self.data.qpos[5], self.data.qpos[6]],
            dtype=np.float64,
        )
        return quat_to_yaw(quaternion)

    @staticmethod
    def _compute_face_yaw(goal: Goal) -> float | None:
        if not goal.waypoints or goal.face_direction is None:
            return None
        gx, gy = goal.waypoints[-1]
        face_map = {
            "front": (0, 1),
            "back": (0, -1),
            "left": (1, 0),
            "right": (-1, 0),
        }
        direction = face_map.get(goal.face_direction)
        if direction is None:
            return None
        target_x = gx + direction[0] * 0.8
        target_y = gy + direction[1] * 0.8
        return float(math.atan2(target_y - gy, target_x - gx))

    def parse_goal(self, command: str, debug_prefix: str | None = None) -> Goal | None:
        robot_pos = self.current_position()
        robot_yaw = self.current_yaw()
        debug_img = None
        if debug_prefix:
            debug_img = debug_prefix + "_vlm_scene.png"
        goal = self.vlm_bridge.parse(
            command,
            mj_model=self.model,
            mj_data=self.data,
            robot_pos=robot_pos,
            robot_yaw=robot_yaw,
            debug_image_path=debug_img,
        )
        if goal is not None:
            logger.info(
                "Parsed goal: %s -> waypoints %s",
                goal.target_name,
                [(f"{x:.1f}", f"{y:.1f}") for x, y in goal.waypoints],
            )
        return goal

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
        self, goal: Goal, command: str, video_renderer: "VideoRenderer",
        face_yaw_override: float | None = None,
    ) -> bool:
        from g1_nav_demo.avoidance import AvoidanceStateMachine

        face_yaw = (
            face_yaw_override if face_yaw_override is not None
            else self._compute_face_yaw(goal)
        )
        self.goal_planner.set_waypoints(goal.waypoints, face_yaw=face_yaw)

        avoidance = AvoidanceStateMachine(
            stop_dist=self.avoidance_stop_dist,
            clear_dist=self.avoidance_clear_dist,
            timeout_steps=self.avoidance_timeout_steps,
            detour_dist=self.avoidance_detour_dist,
        )

        if self.obstacle_human is not None:
            self._configure_obstacle_intercept(goal)

        target_positions = self.default_angles.copy()
        velocity_command = np.zeros(3, dtype=np.float32)
        reached = False
        plan_result = None
        steps_per_render = max(1, self.sim_fps // self.render_fps)

        for step in range(self.max_steps):
            if step % self.decimation == 0:
                current_pos = self.current_position()
                current_yaw = self.current_yaw()

                range_val = _read_forward_range(
                    self.model, self.data, current_pos, current_yaw
                )
                avoidance.step(range_val, current_pos, self.goal_planner, face_yaw)
                video_renderer.obstacle_banner = avoidance.banner

                if avoidance.is_navigating:
                    plan_result = self.goal_planner.compute_command(
                        current_pos, current_yaw
                    )
                    if plan_result.reached:
                        logger.info(
                            "Reached goal at step %d (distance=%.3f)",
                            step, plan_result.distance,
                        )
                        reached = True
                        break
                    velocity_command = np.array(
                        [plan_result.vx, plan_result.vy, plan_result.vyaw],
                        dtype=np.float32,
                    )
                else:
                    velocity_command = np.zeros(3, dtype=np.float32)

                dof_pos = np.array(
                    self.data.qpos[self.leg_qpos_adr], dtype=np.float32
                )
                dof_vel = np.array(
                    self.data.qvel[self.leg_dof_adr], dtype=np.float32
                )
                angular_velocity = np.array(
                    [self.data.qvel[3], self.data.qvel[4], self.data.qvel[5]],
                    dtype=np.float32,
                )
                quaternion = np.array(
                    [self.data.qpos[3], self.data.qpos[4],
                     self.data.qpos[5], self.data.qpos[6]],
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
                self.upper_kps * (self.upper_default_pos - upper_pos)
                - self.upper_kds * upper_vel
            )
            upper_torques = np.clip(
                upper_torques,
                self.upper_ctrl_range[:, 0], self.upper_ctrl_range[:, 1],
            )
            self.data.ctrl[self.upper_actuator_ids] = upper_torques

            if self.obstacle_human is not None:
                self.obstacle_human.step(self.model.opt.timestep)

            mujoco.mj_step(self.model, self.data)

            if step % steps_per_render == 0 and plan_result is not None:
                n_wps = len(self.goal_planner._waypoints)
                wp_idx = self.goal_planner._current_wp_idx
                frame = video_renderer.render_frame(
                    self.data,
                    command=f"{command} [wp {min(wp_idx + 1, n_wps)}/{n_wps}]",
                    distance=plan_result.distance,
                    update_head_camera=True,
                    head_body_id=self.torso_body_id,
                    goal_waypoints=list(self.goal_planner._waypoints),
                    current_wp_idx=wp_idx,
                )
                video_renderer.write_frame(frame)

        if not reached:
            logger.warning("Did not reach goal within %d steps", self.max_steps)
        return reached

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

    


def _init_simulation(args) -> tuple:
    from g1_nav_demo.scene.tabletop_loader import build_merged_scene

    scene_dir = os.path.dirname(os.path.abspath(args.scene_xml))
    repo_root = os.path.abspath(os.path.join(scene_dir, "..", ".."))

    manifest_path = args.tabletop_manifest or os.path.join(scene_dir, "tabletop_items.json")
    hazard_dir = args.hazard_textures_dir or scene_dir
    merged_xml = build_merged_scene(
        room_xml_path=args.scene_xml,
        manifest_path=manifest_path,
        scenario=args.tabletop_scenario,
        hazard_textures_dir=hazard_dir,
    )
    model = mujoco.MjModel.from_xml_path(merged_xml)
    model.opt.timestep = 1.0 / args.sim_fps
    data = mujoco.MjData(model)

    walk_policy = G1WalkPolicy(args.policy_path, device=args.device)
    goal_planner = GoalPlanner()
    vlm_bridge = VLMBridge(model_name=args.vlm_model)

    torso_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "torso_link"
    )

    mujoco.mj_resetData(model, data)

    default_angles = G1WalkPolicy.DEFAULT_ANGLES
    leg_qpos_adr = []
    for joint_name in LEG_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        leg_qpos_adr.append(model.jnt_qposadr[jid])
    leg_qpos_adr = np.array(leg_qpos_adr, dtype=np.intp)

    upper_qpos_adr = []
    upper_default_pos = []
    for joint_name in UPPER_JOINT_NAMES:
        act_name = joint_name.replace("_joint", "")
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        upper_qpos_adr.append(model.jnt_qposadr[jid])
        upper_default_pos.append(UPPER_ACTUATOR_DEFAULTS[act_name][0])
    upper_qpos_adr = np.array(upper_qpos_adr, dtype=np.intp)
    upper_default_pos = np.array(upper_default_pos, dtype=np.float32)

    data.qpos[2] = 0.793
    data.qpos[3] = 1.0
    data.qpos[4] = 0.0
    data.qpos[5] = 0.0
    data.qpos[6] = 0.0

    for i, adr in enumerate(leg_qpos_adr):
        data.qpos[adr] = default_angles[i]

    for i, adr in enumerate(upper_qpos_adr):
        data.qpos[adr] = upper_default_pos[i]

    mujoco.mj_forward(model, data)
    walk_policy.reset()

    session = NavigationSession(
        model=model,
        data=data,
        walk_policy=walk_policy,
        goal_planner=goal_planner,
        vlm_bridge=vlm_bridge,
        sim_fps=args.sim_fps,
        render_fps=args.render_fps,
        max_steps=args.max_steps,
        torso_body_id=torso_body_id,
        avoidance_stop_dist=args.obstacle_stop_dist,
        avoidance_clear_dist=args.obstacle_clear_dist,
        avoidance_timeout_steps=args.obstacle_timeout,
        avoidance_detour_dist=args.obstacle_detour_dist,
    )
    if args.moving_obstacle:
        session._init_obstacle_human()
    agent_loop = AgentLoop(session=session, model_name=args.vlm_model)
    return session, agent_loop


def _run_single_turn(args, session: NavigationSession, agent_loop: AgentLoop) -> None:
    report_json = args.output.rsplit(".", 1)[0] + "_report.json"
    video_renderer = VideoRenderer(
        session.model, output_path=args.output, fps=session.render_fps,
        width=1280, height=480,
    )
    try:
        result = agent_loop.run_turn(args.command, video_renderer, report_json)
        logger.info("Turn complete: %s — %s", result["verdict"], result["message"])
    finally:
        video_renderer.close()


def _run_multiturn(args, session: NavigationSession, agent_loop: AgentLoop) -> None:
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
        video_path = os.path.join(args.output_dir, f"turn_{turn:03d}.mp4")
        report_json = os.path.join(args.output_dir, f"turn_{turn:03d}_report.json")

        video_renderer = VideoRenderer(
            session.model, output_path=video_path, fps=session.render_fps,
            width=1280, height=480,
        )
        try:
            result = agent_loop.run_turn(command, video_renderer, report_json)
            pos = session.current_position()
            print(f"  Turn {turn}: {result['verdict']} — {result['message']}")
            print(f"  Position: ({pos[0]:.2f}, {pos[1]:.2f})")
        finally:
            video_renderer.close()

    print(f"Session ended. {turn} turn(s) completed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="G1 Navigation Demo")
    parser.add_argument("--command", type=str, default=None,
                        help="Navigation command string")
    parser.add_argument("--scene-xml", type=str, default=None,
                        help="Path to MuJoCo scene XML file")
    parser.add_argument("--policy-path", type=str, default=None,
                        help="Path to walking policy JIT checkpoint")
    parser.add_argument("--output", type=str, default="demo_output.mp4",
                        help="Output video file path")
    parser.add_argument("--vlm-model", type=str, default="x-ai/grok-4.3",
                        help="Model name for the VLM API")
    parser.add_argument("--max-steps", type=int, default=20000,
                        help="Maximum simulation steps per turn")
    parser.add_argument("--sim-fps", type=int, default=500,
                        help="Simulation frequency in Hz")
    parser.add_argument("--render-fps", type=int, default=30,
                        help="Video render frequency in Hz")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Torch device for policy inference")
    parser.add_argument("--multiturn", action="store_true",
                        help="Enable interactive multi-turn mode")
    parser.add_argument("--output-dir", type=str, default="demo_output",
                        help="Output directory for per-turn videos (multiturn mode)")
    parser.add_argument("--tabletop-scenario", type=str, default=None,
                        help="Tabletop scenario name from tabletop_items.json")
    parser.add_argument("--hazard-textures-dir", type=str, default=None,
                        help="Directory containing hazard placard images")
    parser.add_argument("--tabletop-manifest", type=str, default=None,
                        help="Path to tabletop_items.json (default: scene/tabletop_items.json)")
    parser.add_argument("--moving-obstacle", action="store_true",
                        help="Enable kinematic human obstacle crossing the robot path")
    parser.add_argument("--obstacle-stop-dist", type=float, default=1.0,
                        help="Range (m) at which robot stops for obstacle (default: 1.0)")
    parser.add_argument("--obstacle-clear-dist", type=float, default=1.2,
                        help="Range (m) at which robot resumes after obstacle clears (default: 1.2)")
    parser.add_argument("--obstacle-timeout", type=int, default=600,
                        help="Control steps blocked before reroute (~12 s at 50 Hz, default: 600)")
    parser.add_argument("--obstacle-detour-dist", type=float, default=1.0,
                        help="Perpendicular detour offset (m) for reroute waypoint (default: 1.0)")
    args = parser.parse_args()

    if args.scene_xml is None:
        args.scene_xml = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "scene", "g1_nav_room.xml"
        )

    if args.policy_path is None:
        parser.error("--policy-path is required")

    if not args.multiturn and args.command is None:
        parser.error("--command is required in single-turn mode")

    if args.command is None:
        args.command = None

    session, agent_loop = _init_simulation(args)

    if args.multiturn:
        _run_multiturn(args, session, agent_loop)
    else:
        _run_single_turn(args, session, agent_loop)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
