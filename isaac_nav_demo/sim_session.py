"""Isaac Sim navigation session.

Drop-in replacement for the MuJoCo NavigationSession.  Public interface is
identical so agent_loop.py works without changes.  Internals replace:
  - mujoco.mj_step      → world.step()
  - qpos / ctrl arrays  → ArticulationView
  - geom position loop  → PhysX raycasting
  - mujoco.Renderer     → omni.isaac.sensor.Camera  (in VideoRenderer)
"""
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from omni.isaac.core import World
    from omni.isaac.core.articulations import ArticulationView
    from isaac_nav_demo.renderer.video_renderer import VideoRenderer
    from isaac_nav_demo.vlm.goal_parser import Goal, VLMBridge
    from isaac_nav_demo.planner.goal_planner import GoalPlanner

logger = logging.getLogger(__name__)

# Joint names match the G1 USD articulation — verify against the actual USD
# by inspecting /World/G1 in Isaac Sim stage after import.
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

UPPER_JOINT_NAMES = [
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

UPPER_DEFAULTS = {
    "waist_yaw_joint":            (0.0,  200.0, 10.0),
    "waist_roll_joint":           (0.0,  200.0, 10.0),
    "waist_pitch_joint":          (0.0,  200.0, 10.0),
    "left_shoulder_pitch_joint":  (0.25,  40.0,  4.0),
    "left_shoulder_roll_joint":   (0.3,   40.0,  4.0),
    "left_shoulder_yaw_joint":    (0.0,   40.0,  4.0),
    "left_elbow_joint":           (0.9,   40.0,  4.0),
    "left_wrist_roll_joint":      (0.0,   20.0,  2.0),
    "left_wrist_pitch_joint":     (0.0,   20.0,  2.0),
    "left_wrist_yaw_joint":       (0.0,   20.0,  2.0),
    "right_shoulder_pitch_joint": (-0.25, 40.0,  4.0),
    "right_shoulder_roll_joint":  (-0.3,  40.0,  4.0),
    "right_shoulder_yaw_joint":   (0.0,   40.0,  4.0),
    "right_elbow_joint":          (0.9,   40.0,  4.0),
    "right_wrist_roll_joint":     (0.0,   20.0,  2.0),
    "right_wrist_pitch_joint":    (0.0,   20.0,  2.0),
    "right_wrist_yaw_joint":      (0.0,   20.0,  2.0),
}


def _quat_to_yaw(quat: np.ndarray) -> float:
    """Isaac Sim quaternion (w, x, y, z) → yaw angle."""
    w, x, y, z = quat
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


class IsaacNavigationSession:
    """Mirrors the MuJoCo NavigationSession interface for the warehouse demo."""

    def __init__(
        self,
        world: "World",
        walk_policy,
        goal_planner: "GoalPlanner",
        vlm_bridge: "VLMBridge",
        obstacle_map: dict,
        overhead_cam,
        sim_fps: int = 200,
        render_fps: int = 30,
        max_steps: int = 10000,
        avoidance_stop_dist: float = 1.0,
        avoidance_clear_dist: float = 1.2,
        avoidance_timeout_steps: int = 75,
        avoidance_detour_dist: float = 1.0,
    ) -> None:
        self.world        = world
        self.walk_policy  = walk_policy
        self.goal_planner = goal_planner
        self.vlm_bridge   = vlm_bridge
        self.obstacle_map = obstacle_map
        self.overhead_cam = overhead_cam
        self.sim_fps      = sim_fps
        self.render_fps   = render_fps
        self.max_steps    = max_steps
        self.avoidance_stop_dist     = avoidance_stop_dist
        self.avoidance_clear_dist    = avoidance_clear_dist
        self.avoidance_timeout_steps = avoidance_timeout_steps
        self.avoidance_detour_dist   = avoidance_detour_dist

        self._art: "ArticulationView | None" = None
        self._joint_names: list[str] = []
        self._leg_indices:   np.ndarray | None = None
        self._upper_indices: np.ndarray | None = None
        self._upper_default_pos: np.ndarray | None = None
        self._upper_kps: np.ndarray | None = None
        self._upper_kds: np.ndarray | None = None
        self._ctrl_range: np.ndarray | None = None
        self._upper_ctrl_range: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Articulation setup
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Call once after world.reset() to resolve articulation joint indices."""
        from omni.isaac.core.articulations import ArticulationView

        self._art = ArticulationView(prim_paths_expr="/World/G1", name="g1_art")
        self.world.scene.add(self._art)
        self.world.reset()

        self._joint_names = list(self._art.joint_names)
        logger.info("G1 joint names: %s", self._joint_names)

        self._leg_indices   = self._resolve_indices(LEG_JOINT_NAMES)
        self._upper_indices = self._resolve_indices(UPPER_JOINT_NAMES)

        upper_defaults, upper_kps, upper_kds = [], [], []
        for jn in UPPER_JOINT_NAMES:
            default, kp, kd = UPPER_DEFAULTS.get(jn, (0.0, 40.0, 4.0))
            upper_defaults.append(default)
            upper_kps.append(kp)
            upper_kds.append(kd)
        self._upper_default_pos = np.array(upper_defaults, dtype=np.float32)
        self._upper_kps         = np.array(upper_kps,     dtype=np.float32)
        self._upper_kds         = np.array(upper_kds,     dtype=np.float32)

        from isaac_nav_demo.walk_policy.g1_walk_policy import G1WalkPolicy
        self._kps  = G1WalkPolicy.KPS
        self._kds  = G1WalkPolicy.KDS
        self._default_angles = G1WalkPolicy.DEFAULT_ANGLES.copy()
        self._decimation = max(
            1,
            int(round(G1WalkPolicy.CONTROL_DT * self.sim_fps)),
        )

        # Fetch actuator limits from the articulation
        limits = self._art.get_dof_limits()[0]    # (n_dof, 2)
        self._ctrl_range       = limits[self._leg_indices]
        self._upper_ctrl_range = limits[self._upper_indices]

        # Set initial pose
        self._apply_default_pose()
        self.walk_policy.reset()
        logger.info("IsaacNavigationSession initialised")

    def _resolve_indices(self, joint_names: list[str]) -> np.ndarray:
        indices = []
        for jn in joint_names:
            try:
                idx = self._joint_names.index(jn)
            except ValueError:
                logger.warning("Joint %r not found in articulation; using 0", jn)
                idx = 0
            indices.append(idx)
        return np.array(indices, dtype=np.intp)

    def _apply_default_pose(self) -> None:
        from isaac_nav_demo.walk_policy.g1_walk_policy import G1WalkPolicy

        n_dof = len(self._joint_names)
        qpos  = np.zeros(n_dof, dtype=np.float32)
        for i, idx in enumerate(self._leg_indices):
            qpos[idx] = G1WalkPolicy.DEFAULT_ANGLES[i]
        for i, idx in enumerate(self._upper_indices):
            qpos[idx] = self._upper_default_pos[i]

        self._art.set_joint_positions(qpos[np.newaxis, :])
        self._art.set_joint_velocities(np.zeros_like(qpos)[np.newaxis, :])

        pos = np.array([[0.5, 0.0, 0.793]])
        self._art.set_world_poses(positions=pos)

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def current_position(self) -> tuple[float, float]:
        pos, _ = self._art.get_world_poses()
        return (float(pos[0, 0]), float(pos[0, 1]))

    def current_yaw(self) -> float:
        _, quats = self._art.get_world_poses()
        return _quat_to_yaw(quats[0])

    # ------------------------------------------------------------------
    # Forward range sensing (replaces MuJoCo geom scan)
    # ------------------------------------------------------------------

    def _read_forward_range(
        self,
        robot_xy: tuple[float, float],
        yaw: float,
        cutoff: float = 4.0,
    ) -> float:
        """Raycast along robot's forward direction; return distance to nearest hit."""
        try:
            import carb
            from omni.physx import get_physx_scene_query_interface

            origin    = carb.Float3(robot_xy[0], robot_xy[1], 0.9)
            direction = carb.Float3(math.cos(yaw), math.sin(yaw), 0.0)
            result    = get_physx_scene_query_interface().raycast_closest(
                origin, direction, cutoff
            )
            return float(result.distance) if result.hit else cutoff
        except Exception:
            # Fallback: bounding-box scan against obstacle_map
            return self._bbox_range(robot_xy, yaw, cutoff)

    def _bbox_range(
        self,
        robot_xy: tuple[float, float],
        yaw: float,
        cutoff: float,
    ) -> float:
        robot = np.array(robot_xy, dtype=np.float64)
        fwd   = np.array([math.cos(yaw), math.sin(yaw)])
        cos_half = math.cos(math.radians(20.0))
        best  = cutoff
        for cx, cy, hx, hy in self.obstacle_map.values():
            obj = np.array([cx, cy])
            to_obj = obj - robot
            dist   = float(np.linalg.norm(to_obj))
            if dist < 0.5 or dist > cutoff:
                continue
            if float(np.dot(to_obj / dist, fwd)) >= cos_half:
                best = min(best, dist)
        return best

    # ------------------------------------------------------------------
    # Goal parsing
    # ------------------------------------------------------------------

    def parse_goal(self, command: str, debug_prefix: str | None = None) -> "Goal | None":
        robot_pos = self.current_position()
        robot_yaw = self.current_yaw()
        debug_img = (debug_prefix + "_vlm_scene.png") if debug_prefix else None
        return self.vlm_bridge.parse(
            command,
            world=self.world,
            overhead_cam=self.overhead_cam,
            robot_pos=robot_pos,
            robot_yaw=robot_yaw,
            debug_image_path=debug_img,
        )

    # ------------------------------------------------------------------
    # Detour
    # ------------------------------------------------------------------

    def compute_detour_goal(self, goal: "Goal") -> "Goal":
        from isaac_nav_demo.vlm.goal_parser import Goal as G
        current_pos = self.current_position()
        next_wp     = self.goal_planner.current_waypoint
        if next_wp is None:
            return goal
        dx = next_wp[0] - current_pos[0]
        dy = next_wp[1] - current_pos[1]
        dist = math.hypot(dx, dy)
        fwd  = np.array([dx / dist, dy / dist]) if dist > 1e-3 else np.array([1.0, 0.0])
        perp = np.array([-fwd[1], fwd[0]])
        detour = np.array(current_pos) + perp * self.avoidance_detour_dist
        remaining = list(self.goal_planner._waypoints[self.goal_planner._current_wp_idx:])
        new_wps   = [(float(detour[0]), float(detour[1]))] + remaining
        return G(
            target_name=goal.target_name,
            waypoints=new_wps,
            face_direction=goal.face_direction,
        )

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_face_yaw(goal: "Goal") -> float | None:
        if not goal.waypoints or goal.face_direction is None:
            return None
        gx, gy = goal.waypoints[-1]
        face_map = {"front": (0, 1), "back": (0, -1), "left": (1, 0), "right": (-1, 0)}
        direction = face_map.get(goal.face_direction)
        if direction is None:
            return None
        tx = gx + direction[0] * 0.8
        ty = gy + direction[1] * 0.8
        return float(math.atan2(ty - gy, tx - gx))

    def run_to_goal_with_renderer(
        self,
        goal: "Goal",
        command: str,
        video_renderer: "VideoRenderer",
        face_yaw_override: float | None = None,
    ) -> tuple[bool, bool]:
        from isaac_nav_demo.avoidance import AvoidanceStateMachine
        from isaac_nav_demo.walk_policy.g1_walk_policy import G1WalkPolicy

        face_yaw = face_yaw_override if face_yaw_override is not None else self._compute_face_yaw(goal)
        self.goal_planner.set_waypoints(goal.waypoints, face_yaw=face_yaw)

        avoidance = AvoidanceStateMachine(
            stop_dist=self.avoidance_stop_dist,
            clear_dist=self.avoidance_clear_dist,
            timeout_steps=self.avoidance_timeout_steps,
            detour_dist=self.avoidance_detour_dist,
        )

        target_positions  = self._default_angles.copy()
        velocity_command  = np.zeros(3, dtype=np.float32)
        reached           = False
        blocked           = False
        plan_result       = None
        steps_per_render  = max(1, self.sim_fps // self.render_fps)

        for step in range(self.max_steps):
            if step % self._decimation == 0:
                current_pos = self.current_position()
                current_yaw = self.current_yaw()

                range_val = self._read_forward_range(current_pos, current_yaw)
                reroute   = avoidance.step(range_val, current_pos, face_yaw)
                video_renderer.obstacle_banner = avoidance.banner

                if reroute:
                    reached = False
                    blocked = True
                    break

                if avoidance.is_navigating:
                    plan_result = self.goal_planner.compute_command(current_pos, current_yaw)
                    if plan_result.reached:
                        logger.info("Reached goal at step %d (dist=%.3f)", step, plan_result.distance)
                        reached = True
                        break
                    velocity_command = np.array(
                        [plan_result.vx, plan_result.vy, plan_result.vyaw], dtype=np.float32
                    )
                else:
                    velocity_command = np.zeros(3, dtype=np.float32)

                # --- Policy inference ---
                joint_pos = self._art.get_joint_positions()[0]
                joint_vel = self._art.get_joint_velocities()[0]
                _, quats  = self._art.get_world_poses()
                quat_wxyz = quats[0]

                dof_pos = joint_pos[self._leg_indices].astype(np.float32)
                dof_vel = joint_vel[self._leg_indices].astype(np.float32)

                # Angular velocity from articulation root
                ang_vel_raw  = self._art.get_angular_velocities()[0]
                angular_velocity = np.array(ang_vel_raw, dtype=np.float32)

                projected_gravity = G1WalkPolicy.compute_projected_gravity(
                    quat_wxyz.astype(np.float32)
                )
                target_positions = self.walk_policy.get_action(
                    projected_gravity=projected_gravity,
                    velocity_command=velocity_command,
                    dof_pos=dof_pos,
                    dof_vel=dof_vel,
                    angular_velocity=angular_velocity,
                )

            # --- PD control ---
            joint_pos = self._art.get_joint_positions()[0]
            joint_vel = self._art.get_joint_velocities()[0]

            dof_pos   = joint_pos[self._leg_indices].astype(np.float32)
            dof_vel   = joint_vel[self._leg_indices].astype(np.float32)
            torques   = self._kps * (target_positions - dof_pos) - self._kds * dof_vel
            torques   = np.clip(torques, self._ctrl_range[:, 0], self._ctrl_range[:, 1])

            upper_pos    = joint_pos[self._upper_indices].astype(np.float32)
            upper_vel    = joint_vel[self._upper_indices].astype(np.float32)
            upper_torques = (
                self._upper_kps * (self._upper_default_pos - upper_pos)
                - self._upper_kds * upper_vel
            )
            upper_torques = np.clip(
                upper_torques, self._upper_ctrl_range[:, 0], self._upper_ctrl_range[:, 1]
            )

            # Write all joint efforts
            n_dof       = len(self._joint_names)
            effort_all  = np.zeros(n_dof, dtype=np.float32)
            for i, idx in enumerate(self._leg_indices):
                effort_all[idx] = torques[i]
            for i, idx in enumerate(self._upper_indices):
                effort_all[idx] = upper_torques[i]
            self._art.set_joint_efforts(effort_all[np.newaxis, :])

            # Step physics
            self.world.step(render=False)

            # Render frame
            if step % steps_per_render == 0 and plan_result is not None:
                pos, _ = self._art.get_world_poses()
                robot_pos3 = pos[0]
                n_wps  = len(self.goal_planner._waypoints)
                wp_idx = self.goal_planner._current_wp_idx
                frame  = video_renderer.render_frame(
                    robot_pos=robot_pos3,
                    robot_yaw=current_yaw,
                    command=f"{command} [wp {min(wp_idx+1, n_wps)}/{n_wps}]",
                    distance=plan_result.distance,
                    goal_waypoints=list(self.goal_planner._waypoints),
                    current_wp_idx=wp_idx,
                )
                video_renderer.write_frame(frame)

        if not reached and not blocked:
            logger.warning("Did not reach goal within %d steps", self.max_steps)
        return reached, blocked

    def idle(
        self,
        duration_steps: int = 500,
        video_renderer: "VideoRenderer | None" = None,
        command: str = "",
        goal_waypoints: list | None = None,
    ) -> None:
        from isaac_nav_demo.walk_policy.g1_walk_policy import G1WalkPolicy

        zero_cmd = np.zeros(3, dtype=np.float32)
        target_positions = self._default_angles.copy()
        steps_per_render = max(1, self.sim_fps // self.render_fps)

        for step in range(duration_steps):
            if step % self._decimation == 0:
                joint_pos = self._art.get_joint_positions()[0]
                joint_vel = self._art.get_joint_velocities()[0]
                _, quats  = self._art.get_world_poses()

                dof_pos  = joint_pos[self._leg_indices].astype(np.float32)
                dof_vel  = joint_vel[self._leg_indices].astype(np.float32)
                ang_vel  = np.array(self._art.get_angular_velocities()[0], dtype=np.float32)
                quat     = quats[0].astype(np.float32)

                proj_grav = G1WalkPolicy.compute_projected_gravity(quat)
                target_positions = self.walk_policy.get_action(
                    projected_gravity=proj_grav,
                    velocity_command=zero_cmd,
                    dof_pos=dof_pos,
                    dof_vel=dof_vel,
                    angular_velocity=ang_vel,
                )

            joint_pos     = self._art.get_joint_positions()[0]
            joint_vel     = self._art.get_joint_velocities()[0]
            dof_pos       = joint_pos[self._leg_indices].astype(np.float32)
            dof_vel       = joint_vel[self._leg_indices].astype(np.float32)
            torques       = self._kps * (target_positions - dof_pos) - self._kds * dof_vel
            torques       = np.clip(torques, self._ctrl_range[:, 0], self._ctrl_range[:, 1])
            upper_pos     = joint_pos[self._upper_indices].astype(np.float32)
            upper_vel     = joint_vel[self._upper_indices].astype(np.float32)
            upper_torques = (
                self._upper_kps * (self._upper_default_pos - upper_pos)
                - self._upper_kds * upper_vel
            )
            upper_torques = np.clip(
                upper_torques, self._upper_ctrl_range[:, 0], self._upper_ctrl_range[:, 1]
            )

            n_dof      = len(self._joint_names)
            effort_all = np.zeros(n_dof, dtype=np.float32)
            for i, idx in enumerate(self._leg_indices):
                effort_all[idx] = torques[i]
            for i, idx in enumerate(self._upper_indices):
                effort_all[idx] = upper_torques[i]
            self._art.set_joint_efforts(effort_all[np.newaxis, :])

            self.world.step(render=False)

            if video_renderer is not None and step % steps_per_render == 0:
                pos, _ = self._art.get_world_poses()
                yaw    = _quat_to_yaw(self._art.get_world_poses()[1][0])
                frame  = video_renderer.render_frame(
                    robot_pos=pos[0],
                    robot_yaw=yaw,
                    command=command,
                    distance=0.0,
                    goal_waypoints=goal_waypoints,
                    current_wp_idx=0,
                )
                video_renderer.write_frame(frame)
