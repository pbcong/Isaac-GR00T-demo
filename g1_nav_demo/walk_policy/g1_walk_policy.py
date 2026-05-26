import numpy as np
import torch


class G1WalkPolicy:
    """
    Wraps the unitree_rl_gym pre-trained G1 walking policy for MuJoCo simulation.

    Observation vector (47 dims):
        [0:3]   angular velocity * ang_vel_scale
        [3:6]   projected gravity (body-frame gravity direction)
        [6:9]   velocity command * cmd_scale
        [9:21]  (dof_pos - default_angles) * dof_pos_scale
        [21:33] dof_vel * dof_vel_scale
        [33:45] last action
        [45:47] sin(phase), cos(phase)

    Action space (12 dims):
        target_dof_pos = action * action_scale + default_angles

    Joint order (per leg, left then right):
        hip_pitch, hip_yaw, hip_roll, knee, ankle_pitch, ankle_roll
    """

    DEFAULT_ANGLES = np.array(
        [
            -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
            -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
        ],
        dtype=np.float32,
    )
    NUM_ACTIONS = 12
    NUM_OBS = 47

    ANG_VEL_SCALE = 0.25
    DOF_POS_SCALE = 1.0
    DOF_VEL_SCALE = 0.05
    ACTION_SCALE = 0.25
    CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)

    GAIT_PERIOD = 0.8

    CONTROL_DT = 0.02  # 50 Hz

    KPS = np.array(
        [100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40],
        dtype=np.float32,
    )
    KDS = np.array(
        [2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2],
        dtype=np.float32,
    )

    def __init__(self, policy_path: str, device: str = "cpu"):
        self.policy = torch.jit.load(policy_path, map_location=device)
        self.policy.eval()
        for m in self.policy.modules():
            if hasattr(m, "flatten_parameters"):
                m.flatten_parameters()
        self.device = device

        self._last_action = np.zeros(self.NUM_ACTIONS, dtype=np.float32)
        self._phase_time = 0.0

    def reset(self) -> None:
        self._last_action = np.zeros(self.NUM_ACTIONS, dtype=np.float32)
        self._phase_time = 0.0

    @staticmethod
    def compute_projected_gravity(quaternion: np.ndarray) -> np.ndarray:
        """Convert quaternion (w, x, y, z) to projected gravity vector."""
        qw, qx, qy, qz = quaternion
        gravity = np.zeros(3, dtype=np.float32)
        gravity[0] = 2.0 * (-qz * qx + qw * qy)
        gravity[1] = -2.0 * (qz * qy + qw * qx)
        gravity[2] = 1.0 - 2.0 * (qw * qw + qz * qz)
        return gravity

    def get_action(
        self,
        projected_gravity: np.ndarray,
        velocity_command: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        angular_velocity: np.ndarray | None = None,
        quaternion: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute target joint positions from policy inference.

        Args:
            projected_gravity: 3-dim gravity vector in body frame.
                               If None, computed from *quaternion*.
            velocity_command: 3-dim command [lin_vel_x, lin_vel_y, ang_vel_yaw].
            dof_pos: 12-dim current joint positions (rad).
            dof_vel: 12-dim current joint velocities (rad/s).
            angular_velocity: 3-dim body angular velocity (rad/s).
                              If None, computed from quaternion derivative.
            quaternion: 4-dim orientation quaternion (w, x, y, z).
                        Only used if angular_velocity or projected_gravity is None.

        Returns:
            12-dim target joint positions (rad) for PD controllers.
        """
        if projected_gravity is None:
            if quaternion is None:
                raise ValueError(
                    "Either projected_gravity or quaternion must be provided"
                )
            projected_gravity = self.compute_projected_gravity(quaternion)

        qj = (dof_pos - self.DEFAULT_ANGLES) * self.DOF_POS_SCALE
        dqj = dof_vel * self.DOF_VEL_SCALE

        obs = np.zeros(self.NUM_OBS, dtype=np.float32)

        if angular_velocity is not None:
            omega_scaled = angular_velocity * self.ANG_VEL_SCALE
        else:
            omega_scaled = np.zeros(3, dtype=np.float32)

        phase = (self._phase_time % self.GAIT_PERIOD) / self.GAIT_PERIOD
        sin_phase = np.float32(np.sin(2.0 * np.pi * phase))
        cos_phase = np.float32(np.cos(2.0 * np.pi * phase))

        obs[0:3] = omega_scaled
        obs[3:6] = projected_gravity
        obs[6:9] = velocity_command * self.CMD_SCALE
        obs[9 : 9 + self.NUM_ACTIONS] = qj
        obs[9 + self.NUM_ACTIONS : 9 + 2 * self.NUM_ACTIONS] = dqj
        obs[9 + 2 * self.NUM_ACTIONS : 9 + 3 * self.NUM_ACTIONS] = self._last_action
        obs[9 + 3 * self.NUM_ACTIONS : 9 + 3 * self.NUM_ACTIONS + 2] = [
            sin_phase,
            cos_phase,
        ]

        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.policy(obs_tensor).cpu().numpy().squeeze()

        self._last_action = action.copy()

        target_dof_pos = action * self.ACTION_SCALE + self.DEFAULT_ANGLES

        self._phase_time += self.CONTROL_DT

        return target_dof_pos