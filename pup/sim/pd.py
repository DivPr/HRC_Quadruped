"""A motor driver computes torque from position and velocity errors."""

import mujoco
import numpy as np


class PDController:
    """Joint proportional-derivative control with physical torque limits."""

    def __init__(self, kp: np.ndarray | float, kd: np.ndarray | float,
                 torque_limit: float = 20.0) -> None:
        """Store scalar or (12,) gains in Nm/rad and Nm/(rad/s), limit in Nm."""
        # ===== TODO(student): Store the PD gains and torque limit =====
        self.kp = kp
        self.kd = kd
        self.torque_limit = torque_limit
        # ===== end TODO =====

    def __call__(self, q: np.ndarray, qd: np.ndarray, q_des: np.ndarray,
                 qd_des: np.ndarray | None = None) -> np.ndarray:
        """Return (12,) clipped torques, Nm, for (12,) angles/rates in rad/rad/s."""
        # ===== TODO(student): Calculate and clip joint torques =====
        if qd_des is None:
            qd_des = np.zeros_like(qd)
        torque = self.kp * (q_des - q) + self.kd * (qd_des - qd)
        torque = np.clip(torque, -self.torque_limit, self.torque_limit)
        return torque
        # ===== end TODO =====


def joint_state(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    """Return copies of actuated q (12,), rad, and qd (12,), rad/s; root is free."""
    # ===== TODO(student): Slice joint positions and velocities past the free root =====
    q = data.qpos[7:].copy()
    qd = data.qvel[6:].copy()
    return q, qd
    # ===== end TODO =====
