"""A motor driver computes torque from position and velocity errors."""

import mujoco
import numpy as np


class PDController:
    """Joint proportional-derivative control with physical torque limits."""

    def __init__(self, kp: np.ndarray | float, kd: np.ndarray | float,
                 torque_limit: float = 20.0) -> None:
        """Store scalar or (12,) gains in Nm/rad and Nm/(rad/s), limit in Nm."""
        # ===== TODO(student): Store the PD gains and torque limit =====
        raise NotImplementedError(
            "Stage 1: Store the PD gains and torque limit. See docs/01_mujoco_and_pd.md")
        # ===== end TODO =====

    def __call__(self, q: np.ndarray, qd: np.ndarray, q_des: np.ndarray,
                 qd_des: np.ndarray | None = None) -> np.ndarray:
        """Return (12,) clipped torques, Nm, for (12,) angles/rates in rad/rad/s."""
        # ===== TODO(student): Calculate and clip joint torques =====
        raise NotImplementedError(
            "Stage 1: Calculate and clip joint torques. See docs/01_mujoco_and_pd.md")
        # ===== end TODO =====


def joint_state(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    """Return copies of actuated q (12,), rad, and qd (12,), rad/s; root is free."""
    # ===== TODO(student): Slice joint positions and velocities past the free root =====
    raise NotImplementedError(
        "Stage 1: Slice joint positions and velocities past the free root. See docs/01_mujoco_and_pd.md")
    # ===== end TODO =====
