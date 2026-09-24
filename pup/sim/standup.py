"""Raise Pup smoothly from crouch before asking it to walk."""

from contextlib import nullcontext

import mujoco
import numpy as np

from pup.sim.pd import PDController, joint_state
from pup.sim.viewer import load_scene, reset_to_keyframe


def stand_up(duration_s: float = 3.0, headless: bool = True,
             kp: float = 10.0, kd: float = 1.0) -> dict:  # TODO(student): tune
    """Return final_height (m), max_roll/max_pitch (rad), and fell (bool).

    Interpolate (12,) target angles from crouch to home in one second;
    then hold until duration_s.

    The default gains above are the spring-2026 quadruped's (kp=10). Pup is
    heavier -- run it, watch it sag, and tune them (Stage 1, task 3). The
    test reads whatever defaults you leave in the signature.
    """
    # ===== TODO(student): Interpolate from crouch to home and measure stability =====
    raise NotImplementedError(
        "Stage 1: Interpolate from crouch to home and measure stability. See docs/01_mujoco_and_pd.md")
    # ===== end TODO =====
