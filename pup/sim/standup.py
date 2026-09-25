"""Raise Pup smoothly from crouch before asking it to walk."""

from contextlib import nullcontext
import time

import mujoco
import numpy as np

from pup.sim import viewer
from pup.sim.pd import PDController, joint_state
from pup.sim.viewer import load_scene, reset_to_keyframe


def stand_up(duration_s: float = 3.0, headless: bool = True,
             kp: float = 40.0, kd: float = 1.0) -> dict:  # TODO(student): tune
    """Return final_height (m), max_roll/max_pitch (rad), and fell (bool).

    Interpolate (12,) target angles from crouch to home in one second;
    then hold until duration_s.

    The default gains above are the spring-2026 quadruped's (kp=10). Pup is
    heavier -- run it, watch it sag, and tune them (Stage 1, task 3). The
    test reads whatever defaults you leave in the signature.
    """
    # ===== TODO(student): Interpolate from crouch to home and measure stability =====
    model, data = load_scene()

    reset_to_keyframe(model, data, "crouch")

    q_crouch = model.keyframe("crouch").qpos[7:].copy()
    q_home = model.keyframe("home").qpos[7:].copy()

    controller = PDController(kp, kd)

    max_roll = 0.0
    max_pitch = 0.0
    fell = False

    if headless:
        context = nullcontext(None)
    else:
        from mujoco.viewer import launch_passive
        context = launch_passive(model, data)

    with context as viewer:
        while data.time < duration_s:
            started = time.perf_counter()

            alpha = np.clip(data.time / 1.0, 0.0, 1.0)
            q_des = (1.0 - alpha) * q_crouch + alpha * q_home

            q, qd = joint_state(model, data)
            torque = controller(q, qd, q_des)
            data.ctrl[:] = torque

            mujoco.mj_step(model, data)

            w, x, y, z = data.qpos[3:7]

            roll = np.arctan2(
                2 * (w * x + y * z),
                1 - 2 * (x * x + y * y)
            )

            pitch = np.arcsin(
                np.clip(
                    2 * (w * y - z * x),
                    -1.0,
                    1.0
                )
            )

            max_roll = max(max_roll, abs(roll))
            max_pitch = max(max_pitch, abs(pitch))

            if data.qpos[2] < 0.12 or not np.all(np.isfinite(data.qpos)):
                fell = True

            if viewer is not None:
                viewer.sync()
                elapsed = time.perf_counter() - started
                time.sleep(max(0.0, model.opt.timestep - elapsed))

    return {
        "final_height": float(data.qpos[2]),
        "max_roll": float(max_roll),
        "max_pitch": float(max_pitch),
        "fell": bool(fell),
    }
    # ===== end TODO =====
