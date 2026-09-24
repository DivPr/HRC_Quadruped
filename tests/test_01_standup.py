"""Stage 1: stand up from a crouch and hold, without oscillating or falling."""

import inspect

import mujoco
import numpy as np
import pytest

from pup.sim.pd import PDController
from pup.sim.standup import stand_up

# Generous window; the doc suggests kp=40, kd=1.0, which sits comfortably inside.
KP_RANGE = (35.0, 80.0)
KD_RANGE = (0.3, 2.5)


def _check(metrics: dict, home_height: float) -> None:
    """Assert one stand_up() result meets the Stage 1 acceptance criteria."""
    assert metrics["fell"] is False
    assert abs(metrics["final_height"] - home_height) < 0.03
    assert metrics["max_roll"] < 0.15
    assert metrics["max_pitch"] < 0.15


@pytest.mark.parametrize("kp,kd", [(40, 1.0), (50, 0.5), (60, 1.5)])
def test_stand_up(mj_model, kp, kd):
    """The reference behaviour works across the whole accepted gain window."""
    _check(stand_up(kp=kp, kd=kd), mj_model.keyframe("home").qpos[2])


def test_default_gains_are_tuned(mj_model):
    """Task 3: the gains you leave as stand_up's defaults must themselves work."""
    metrics = stand_up()  # runs first, so an unwritten stand_up reads as NOT STARTED
    signature = inspect.signature(stand_up)
    kp = float(signature.parameters["kp"].default)
    kd = float(signature.parameters["kd"].default)
    assert KP_RANGE[0] <= kp <= KP_RANGE[1], f"kp={kp} outside {KP_RANGE}"
    assert KD_RANGE[0] <= kd <= KD_RANGE[1], f"kd={kd} outside {KD_RANGE}"
    _check(metrics, mj_model.keyframe("home").qpos[2])


def test_no_oscillation_after_settling(mj_model):
    """Once the ramp is over the trunk height must be steady, not ringing."""
    metrics = stand_up(duration_s=4.0)
    assert metrics["fell"] is False
    heights = []
    data = mujoco.MjData(mj_model)
    mujoco.mj_resetDataKeyframe(mj_model, data, mj_model.keyframe("home").id)
    controller = PDController(40, 1.0)
    home = mj_model.keyframe("home").qpos
    for _ in range(1250):  # 5 s at 250 Hz
        data.ctrl[:] = controller(data.qpos[7:], data.qvel[6:], home[7:])
        mujoco.mj_step(mj_model, data)
        heights.append(data.qpos[2])
    assert np.max(np.abs(np.array(heights) - home[2])) < 0.03
    assert np.std(np.array(heights)[-250:]) < 2e-3, "trunk height still oscillating"
