import mujoco
import numpy as np

from pup.envs.constants import MJX_SCENE
from pup.sim.pd import PDController, joint_state
from pup.sim.viewer import load_scene, run


def test_pd():
    zeros = np.zeros(12)
    controller = PDController(25, 0.5)
    np.testing.assert_equal(controller(zeros, zeros, zeros), zeros)
    np.testing.assert_equal(controller(zeros, zeros, np.ones(12)), 20)
    np.testing.assert_equal(controller(zeros, np.ones(12), zeros), -0.5)
    np.testing.assert_equal(controller(np.ones(12), zeros, zeros), -20)
    np.testing.assert_equal(controller(zeros, zeros, zeros, np.ones(12)), 0.5)
    gains = PDController(np.arange(12), np.ones(12))
    np.testing.assert_equal(gains(zeros, zeros, np.ones(12)), np.arange(12))


def test_joint_state(mj_model, mj_data_home):
    q, qd = joint_state(mj_model, mj_data_home)
    np.testing.assert_equal(q, mj_data_home.qpos[7:])
    assert q.shape == qd.shape == (12,)
    q[:] = 99
    assert mj_data_home.qpos[7] != 99


def test_matches_position_actuator():
    model, data = load_scene(MJX_SCENE)
    controller = PDController(25, 0.5)
    for index in range(200):
        data.ctrl[:] = model.keyframe("home").qpos[7:] + 0.05*np.sin(index/20)
        mujoco.mj_forward(model, data)
        expected = controller(data.qpos[7:], data.qvel[6:], data.ctrl)
        np.testing.assert_allclose(data.actuator_force, expected, atol=1e-3)
        mujoco.mj_step(model, data)


def test_provided_headless_loop():
    from pup.envs.constants import SCENE
    _, data = run(SCENE, lambda model, data, t: np.zeros(12),
                  duration_s=0.02, headless=True, realtime=False)
    assert data.time >= 0.02
