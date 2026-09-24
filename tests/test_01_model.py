import mujoco
import numpy as np
from mujoco import mjx

from pup.envs.constants import CONTACT_SENSORS, MJX_SCENE, SENSOR_DIMS


def test_model(mj_model):
    assert (mj_model.nq, mj_model.nv, mj_model.nu) == (19, 18, 12)
    assert 11 < mj_model.body_mass.sum() < 13
    assert mj_model.nmesh == 0
    for name, dimension in SENSOR_DIMS.items():
        assert mj_model.sensor(name).dim == dimension
    assert mj_model.keyframe("home").qpos[2] > 0.25
    mjx.put_model(mj_model, impl="jax")
    mjx.put_model(mujoco.MjModel.from_xml_path(str(MJX_SCENE)), impl="jax")


def test_only_foot_floor_collisions(mj_model, mj_data_home):
    # Inspect collision masks, then read actual contacts exclusively via sensors.
    for index in range(mj_model.ngeom):
        name = mj_model.geom(index).name
        if name != "floor" and not name.endswith("_foot"):
            assert mj_model.geom_contype[index] == mj_model.geom_conaffinity[index] == 0
    from pup.sim.pd import PDController
    controller = PDController(40, 1)
    for _ in range(100):
        mj_data_home.ctrl[:] = controller(mj_data_home.qpos[7:], mj_data_home.qvel[6:],
                                          mj_model.keyframe("home").qpos[7:])
        mujoco.mj_step(mj_model, mj_data_home)
    assert sum(float(mj_data_home.sensor(name).data[0]) for name in CONTACT_SENSORS) == 4
    assert np.isfinite(mj_data_home.qpos).all()


def test_constants_agree_with_config():
    """ACTION_SCALE and DEFAULT_POSE are duplicated for the numpy side; keep them in sync."""
    import numpy as np

    from pup.envs.config import default_config
    from pup.envs.constants import ACTION_SCALE, DEFAULT_POSE, JOINT_NAMES, OBS_LAYOUT
    from pup.sim.viewer import load_scene

    assert ACTION_SCALE == default_config().action_scale
    model, _ = load_scene()
    np.testing.assert_allclose(DEFAULT_POSE, model.keyframe("home").qpos[7:], atol=1e-9)
    assert JOINT_NAMES == tuple(model.joint(i).name for i in range(1, model.njnt))
    widths = [int(part.split("[")[1].rstrip("]")) for part in OBS_LAYOUT.split("|")]
    assert sum(widths) == 45
