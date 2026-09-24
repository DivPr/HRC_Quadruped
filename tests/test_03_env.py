import time

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from pup.envs.randomize import domain_randomize


def test_reset_observation(env, rng):
    state = jax.jit(env.reset)(rng)
    assert state.obs.shape == (45,)
    assert np.isfinite(state.obs).all()
    assert set(state.metrics) == set(env._config.reward_config.scales)
    np.testing.assert_allclose(state.obs[3:6], [0, 0, -1], atol=1e-6)
    np.testing.assert_array_equal(state.obs[6:9], state.info["command"])
    np.testing.assert_array_equal(state.obs[9:21], state.data.qpos[7:]-env._default_pose)
    np.testing.assert_array_equal(state.obs[21:33], state.data.qvel[6:])


def test_step_and_stand(env, rng):
    state = jax.jit(env.reset)(rng)
    tree = jax.tree_util.tree_structure(state)
    step = jax.jit(env.step)
    for _ in range(50):
        state = step(state, jnp.zeros(12))
        assert state.done == 0
        assert np.isfinite(state.reward)
        assert abs(float(state.data.qpos[2]) - env._home[2]) < 0.05
    assert jax.tree_util.tree_structure(state) == tree


def test_upside_down(env, rng):
    state = jax.jit(env.reset)(rng)
    data = state.data.replace(qpos=state.data.qpos.at[3:7].set(jnp.array([0., 1, 0, 0])))
    data = jax.jit(mjx.forward)(env.mjx_model, data)
    assert env._get_termination(data)
    result = jax.jit(env.step)(state.replace(data=data), jnp.zeros(12))
    assert result.done == 1
    assert env._get_termination(data.replace(qpos=data.qpos.at[2].set(jnp.nan)))


def test_batched_cpu(env, rng):
    started = time.perf_counter()
    state = jax.jit(jax.vmap(env.reset))(jax.random.split(rng, 16))
    step = jax.jit(jax.vmap(env.step))
    for _ in range(10):
        state = step(state, jnp.zeros((16, 12)))
    state.reward.block_until_ready()
    elapsed = time.perf_counter() - started
    print(f"16 environments, reset + ten steps including compilation: {elapsed:.2f}s")
    assert elapsed < 90
    assert state.obs.shape == (16, 45)
    assert np.isfinite(state.obs).all()


def test_commands(env, rng):
    commands = jax.jit(jax.vmap(env.sample_command))(jax.random.split(rng, 10000))
    assert np.all(commands >= np.array(env._config.command_config.minimum))
    assert np.all(commands <= np.array(env._config.command_config.maximum))
    assert 0.08 < np.mean(np.all(commands == 0, axis=1)) < 0.12


def test_randomization(env, rng):
    model, axes = domain_randomize(env.mjx_model, jax.random.split(rng, 8))
    assert axes.body_mass == 0
    assert model.body_mass.shape == (8, env.mj_model.nbody)
    assert np.ptp(model.body_mass[:, 1]) > 0
    floor = env.mj_model.geom("floor").id
    assert np.ptp(model.geom_friction[:, floor, 0]) > 0, "floor friction not randomized"
    # <position> actuators keep biasprm[1] == -gainprm[0]; scaling must preserve it.
    np.testing.assert_allclose(model.actuator_gainprm[:, :, 0],
                               -model.actuator_biasprm[:, :, 1])


def test_registered_with_playground():
    """`registry.load("PupJoystickFlat")` must work, as Playground's tooling expects."""
    from mujoco_playground import registry

    import pup.envs  # noqa: F401  -- importing is what registers the environment

    config = registry.get_default_config("PupJoystickFlat")
    assert config.ctrl_dt == 0.02
    assert config.sim_dt == 0.004
    loaded = registry.load("PupJoystickFlat")
    assert loaded.action_size == 12
    assert loaded.dt == config.ctrl_dt
