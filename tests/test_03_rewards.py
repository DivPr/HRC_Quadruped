import jax.numpy as jnp
import numpy as np

from pup.envs import rewards


def test_linear_tracking():
    command = jnp.array([1., 0.5, 0.2])
    assert rewards.reward_tracking_lin_vel(command, command) == 1
    np.testing.assert_allclose(rewards.reward_tracking_lin_vel(command, jnp.zeros(3)),
                               np.exp(-1.25/0.25), rtol=1e-6)


def test_angular_tracking():
    command = jnp.array([1., 0.5, 0.2])
    assert rewards.reward_tracking_ang_vel(command, command) == 1
    np.testing.assert_allclose(rewards.reward_tracking_ang_vel(command, jnp.zeros(3)),
                               np.exp(-0.04/0.25), rtol=1e-6)


def test_action_rate():
    ones, zeros = jnp.ones(12), jnp.zeros(12)
    assert rewards.cost_action_rate(ones, ones, ones) == 0
    assert rewards.cost_action_rate(ones, zeros, zeros) == 24
    assert rewards.cost_action_rate(2*ones, ones, zeros) == 12


def test_provided_terms():
    assert rewards.cost_orientation(jnp.array([0., 0, -1])) == 0
    assert rewards.cost_lin_vel_z(jnp.array([1., 2, 3])) == 9
    assert rewards.cost_ang_vel_xy(jnp.array([1., 2, 3])) == 5
    assert rewards.cost_torques(jnp.ones(12)) == 12
    # Playground's per-joint weights [1, 1, 0.1] x 4 sum to 8.4, not 12.
    np.testing.assert_allclose(
        rewards.cost_joint_pose_deviation(jnp.ones(12), jnp.zeros(12)), 8.4, rtol=1e-6)
    assert rewards.cost_stand_still(jnp.zeros(3), jnp.ones(12), jnp.zeros(12)) == 12
    assert rewards.cost_termination(jnp.bool_(True)) == 1
    np.testing.assert_allclose(rewards.reward_feet_air_time(jnp.ones(4)*0.2,
                               jnp.ones(4, bool), jnp.ones(3)), 0.4)
