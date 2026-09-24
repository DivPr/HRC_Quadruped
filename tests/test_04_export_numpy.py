"""Stage 4: the exported .npz must reproduce Brax inference bit-for-bit."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import networks as ppo_networks

from pup.envs.constants import ACTION_SCALE, CHECKPOINTS, DEFAULT_POSE
from pup.policy.mlp_numpy import NumpyPolicy
from pup.train.export import export_policy


@pytest.fixture(scope="module")
def brax_policy():
    """Build an untrained-but-randomized PPO policy plus a populated normalizer."""
    network = ppo_networks.make_ppo_networks(
        observation_size=45, action_size=12,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=(32, 32, 32), value_hidden_layer_sizes=(32, 32))
    params = network.policy_network.init(jax.random.PRNGKey(1))
    normalizer = running_statistics.init_state(specs.Array((45,), jnp.dtype("float32")))
    samples = jax.random.normal(jax.random.PRNGKey(2), (512, 45)) * 2.0 + 0.5
    normalizer = running_statistics.update(normalizer, samples)
    inference = ppo_networks.make_inference_fn(network)
    return network, params, normalizer, inference


def test_export_matches_jax_inference(brax_policy, tmp_path):
    _, params, normalizer, inference = brax_policy
    path = export_policy(params, normalizer, tmp_path / "policy.npz")
    policy = NumpyPolicy.load(path)
    jax_policy = jax.jit(inference((normalizer, params), deterministic=True))
    observations = np.asarray(jax.random.normal(jax.random.PRNGKey(3), (256, 45)))
    key = jax.random.PRNGKey(0)
    for obs in observations:
        expected = np.asarray(jax_policy(jnp.asarray(obs), key)[0])
        actual = policy(obs)
        assert actual.shape == (12,)
        assert np.max(np.abs(actual - expected)) < 1e-4


def test_joint_targets_and_metadata(brax_policy, tmp_path):
    _, params, normalizer, _ = brax_policy
    policy = NumpyPolicy.load(export_policy(params, normalizer, tmp_path / "p.npz"))
    assert (policy.obs_size, policy.action_size) == (45, 12)
    np.testing.assert_allclose(policy.default_pose, DEFAULT_POSE, atol=1e-6)
    obs = np.zeros(45)
    np.testing.assert_allclose(policy.joint_targets(obs),
                               DEFAULT_POSE + ACTION_SCALE * policy(obs), atol=1e-9)
    assert np.all(np.abs(policy(obs)) <= 1.0)


def test_reference_checkpoint_loads():
    """The Stage 5 escape hatch must be a well-formed, loadable policy."""
    path = CHECKPOINTS / "pup_joystick_flat_reference.npz"
    assert path.exists(), "reference checkpoint missing; see checkpoints/README.md"
    policy = NumpyPolicy.load(path)
    assert (policy.obs_size, policy.action_size) == (45, 12)
    assert policy(np.zeros(45)).shape == (12,)
    assert policy.joint_targets(np.zeros(45)).shape == (12,)
