"""Verified Playground Go1 PPO parameters, with small CPU and T4 modes."""

from mujoco_playground.config import locomotion_params


def full() -> dict:
    """Return installed Go1 settings: 200M steps, 8192 environments, large MLPs."""
    config = locomotion_params.brax_ppo_config("Go1JoystickFlatTerrain").to_dict()
    # Pup deliberately has a flat observation, without an asymmetric critic.
    config["network_factory"]["value_obs_key"] = "state"
    return config


def cpu_smoke() -> dict:
    """Return 20k steps across eight robots; proves training, not walking."""
    config = full()
    config.update(num_timesteps=20_000, num_envs=8, batch_size=8,
                  num_minibatches=1, num_updates_per_batch=1, unroll_length=10,
                  num_evals=2, num_eval_envs=8, episode_length=100,
                  num_resets_per_eval=0)
    config["network_factory"].update(policy_hidden_layer_sizes=(32, 32),
                                      value_hidden_layer_sizes=(32, 32))
    return config


def t4_fast() -> dict:
    """Return a 30M-step fallback; T4 runtime must be measured before release."""
    config = full()
    config.update(num_timesteps=30_000_000, num_envs=2048, num_evals=5,
                  num_resets_per_eval=0, num_eval_envs=32)
    return config


def cpu_reference() -> dict:
    """Return the laptop-CPU recipe used to train the committed reference policy.

    Identical to :func:`full` apart from a smaller environment batch (2048 instead
    of 8192, which Brax compensates for by collecting four unroll rounds per
    training step) and 60M instead of 200M environment steps.
    """
    config = full()
    config.update(num_timesteps=60_000_000, num_envs=2048, num_evals=20,
                  num_resets_per_eval=0, num_eval_envs=64)
    return config
