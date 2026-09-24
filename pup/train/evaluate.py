"""Measure body-frame command tracking using privileged simulator sensors."""

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from pup.envs.math_utils import get_sensor_data, quat_inv, rotate

COMMANDS = ((0.5, 0., 0.), (1., 0., 0.), (0., 0.5, 0.), (0., 0., 1.), (0., 0., 0.))


def summarize_tracking(command: Sequence[float], velocities: np.ndarray,
                         fell: np.ndarray, lengths: np.ndarray) -> dict:
    """Summarize (N,3) vx/vy/yaw samples, episode fall flags and control-step lengths."""
    error = np.abs(velocities - np.asarray(command))
    return dict(command=list(command), mean_abs_error=error.mean(axis=0).tolist(),
                mean_abs_vy=float(np.abs(velocities[:, 1]).mean()),
                mean_speed=float(np.linalg.norm(velocities[:, :2], axis=1).mean()),
                fall_rate=float(np.mean(fell)), mean_episode_length=float(np.mean(lengths)))


def walking_passes(results: dict) -> bool:
    """Apply the spec's forward, turning and stationary acceptance thresholds."""
    entries = {tuple(row["command"]): row for row in results["commands"]}
    if not all(key in entries for key in [(1., 0., 0.), (0., 0., 1.), (0., 0., 0.)]):
        return False
    forward, turn, stand = (entries[key] for key in [(1., 0., 0.), (0., 0., 1.), (0., 0., 0.)])
    return bool(forward["mean_abs_error"][0] < 0.2 and forward["mean_abs_vy"] < 0.1
                and forward["fall_rate"] == 0 and turn["mean_abs_error"][2] < 0.3
                and stand["mean_speed"] < 0.1)


def evaluate(env, inference_fn: Callable, params,
             commands: Sequence = COMMANDS, n_episodes: int = 5, seed: int = 0) -> dict:
    """Roll deterministic policies; report per-command errors (m/s, m/s, rad/s).

    Each episode ends on first fall or env.episode_length. Fixed commands disable
    random resampling. All live samples, including the startup transient, count.
    """
    policy = inference_fn(params, deterministic=True)
    horizon = env._config.episode_length

    @jax.jit
    def rollout(key, command):
        state = env.reset(key)
        info = {**state.info, "command": command, "fixed_command": jnp.bool_(True)}
        state = state.replace(info=info, obs=env._get_obs(state.data, info))
        def advance(state, _):
            active = state.done == 0
            action, _ = policy(state.obs, state.info["rng"])
            state = jax.lax.cond(active, lambda s: env.step(s, action), lambda s: s, state)
            velocity = get_sensor_data(env.mj_model, state.data, "global_linvel")
            local = rotate(velocity, quat_inv(state.data.qpos[3:7]))
            yaw = get_sensor_data(env.mj_model, state.data, "gyro")[2]
            return state, (jnp.array([local[0], local[1], yaw]), active)
        state, (velocities, active) = jax.lax.scan(advance, state, None, length=horizon)
        return velocities, active, state.done

    results = []
    keys = jax.random.split(jax.random.PRNGKey(seed), len(commands)*n_episodes)
    for index, command in enumerate(commands):
        samples, falls, lengths = [], [], []
        for episode in range(n_episodes):
            velocity, active, done = rollout(keys[index*n_episodes+episode], jnp.array(command))
            active = np.asarray(active, bool)
            samples.append(np.asarray(velocity)[active])
            falls.append(float(done))
            lengths.append(int(active.sum()))
        results.append(summarize_tracking(command, np.concatenate(samples),
                                          np.asarray(falls), np.asarray(lengths)))
    result = dict(seed=seed, n_episodes=n_episodes, commands=results)
    result["walking_passes"] = walking_passes(result)
    return result
