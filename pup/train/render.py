"""Headless rendering. Set MUJOCO_GL before importing MuJoCo on Linux."""

from pathlib import Path

import jax
import jax.numpy as jnp
import mediapy
import mujoco
import numpy as np
from PIL import Image


def render_rollout(env, inference_fn, params, out_path: str | Path,
                    duration_s: float = 4.0, seed: int = 0) -> None:
    """Render deterministic (0.5,0,0) tracking to GIF or MP4 at 25 fps."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = jax.jit(env.reset)(jax.random.PRNGKey(seed))
    info = {**state.info, "command": jnp.array([0.5, 0., 0.]), "fixed_command": jnp.bool_(True)}
    state = state.replace(info=info, obs=env._get_obs(state.data, info))
    policy = jax.jit(inference_fn(params, deterministic=True))
    step = jax.jit(env.step)
    data = mujoco.MjData(env.mj_model)
    frames = []
    with mujoco.Renderer(env.mj_model, height=360, width=480) as renderer:
        for index in range(round(duration_s/env.dt)):
            action, _ = policy(state.obs, state.info["rng"])
            state = step(state, action)
            if index % 2 == 0:
                data.qpos[:] = np.asarray(state.data.qpos)
                data.qvel[:] = np.asarray(state.data.qvel)
                mujoco.mj_forward(env.mj_model, data)
                camera = mujoco.MjvCamera()
                camera.lookat[:] = data.qpos[:3]
                camera.distance, camera.azimuth, camera.elevation = 1.25, 135, -20
                renderer.update_scene(data, camera=camera)
                frames.append(renderer.render().copy())
            if state.done:
                break
    if path.suffix == ".gif":
        images = [Image.fromarray(frame) for frame in frames]
        images[0].save(path, save_all=True, append_images=images[1:], duration=40, loop=0)
    else:
        mediapy.write_video(str(path), frames, fps=25)
