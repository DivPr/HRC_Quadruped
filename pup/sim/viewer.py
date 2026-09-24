"""Provided: load a scene and run a control loop, with or without a window.

This is the harness for Stage 1. You write `controller_fn`; this file handles
loading, keyframes, the passive viewer, and real-time pacing.

    from pup.sim.viewer import run
    run(SCENE, lambda model, data, t: my_torques, duration_s=5.0)

**macOS:** the passive viewer needs `mjpython`, not `python`, because macOS
requires GUI event loops on the main thread:

    uv run mjpython -c "from pup.sim.viewer import run; ..."

Headless runs (everything the tests do) use plain `python` everywhere.
"""

import time
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path

import mujoco
import numpy as np

from pup.envs.constants import SCENE


def reset_to_keyframe(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> None:
    """Reset time and state to a named keyframe (`home` or `crouch`), then refresh sensors.

    `mj_forward` is what makes `data.sensor(...)` valid immediately after the
    reset; without it you read whatever was in the buffer before.
    """
    mujoco.mj_resetDataKeyframe(model, data, model.keyframe(name).id)
    mujoco.mj_forward(model, data)


def load_scene(path: str | Path = SCENE) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load a scene XML and return `(model, data)` positioned at the `home` keyframe.

    Args:
      path: scene XML. Defaults to `scene_flat.xml` (torque `<motor>` actuators);
        pass `MJX_SCENE` for the `<position>`-actuator version.
    """
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    reset_to_keyframe(model, data, "home")
    return model, data


def run(scene_path: str | Path, controller_fn: Callable,
        duration_s: float | None = None, headless: bool = False,
        realtime: bool = True) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Step a scene under a controller and return the final `(model, data)`.

    Args:
      scene_path: scene XML to load.
      controller_fn: `f(model, data, t_seconds) -> np.ndarray` of shape (12,).
        In `scene_flat.xml` the return value is **torque in Nm**; in
        `scene_flat_mjx.xml` it is a **target angle in rad**.
      duration_s: simulated seconds to run, or None to run until the window is
        closed (not allowed when `headless`).
      headless: skip the viewer entirely. What the tests use.
      realtime: sleep so simulated time tracks wall-clock time. Turn this off
        for batch runs; it is what makes a headless test take 0.02 s instead of
        real seconds.
    """
    model, data = load_scene(scene_path)
    if headless and duration_s is None:
        raise ValueError("Headless runs need a finite duration_s, or they never stop.")
    if headless:
        context = nullcontext(None)
    else:
        from mujoco.viewer import launch_passive
        context = launch_passive(model, data)
    with context as viewer:
        while duration_s is None or data.time < duration_s:
            if viewer is not None and not viewer.is_running():
                break
            started = time.perf_counter()
            data.ctrl[:] = np.asarray(controller_fn(model, data, data.time))
            mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()
            if realtime:
                time.sleep(max(0.0, model.opt.timestep - (time.perf_counter() - started)))
    return model, data
