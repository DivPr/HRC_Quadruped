"""Shared CPU fixtures, plus the "not started yet" reporting the docs promise.

Any test that fails with `NotImplementedError` is reported as **NOT STARTED**
rather than **FAILED**, so `pytest` output tells you at a glance which stages
you have not begun versus which ones you have begun and broken. The per-stage
summary is printed at the end of every run and written to `.pup_progress.json`
for `scripts/progress.py`.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

STAGES = {
    "test_01": "Stage 1  MuJoCo + PD controller",
    "test_02": "Stage 2  JAX for robotics",
    "test_03": "Stage 3  MJX environment",
    "test_04": "Stage 4  Brax PPO + export",
    "test_05": "Stage 5  ROS 2 sim2sim",
}
PROGRESS_PATH = Path(__file__).resolve().parents[1] / ".pup_progress.json"
_RESULTS: dict[str, str] = {}


def _stage_of(nodeid: str) -> str:
    """Map a pytest node id such as ``tests/test_03_env.py::test_x`` to a stage key."""
    name = nodeid.split("::")[0].rsplit("/", 1)[-1]
    return name[: len("test_00")] if name.startswith("test_") else "other"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Reclassify NotImplementedError failures as "not started" instead of failures."""
    outcome = yield
    report = outcome.get_result()
    if report.when != "call":
        return
    not_started = (call.excinfo is not None
                   and call.excinfo.errisinstance(NotImplementedError))
    if not_started:
        report.outcome = "skipped"
        report.longrepr = (str(item.fspath), item.location[1] or 0,
                           "NOT STARTED: student task not implemented yet")
    status = "not_started" if not_started else report.outcome
    previous = _RESULTS.get(report.nodeid)
    if previous != "failed":
        _RESULTS[report.nodeid] = status


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print the per-stage checklist and persist it for scripts/progress.py."""
    del exitstatus, config
    if not _RESULTS:
        return
    stages: dict[str, dict[str, int]] = {}
    for nodeid, status in _RESULTS.items():
        counts = stages.setdefault(_stage_of(nodeid), {})
        counts[status] = counts.get(status, 0) + 1
    write = terminalreporter.write_line
    write("")
    write("Pup onboarding progress")
    for key in sorted(stages):
        counts = stages[key]
        label = STAGES.get(key, key)
        passed = counts.get("passed", 0)
        failed = counts.get("failed", 0)
        not_started = counts.get("not_started", 0)
        skipped = counts.get("skipped", 0)
        if failed:
            mark = "✗ FAILED"
        elif not_started and not passed:
            mark = "▷ NOT STARTED"
        elif not_started:
            mark = "◑ IN PROGRESS"
        else:
            mark = "✓ PASSED"
        write(f"  {mark:<15} {label:<34} "
              f"{passed} passed, {failed} failed, {not_started} not started, "
              f"{skipped} skipped")
    PROGRESS_PATH.write_text(json.dumps(
        {"stages": {key: stages[key] for key in sorted(stages)},
         "labels": STAGES, "results": _RESULTS}, indent=2) + "\n")


def pytest_runtest_setup(item):
    """Skip `gpu` tests without a GPU and `ros` tests outside the ROS 2 container."""
    if any(mark.name == "gpu" for mark in item.iter_markers()):
        import jax
        if jax.default_backend() != "gpu":
            pytest.skip("needs a GPU (jax.default_backend() != 'gpu')")
    if any(mark.name == "ros" for mark in item.iter_markers()):
        if importlib.util.find_spec("rclpy") is None:
            pytest.skip("needs the ROS 2 container (see docs/05_ros2_sim2sim.md)")


@pytest.fixture
def mj_model():
    """The plain-MuJoCo flat scene model (torque actuators)."""
    from pup.sim.viewer import load_scene
    return load_scene()[0]


@pytest.fixture
def mj_data_home(mj_model):
    """An `MjData` for `mj_model`, reset to the `home` keyframe."""
    import mujoco

    from pup.sim.viewer import reset_to_keyframe
    data = mujoco.MjData(mj_model)
    reset_to_keyframe(mj_model, data, "home")
    return data


@pytest.fixture
def rng():
    """A fixed JAX PRNG key, so failures are reproducible."""
    import jax
    return jax.random.PRNGKey(0)


@pytest.fixture(scope="module")
def env():
    """PupJoystickFlat with observation noise and reset noise switched off."""
    from pup.envs.config import default_config
    from pup.envs.pup_joystick import PupJoystick
    config = default_config()
    config.noise_config.level = 0.0
    config.reset_noise = 0.0
    return PupJoystick(config)
