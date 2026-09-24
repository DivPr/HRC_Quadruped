"""Register Pup when Playground is installed; keep constants usable in ROS Python."""

import importlib.util


def _make_environment(config=None, config_overrides=None):
    from pup.envs.pup_joystick import PupJoystick
    return PupJoystick(config=config, config_overrides=config_overrides)


def _make_config():
    from pup.envs.config import default_config
    return default_config()


if importlib.util.find_spec("mujoco_playground") is not None:
    from mujoco_playground import locomotion
    locomotion.register_environment("PupJoystickFlat", _make_environment, _make_config)
