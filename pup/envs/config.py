"""Every number the MJX environment uses, in one place. Provided -- do not tune.

Mirrors Playground's `go1/joystick.py::default_config()`, trimmed to the reward
terms this onboarding uses. Read it; `docs/03_mjx_environment.md` explains what
each block buys you.
"""

from ml_collections import ConfigDict


def default_config() -> ConfigDict:
    """Return a fresh config; time in s, angles rad, distances m, velocities m/s."""
    return ConfigDict(dict(
        # --- Timing. n_substeps = ctrl_dt / sim_dt = 5: the policy runs at 50 Hz,
        # physics and the <position> actuators' PD loop at 250 Hz.
        ctrl_dt=0.02,
        sim_dt=0.004,
        episode_length=1000,      # 20 s at 50 Hz
        action_repeat=1,

        # --- Action. motor_targets = default_pose + action * action_scale, so a
        # freshly initialized policy (outputs near zero) stands in the home pose.
        action_scale=0.5,

        # --- Informational only: the gains actually live in scene_flat_mjx.xml's
        # <position kp kv> actuators. They are repeated here so training code can
        # log them and the ROS 2 node can send matching gains.
        Kp=25.0,
        Kd=0.5,

        # --- Backend. "warp" needs an NVIDIA GPU; "jax" runs everywhere.
        impl="jax",
        naconmax=32768,           # contact-buffer size for the MJX backend
        njmax=40,                 # constraint-buffer size
        reset_noise=0.01,         # rad of uniform jitter on the initial joint pose

        # --- Observation noise, applied per block in _get_obs. Set level=0.0 to
        # disable it (the tests do). Without noise a policy overfits to a
        # perfectly clean sensor suite that no robot has.
        noise_config=dict(level=1.0, gyro=0.2, gravity=0.05,
                          joint_pos=0.03, joint_vel=1.5),

        # --- Commands: (vx m/s, vy m/s, wz rad/s), resampled every 250 control
        # steps (5 s). sample_command also emits an all-zero command 10 % of the
        # time so the robot learns to stand still.
        command_config=dict(minimum=[-1.0, -0.8, -1.2],
                            maximum=[1.5, 0.8, 1.2],
                            resample_steps=250),

        # --- Reward scales. Playground's Go1 flat-terrain values, unchanged.
        # Positive = reward, negative = cost. Students do not tune these; see the
        # sidebar in docs/03_mjx_environment.md for what each one is buying.
        reward_config=dict(scales=dict(
            tracking_lin_vel=1.0,     # the actual task
            tracking_ang_vel=0.5,     # turning
            lin_vel_z=-0.5,           # stop bouncing the trunk
            ang_vel_xy=-0.05,         # stop rolling/pitching the trunk
            orientation=-5.0,         # stay level; falling is expensive
            torques=-0.0002,          # energy -- tiny, or it learns to do nothing
            action_rate=-0.01,        # smoothness; this is what transfers to hardware
            feet_air_time=0.1,        # take real steps instead of shuffling
            stand_still=-0.5,         # hold the pose when commanded zero
            pose=0.5,                 # stay near the default joint configuration
            termination=-1.0,         # falling is bad
        )),
    ))
