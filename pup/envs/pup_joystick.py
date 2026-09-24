"""A small Playground environment: provide plumbing, implement control concepts."""

from typing import Any

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from mujoco_playground._src import mjx_env

from pup.envs import rewards
from pup.envs.config import default_config
from pup.envs.constants import CONTACT_SENSORS, MJX_SCENE
from pup.envs.math_utils import get_sensor_data, quat_inv, rotate
from pup.jaxlab.ex3_quaternions import gravity_in_body_frame


class PupJoystick(mjx_env.MjxEnv):
    """Track desired body vx, vy (m/s), and yaw rate (rad/s), at 50 Hz."""

    def __init__(self, config=None, config_overrides: dict | None = None) -> None:
        """Load primitive-only Pup with position actuators and contact sensors."""
        super().__init__(config or default_config(), config_overrides)
        self._mj_model = mujoco.MjModel.from_xml_path(str(MJX_SCENE))
        self._mj_model.opt.timestep = self._config.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._default_pose = jnp.array(self._mj_model.keyframe("home").qpos[7:])
        self._home = jnp.array(self._mj_model.keyframe("home").qpos)
        self._joint_limits = jnp.array(self._mj_model.jnt_range[1:])
        self._trunk_id = self._mj_model.body("trunk").id
        self._imu_id = self._mj_model.site("imu").id
        self._contact_addresses = jnp.array([
            self._mj_model.sensor(name).adr[0] for name in CONTACT_SENSORS])
        noise = self._config.noise_config
        self._noise_scale = jnp.concatenate([
            jnp.full(3, noise.gyro), jnp.full(3, noise.gravity), jnp.zeros(3),
            jnp.full(12, noise.joint_pos), jnp.full(12, noise.joint_vel), jnp.zeros(12)])

    @property
    def xml_path(self) -> str:
        """Return the absolute scene path."""
        return str(MJX_SCENE)

    @property
    def action_size(self) -> int:
        """Return the number of unitless joint actions."""
        return 12

    @property
    def mj_model(self) -> mujoco.MjModel:
        """Return the CPU model used for sensor addresses and rendering."""
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        """Return the model used by the chosen MJX backend."""
        return self._mjx_model

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Return initial State with (45,) observation; rng is a JAX PRNG key."""
        rng, pose_key, command_key = jax.random.split(rng, 3)
        qpos = self._home.at[7:].add(self._config.reset_noise *
                                    jax.random.uniform(pose_key, (12,), minval=-1, maxval=1))
        data = mjx_env.make_data(self.mj_model, qpos=qpos, ctrl=self._default_pose,
                                 impl=self._config.impl, naconmax=self._config.naconmax,
                                 njmax=self._config.njmax)
        data = mjx.forward(self.mjx_model, data)
        info = dict(rng=rng, command=self.sample_command(command_key),
                    step=jnp.int32(0), last_act=jnp.zeros(12), last_last_act=jnp.zeros(12),
                    feet_air_time=jnp.zeros(4), last_contact=jnp.zeros(4, dtype=bool),
                    fixed_command=jnp.bool_(False))
        metrics = {name: jnp.zeros(()) for name in self._config.reward_config.scales}
        return mjx_env.State(data, self._get_obs(data, info), jnp.zeros(()),
                             jnp.zeros(()), metrics, info)

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        """Return noisy (45,) gyro/gravity/command/q-offset/qd/last_action.

        Units: rad/s (3), unit direction (3), m/s,m/s,rad/s (3), rad (12),
        rad/s (12), unitless (12). Sensor quaternions are wxyz (4,).
        """
        # ===== TODO(student): Build the exact 45-dimensional observation =====
        raise NotImplementedError(
            "Stage 3: Build the exact 45-dimensional observation. See docs/03_mjx_environment.md")
        # ===== end TODO =====

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        """Return scalar bool for upside-down, height <0.12 m, or nonfinite qpos."""
        # ===== TODO(student): Detect falls and invalid simulation states =====
        raise NotImplementedError(
            "Stage 3: Detect falls and invalid simulation states. See docs/03_mjx_environment.md")
        # ===== end TODO =====

    def sample_command(self, rng: jax.Array) -> jax.Array:
        """Sample (3,) vx,vy,yaw within config ranges, with 10% exactly zero."""
        # ===== TODO(student): Sample a bounded command including standing =====
        raise NotImplementedError(
            "Stage 3: Sample a bounded command including standing. See docs/03_mjx_environment.md")
        # ===== end TODO =====

    def _update_feet(self, data: mjx.Data, info: dict) -> tuple:
        """Read (4,) contact flags from sensors; accumulate swing durations in s."""
        contact = data.sensordata[self._contact_addresses] > 0
        first_contact = (info["feet_air_time"] > 0) & (contact | info["last_contact"])
        info = {**info, "feet_air_time": info["feet_air_time"] + self.dt}
        return info, contact, first_contact

    def _maybe_resample_command(self, rng: jax.Array, info: dict, step: jax.Array) -> dict:
        """Refresh commands every configured number of 50 Hz control steps."""
        rng, key = jax.random.split(rng)
        resample = (step % self._config.command_config.resample_steps == 0)
        resample &= ~info["fixed_command"]
        return {**info, "rng": rng,
                "command": jnp.where(resample, self.sample_command(key), info["command"])}

    def _reward_terms(self, data: mjx.Data, action: jax.Array,
                       info: dict, done: jax.Array, first_contact: jax.Array) -> dict:
        """Feed each reward term its inputs; returns named, unscaled scalars (provided)."""
        world_velocity = get_sensor_data(self.mj_model, data, "global_linvel")
        local_velocity = rotate(world_velocity, quat_inv(data.qpos[3:7]))
        angular_velocity = get_sensor_data(self.mj_model, data, "gyro")
        return dict(
            tracking_lin_vel=rewards.reward_tracking_lin_vel(info["command"], local_velocity),
            tracking_ang_vel=rewards.reward_tracking_ang_vel(info["command"], angular_velocity),
            lin_vel_z=rewards.cost_lin_vel_z(world_velocity),
            ang_vel_xy=rewards.cost_ang_vel_xy(angular_velocity),
            orientation=rewards.cost_orientation(gravity_in_body_frame(data.qpos[3:7])),
            torques=rewards.cost_torques(data.actuator_force),
            action_rate=rewards.cost_action_rate(action, info["last_act"], info["last_last_act"]),
            feet_air_time=rewards.reward_feet_air_time(info["feet_air_time"], first_contact,
                                                       info["command"]),
            stand_still=rewards.cost_stand_still(info["command"], data.qpos[7:], self._default_pose),
            pose=jnp.exp(-rewards.cost_joint_pose_deviation(data.qpos[7:], self._default_pose)),
            termination=rewards.cost_termination(done))

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Advance 0.02 s with (12,) unitless actions; return the same State tree."""
        info = dict(state.info)
        # ===== TODO(student): Apply targets, step physics, and assemble scaled rewards =====
        raise NotImplementedError(
            "Stage 3: Apply targets, step physics, and assemble scaled rewards. See docs/03_mjx_environment.md")
        # ===== end TODO =====
        # Brax wrappers add their own metric keys; update, never replace.
        metrics = {**state.metrics, **scaled}
        info.update(feet_air_time=info["feet_air_time"] * ~contact,
                    last_contact=contact, step=info["step"] + 1)
        info = self._maybe_resample_command(info["rng"], info, info["step"])
        obs = self._get_obs(data, info)
        return state.replace(data=data, obs=obs, reward=reward,
                              done=done.astype(jnp.float32), metrics=metrics, info=info)
