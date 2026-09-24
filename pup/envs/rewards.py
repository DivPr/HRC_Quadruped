"""Unscaled scalar rewards. Costs are positive and receive negative scales."""

import jax
import jax.numpy as jnp


def reward_tracking_lin_vel(command: jax.Array, local_linvel: jax.Array,
                            sigma: float = 0.25) -> jax.Array:
    """Return exp(-xy error²/sigma), scalar; inputs (3,), m/s and rad/s."""
    # ===== TODO(student): Reward tracking the commanded body-frame xy velocity =====
    raise NotImplementedError(
        "Stage 3: Reward tracking the commanded body-frame xy velocity. See docs/03_mjx_environment.md")
    # ===== end TODO =====


def reward_tracking_ang_vel(command: jax.Array, ang_vel: jax.Array,
                            sigma: float = 0.25) -> jax.Array:
    """Return yaw tracking score, scalar; command (3,), angular velocity (3,), rad/s."""
    # ===== TODO(student): Reward tracking the commanded yaw rate =====
    raise NotImplementedError(
        "Stage 3: Reward tracking the commanded yaw rate. See docs/03_mjx_environment.md")
    # ===== end TODO =====


def cost_action_rate(act: jax.Array, last_act: jax.Array,
                     last_last_act: jax.Array) -> jax.Array:
    """Sum squared first and second differences of three unitless (12,) actions."""
    # ===== TODO(student): Penalize changes in actions and their first differences =====
    raise NotImplementedError(
        "Stage 3: Penalize changes in actions and their first differences. See docs/03_mjx_environment.md")
    # ===== end TODO =====


def cost_lin_vel_z(velocity: jax.Array) -> jax.Array:
    """Square vertical velocity from (3,) world velocity, (m/s)²."""
    return velocity[2] ** 2


def cost_ang_vel_xy(velocity: jax.Array) -> jax.Array:
    """Sum squared roll/pitch rates from (3,) body angular velocity, (rad/s)²."""
    return jnp.sum(velocity[:2] ** 2)


def cost_orientation(gravity: jax.Array) -> jax.Array:
    """Penalize tilt using xy components of unit gravity (3,); scalar, unitless."""
    return jnp.sum(gravity[:2] ** 2)


def cost_torques(torques: jax.Array) -> jax.Array:
    """Return sum of squared actuator torques (12,), Nm²."""
    return jnp.sum(torques ** 2)


# Playground's Go1 weights: knees are cheap to move, hips are not. Walking needs
# large knee excursions, so weighting all twelve joints equally here teaches the
# robot to stand very still indeed.
POSE_WEIGHTS = jnp.array([1.0, 1.0, 0.1] * 4)


def cost_joint_pose_deviation(q: jax.Array, default_pose: jax.Array) -> jax.Array:
    """Return weighted squared distance between two (12,) poses, rad²."""
    return jnp.sum(jnp.square(q - default_pose) * POSE_WEIGHTS)


def cost_stand_still(command: jax.Array, q: jax.Array,
                      default_pose: jax.Array) -> jax.Array:
    """Penalize joint displacement (rad) only when the (3,) command is near zero."""
    return jnp.sum(jnp.abs(q - default_pose)) * (jnp.linalg.norm(command) < 0.01)


def cost_termination(done: jax.Array) -> jax.Array:
    """Convert a scalar fall flag to a unitless floating-point cost."""
    return done.astype(jnp.float32)


def reward_feet_air_time(air_time: jax.Array, first_contact: jax.Array,
                          command: jax.Array) -> jax.Array:
    """Reward completed swings above 0.1 s; (4,) times/flags, (3,) command."""
    return jnp.sum((air_time - 0.1) * first_contact) * (jnp.linalg.norm(command) > 0.01)
