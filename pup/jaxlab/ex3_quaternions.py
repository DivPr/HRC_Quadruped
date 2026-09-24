"""Exercise 3: MuJoCo uses WXYZ, not ROS XYZW.

A quaternion q_wb rotates a body-frame vector into the world frame.
Its inverse rotates a world-frame vector into the body frame.
Unit quaternions have length one. q and -q describe the same rotation.
Our gravity observation is a direction [0,0,-1], not acceleration -9.81.

Worked example (provided):
    q_wb = jnp.array([1., 0., 0., 0.])
    vector = jnp.array([0., 0., -1.])
    # Identity orientation leaves vector unchanged.

Use the cross-product identity:
    temporary = 2 * cross(q.xyz, vector)
    rotated = vector + q.w * temporary + cross(q.xyz, temporary)
For a unit quaternion the inverse negates xyz and preserves w.
Compose inverse and rotate to express world gravity in the body.
Do not construct Euler angles: their singularities add unnecessary work.
The tests compare random unit quaternions with MuJoCo's own C routine.
See docs/02_jax_for_robotics.md for a worked 90-degree example.
"""

import jax
import jax.numpy as jnp


def quat_rotate(q: jax.Array, v: jax.Array) -> jax.Array:
    """Rotate v (3,) by unit wxyz q (4,); return (3,) with v's units."""
    # ===== TODO(student): Rotate a vector by a unit quaternion =====
    raise NotImplementedError(
        "Stage 2: Rotate a vector by a unit quaternion. See docs/02_jax_for_robotics.md")
    # ===== end TODO =====


def quat_inv(q: jax.Array) -> jax.Array:
    """Return the inverse (4,) of a unit wxyz quaternion (dimensionless)."""
    # ===== TODO(student): Invert a unit quaternion =====
    raise NotImplementedError(
        "Stage 2: Invert a unit quaternion. See docs/02_jax_for_robotics.md")
    # ===== end TODO =====


def gravity_in_body_frame(q_wb: jax.Array) -> jax.Array:
    """Map unit wxyz q_wb (4,) to the body-frame down direction (3,), unitless."""
    # ===== TODO(student): Express the world gravity direction in the body frame =====
    raise NotImplementedError(
        "Stage 2: Express the world gravity direction in the body frame. See docs/02_jax_for_robotics.md")
    # ===== end TODO =====
