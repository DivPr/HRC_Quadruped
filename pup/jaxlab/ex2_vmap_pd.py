"""Exercise 2: one controller, thousands of robots.

Write one pure controller first, then map its batch axis with vmap.
Axes marked None are shared (gains); axes marked 0 index robots.
The mapped function is still pure and can be compiled with jit.

Worked example (provided):
    def squared_norm(vector):
        return jnp.sum(vector**2)
    norms = jax.jit(jax.vmap(squared_norm))
    norms(jnp.ones((128, 3)))  # shape (128,), all 3

PyTree in five minutes:
A dict of arrays is a PyTree; jit and vmap visit its array leaves.
For example {'q': zeros((128, 12)), 'qd': zeros((128, 12))} can be batched.
Playground State is a dataclass PyTree with data, obs, reward, done and info.
Its structure and each leaf shape must stay fixed through a rollout.
Do not add an info key only after the robot falls.

Different batch sizes compile different programs; the same shape reuses one.
This exercise uses radians, rad/s and Nm just like the numpy PD controller.
"""

import jax
import jax.numpy as jnp


def pd_torques(q: jax.Array, qd: jax.Array, q_des: jax.Array,
               kp: float = 25.0, kd: float = 0.5) -> jax.Array:
    """Map (12,) q, qd, q_des in rad/rad/s to (12,) torques clipped at 20 Nm."""
    # ===== TODO(student): Port the joint PD equation to JAX =====
    raise NotImplementedError(
        "Stage 2: Port the joint PD equation to JAX. See docs/02_jax_for_robotics.md")
    # ===== end TODO =====


def batched_pd(q: jax.Array, qd: jax.Array, q_des: jax.Array,
               kp: float = 25.0, kd: float = 0.5) -> jax.Array:
    """Map three (N,12) states/targets to (N,12) torques, sharing scalar gains."""
    # ===== TODO(student): Vectorize one PD controller over robots =====
    raise NotImplementedError(
        "Stage 2: Vectorize one PD controller over robots. See docs/02_jax_for_robotics.md")
    # ===== end TODO =====
