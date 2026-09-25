"""Exercise 1: write calculations that can be traced.

JAX first traces your Python into an array program, then compiles that program.
A traced value is a placeholder, so Python cannot ask whether it is positive.
Use array selection instead. Arrays are immutable; updates return new arrays.
A traced loop bound needs a JAX loop whose carried value keeps its shape.

Worked example (provided):
    def positive_part(values):
        return jnp.where(values > 0, values, 0)
    compiled = jax.jit(positive_part)
    compiled(jnp.array([-2., 3.]))  # [0., 3.]

Broken starting ideas to repair:
    if x > 0: return x
    values[index] = value
    for index in range(count): total += index

You will hit exactly these three errors in Stage 3.
See docs/02_jax_for_robotics.md. Each function has a test with traced inputs.
Keep input shapes stable between calls to reuse the compiled program.
"""

import jax
import jax.numpy as jnp


def absolute_value(x: jax.Array) -> jax.Array:
    """Return elementwise absolute values, same shape and units as x."""
    # ===== TODO(student): Replace Python branching with array selection =====
    return jnp.where(x > 0, x, -x)
    # ===== end TODO =====


def replace_element(values: jax.Array, index: jax.Array, value: jax.Array) -> jax.Array:
    """Return a new (N,) array with one element replaced; preserve input/units."""
    # ===== TODO(student): Perform an immutable indexed update =====
    return values.at[index].set(value)
    # ===== end TODO =====


def sum_indices(count: jax.Array) -> jax.Array:
    """Sum integers 0 through count-1; scalar int input/output, no units."""
    # ===== TODO(student): Sum using a traced loop bound =====
    return jax.lax.fori_loop(0, count, lambda i, total: total + i, 0)
    # ===== end TODO =====
