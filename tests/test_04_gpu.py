"""GPU-only checks. Skipped automatically unless jax.default_backend() == 'gpu'.

There is nothing here a member has to write. The point is that `pytest -m gpu`
is a real command with real content, so someone with a local NVIDIA card can
confirm their setup before starting a 200M-step run rather than after.
"""

import time

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.gpu


def test_backend_is_gpu():
    """Sanity: the marker only runs when a GPU is actually present."""
    assert jax.default_backend() == "gpu"
    assert any(device.platform == "gpu" for device in jax.devices())


def test_vmapped_env_throughput(env):
    """1024 environments must step meaningfully faster than real time on a GPU."""
    count = 1024
    keys = jax.random.split(jax.random.PRNGKey(0), count)
    state = jax.jit(jax.vmap(env.reset))(keys)
    step = jax.jit(jax.vmap(env.step))
    action = jnp.zeros((count, 12))
    state = step(state, action)
    state.reward.block_until_ready()

    started = time.perf_counter()
    for _ in range(50):
        state = step(state, action)
    state.reward.block_until_ready()
    steps_per_second = 50 * count / (time.perf_counter() - started)
    print(f"{steps_per_second:,.0f} env-steps/s with {count} environments")
    # A T4 manages far more than this; the bar is only "clearly GPU-accelerated".
    assert steps_per_second > 50_000
