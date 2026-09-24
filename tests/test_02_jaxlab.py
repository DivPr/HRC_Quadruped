import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from pup.jaxlab import ex1_pure_functions as pure
from pup.jaxlab.ex2_vmap_pd import batched_pd, pd_torques
from pup.jaxlab.ex3_quaternions import gravity_in_body_frame, quat_inv, quat_rotate
from pup.sim.pd import PDController


def test_absolute_value():
    np.testing.assert_array_equal(jax.jit(pure.absolute_value)(jnp.array([-2, 0, 3])), [2, 0, 3])


def test_replace_element():
    values = jnp.arange(4)
    np.testing.assert_array_equal(jax.jit(pure.replace_element)(values, 2, 9), [0, 1, 9, 3])
    np.testing.assert_array_equal(values, np.arange(4))


def test_sum_indices():
    compiled = jax.jit(pure.sum_indices)
    assert compiled(0) == 0
    assert compiled(7) == 21


@pytest.mark.parametrize("count", [1, 128, 4096])
def test_batched_pd(count):
    random = np.random.default_rng(count)
    q, qd, target = random.normal(size=(3, count, 12)).astype(np.float32)
    expected = PDController(25, 0.5)(q, qd, target)
    traces = []
    def counted(q, qd, target):
        traces.append(1)
        return batched_pd(q, qd, target)
    compiled = jax.jit(counted)
    for _ in range(2):
        np.testing.assert_allclose(compiled(q, qd, target), expected, atol=1e-5)
    assert len(traces) == 1
    np.testing.assert_allclose(pd_torques(q[0], qd[0], target[0]), expected[0], atol=1e-5)


def test_quaternion_rotation():
    random = np.random.default_rng(12)
    quaternions = random.normal(size=(128, 4))
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    vectors = random.normal(size=(128, 3))
    expected = np.zeros_like(vectors)
    for q, v, out in zip(quaternions, vectors, expected):
        mujoco.mju_rotVecQuat(out, v, q)
    actual = jax.jit(jax.vmap(quat_rotate))(quaternions, vectors)
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_inverse():
    q = jnp.array([0.5, 0.5, 0.5, 0.5])
    vector = jnp.array([1., 2., 3.])
    np.testing.assert_allclose(quat_rotate(quat_inv(q), quat_rotate(q, vector)), vector)


def test_gravity():
    np.testing.assert_array_equal(gravity_in_body_frame(jnp.array([1., 0, 0, 0])), [0, 0, -1])
    np.testing.assert_array_equal(gravity_in_body_frame(jnp.array([0., 1, 0, 0])), [0, 0, 1])
