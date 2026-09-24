# Stage 2 — JAX for robotics

## Why this exists

In Stage 3 you write an environment that runs **8192 robots at once** on one
GPU. That is only possible because MJX is written in JAX, and JAX gets its speed
by *tracing* your Python once into a computation graph and compiling that graph.
The price is that your Python has to be traceable: no side effects, no branching
on values, no growing lists.

## Concepts you need

**Tracing.** When you call `jax.jit(f)(x)`, JAX runs `f` once with a *tracer* in
place of `x`. A tracer knows its shape and dtype but has no value. Everything
your Python does to that tracer is recorded; everything else (prints, list
appends, `if` on the tracer) either happens once at trace time or fails.

**The three rules:**

1. **No Python control flow on traced values.** `if x > 0:` cannot work — `x > 0`
   is a tracer, not a bool. Use `jnp.where(cond, a, b)`, which computes both
   branches and selects.
2. **No mutation.** JAX arrays are immutable. `values[i] = v` fails; use
   `values.at[i].set(v)`, which returns a new array (and compiles to an in-place
   update, so it is not slow).
3. **No Python loops with traced bounds.** `for i in range(n)` where `n` is
   traced cannot be unrolled. Use `jax.lax.fori_loop(lo, hi, body, init)`. (A
   Python loop with a *constant* bound is fine)

**`vmap`.** `jax.vmap(f)` turns a function written for one robot into a function
that runs on a batch, without a loop. `in_axes` says which arguments are batched:
`0` means "index the first axis", `None` means "shared across the batch".

> **PyTree in five minutes**
>
> A PyTree is any nested structure of containers (dict, list, tuple, dataclass)
> whose leaves are arrays. `jit`, `vmap`, `grad` and `lax.scan` all accept and
> return PyTrees — they just apply themselves leaf by leaf.
>
> ```python
> state = {"q": jnp.zeros((128, 12)), "qd": jnp.zeros((128, 12))}
> jax.tree_util.tree_map(lambda x: x.shape, state)
> # {'q': (128, 12), 'qd': (128, 12)}
> ```
>
> `mjx_env.State` (Stage 3) is a PyTree: `data`, `obs`, `reward`, `done`,
> `metrics`, `info`. Two consequences bite in Stage 3:
> **(a)** the *structure* must be identical in and out of `step` — you may not
> add an `info` key only when the robot falls; **(b)** every leaf must keep the
> same shape and dtype, or JAX recompiles.

**Quaternions.** MuJoCo uses **`wxyz`** order. ROS uses **`xyzw`**. A unit
quaternion `q_wb` rotates a body-frame vector into world. Its inverse (negate
`xyz`, keep `w`) goes the other way. Rotating without building a matrix:

```
t = 2 * cross(q.xyz, v)
v_rotated = v + q.w * t + cross(q.xyz, t)
```

> **Reading:** JAX "🔪 Sharp Bits", the `jit`/`vmap` tutorials, and the MuJoCo
> quaternion conventions note ([reading packet](reading_packet.md), Stage 2).

## Your task

Three files, each with a ~20-line docstring teaching the concept plus a worked
example. Read the docstring first; the answer is nearly in it.

### 1. `pup/jaxlab/ex1_pure_functions.py`

Fix three deliberately-broken ideas, one per rule:

| Function | Broken idea | Rule |
|---|---|---|
| `absolute_value(x)` | `if x > 0: return x` | use `jnp.where` |
| `replace_element(values, index, value)` | `values[index] = value` | use `.at[].set()` |
| `sum_indices(count)` | `for i in range(count)` with traced `count` | use `jax.lax.fori_loop` |

You will hit exactly these three errors in Stage 3, in `sample_command`,
`_update_feet`, and anything you try to accumulate.

### 2. `pup/jaxlab/ex2_vmap_pd.py`

- `pd_torques(q, qd, q_des, kp=25.0, kd=0.5)` — your Stage 1 controller in
  `jnp`, for a single `(12,)` robot, clipped at ±20 Nm.
- `batched_pd(...)` — the same thing over `(N, 12)` inputs via `jax.vmap`, with
  the scalar gains shared (`in_axes=(0, 0, 0, None, None)`).

The test checks `N = 1, 128, 4096` against your numpy `PDController` and asserts
that `jax.jit` traces it **once** across two calls with the same shapes. If it
traces twice, you changed a shape or a dtype between calls.

### 3. `pup/jaxlab/ex3_quaternions.py`

- `quat_rotate(q, v)` — rotate `(3,)` by unit `wxyz` `(4,)`.
- `quat_inv(q)` — inverse of a unit quaternion.
- `gravity_in_body_frame(q_wb)` — the world down-direction `[0, 0, −1]`
  expressed in the body frame. **A direction, not an acceleration**: no 9.81.

Sanity checks you can do in your head: identity orientation → `[0, 0, −1]`;
upside down (`q = [0, 1, 0, 0]`, a 180° roll) → `[0, 0, +1]`. The test compares
128 random quaternions against MuJoCo's own `mju_rotVecQuat`, so there is no
arguing with it.

## How you'll know you're done

```bash
uv run pytest tests/test_02_jaxlab.py -v      # < 60 s, CPU
```

## Common mistakes

- **`jnp.where(x > 0, x, 0)` returning the wrong dtype.** Mixing a float array
  with a Python `int` literal is fine in these exercises, but in Stage 3 a
  dtype change between steps forces a recompile every step and your training
  crawls. If training looks 100× too slow, this is why.
- **Using `jax.lax.fori_loop` where a Python loop would do.** Constant bounds
  don't need it. Don't cargo-cult.
- **`xyzw`.** MuJoCo is `wxyz`. Say it out loud once.
- **Forgetting the quaternion inverse.** `quat_rotate(q_wb, [0,0,-1])` gives you
  the body's down-vector expressed in the *world*, which is a constant and
  useless. You want the other direction.
- **Normalizing when you shouldn't.** MuJoCo's `framequat` sensor is already
  unit-norm. Renormalizing costs a `sqrt` per step in the inner loop.