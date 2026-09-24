# Stage 3 — The MJX environment

## Why this exists

An RL environment is a contract: given a state and an action, produce the next
state, an observation, a reward, and a "did the episode end" flag. Everything
else in reinforcement learning is downstream of that contract, and almost every
failure that looks like "PPO didn't learn" is actually a bug in the environment.

Writing one for MJX adds a second constraint: the whole thing must be a **pure,
jittable function**, because Brax will `vmap` it over 8192 robots and `scan` it
over 20 timesteps at a time. Stage 2 was the rehearsal; this is the performance.

We build it the way MuJoCo Playground builds `go1/joystick.py`, deliberately.
NEMO's environment is derived from Playground, so the shapes of the files, the
names of the reward terms, and the structure of `info` will all look familiar
the first time you open NEMO's repository. Where this repo differs from
Playground it is noted in the code.

## Concepts you need

**The joystick task.** The robot is given a command `(vx, vy, wz)` — forward
speed, lateral speed, yaw rate — resampled every 5 s, and is rewarded for
matching it. That is it. Walking is not in the reward function anywhere; a gait
emerges because it is the cheapest way to track a velocity command without
falling over.

**Control rate vs physics rate.** `ctrl_dt = 0.02` (50 Hz policy),
`sim_dt = 0.004` (250 Hz physics), so `n_substeps = 5`. Each `env.step` runs
five physics steps with the same action held constant, and the `<position>`
actuators close the PD loop on every one of those five.

**Action → target.** The policy outputs 12 numbers in `[−1, 1]`. They are
*offsets from the default pose*, not absolute angles:

```python
motor_targets = default_pose + action * action_scale     # action_scale = 0.5
```

This is a big deal for learning. A randomly-initialized policy outputs roughly
zero, which means "stand in the home pose" — a sane starting behaviour — rather
than "fold all joints to their limits". `action_scale` sets how far from that
pose the policy may reach: 0.5 rad, matching Playground's Go1.

**Reading contacts.** Do **not** poke around in `mjx.Data.contact`. Playground
moved off that and so do we: the model declares one `<contact>` sensor per foot
against the floor, and you read `data.sensordata` at the sensor's address. This
works identically on the JAX and Warp backends, which is the whole point.

**Sensors.** `gyro` and `accelerometer` on the `imu` site, `orientation`
(`framequat`), `upvector` (`framezaxis`), plus `global_linvel` / `global_angvel`.
The last two are **privileged**: they are ground truth from the simulator that a
real robot cannot measure. They may appear in *rewards* and *evaluation* but
never in the observation.

> **Reading:** MuJoCo Playground's `go1/joystick.py`, the Playground technical
> report, the MJX documentation, and the MuJoCo contact-sensor docs
> ([reading packet](reading_packet.md), Stage 3). Playground's source is
> installed — read it:
> `.venv/lib/python3.11/site-packages/mujoco_playground/_src/locomotion/go1/joystick.py`

## What is provided

You do not have to write plumbing. These are done for you and worth reading:

| File | What it gives you |
|---|---|
| `envs/config.py` | `default_config()` — every timing, noise, command and reward-scale number |
| `envs/randomize.py` | `domain_randomize(model, rng)` → `(model, in_axes)`, Playground's pattern |
| `envs/math_utils.py` | one import site for `get_sensor_data`, `quat_inv`, `rotate` |
| `envs/__init__.py` | registers `PupJoystickFlat` so `registry.load(...)` works |
| `envs/rewards.py` | eight fully-implemented reward terms to use as templates |
| `pup_joystick.py` | `__init__`, `reset`, `_update_feet`, `_maybe_resample_command`, `_reward_terms`, and the `step` skeleton around your TODO block |

`cost_orientation` is provided specifically as the *simplest possible template*
for your tracking terms — one line, clear units, a docstring that states shapes.
Copy its shape. `reward_feet_air_time` is provided because its bookkeeping is
fiddly and teaches you little on a first pass.

## Your task

Six functions and one delimited block.

### 1. `_get_obs(data, info) -> jnp.ndarray[45]`

**This exact order.** The ROS 2 node in Stage 5 must rebuild the identical
vector from ROS messages, and there is no way to detect a mismatch except by
watching the robot fall over.

| Slice | Contents | Units |
|---|---|---|
| `[0:3]` | `gyro` — body angular velocity | rad/s |
| `[3:6]` | gravity direction in the body frame | unit vector |
| `[6:9]` | `info["command"]` — (vx, vy, wz) | m/s, m/s, rad/s |
| `[9:21]` | `qpos[7:] − default_pose` | rad |
| `[21:33]` | `qvel[6:]` | rad/s |
| `[33:45]` | `info["last_act"]` | unitless, [−1, 1] |

Then add uniform noise scaled per-block by `config.noise_config`
(`gyro`, `gravity`, `joint_pos`, `joint_vel`, all multiplied by
`noise_config.level`) using `jax.random` and the key in `info["rng"]`. The
command and the last action are *not* noised — they are things you know exactly.

`__init__` has already built `self._noise_scale`, a `(45,)` vector with the
right per-block magnitudes and zeros where no noise belongs, so this is one
`jax.random.uniform` call and one multiply.

For the gravity block, reuse your Stage 2 `gravity_in_body_frame`. Playground's
`mjx_env` has an equivalent; they compute the same thing, and using yours means
Stage 5's numpy port is provably the same function.

### 2. `_get_termination(data) -> bool`

Return `True` if **any** of:

- the trunk's up-vector has `z < 0` (flipped over) — use the `upvector` sensor;
- trunk height `qpos[2] < 0.12` m;
- any element of `qpos` is not finite.

The third one looks paranoid and isn't: an unstable contact can produce `NaN`,
and a `NaN` reward silently destroys the policy network in one gradient step.
Terminating is how you notice.

### 3. `sample_command(rng) -> jnp.ndarray[3]`

Uniform inside `config.command_config` (`vx ∈ [−1.0, 1.5]`, `vy ∈ [−0.8, 0.8]`,
`wz ∈ [−1.2, 1.2]`), except that **10% of samples are exactly `[0, 0, 0]`**.
Standing still is a skill; if you never command it, the policy never learns it,
and your robot jogs in place forever.

Split the key: one draw for the velocities, one for the "is this a zero
command?" coin flip. Select with `jnp.where`, not `if` — rule 1 from Stage 2.

### 4. The core of `step(state, action)`

One delimited TODO block, about a dozen lines. Everything before and after it
(command resampling, air-time reset, metrics assembly, `State` construction) is
provided, and the lines after the block tell you what names it must leave
behind: `data`, `contact`, `done`, `reward` and `scaled`.

In order:

1. Turn the action into joint targets: `default_pose + action * action_scale`
   (`self._default_pose`, `self._config.action_scale`).
2. Advance the physics with `mjx_env.step(self.mjx_model, state.data,
   motor_targets, self.n_substeps)` — five 4 ms steps with the targets held
   constant, the `<position>` actuators closing the PD loop on each one.
3. Update the foot bookkeeping with the provided `self._update_feet(data, info)`.
   Read its signature: it returns three things, and the order matters.
4. Compute `done` with your `_get_termination`.
5. Get the unscaled reward terms from the provided
   `self._reward_terms(...)` — check what it wants and in what order.
6. Multiply each term by its scale from
   `self._config.reward_config.scales`, keeping the same keys. Sum them,
   multiply by `self.dt`, and clip to `[0, 10000]`.
7. Roll the action history forward: the old `last_act` becomes
   `last_last_act`, and `action` becomes `last_act`.

Two things in step 6 look odd and are deliberate:

- **`* self.dt`.** Reward terms are per-second rates; multiplying by the control
  timestep makes total episode return independent of the control rate.
- **`jnp.clip(..., 0.0, 10000.0)`.** Playground clips the per-step reward at
  zero. Without it, a robot that has already failed keeps accumulating negative
  reward, and PPO learns that terminating immediately is optimal.

`_reward_terms` is provided and already hands each term the right arguments —
including rotating the world-frame velocity into the body frame. Your job here
is the scaling and the summation, not the plumbing.

### 5–7. Three reward terms in `rewards.py`

```python
reward_tracking_lin_vel(command, local_linvel, sigma=0.25)
    -> exp(-‖command[:2] − local_linvel[:2]‖² / sigma)

reward_tracking_ang_vel(command, ang_vel, sigma=0.25)
    -> exp(-(command[2] − ang_vel[2])² / sigma)

cost_action_rate(act, last_act, last_last_act)
    -> Σ(act − last_act)² + Σ(act − 2·last_act + last_last_act)²
```

The exponential shape matters. A quadratic *penalty* on velocity error is
unbounded below and dominates everything else early in training, when the robot
is bad at everything; an exponential *reward* is bounded in `[0, 1]`, so the
policy gets a smooth gradient toward the command without being punished into
inaction. `sigma` sets how forgiving it is.

`cost_action_rate` penalizes both the first and second differences: the first
discourages twitching, the second discourages the sawtooth oscillation a policy
will otherwise happily learn because it looks like zero net motion.

### The reward scales (provided — do not tune these)

| Term | Scale | What it buys you |
|---|---|---|
| `tracking_lin_vel` | +1.0 | the actual task |
| `tracking_ang_vel` | +0.5 | turning |
| `lin_vel_z` | −0.5 | stop bouncing the trunk |
| `ang_vel_xy` | −0.05 | stop rolling/pitching the trunk |
| `orientation` | −5.0 | keep the body level (big, because falling is expensive) |
| `torques` | −0.0002 | energy; tiny, or it learns to do nothing |
| `action_rate` | −0.01 | smoothness — this is what makes it transfer to hardware |
| `feet_air_time` | +0.1 | take real steps instead of shuffling |
| `stand_still` | −0.5 | hold the pose when commanded zero |
| `pose` | +0.5 | stay near the default joint configuration |
| `termination` | −1.0 | falling is bad |

These are Playground's Go1 flat-terrain values. They are in `config.py`; the
reference run in `checkpoints/` used them unchanged. Tuning reward scales is a
real skill and a genuinely bad way to spend your first week — if your robot
doesn't walk, the bug is in the environment, not the weights.

## How you'll know you're done

```bash
uv run pytest tests/test_03_env.py tests/test_03_rewards.py -v
```

- `reset` gives a finite `(45,)` observation; `metrics` keys equal the reward
  scale keys.
- `env.step` is jittable, returns finite reward and `done ∈ {0, 1}`, and — this
  one is required by Brax's wrappers — returns a `State` whose **PyTree
  structure is identical** to the input's.
- `jax.vmap` over 16 environments plus 10 steps finishes in under 90 s on CPU
  (the reference takes ~15 s on a laptop, including compilation; the limit is
  3× that, so a slow machine is not a failure).
- **Zero action from `home` for 50 steps does not fall**, and trunk height stays
  within 0.05 m. This is the test that proves your keyframe, your actuators and
  your `default_pose` all agree. If this fails, nothing downstream can work.
- Upside down → `done == 1` after one step; `NaN` in `qpos` → terminated.
- Each reward term against hand-computed values (perfect tracking → exactly 1.0;
  constant actions → action rate exactly 0).
- 10 000 commands stay in range and 8–12 % are exactly zero.

## Common mistakes

- **Joint slicing, again.** `qpos[7:]` and `qvel[6:]`, inside MJX too. The free
  joint does not go away just because you are in JAX now.
- **World-frame velocity in the tracking reward.** `global_linvel` is in the
  *world* frame. The command is in the *body* frame. Rotate before comparing —
  `_reward_terms` does this for you, but when you write your own environment
  later, this is the bug that makes a policy that only walks north.
- **Forgetting to multiply by the scales.** Everything trains, badly, and the
  learning curve looks plausible. Check that `metrics` values are the *scaled*
  ones.
- **Adding a key to `info` conditionally.** The PyTree structure must be
  identical every step. Allocate every key in `reset`.
- **Reusing `info["rng"]` without splitting.** Your "noise" becomes a constant.
- **Reading `data.contact`.** Use the contact sensors.
- **Putting `global_linvel` in the observation.** It will train beautifully and
  be worthless on hardware. This is the classic sim2real own-goal, and it is why
  the privileged sensors are labelled as such in `constants.py`.

