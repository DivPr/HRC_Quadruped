# FAQ and troubleshooting

Mostly seeded from failures hit while building this repository. If you burn more
than twenty minutes on something that isn't here, open a PR adding it — that is
a genuinely useful contribution.

---

## Setup

**`uv sync` fails resolving `jaxlib`.**
You are on Python 3.13. `jaxlib==0.8.2` has no wheel for it.
`uv python install 3.11 && uv sync`.

**`ModuleNotFoundError: No module named 'pup'`.**
Run from the repository root, and via `uv run python ...`. If you installed
manually, `pip install -e .`.

**`import mujoco_mjx` fails.**
There is no such module. The package is distributed as `mujoco-mjx` and imported
as `from mujoco import mjx`.

**The MuJoCo viewer window never appears on macOS.**
Use `mjpython`, not `python`: `uv run mjpython your_script.py`. macOS requires
GUI event loops on the main thread and `mjpython` arranges that. Headless runs
(every test) are unaffected.

---

## MuJoCo

**The robot slowly sinks / never reaches the target height.**
Steady-state error from a pure PD controller holding 12 kg. Raise `kp`, or add
feed-forward gravity compensation. It is physics, not a bug.

**The robot vibrates or explodes.**
`kp` too high relative to `kd`, or your timestep is too large. With
`timestep=0.004` and `armature=0.03`, gains up to ~100 are stable here.

**Everything is off by one joint.**
`qpos[7:]`, `qvel[6:]`. Read that twice. The free joint is 7 in `qpos` and 6 in
`qvel`.

**Contacts appear between the thighs and the trunk.**
Check `contype`/`conaffinity`. In `pup.xml` only the feet (`contype=2
conaffinity=1`) and the floor (`contype=1 conaffinity=2`) can collide. Two geoms
collide only if `contype1 & conaffinity2` or `contype2 & conaffinity1` is
nonzero.

**`actuator_force` doesn't match my PD output.**
You are in `scene_flat.xml`, which has `<motor>` actuators — there
`actuator_force` *is* your `ctrl`. The comparison only makes sense in
`scene_flat_mjx.xml`, which has `<position>` actuators.

---

## JAX / MJX

**`ConcretizationTypeError: Abstract tracer value encountered`.**
You used a traced array in Python control flow — `if`, `while`, `bool()`,
`int()`, `.item()`, or an f-string with a value. Use `jnp.where`, `jax.lax.cond`
or `jax.lax.fori_loop`.

**`TypeError: JAX arrays are immutable`.**
`x.at[i].set(v)`, not `x[i] = v`.

**Training is 100× slower than expected / it recompiles every step.**
Something changes shape or dtype between steps. Common causes: a Python `int`
sneaking into `info` (becomes weak-typed and then promotes), a list that grows,
or a conditional key in a dict. Debug with `JAX_LOG_COMPILES=1`:

```bash
JAX_LOG_COMPILES=1 uv run python -m pup.train.train_ppo --config cpu_smoke
```
More than a handful of compile lines means you have a leak.

**`scan body function carry input and carry output must have the same pytree
structure`.**
Your `step` returned a `State` with a different structure than it received.
Usually a missing or extra `metrics` / `info` key. Brax's wrappers *add* keys to
`metrics` (`reward` among them), so **update the incoming dict, never replace
it**:

```python
metrics = {**state.metrics, **scaled}     # right
metrics = scaled                          # wrong: drops the wrappers' keys
```

**`NaN` in the reward.**
Almost always: (a) a torque or velocity blew up because termination isn't firing
(check `_get_termination`), (b) a divide by a norm that can be zero, or (c)
`exp` of a large positive number. Add the non-finite check to termination and
run with `JAX_DEBUG_NANS=1` to get a traceback at the source.

**MJX is slower on CPU than plain MuJoCo.**
Expected for one robot — MJX pays a fixed vectorization cost. It wins at
thousands. The tests use ≤16 environments on purpose.

**Warp prints `solver iterations limit reached` on every step.**
Expected. `scene_flat_mjx.xml` sets `iterations="1"` on purpose (Playground's Go1
value — accuracy traded for speed across thousands of robots). The JAX backend
does the same thing quietly. Silence it with `model.opt.warn_overflow = 0`.

**`jax.default_backend()` says `cpu` and I have an NVIDIA GPU.**
`uv sync --extra gpu`, and check that your driver supports the CUDA version JAX
was built against.

---

## Training

**Reward goes up, then episode length collapses.**
The policy found something that scores well briefly and then falls. Look at
which term is climbing — usually `feet_air_time` or the tracking term paired
with an unpunished orientation.

**Reward plateaus low with full-length episodes.**
It learned to stand still. Check that your tracking rewards are actually scaled
and summed, that `sample_command` isn't returning zeros far more than 10% of the
time, and that `stand_still`'s condition is on command magnitude.

**Colab disconnected halfway.**
Checkpoints are written at every eval. Re-run the notebook's training cell with
`--out` pointed at the same directory, or just accept the last checkpoint and
export it. Do not close the tab; free-tier Colab reclaims idle sessions.

**`MUJOCO_GL` errors when rendering on Colab or a server.**
Set it *before* MuJoCo is imported: `os.environ["MUJOCO_GL"] = "egl"` on a GPU
box, `"osmesa"` for software rendering. The notebook already does this in its
first cell.

**`ValueError: Checkpoint path should be absolute` when exporting.**
Orbax refuses relative paths. `pup/train/export.py` calls `.resolve()` for you;
if you load a checkpoint yourself, do the same.

**`uv sync --locked` fails in CI but works locally.**
Someone changed `pyproject.toml` without re-running `uv lock`. Run `uv lock`,
commit the result, and check with `uv lock --check`. (This happened during the
build: the `gpu` extra had resolved `jaxlib` 0.10.2 against a 0.8.2 pin.)

**My exported policy behaves differently from the JAX one.**
In order of likelihood: you forgot the observation normalizer; you used ReLU
instead of swish; you applied `tanh` to all 24 outputs instead of the first 12;
you sorted layer names as strings.

---

## ROS 2

**`/opt/ros/jazzy/setup.bash: AMENT_TRACE_SETUP_FILES: unbound variable`.**
Your script has `set -u`. ROS 2's setup scripts read unset variables on purpose.
Use `set -eo pipefail` without `-u`, or `set +u` around the `source`.

**`ModuleNotFoundError: No module named 'pup'` inside the container.**
`PYTHONPATH=/ws` should be set by the image. Check with `echo $PYTHONPATH`, and
that you started the container through `docker/compose.yaml` (which bind-mounts
the repository at `/ws`).

**`colcon build` says "no packages found".**
You are not in `ros2_ws/`. Build from `/ws/ros2_ws`, which contains `src/`.

**I edited a Python file but nothing changed.**
`--symlink-install` links files that existed at build time. New files, new entry
points and `setup.py` edits all need a real rebuild. Restarting the node is
enough for edits to existing files.

**The nodes cannot see each other.**
DDS discovery uses multicast, which does not traverse Docker's default bridge
network. Use `network_mode: host` (already in `compose.yaml`) and the same
`ROS_DOMAIN_ID` in every terminal.

**`ros2 topic list` shows the topic but `echo` prints nothing.**
QoS mismatch. `/pup/joint_states` and `/pup/imu` are published best-effort;
subscribe with `QoSPresetProfiles.SENSOR_DATA`, or
`ros2 topic echo /pup/imu --qos-reliability best_effort`.

**Do not source ROS 2 inside the uv venv.**
`rclpy` is built against the system interpreter. Mixing them produces import
errors that look like corruption. Two Pythons, two terminals.

**`ros2 topic hz /pup/joint_command` reports ~250 Hz.**
You wired the policy to the physics timer. It must run at 50 Hz.

**The robot twitches or falls immediately under the policy, but Stage 4's eval
was fine.**
In order of likelihood:
1. Quaternion convention — `sensor_msgs/Imu` is **`xyzw`**, MuJoCo is `wxyz`.
2. Joint ordering — reorder `JointState` by name into `JOINT_NAMES` order.
3. `last_action` holding joint targets in radians instead of the `[−1, 1]`
   action.
4. Publishing at the wrong rate.

Debug it by printing your ROS observation and the environment's observation for
the same pose, side by side, and diffing block by block. The layout string in
the exported `.npz` (`obs_layout`) tells you what each block should be.

**GUI passthrough doesn't work.**
Linux: `xhost +local:docker`, uncomment the `DISPLAY` and `/tmp/.X11-unix`
lines. macOS: XQuartz, "Allow connections from network clients", and
`xhost + 127.0.0.1`. Everything except the picture works headless; the eval
script needs no display at all.
