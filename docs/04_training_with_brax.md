# Stage 4 — Training with Brax PPO

> **Start the Colab run before you read the rest of this page.** Open
> `notebooks/train_pup_colab.ipynb`, point the first cell at your fork, run all
> cells, and come back. It trains for tens of minutes while you do tasks 4 and 5
> on your laptop. The doc is written assuming you did this.

## Why this exists

You have an environment. Now you run the club's actual training loop on it. The
point is not to invent an algorithm — Brax's PPO is a good implementation and
you should use it as-is — the point is to understand the five or six knobs that
actually matter, to read a learning curve honestly, and to get the trained
weights *out* of JAX into something a robot can run.

That last part is underrated. A policy that only exists as a Brax checkpoint is
not deployable. Stage 5's ROS 2 node has no JAX, no GPU and a 20 ms budget. So
the deliverable here is a 200 KB `.npz` and thirty lines of numpy that reproduce
Brax's inference exactly.

## Concepts you need

**PPO in one paragraph.** Collect a batch of experience with the current policy;
estimate how much better each action was than average (the *advantage*, from a
learned value function via GAE); nudge the policy toward better-than-average
actions, but clip the update so no single batch moves the policy too far; repeat.
The clipping is the entire idea — it makes an on-policy method tolerate several
gradient steps per batch of data.

**The knobs, and what they actually do:**

| Parameter | Value | Meaning |
|---|---|---|
| `num_envs` | 8192 | robots simulated in parallel. Bigger = less gradient noise, more memory |
| `unroll_length` | 20 | steps collected per env per iteration → 163 840 transitions per batch |
| `batch_size` × `num_minibatches` | 256 × 32 | how that batch is chopped for SGD |
| `num_updates_per_batch` | 4 | gradient passes over the same data before throwing it away |
| `discounting` | 0.97 | at 50 Hz this is a ~0.7 s horizon. Locomotion is a short-horizon problem |
| `entropy_cost` | 1e-2 | keeps the policy stochastic so it keeps exploring. Too low → premature convergence to a limp |
| `learning_rate` | 3e-4 | the usual |
| `normalize_observations` | True | running mean/std over observations. **Must be exported with the weights** |
| `num_timesteps` | 200M | total environment steps |

**Where the numbers come from.** `ppo_params.full()` does not hard-code them —
it calls `locomotion_params.brax_ppo_config("Go1JoystickFlatTerrain")` from the
installed Playground and returns *that*, so the config cannot drift away from
upstream. On the currently pinned version that is 200M steps, 8192 envs, and
`(512, 256, 128)` hidden layers for both networks.

**The configs you can run:**

| Config | Steps | Envs | Where | Purpose |
|---|---|---|---|---|
| `cpu_smoke` | 20 k | 8 | your laptop | proves the plumbing runs. Reward is meaningless |
| `t4_fast` | 30 M | 2048 | Colab T4 | fallback if `full` doesn't fit your session |
| `full` | 200 M | 8192 | Colab T4 / local GPU | the real thing |
| `cpu_reference` | 60 M | 2048 | a laptop CPU, ~80 min | how `checkpoints/` was made |

> **Reading:** the PPO paper, Brax's `training/agents/ppo`, Playground's
> `learning/train_jax_ppo.py` ([reading packet](reading_packet.md), Stage 4).

## Your task

### 1. Complete the wiring block in `pup/train/train_ppo.py`

~8 lines inside one TODO block. Everything around it — argument parsing, CSV
logging of every `eval/episode_*` metric, checkpointing, and the final
`evaluate` + `render` calls — is provided. You need to:

- build a `PupJoystick` from the config;
- build a network factory: `functools.partial(ppo_networks.make_ppo_networks,
  **parameters.pop("network_factory"))`;
- call `ppo.train(...)` with `environment=`, `wrap_env_fn=wrapper.wrap_for_brax_training`,
  `randomization_fn=domain_randomize`, `network_factory=`, `seed=`,
  `progress_fn=progress`, `save_checkpoint_path=str(output / "checkpoints")`,
  `restore_checkpoint_path=` (see below), and `**parameters`.

`restore_checkpoint_path` is what makes `--restore runs/colab/checkpoints/<step>`
work, so pass
`str(Path(restore).resolve()) if restore else None`. Orbax rejects relative
paths, hence the `.resolve()`. If your Colab session dies at 120M steps, that
flag is the difference between losing an hour and losing none.

Read Playground's `learning/train_jax_ppo.py` alongside it — that is the file
this block is modelled on, and it is installed at
`.venv/lib/python3.11/site-packages/mujoco_playground/learning/train_jax_ppo.py`.

`wrap_for_brax_training` is doing real work: episode truncation, auto-reset, and
domain randomization per environment. Not using it is how you get an env that
trains for exactly one episode.

### 2. Run `cpu_smoke` locally

```bash
uv run python -m pup.train.train_ppo --config cpu_smoke --out runs/smoke --no-render
```

Under a minute. The reward will be garbage. What you are checking is that
`runs/smoke/` gains a `checkpoints/` directory, a `learning_curve.csv`, a
`policy.pkl` and an `eval.json`.

### 3. Run `full` on Colab (or a local GPU)

The notebook clones your fork, installs with `uv`, asserts the GPU is visible,
trains, plots the curve, renders a video, and zips `runs/` for download. Commit
into your repository under `results/`:

- `results/learning_curve.png`
- `results/eval_training.json`
- `results/training.gif`

**Acceptance criteria for "it walks"** (computed by `evaluate.py`, reported in
`eval.json` as `walking_passes`):

| Command | Requirement |
|---|---|
| `(1.0, 0, 0)` | mean \|v_x − 1.0\| < 0.2 m/s, mean \|v_y\| < 0.1 m/s, 0/5 falls |
| `(0, 0, 1.0)` | mean \|ω_z − 1.0\| < 0.3 rad/s |
| `(0, 0, 0)` | mean speed < 0.1 m/s |

Paste the JSON into `SUBMISSION.md`.

**Reading the curve.** `eval/episode_reward` climbing while
`eval/avg_episode_length` stays pinned at 1000 is what success looks like. If
episode length collapses, it is falling. If reward plateaus low with full-length
episodes, it has found a local optimum where standing still beats trying to
walk — usually a sign the tracking reward is being swamped.

### 4. `pup/train/export.py` — `export_policy(params, normalizer_params, out_path)`

Everything you need to know is stated outright, because walking a PyTree is the
skill here, not guessing a format:

- Brax gives you `params = (normalizer_params, policy_params, value_params)`.
- `normalizer_params.mean` and `.std` are both `(45,)`.
- `policy_params` is `{'params': {'hidden_0': {'kernel': (45, 512), 'bias': (512,)}, ...}}`.
  Layers are named `hidden_0 … hidden_N`; sort by that trailing integer, do not
  trust dict order.
- The last layer emits `2 * action_size = 24`: mean first, then std.
- Hidden activation is **swish** (`x * sigmoid(x)`); the last layer is linear.
- The deterministic action is `tanh(mean)`.

Look at the tree yourself:

```python
import jax
from brax.io import model
params = model.load_params("runs/colab/policy.pkl")
print(jax.tree_util.tree_map(lambda x: x.shape, params[1]))
```

Write an `.npz` with `obs_mean`, `obs_std`, `kernel_i`/`bias_i` in forward
order, and the metadata `n_layers`, `obs_size`, `action_size`, `action_scale`,
`default_pose`, `hidden_activation`, `obs_layout`. The metadata is not
bureaucracy — `obs_layout` is the string that lets a future you check, from the
file alone, that the ROS node is building the right vector.

### 5. `pup/policy/mlp_numpy.py` — `class NumpyPolicy`

Pure numpy. No `import jax` anywhere in this file or anything it imports; the
test will not catch that but the ROS 2 container will.

```python
NumpyPolicy.load(path) -> NumpyPolicy
policy(obs: np.ndarray[45]) -> np.ndarray[12]        # in [-1, 1]
policy.joint_targets(obs) -> np.ndarray[12]          # default_pose + action_scale * action, rad
```

Forward pass: normalize with `(obs − mean) / std`, then swish-activated hidden
layers, then a linear output layer, then `tanh` on the first 12 outputs.

## How you'll know you're done

```bash
uv run pytest tests/test_04_export_numpy.py -v      # fast
uv run pytest tests/test_04_train_smoke.py -v -m slow   # under a minute
```

`test_04_export_numpy.py` builds a real Brax policy, exports it, and requires
`|NumpyPolicy(obs) − brax_inference(obs)| < 1e-4` over **256 random
observations**. 1e-4 is tight on purpose: if you get the activation or the
normalizer wrong, you will pass a spot-check at one observation and fail here.

## Common mistakes

- **Forgetting the observation normalizer.** The most common export bug by a
  wide margin. Without it your actions are garbage but plausible-looking
  garbage, and you will blame Stage 5.
- **ReLU instead of swish.** Both give smooth-ish output. Only one matches.
- **Taking `tanh` of all 24 outputs.** The last 12 are the standard deviation.
- **Sorting layer names as strings.** `hidden_10` sorts before `hidden_2`. It
  doesn't bite at 5 layers; it will bite someone eventually.
- **Exporting the value network.** You don't need it at run time.
- **Expecting `cpu_smoke` to walk.** It cannot. It exists to prove the pipeline.
- **Colab disconnecting.** Checkpoints are written every eval; the notebook
  downloads the whole `runs/` directory. Don't close the tab.

## Escape hatch

`checkpoints/pup_joystick_flat_reference.npz` is a trained, exported policy. If
your training will not converge, use it for Stage 5 and **say so in
`SUBMISSION.md`**.