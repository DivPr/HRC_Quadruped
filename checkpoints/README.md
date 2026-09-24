# Checkpoints — the Stage 5 escape hatch

## What's here

`pup_joystick_flat_reference.npz` — a trained, exported Pup joystick policy, in
the format `pup/train/export.py` produces and `pup/policy/mlp_numpy.py` loads.

## Why it exists

**No stage may be a dead end.** Stage 5 (ROS 2) is worth doing even if Stage 4
(training) did not converge for you — the two teach completely different things,
and getting stuck on one should never cost you the other.

So if your own policy doesn't walk: use this one.

```bash
ros2 launch pup_bringup sim2sim.launch.py \
  policy_path:=/ws/checkpoints/pup_joystick_flat_reference.npz
```

That is the default path in the launch file, so this is also what happens if you
pass nothing.

**Say so in `SUBMISSION.md`.** There is a checkbox for it. Declaring an escape
hatch costs you nothing; quietly pretending you trained it costs you the
reviewer's trust, and they can tell — your `results/` has to contain a learning
curve and an eval JSON from your own run either way.

## Using your own instead

```bash
uv run python -m pup.train.export \
  --checkpoint runs/colab/policy.pkl \
  --out results/my_policy.npz
```

Then point `policy_path` at it. Commit it — it is a couple of hundred kilobytes.

## What's inside the file

```python
import numpy as np
archive = np.load("checkpoints/pup_joystick_flat_reference.npz")
print(archive.files)
print(str(archive["obs_layout"]))
```

`obs_mean` / `obs_std` (the observation normalizer), `kernel_i` / `bias_i` for
each MLP layer in forward order, and metadata: `n_layers`, `obs_size`,
`action_size`, `action_scale`, `default_pose`, `hidden_activation`,
`obs_layout`.

`obs_layout` is the important one. It is the authoritative statement of what the
45 numbers mean, travelling inside the file with the weights, so that a policy
and a robot can never quietly disagree about it.
