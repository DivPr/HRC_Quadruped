# Submission — HRC Software Onboarding Fall 2026

**Name:**
**Discord handle:**
**Repository:**
**Date:**

---

## Stage checklist

- [ ] **Stage 0** — Setup. `scripts/check_setup.py` exits 0.
- [ ] **Stage 1** — MuJoCo + PD controller. Tests green.
- [ ] **Stage 2** — JAX exercises. Tests green.
- [ ] **Stage 3** — MJX environment. Tests green.
- [ ] **Stage 4** — Brax PPO. Trained a policy, exported it, `NumpyPolicy` matches.
- [ ] **Stage 5** — ROS 2 sim2sim. Smoke test green, teleop GIF recorded.

## `scripts/progress.py` output

<details>
<summary>paste the full output here</summary>

```
$ uv run python scripts/progress.py --slow

(paste)
```

</details>

## Stage 4 — training results

**Config used:** (`full` / `t4_fast` / other) &nbsp; **Seed:** &nbsp;
**Where it ran:** (Colab T4 / local GPU / …) &nbsp; **Wall-clock:**

Learning curve: `results/learning_curve.png`
Rollout GIF: `results/training.gif`

```json
(paste results/eval_training.json)
```

Did it meet the acceptance criteria (`"walking_passes": true`)?

## Stage 5 — sim2sim results

Teleop GIF: `results/sim2sim_teleop.gif`

```json
(paste results/eval_sim2sim.json)
```

Measured `/pup/joint_command` rate from `ros2 topic hz`:

## Escape hatches used

- [ ] I used `checkpoints/pup_joystick_flat_reference.npz` for Stage 5 instead
      of my own policy.
- [ ] Other (describe):

*(Using one is fine. Not declaring one is not.)*

## Reflection — Stage 5

**List two ways sim2sim can pass while real hardware still fails, and what you
would add to the sim node to catch each.**

>

## What was hardest?

One paragraph. This is will help us improve onboarding.

>

## Time spent

| Stage | Hours |
|---|---|
| 0 Setup | |
| 1 MuJoCo + PD | |
| 2 JAX | |
| 3 MJX env | |
| 4 Brax + export | |
| 5 ROS 2 | |
| **Total** | |

---

**Then DM the software lead (Henry Tsay) on Discord or show during a meeting.**
