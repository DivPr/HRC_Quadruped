# Submission — HRC Software Onboarding Fall 2026

**Name: Divyansh Pramanick**
**Discord handle: divpr**
**Repository: HRC_Quadruped**
**Date: 9/25/26**

---

## Stage checklist

- [X] **Stage 0** — Setup. `scripts/check_setup.py` exits 0.
- [X] **Stage 1** — MuJoCo + PD controller. Tests green.
- [X] **Stage 2** — JAX exercises. Tests green.
- [X] **Stage 3** — MJX environment. Tests green.
- [X] **Stage 4** — Brax PPO. Trained a policy, exported it, `NumpyPolicy` matches.
- [X] **Stage 5** — ROS 2 sim2sim. Smoke test green, teleop GIF recorded.

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
{
  "seed": 0,
  "n_episodes": 5,
  "commands": [
    {
      "command": [
        0.5,
        0.0,
        0.0
      ],
      "mean_abs_error": [
        0.05475535339564085,
        0.041102223966037854,
        0.07793366386489652
      ],
      "mean_abs_vy": 0.041102223098278046,
      "mean_speed": 0.5356212854385376,
      "fall_rate": 0.0,
      "mean_episode_length": 1000.0
    },
    {
      "command": [
        1.0,
        0.0,
        0.0
      ],
      "mean_abs_error": [
        0.057604559803009034,
        0.04405815975208534,
        0.07267442484307103
      ],
      "mean_abs_vy": 0.04405815899372101,
      "mean_speed": 1.007077693939209,
      "fall_rate": 0.0,
      "mean_episode_length": 1000.0
    },
    {
      "command": [
        0.0,
        0.5,
        0.0
      ],
      "mean_abs_error": [
        0.03444756852104328,
        0.04398006879687309,
        0.06033352183692623
      ],
      "mean_abs_vy": 0.4859009385108948,
      "mean_speed": 0.4881531298160553,
      "fall_rate": 0.0,
      "mean_episode_length": 1000.0
    },
    {
      "command": [
        0.0,
        0.0,
        1.0
      ],
      "mean_abs_error": [
        0.04894600837427424,
        0.03645938830001396,
        0.060944178640842436
      ],
      "mean_abs_vy": 0.036459386348724365,
      "mean_speed": 0.06688454747200012,
      "fall_rate": 0.0,
      "mean_episode_length": 1000.0
    },
    {
      "command": [
        0.0,
        0.0,
        0.0
      ],
      "mean_abs_error": [
        0.036015412775363076,
        0.03653747468980873,
        0.09533157039500074
      ],
      "mean_abs_vy": 0.036537475883960724,
      "mean_speed": 0.05715639516711235,
      "fall_rate": 0.0,
      "mean_episode_length": 1000.0
    }
  ],
  "walking_passes": true
}

```

Did it meet the acceptance criteria (`"walking_passes": true`)?

Yes

## Stage 5 — sim2sim results

Teleop GIF: `results/sim2sim_teleop.gif`

```json
{
  "duration_s": 20.0,
  "min_trunk_height": 0.2890430756311453,
  "fell": false,
  "commands": [
    {
      "command": [
        0.5,
        0.0,
        0.0
      ],
      "n_samples": 151,
      "mean_velocity": [
        0.4123076407252784,
        0.009521364327305765,
        -0.013366876474545746
      ],
      "mean_abs_error": [
        0.08769235927472158,
        0.009521364327305765,
        0.013366876474545746
      ],
      "mean_speed": 0.41265564838921,
      "threshold": 0.25,
      "passed": true
    },
    {
      "command": [
        1.0,
        0.0,
        0.0
      ],
      "n_samples": 150,
      "mean_velocity": [
        0.8977797348345914,
        0.03136696075311743,
        -0.02063935479178386
      ],
      "mean_abs_error": [
        0.10222026516540861,
        0.03136696075311743,
        0.02063935479178386
      ],
      "mean_speed": 0.8985635063146062
    },
    {
      "command": [
        0.0,
        0.0,
        1.0
      ],
      "n_samples": 150,
      "mean_velocity": [
        -0.03392524645033148,
        0.015469350766372372,
        0.9571070696983194
      ],
      "mean_abs_error": [
        0.03392524645033148,
        0.015469350766372372,
        0.04289293030168062
      ],
      "mean_speed": 0.04229662659102842
    },
    {
      "command": [
        0.0,
        0.0,
        0.0
      ],
      "n_samples": 151,
      "mean_velocity": [
        -2.5591037769556762e-05,
        -0.0001885054794334371,
        -0.00011395237403962418
      ],
      "mean_abs_error": [
        2.5591037769556762e-05,
        0.0001885054794334371,
        0.00011395237403962418
      ],
      "mean_speed": 0.00019100586290358746
    }
  ],
  "passed": true
}

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
| 0 Setup | .5 hours|
| 1 MuJoCo + PD |2 hours|
| 2 JAX |1 hour|
| 3 MJX env |2 hours|
| 4 Brax + export |5 hours|
| 5 ROS 2 |1.5|
| **Total** |12 hours|

---

**Then DM the software lead (Henry Tsay) on Discord or show during a meeting.**
