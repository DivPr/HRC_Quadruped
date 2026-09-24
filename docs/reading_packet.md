# Reading packet

**You do
not need to read all of this.** Each item is tagged with the stage it supports;
read those items when you reach that stage. The two marked ★ are the ones worth
reading properly.

Tags: **[S1]** MuJoCo/PD · **[S2]** JAX · **[S3]** MJX env · **[S4]** training ·
**[S5]** ROS 2 · **[BG]** background, not needed to finish

---

## Simulators and physics

**[S1]** **MuJoCo documentation — Overview and Computation.**
<https://mujoco.readthedocs.io/en/stable/overview.html>
Read "Overview" fully and skim "Computation". You want `mjModel` vs `mjData`,
generalized coordinates, and what a step does.

**[S1]** **MuJoCo XML reference (MJCF).**
<https://mujoco.readthedocs.io/en/stable/XMLreference.html>
A reference, not a read. Look up `<geom>`, `<joint>`, `<motor>`, `<position>`,
`<keyframe>` as needed.

**[S1][S3]** **MuJoCo sensors, including contact sensors.**
<https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor-contact>
The supported way to read contacts, and what `reduce` and `data` do.

**[S3]** **MJX documentation.**
<https://mujoco.readthedocs.io/en/stable/mjx.html>
Especially "Feature parity" and the performance-tuning notes — they explain why
the MJX scene has `iterations="1" ls_iterations="5"` and only four possible
contacts.

**[S3][S4]** ★ **MuJoCo Playground technical report.**
<https://arxiv.org/abs/2502.08844>
The single most relevant paper to what this club does. Read it.

**[S3][S4]** **MuJoCo Playground repository — `locomotion/go1/joystick.py` and
`learning/train_jax_ppo.py`.**
<https://github.com/google-deepmind/mujoco_playground>
Your Stage 3 and Stage 4 files are modelled on these two. They are installed
locally under `.venv/lib/python3.11/site-packages/mujoco_playground/`.

**[BG]** **MuJoCo Warp, and the Playground discussion of it.**
<https://github.com/google-deepmind/mujoco_warp>
Where MJX performance work is happening. Every environment here already accepts
`--impl warp`; this is the context for why that flag exists.

## JAX

**[S2]** ★ **JAX — "🔪 JAX — The Sharp Bits".**
<https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html>
Pure functions, immutability, control flow, PRNG keys. This is the one that
saves you the most time.

**[S2]** **JAX quickstart: `jit`, `vmap`, `grad`.**
<https://docs.jax.dev/en/latest/quickstart.html>

**[BG]** **`jax.lax.scan` documentation.**
<https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html>

## Reinforcement learning

**[S4]** **Proximal Policy Optimization Algorithms** — Schulman et al., 2017.
<https://arxiv.org/abs/1707.06347>
Read sections 1–4. The clipped surrogate objective is the whole idea.

**[S4]** **Brax — `brax/training`.**
<https://github.com/google/brax>
Note the project's own statement that only the `training` subdirectory is
actively maintained; the physics half has been superseded by MJX.

**[S3][S4]** **Learning Quadrupedal Locomotion over Challenging Terrain** —
Lee et al., 2020. <https://arxiv.org/abs/2010.11251>
Where the reward-term vocabulary you see in `rewards.py` comes from.

**[BG]** ★ **Sim-to-Real Learning of All Common Bipedal Gaits via Periodic
Reward Composition** — Siekmann et al., 2021.
<https://arxiv.org/abs/2011.01387>
The "important paper". Nothing here depends on it, and you should read it anyway
if you care about bipeds — which, on this team, you do. It is where the club's
thinking about gait rewards comes from.

**[BG]** **Advancing Humanoid Locomotion: Mastering Challenging Terrains with
Denoising World Model Learning** (DWL) — Gu et al., 2024.
<https://arxiv.org/abs/2408.14472>
Foot-height and terrain-adaptive reward design for humanoids.

## Legged control theory (background, not required)

**[BG]** **scaron.info — biped walking posts.**
<https://scaron.info/robotics/>
Stéphane Caron's notes on ZMP, capture point, and linear inverted pendulum
models. Excellent, readable, and the classical counterpoint to everything in
this repository.

**[BG]** **Bipedal Walking Control using ALIP and MPC** — Gong & Grizzle.
<https://arxiv.org/abs/2109.14862>
Model-predictive control on the angular-momentum LIP. What NEMO would do if it
weren't learning.

## ROS 2

**[S5]** **ROS 2 Jazzy — beginner CLI tools.**
<https://docs.ros.org/en/jazzy/Tutorials/Beginner-CLI-Tools.html>
Nodes, topics, `ros2 topic echo/hz`, `rqt_graph`. Ninety minutes, and it covers
most of what Stage 5 needs.

**[S5]** **Writing a simple publisher and subscriber (Python).**
<https://docs.ros.org/en/jazzy/Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Py-Publisher-And-Subscriber.html>

**[S5]** **Creating custom msg and srv files.**
<https://docs.ros.org/en/jazzy/Tutorials/Beginner-Client-Libraries/Custom-ROS2-Interfaces.html>
Read this to understand what `pup_interfaces` is and why it is its own package.
You will need it the first time you add a message to NEMO.

**[S5]** **Creating a launch file.**
<https://docs.ros.org/en/jazzy/Tutorials/Intermediate/Launch/Creating-Launch-Files.html>

**[S5]** **About Quality of Service settings.**
<https://docs.ros.org/en/jazzy/Concepts/Intermediate/About-Quality-of-Service-Settings.html>
Read this the moment a topic exists but delivers nothing.

> The links above point at **Jazzy**, matching `docker/ros2/Dockerfile`. If the
> club moves NEMO to a newer LTS, change the distro in the Dockerfile and the
> `jazzy` in these URLs together.
