# Glossary

Terms in the order you meet them, not alphabetically. Skim it now; come back
when a word bites.

## Simulation

**`qpos`** — generalized positions. For Pup: 3 trunk position + 4 trunk
quaternion (`wxyz`) + 12 joint angles = 19.

**`qvel`** — generalized velocities. 3 linear + 3 angular + 12 joint rates = 18.
**It is one shorter than `qpos`** because a quaternion uses 4 numbers to encode
3 rotational degrees of freedom.

**`ctrl`** — actuator inputs, one per actuator. Meaning depends on the actuator:
torque (Nm) for `<motor>`, target angle (rad) for `<position>`.

**Generalized coordinates** — the minimal set of numbers describing a
mechanism's configuration, following its joint structure, rather than every
body's full 6-DoF pose.

**Free joint** — a 6-DoF joint connecting a body to the world. Pup's trunk has
one; that is what makes it a floating-base robot rather than a fixed arm.

**`mjModel` / `mjData`** — the model is constant (masses, geometry, gains); the
data is the state that changes each step. In MJX these become `mjx.Model` and
`mjx.Data`, which are JAX PyTrees.

**Keyframe** — a named saved state in the XML (`home`, `crouch`). Reset to one
with `mj_resetDataKeyframe`.

**`sim_dt` vs `ctrl_dt`** — physics timestep (0.004 s = 250 Hz) vs policy
timestep (0.02 s = 50 Hz). `n_substeps = ctrl_dt / sim_dt = 5`: the action is
held constant across five physics steps.

**Contact sensor** — a MuJoCo `<sensor><contact/></sensor>` that reports whether
two named geoms are touching. The supported way to read contacts in MJX;
inspecting `mjx.Data.contact` directly is not.

## Control

**PD control** — `tau = kp*(q_des − q) + kd*(qd_des − qd)`. `kp` is stiffness
(Nm/rad), `kd` is damping (Nm/(rad/s)).

**Torque control vs position control** — you send a torque, or you send a target
angle and something else computes the torque. Learned policies almost always
output position targets, because the PD loop underneath does the fast
stabilization the policy would otherwise have to learn.

**Steady-state error** — a pure PD controller holding a load against gravity
settles slightly below its target, because at zero error it produces zero force.
Real systems add gravity compensation or a feed-forward term.

**Projected gravity** — the world down-direction expressed in the body frame. A
unit 3-vector; `[0, 0, −1]` when level. How a legged robot knows which way is
up.

## JAX

**PyTree** — any nested structure of dicts/lists/tuples/dataclasses whose leaves
are arrays. JAX transforms operate leaf-wise on PyTrees.

**Tracing** — running your function once with abstract placeholders to record
the operations. Why `if x > 0` on an array fails inside `jit`.

**`jit`** — trace-and-compile. Recompiles whenever input shapes or dtypes
change, so keep them fixed.

**`vmap`** — auto-vectorize a single-example function over a batch axis.
`in_axes=0` batches an argument; `None` shares it.

**`scan`** — a compiled loop with a carried state. Traces the body once no
matter how many iterations run.

**`grad`** — reverse-mode automatic differentiation.

**Pure function** — same inputs, same outputs, no side effects. Required for
`jit` to be correct.

## Reinforcement learning

**Environment** — the object exposing `reset()` and `step(state, action)`.

**Observation** — what the policy sees. Here: 45 numbers, all of them things a
real robot could measure.

**Privileged information** — quantities available only in simulation (true base
velocity, contact forces, friction). Fine in rewards, evaluation, or a critic;
never in the policy's observation.

**Action** — the policy's output. Here: 12 numbers in `[−1, 1]`, scaled and
added to the default pose to become joint targets.

**Episode / rollout / horizon** — one run from reset to termination; the
recorded trajectory; the maximum length (1000 steps = 20 s here).

**Termination vs truncation** — the robot fell (a real failure, no bootstrapped
value) versus the episode hit the time limit (not a failure; the value function
must still bootstrap). Getting these confused biases the value function badly.

**Reward shaping** — adding terms besides the true objective to make learning
tractable. Every term below `tracking_lin_vel` in `config.py` is shaping.

**PPO** — Proximal Policy Optimization. On-policy; clips the policy update so
several gradient steps can be taken per batch.

**Advantage** — how much better an action was than the value function expected.
Estimated with GAE.

**Clip ratio** — the bound on how far one update may move the action
probabilities.

**Entropy cost** — a bonus for keeping the policy random, which keeps it
exploring.

**Observation normalization** — dividing observations by a running mean/std so
every input is O(1). Must be exported alongside the weights.

**Domain randomization** — perturbing friction, mass, gains and so on across
parallel environments so the policy cannot overfit to one exact robot. The
cheapest sim2real insurance there is.

**Sim2sim** — running a trained policy against a different simulator through the
deployment interfaces. **Sim2real** — running it on the robot.

## ROS 2

**Node** — a process doing one job.

**Topic** — a named, typed, many-to-many channel.

**Message** — the struct sent on a topic, defined in a `.msg` file.

**Parameter** — a named setting on a node, set from the CLI or a launch file.

**Launch file** — a Python script that starts several nodes with parameters.

**QoS (Quality of Service)** — per-topic delivery policy: reliability, history
depth, durability. Mismatched QoS between publisher and subscriber means
messages silently do not arrive.

**DDS** — the middleware underneath ROS 2. Discovers peers over multicast, which
is why Docker needs `network_mode: host`.

**`colcon`** — the build tool. `--symlink-install` links Python sources instead
of copying, so edits take effect without a rebuild.

**Workspace / overlay** — `ros2_ws/` with its `src/`; sourcing
`install/setup.bash` overlays your packages on top of `/opt/ros/jazzy`.

**`ament_cmake` vs `ament_python`** — build types. Message packages need the
former (code generation); pure-Python node packages use the latter.
